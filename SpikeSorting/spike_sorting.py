"""
Neuroplatform Utils - Utility functions for the Neuroplatform project.

This is intended to be a plug-and-play utility module for V1, without any changes to the main Neuroplatform codebase.

SpikeSorting : Spike sorting utility functions. Author : Cyril Achard, based on code by Gregoio Rebecchi, August 2024
"""
import numpy as np
import pandas as pd
import multiprocessing
from datetime import timedelta
from tqdm import tqdm
from sklearn.cluster import OPTICS, HDBSCAN
from joblib import Parallel, delayed, parallel_config
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import ListedColormap
from IPython.display import display
import logging

from neuroplatform import Database

COLORBLIND_MAP = ListedColormap(
    [
        "#006BA4",
        "#FF800E",
        "#ABABAB",
        "#595959",
        "#5F9ED1",
        "#C85200",
        "#898989",
        "#A2C8EC",
        "#FFBC79",
        "#CFCFCF",
    ]
)
try:
    import seaborn as sns

    sns.set_palette(COLORBLIND_MAP.colors)
except ImportError:
    pass


### Spike Sorting

class SpikeSortingNotRun(ValueError):
    def __init__(self, message="Please run the spike sorting algorithm first.", *args):
        super().__init__(message, *args)
        self.message = message

class DimensionalityReduction:
    def __init__(self, method_name: str, n_components: int = None, **kwargs):
        """
        Small wrapper class to handle dimensionality reduction methods and their results.

        Args:
            method_name (str): Name of the method to use. Must be either "PCA" or "ICA".
            n_components (int): Number of components to keep after dimensionality reduction.
            **kwargs: Additional keyword arguments to pass to the model.

        Attributes:
            scaler (StandardScaler): Scaler to use for normalization or preprocessing. If None, no preprocessing is done.
            dimred_data (np.array): Transformed data.
            model_components (np.array): Array of components from the model.
            explained_variance (np.array): Explained variance. Only available for PCA.
            fit_data_df (pd.DataFrame): DataFrame containing the transformed data.
        """
        self._n_components = n_components
        self._method_name = None
        self._model = None
        ####
        self.scaler = None  # used for normalization or preprocessing
        self.dimred_data = None
        self.model_components = None
        self.explained_variance = None
        self.fit_data_df = None
        self._method_kwargs = kwargs

        self.method_name = method_name

        if self.method_name == "PCA":
            if self._n_components is None:
                self.n_components = 3
                try:
                    self._model = PCA(n_components=n_components, **kwargs)
                except TypeError:
                    self._model = PCA(n_components=n_components)
        elif self.method_name == "ICA":
            if self._n_components is None:
                self.n_components = 3
            try:
                self._model = FastICA(n_components=n_components, whiten_solver="eigh")
            except TypeError:
                self._model = FastICA(n_components=n_components, whiten_solver="eigh")
        else:
            # possibility to add any other method here later
            # by setting self._model = model(n_components) or something similar
            raise ValueError("dimred_method must be either 'PCA' or 'ICA'")

    @property
    def method_name(self):
        return self._method_name

    @method_name.setter
    def method_name(self, method, **kwargs):
        """Sets the method to use for dimensionality reduction.

        Args:
            method (str): Name of the method to use. Must be either "PCA" or "ICA".
            **kwargs: Additional keyword arguments to pass to the model.
        """
        if method not in ["PCA", "ICA"]:
            raise ValueError("method_name must be either 'PCA' or 'ICA'")
        self._method_name = method
        model = PCA if method == "PCA" else FastICA
        try:
            self._model = model(n_components=self.n_components, **kwargs)
        except TypeError:
            self._model = model(n_components=self.n_components)

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, n):
        if n is not None:
            if n < 1:
                raise ValueError("n_components must be greater than 0")
        self._n_components = n
        model = PCA if self.method_name == "PCA" else FastICA
        if self._method_kwargs is not None:
            self._model = model(n_components=n, **self._method_kwargs)
        else:
            self._model = model(n_components=n)

    @property
    def model(self):
        return self._model

    def fit(self, data):
        if self.scaler is not None:
            data = self.scaler.fit_transform(data)

        self.dimred_data = self.model.fit_transform(data)

        self.model_components = self._model.components_
        if self.method_name == "PCA":
            self.explained_variance = self.model.explained_variance_ratio_

        self.fit_data_df = pd.DataFrame(
            self.dimred_data,
            columns=[f"C{i+1}" for i in range(self.dimred_data.shape[1])],
        )
        return self.dimred_data

class MergedSpikeSorting:
    """Class to merge the results from two SpikeSorting instances."""

    def __init__(
        self,
        s1: "SpikeSorting",
        s2: "SpikeSorting",
        recompute_dimred: bool = False,
        dimred_method: str = "ICA",
        n_components: int = None,
    ):
        """
        Merges the results from two SpikeSorting instances.

        Args:
            s1 (SpikeSorting): First SpikeSorting instance.
            s2 (SpikeSorting): Second SpikeSorting instance.
            recompute_dimred (bool): Whether to recompute the dimensionality reduction. Default is False.
            dimred_method (str): Dimensionality reduction method to use. Must be either "PCA" or "ICA". Default is "PCA".
            n_components (int): Number of components to keep after dimensionality reduction. Default is 3.

        Attributes:
            s1 (SpikeSorting): First SpikeSorting instance.
            s2 (SpikeSorting): Second SpikeSorting instance.
            processed_spike_events_df (pd.DataFrame): DataFrame containing the spike events with clustering results.
            raw_spikes_df (pd.DataFrame): DataFrame containing the raw spikes.
        """
        self.s1 = s1
        self.s2 = s2
        self.dimred = DimensionalityReduction(
            method_name=dimred_method, n_components=n_components
        )

        if s1.processed_spike_events_df is None or s2.processed_spike_events_df is None:
            raise ValueError("Both SpikeSorting instances must have been run.")
        if s1.raw_spikes_df is None or s2.raw_spikes_df is None:
            raise ValueError("Both SpikeSorting instances must have been run.")
        if (
            s1.dimred_method_spikes.fit_data_df is None
            or s2.dimred_method_spikes.fit_data_df is None
        ):
            raise ValueError("No clustering results found.")

        spikes_s1 = s1.raw_spikes_df.copy()
        spikes_s2 = s2.raw_spikes_df.copy()
        self.raw_spikes_df = self._merge_dfs(spikes_s1, spikes_s2)

        events_s1 = s1.processed_spike_events_df.copy()
        events_s2 = s2.processed_spike_events_df.copy()
        self.processed_spike_events_df = self._merge_dfs(events_s1, events_s2)

        if recompute_dimred:
            self._fit_dimred()
            self._add_clustering_to_dimred_df()
        else:
            s1_dimred = s1.dimred_method_spikes.fit_data_df.copy()
            s2_dimred = s2.dimred_method_spikes.fit_data_df.copy()
            self.dimred.fit_data_df = self._merge_dfs(s1_dimred, s2_dimred)

    def _merge_dfs(self, df1, df2, keep_artifacts=False):
        df1["Sorter_id"] = 1
        df2["Sorter_id"] = 2
        df = pd.concat([df1, df2], ignore_index=True)
        if "is_artifact" in df.columns:
            df = df[~df["is_artifact"]] if not keep_artifacts else df
        return df

    def _fit_dimred(self):
        spike_df = self.raw_spikes_df.copy()
        spike_df = spike_df.groupby("Event_time")["Amplitude"].apply(list)
        spike_matrix = np.array(spike_df.tolist())
        self.dimred.fit(spike_matrix)

    def _add_clustering_to_dimred_df(self):
        self.dimred.fit_data_df = pd.merge(
            left=self.dimred.fit_data_df,
            right=self.processed_spike_events_df["Sorter_id"],
            left_index=True,
            right_index=True,
        )
        self.dimred.fit_data_df = pd.merge(
            left=self.dimred.fit_data_df,
            right=self.processed_spike_events_df["Cluster"],
            left_index=True,
            right_index=True,
        )

    def plot_clustering_comparison_in_latent_space(self, color_clusters=True):
        """Plots the clustering results in the latent space for both SpikeSorting instances.

        Returns:
            If show is False, returns the plotly figure.
        """
        if self.dimred.fit_data_df is None:
            raise ValueError("No clustering results found.")
        try:
            import plotly.express as px

            data = self.dimred.fit_data_df.copy()
            data.Cluster = data.Cluster.astype(str)
            data.Sorter_id = data.Sorter_id.astype(str)
            color = "Cluster" if color_clusters else "Sorter_id"
            symbol = "Sorter_id" if color_clusters else "Cluster"
            fig = px.scatter_3d(
                data,
                x="C1",
                y="C2",
                z="C3",
                color=color,
                symbol=symbol,
                opacity=0.7,
                title="Clustering comparison in latent space",
            )
            fig.show()
        except ImportError:
            raise ImportError("The plotly package is required to run this method.")

    def plot_clustered_spikes_comparison(
        self, palette="colorblind", show_outliers=True
    ):
        """Plots the clustered spikes for both SpikeSorting instances.

        Args:
            palette (str): Palette to use for the plot. Default is 'colorblind'.
        """
        if self.raw_spikes_df is None:
            raise ValueError("No raw spikes to plot.")
        if "Cluster" not in self.raw_spikes_df.columns:
            raise ValueError("No clustering results found.")
        if show_outliers:
            min_label = -1
        else:
            min_label = 0
        plt.figure(figsize=(10, 6))
        title = "Clustered spikes comparison"
        plt.title(title)

        data = self.raw_spikes_df[self.raw_spikes_df["Cluster"] >= min_label].copy()
        sns.lineplot(
            data=data,
            x="Spike_time",
            y="Amplitude",
            hue="Cluster",
            style="Sorter_id",
            palette=palette,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (uV)")
        plt.show()

class SpikeSorting:
    try:
        import seaborn as sns
    except ImportError:
        raise ModuleNotFoundError(
            "Please install Seaborn to use the SpikeSorting class. Make sure to reload your kernel after installing."
        )

    def __init__(
        self,
        realign: bool = True,
        time_before: float = 0.5,
        time_after: float = 1,
        artifact_threshold: float = 750.0,
        filter_artifacts: bool = True,
        dimred_method: str = "PCA",
        n_components: int = None,
        cluster_method: str = "HDBSCAN",
        clustering_kwargs: dict = None,
        n_jobs: int = 8,  # multiprocessing.cpu_count() # using -1 is by far the fastest, but can fail in IPython
    ):
        """
        Spike sorting algorithm using PCA followed by clustering (HDBSCAN or OPTICS).

        Args:
            realign (bool): Whether to realign spikes based on amplitude. Default is True.
            time_before (float): Time before the event to include in the window (in ms). Default is 0.5.
            time_after (float): Time after the event to include in the window (in ms). Default is 1.
            artifact_threshold (float): Threshold to detect artifacts (in uV). Default is 1000.
            filter_artifacts (bool): Whether to filter out artifacts. Default is True.
            dimred_method (str): Dimensionality reduction method to use. Must be either "PCA" or "ICA". Default is "PCA".
            n_components (int): Number of components to keep after dimensionality reduction. Default is 3.
            cluster_method (str): Clustering method to use. Must be either "HDBSCAN" or "OPTICS". Default is "HDBSCAN".
            clustering_kwargs (dict): Additional keyword arguments to pass to the clustering method. Default is None.
            n_jobs (int): Number of jobs to run in parallel. Default is the number of CPUs.

        Attributes:
            fs_id (str): ID of the experiment ('fs#ID').
            electrode_id (int): Electrode number.
            realign (bool): Whether to realign spikes based on amplitude.
            time_before (float): Time before the event to include in the window (in ms).
            time_after (float): Time after the event to include in the window (in ms).
            filter_artifacts (bool): Whether to filter out artifacts.
            artifact_threshold (float): Threshold to detect artifacts (in uV).
            n_jobs (int): Number of jobs to run in parallel.
            n_components (int): Number of components to keep after dimensionality reduction.
            raw_spikes_df (pd.DataFrame): DataFrame containing the raw spikes.
            processed_spike_events_df (pd.DataFrame): DataFrame containing the spike events with clustering results.
            logger (logging.Logger): Logger to use for logging.
        """
        ### Experiment parameters
        self.fs_id = None
        self.electrode_id = None
        ### Processing parameters
        self.realign = realign
        self.time_before = time_before
        self.time_after = time_after
        self.filter_artifacts = filter_artifacts  # removes spikes with amplitude above threshold, BOTH stimulations and artifacts
        self.artifact_threshold = artifact_threshold
        self.n_jobs = n_jobs
        ### I/O attributes
        self._db = Database()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        ### Dimensionality reduction attributes
        self._n_components = n_components
        self.dimred_method_spikes = DimensionalityReduction(  # spikes only
            method_name=dimred_method, n_components=n_components
        )
        self.dimred_method_artifacts = DimensionalityReduction(  # artifacts only
            method_name=dimred_method, n_components=n_components
        )
        ### Clustering attributes
        self._start_timestamp = None
        self._stop_timestamp = None
        self._clustering_model = None  # clustering model for spikes
        self._clustering = None  # clustering data
        self._clustering_model_artifacts = None  # clustering model for artifacts
        self._clustering_artifacts = None  # clustering data for artifacts
        self._hdbscan_default_min_cluster_size: int = 20
        self._hdbscan_default_min_samples: int = 5
        self._optics_default_min_samples = 5
        self._optics_default_xi = 0.075
        self._optics_default_min_cluster_size = 0.05
        ## Processed data
        self.raw_spikes_df = None
        self.processed_spike_events_df = None
        self._query_spike_events = None

        if cluster_method not in ["HDBSCAN", "OPTICS"]:
            raise ValueError("Cluster_method must be either 'HDBSCAN' or 'OPTICS'")
        if clustering_kwargs is not None:
            cluster_method = HDBSCAN if cluster_method == "HDBSCAN" else OPTICS
            self._init_clustering(cluster_method, clustering_kwargs)
        elif cluster_method == "HDBSCAN":
            self._clustering_model = HDBSCAN(
                min_cluster_size=self._hdbscan_default_min_cluster_size,
                min_samples=self._hdbscan_default_min_samples,
            )
            self._clustering_model_artifacts = HDBSCAN(
                min_cluster_size=self._hdbscan_default_min_cluster_size,
                min_samples=self._hdbscan_default_min_samples,
            )
        elif cluster_method == "OPTICS":
            self._clustering_model = OPTICS(
                min_samples=self._optics_default_min_samples,
                xi=self._optics_default_xi,
                min_cluster_size=self._optics_default_min_cluster_size,
            )
            self._clustering_model_artifacts = OPTICS(
                min_samples=self._optics_default_min_samples,
                xi=self._optics_default_xi,
                min_cluster_size=self._optics_default_min_cluster_size,
            )

        # check time_before and time_after
        if time_before < 0 or time_after < 0:
            raise ValueError("time_before and time_after must be positive")
        # check n_jobs
        if n_jobs < 1:
            raise ValueError(
                f"n_jobs must be between 1 and the number of CPUs({multiprocessing.cpu_count()}). Got {n_jobs}"
            )
        elif n_jobs > multiprocessing.cpu_count():
            self.n_jobs = max(multiprocessing.cpu_count() // 2, 8)
            self.logger.info(
                f"n_jobs set to {self.n_jobs} to avoid sending too many requests the system"
            )

        if artifact_threshold <= 500:
            warnings.warn(
                "The artifact threshold is low enough that it may remove some real spikes. Consider increasing it."
            )

        if cluster_method not in ["HDBSCAN", "OPTICS"]:
            raise ValueError("cluster_method must be either 'HDBSCAN' or 'OPTICS'")

    @property
    def n_components(self):
        return self._n_components

    @n_components.setter
    def n_components(self, n):
        if n is not None:
            if n < 1:
                raise ValueError("n_components must be greater than 0")
        self.logger.info(f"Setting n_components to {n} for spikes and artifacts.")
        self._n_components = n
        self.dimred_method_spikes.n_components = n
        self.dimred_method_artifacts.n_components = n

    @property
    def dimred_method(self):
        """
        Returns the dimensionality reduction method(s) used for spikes and artifacts.

        If the same method is used for both spikes and artifacts, returns a single method name.
        If different methods are used, returns a tuple of method names.

        Returns:
            str or tuple: The method name(s) for dimensionality reduction.
        """
        self.logger.info(f"Method for spikes: {self.dimred_method_spikes.method_name}")
        self.logger.info(
            f"Method for artifacts: {self.dimred_method_artifacts.method_name}"
        )

        if (
            self.dimred_method_spikes.method_name
            == self.dimred_method_artifacts.method_name
        ):
            return self.dimred_method_spikes.method_name

        return (
            self.dimred_method_spikes.method_name,
            self.dimred_method_artifacts.method_name,
        )

    @dimred_method.setter
    def dimred_method(self, method):
        if method not in ["PCA", "ICA"]:
            raise ValueError("dimred_method must be either 'PCA' or 'ICA'")
        self.logger.info(
            f"Setting dimensionality reduction method to {method} for spikes and artifacts."
        )
        self.dimred_method_spikes.method_name = method
        self.dimred_method_artifacts.method_name = method

    @classmethod
    def create_spike_sorting(
        cls, start_timestamp, stop_timestamp, fs_id, electrode_nb, **kwargs
    ):
        """Runs the spike sorting algorithm.

        Args:
            start_timestamp (datetime): Start timestamp of the recording.
            stop_timestamp (datetime): Stop timestamp of the recording.
            fs_id (str): ID of the experiment ('fs#ID')
            electrode_nb (int): Electrode number.
            **kwargs: Additional keyword arguments to pass to the SpikeSorting class.

        Returns:
            SpikeSorting: SpikeSorting instance.
        """
        spike_sorting = cls(**kwargs)
        spike_sorting.electrode_id = electrode_nb
        spike_sorting.fs_id = fs_id
        spike_sorting.run_spike_sorting(
            start_timestamp, stop_timestamp, fs_id, electrode_nb
        )
        return spike_sorting

    def __add__(self, other: "SpikeSorting") -> MergedSpikeSorting:
        """Merges two SpikeSorting instances."""
        if not isinstance(other, SpikeSorting):
            raise ValueError("Can only merge two SpikeSorting instances.")
        return MergedSpikeSorting(self, other)

    def run_spike_sorting(
        self, start_timestamp, stop_timestamp, electrode_nb, fs_id=None, exist_ok=False
    ):
        """Runs the spike sorting algorithm.

        Args:
            start_timestamp (datetime): Start timestamp of the recording.
            stop_timestamp (datetime): Stop timestamp of the recording.
            electrode_nb (int): Electrode number.
            fs_id (str): ID of the experiment ('fs#ID'). If None, will attempt to query the spike events by electrode number only.
            exist_ok (bool): If True, will not re-run the spike event fetching if the time window is the same. Default is False.

        Returns:
            pd.DataFrame: DataFrame containing the spike events with clustering results.
        """

        if stop_timestamp < start_timestamp:
            raise ValueError("stop_timestamp must be greater than start_timestamp")

        self.logger.info("Running spike sorting algorithm")
        # TODO fix fs_id being None

        recompute = True

        def check_results_exist():
            if self.processed_spike_events_df is not None:
                if not self.processed_spike_events_df.empty:
                    self.logger.info(
                        "Spike sorting already run for this time window. Skipping."
                    )
                    return True
            return False

        if exist_ok:
            if (
                self._start_timestamp == start_timestamp
                and self._stop_timestamp == stop_timestamp
                and self.electrode_id == electrode_nb
                and (fs_id is None or self.fs_id == fs_id)
            ):
                recompute = not check_results_exist()

        self._start_timestamp = start_timestamp
        self._stop_timestamp = stop_timestamp
        self.electrode_id = electrode_nb
        self.fs_id = fs_id
        if self.raw_spikes_df is None or self.raw_spikes_df.empty or recompute:
            self.processed_spike_events_df = None
            self.logger.info(
                f"Fetching spike events between {start_timestamp} and {stop_timestamp}..."
            )
            try:
                self.processed_spike_events_df = self._db.get_spike_event_electrode(
                    start_timestamp, stop_timestamp, electrode_nb
                )
            except AttributeError:
                if fs_id is None:
                    raise ValueError(
                        "Fetching spike events by electrode number is not supported. Please provide fs_id."
                        "Please request the get_spike_event_electrode method to be added to your current neuroplatform version if you cannot provide fs_id."
                    )
                self.processed_spike_events_df = self._db.get_spike_event(
                    start_timestamp, stop_timestamp, fs_id
                )
                self.processed_spike_events_df = self.processed_spike_events_df[
                    self.processed_spike_events_df["channel"] == electrode_nb
                ]

            self._query_spike_events = self.processed_spike_events_df.copy()

            if (
                self.processed_spike_events_df is None
                or self.processed_spike_events_df.empty
            ):
                warnings.warn("No spike events found. Exiting.")
                return
            if len(self.processed_spike_events_df) > 1e5:
                self.logger.warning(
                    "A lot of spike events were queried. Consider narrowing the time window."
                )
            if len(self._query_spike_events) > 1e6:
                raise ValueError(
                    "Too many spike events queried. Please narrow the time window."
                )
            self.logger.info(f"Extracting raw spikes for electrode {electrode_nb}...")
            self.raw_spikes_df = self._process_spikes_parallel()
        else:
            self.logger.info("Raw spikes already processed. Skipping.")

        if len(self.processed_spike_events_df) < 100:
            self.logger.warning(
                "Few spike events found. Consider increasing the time window or choosing an electrode with more activity."
            )
        if len(self.processed_spike_events_df) < 10:
            self.logger.error(
                "Too few spike events found, consider changing the time window or choosing another electrode with more activity."
            )
            self.logger.info("All events will be considered as outliers.")


        self._fit_dimred()
        # fit dimred to artifacts
        self._fit_dimred(on_artifacts=True)

        self.logger.info("Fitting clustering model...")
        self._fit_clustering()
        self._fit_clustering(on_artifacts=True)

        self.logger.info("Recording clustering results...")
        if len(self.processed_spike_events_df) < 10:
            Clustering = type("Clustering", (), {"labels_": np.zeros(len(self.processed_spike_events_df)) - 1})
            self._clustering = Clustering()
        self._add_clustering_to_raw_df()
        self._add_clustering_to_event_df()
        self._add_clustering_to_dimred_df()

        self.logger.info("Done.")
        return self.processed_spike_events_df

    def plot_explained_variance(self):
        """Plots the explained variance of the PCA."""
        if self.dimred_method_spikes.explained_variance is None:
            raise ValueError(
                "No explained variance to plot. Make sure the spike sorting was run using PCA."
            )
        plt.figure(figsize=(10, 6))
        x = range(1, len(self.dimred_method_spikes.explained_variance) + 1)
        plt.plot(x, self.dimred_method_spikes.explained_variance, marker="o")
        plt.axvline(
            self.dimred_method_spikes.n_components, color="red", linestyle="--", lw=0.5
        )
        plt.legend(["Explained variance", "Selected components"])
        plt.xticks(x)
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.title("Explained variance of the PCA")
        plt.show()

    def plot_spike_clustering_in_latent_space(self, show=True):
        """Plots the clustering results in the latent space.

        Args:
            show (bool): If True, shows the plot. Default is True.

        Returns:
            If show is False, returns the plotly figure.
        """
        plot = self._plot_3d_cluster(self.dimred_method_spikes.fit_data_df, show)
        if plot is not None:
            return plot

    def plot_artifact_clustering_in_latent_space(self, show=True):
        """Plots the clustering results in the latent space for artifacts only.

        Args:
            show (bool): If True, shows the plot. Default is True.

        Returns:
            If show is False, returns the plotly figure.

        """
        if self.dimred_method_artifacts.fit_data_df is None:
            self.logger.warning(
                "No artifact data to plot. It is possible that there are too few artifacts to properly reduce dimensionality."
            )
            return
        plot = self._plot_3d_cluster(self.dimred_method_artifacts.fit_data_df, show)
        if plot is not None:
            return plot

    def plot_clustered_spikes(self, palette="colorblind", show_outliers=True):
        """Plots the clustered spikes.

        Args:
            palette (str): Palette to use for the plot. Default is 'colorblind'.
        """
        if self.raw_spikes_df is None:
            self.logger.error("No raw spikes to plot.")
            raise SpikeSortingNotRun()
        if "Cluster" not in self.raw_spikes_df.columns:
            self.logger.error("No clustering results found.")
            raise SpikeSortingNotRun()
        if show_outliers:
            min_label = -1
        else:
            min_label = 0
        plt.figure(figsize=(10, 6))
        title = f"Clustered spikes between {self._start_timestamp} and {self._stop_timestamp}"
        plt.title(title)

        data = self.raw_spikes_df[self.raw_spikes_df["Cluster"] >= min_label].copy()
        sns.lineplot(
            data=data,
            x="Spike_time",
            y="Amplitude",
            hue="Cluster",
            palette=palette,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (uV)")
        plt.show()

    def plot_clustered_artifacts(self, palette="colorblind"):
        """Plots the clustered artifacts.

        Args:
            palette (str): Palette to use for the plot. Default is 'colorblind'.
        """
        if self.raw_spikes_df is None:
            self.logger.error("No raw spikes to plot.")
            raise SpikeSortingNotRun()
        if "Cluster" not in self.raw_spikes_df.columns:
            self.logger.error("No clustering results found.")
            raise SpikeSortingNotRun()
        data = self.raw_spikes_df[self.raw_spikes_df["Cluster"] < -1].copy()
        if data.empty:
            self.logger.warning("No artifacts found.")
            return
        plt.figure(figsize=(10, 6))
        title = f"Clustered artifacts between {self._start_timestamp} and {self._stop_timestamp}"
        plt.title(title)
        sns.lineplot(
            data=data,
            x="Spike_time",
            y="Amplitude",
            hue="Cluster",
            palette=palette,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (uV)")
        plt.show()

    def plot_raw_outlier_spikes(
        self, show_artifacts=False, use_alpha_scaling=False, use_size_scaling=False
    ):
        """Plots all the outlier spikes or artifacts.

        Args:
            palette (str): Palette to use for the plot. Default is 'colorblind'.
            show_artifacts (bool): If True, shows the artifacts as well as the outliers. Default is False.
            use_alpha_scaling (bool): If True, scales the alpha value based on the number of spikes. Default is False.
            use_size_scaling (bool): If True, scales the size of the spikes based on the number of spikes. Default is False.
        """
        # REQUIRES seaborn, which is not installed by default
        if show_artifacts:
            data = self.raw_spikes_df[self.raw_spikes_df["Cluster"] <= -1]
        else:
            data = self.raw_spikes_df[self.raw_spikes_df["Cluster"] == -1]
        self._plot_raw_spikes(
            data, use_alpha_scaling=use_alpha_scaling, use_size_scaling=use_size_scaling
        )

    def plot_raw_spikes_for_cluster(
        self, cluster, use_alpha_scaling=True, use_size_scaling=True
    ):
        """Plots the raw spikes for a given cluster.

        Args:
            cluster (int): Cluster to plot.
            use_alpha_scaling (bool): If True, scales the alpha value based on the number of spikes. Default is False.
            use_size_scaling (bool): If True, scales the size of the spikes based on the number of spikes. Default is False.
        """
        if self.raw_spikes_df is None:
            self.logger.error("No raw spikes to plot.")
            raise SpikeSortingNotRun()
        if cluster not in self.raw_spikes_df["Cluster"].unique():
            raise ValueError(f"Cluster {cluster} not found in the clustering results.")
        data = self.raw_spikes_df[self.raw_spikes_df["Cluster"] == cluster]
        self._plot_raw_spikes(
            data, use_alpha_scaling=use_alpha_scaling, use_size_scaling=use_size_scaling
        )

    def plot_raw_spikes_for_clusters(
        self, clusters, use_alpha_scaling=True, use_size_scaling=True
    ):
        """Plots the raw spikes for a given cluster.

        Args:
            clusterd (list): List of clusters to plot.
            use_alpha_scaling (bool): If True, scales the alpha value based on the number of spikes. Default is False.
            use_size_scaling (bool): If True, scales the size of the spikes based on the number of spikes. Default is False.
        """
        if self.raw_spikes_df is None:
            self.logger.error("No raw spikes to plot.")
            raise SpikeSortingNotRun()
        if not all([c in self.raw_spikes_df["Cluster"].unique() for c in clusters]):
            raise ValueError(
                f"A cluster was not found in the clustering results. Available : {self.raw_spikes_df['Cluster'].unique()}"
            )
        data = self.raw_spikes_df[self.raw_spikes_df["Cluster"].isin(clusters)]
        self._plot_raw_spikes(
            data, use_alpha_scaling=use_alpha_scaling, use_size_scaling=use_size_scaling
        )

    ##############################
    ##############################

    def _init_clustering(self, cluster_method, clustering_kwargs):
        try:
            self._clustering_model = cluster_method(**clustering_kwargs)
            self._clustering_model_artifacts = cluster_method(**clustering_kwargs)
        except TypeError as e:
            warnings.warn(
                f"Error while initializing clustering: {e}. Using default parameters."
            )

    def _extract_raw_from_event(self, event_time, electrode_ID):
        """Extracts raw data from a given time window.

        Args:
            event_time (datetime): Time of the spike event on which the time window is centered.
            electrode_ID (int): ID of the electrode on which to extract the data.
            time_before (float): Number of ms to extract before the event. Default is 0.5 ms.
            t2 (datetime): Number of ms to extract after the event. Default is 0.5 ms.

        Returns:
            df_time_window (pd.DataFrame): DataFrame containing the raw data.
        """
        t_start = event_time - timedelta(milliseconds=self.time_before)
        t_end = event_time + timedelta(milliseconds=self.time_after)
        raw = self._db.get_raw_spike(t_start, t_end, electrode_ID)
        raw["Spike_time"] = np.array(
            [round(i * 30.3 * 1e-3, 4) for i in range(len(raw["Amplitude"]))]
        )
        return raw

    def _process_spike_event_row(self, row):
        """Processes a single row of the spike events DataFrame."""
        try:
            df_time_window = self._extract_raw_from_event(row["Time"], row["channel"])

            # Realign the spikes if the amplitude is below a certain threshold and realign is True
            if (df_time_window["Amplitude"] < -50).any() and self.realign:
                amp_min_index = np.argmin(df_time_window["Amplitude"])
                time_event = df_time_window.iloc[amp_min_index]["Time"]
            else:
                # Otherwise we just center on the event timestamp
                time_event = row["Time"]

            df_time_window = self._extract_raw_from_event(time_event, row["channel"])
            # Check if some spikes are artifacts
            if (
                self.filter_artifacts
                and (df_time_window["Amplitude"].abs() > self.artifact_threshold).any()
            ):
                self.logger.debug(
                    f"Artifact detected for spike at {row['Time']} with amplitude {df_time_window['Amplitude'].max()}"
                )
                is_artifact = True
            else:
                is_artifact = False
            processed_row = df_time_window[
                "Amplitude"
            ].tolist()  # Ensure the return value is picklable for parallel processing
            # cast to list of floats to avoid pickling error (is this still necessary?)
            # "pickling errors" were actually the server refusing requests when too many were sent at once
            # because of n_jobs being high
            processed_row = list([float(i) for i in processed_row])
            event_time = row["Time"]
            timepoints = df_time_window["Spike_time"].tolist()
            return (  # could go back to returning a df or a dict, this is less readable
                [event_time] * len(processed_row),
                processed_row,
                timepoints,
                [is_artifact] * len(processed_row),
            )
        except Exception as e:
            self.logger.error(f"Error while processing spike event row: {str(e)}")
            return None

    def _process_spikes_parallel(self):
        """
        Process spikes to extract raw data.

        Returns:
            filtered_results_df['Amplitude'].tolist() (list): List of raw spikes.
        """
        self.logger.debug(f"Lauching process_spikes_parallel with {self.n_jobs} jobs")

        df_spike_events_select = self.processed_spike_events_df.copy()

        with parallel_config(backend="multiprocessing", n_jobs=self.n_jobs):
            results = Parallel()(
                delayed(self._process_spike_event_row)(row)
                for _, row in tqdm(
                    df_spike_events_select.iterrows(),
                    total=df_spike_events_select.shape[0],
                )
            )
        # remove None values
        # results = [r for r in results if r is not None]
        time_event, results, timepoints, is_artifact = zip(*results)
        results_df = (
            pd.DataFrame(results)
            .stack()
            .reset_index(drop=True)
            .to_frame(name="Amplitude")
        )
        # Filter out None values
        filtered_results_df = results_df.dropna()

        is_artifact_series = pd.Series(
            [item for sublist in is_artifact for item in sublist]
        )
        filtered_results_df["is_artifact"] = is_artifact_series
        self.logger.debug(f"Processed {len(results)} spikes")
        self.logger.debug(f"Len of first results : {len(results[0])}")

        filtered_results_df["Event_time"] = np.concatenate(time_event)
        assert (
            filtered_results_df["Event_time"].unique().shape[0]
            == df_spike_events_select["Time"].unique().shape[0]
        )
        filtered_results_df["Spike_time"] = np.concatenate(timepoints)
        # add to processed spike events the artifacts status
        is_event_artifact = filtered_results_df.copy()
        is_event_artifact = is_event_artifact.groupby("Event_time")[
            "is_artifact"
        ].apply(lambda x: x.all())
        df_spike_events_select = df_spike_events_select.merge(
            is_event_artifact, left_on="Time", right_index=True, how="left"
        )

        self.processed_spike_events_df = df_spike_events_select  # .dropna(inplace=True)
        self.raw_spikes_df = filtered_results_df
        self.logger.info(
            f"Found {len(self.processed_spike_events_df[self.processed_spike_events_df['is_artifact']])} spikes"
        )
        self.logger.info(
            f"Found {len(self.processed_spike_events_df[self.processed_spike_events_df['is_artifact']])} artifacts"
        )

        return filtered_results_df

    def _get_events_in_time_window(self, start_timestamp, stop_timestamp):
        """Fetches the spike events in the time window."""

        if stop_timestamp < start_timestamp:
            raise ValueError("stop_timestamp must be greater than start_timestamp")
        if (
            start_timestamp < self._start_timestamp
            or stop_timestamp > self._stop_timestamp
        ):
            raise ValueError("Time window must be within the original time window.")

        return self.processed_spike_events_df[
            (self.processed_spike_events_df["Time"] >= start_timestamp)
            & (self.processed_spike_events_df["Time"] <= stop_timestamp)
        ]

    def _plot_latent_space_time_window(
        self, start_timestamp, stop_timestamp, show=False
    ):
        """Plots the latent space for a given time window."""
        if self.dimred_method_spikes.fit_data_df is None:
            self.logger.error("No data to plot.")
            raise SpikeSortingNotRun()
        data = self._get_events_in_time_window(start_timestamp, stop_timestamp)
        if data.empty:
            self.logger.warning("No spikes found in the time window.")
            return
        dimred_data = self.dimred_method_spikes.fit_data_df.loc[data.index]
        return display(self._plot_3d_cluster(dimred_data, show))

    def _fit_dimred(self, on_artifacts=False):
        """Fits a model to the raw spikes with the aim to reduce dimensionality.

        Args:
            filtered_results_df (pd.DataFrame): DataFrame containing the processed raw spikes. Columns should be "Amplitude", "Spike" and "Event_time".
            on_artifacts (bool): If True, runs PCA on artifacts only. Default is False.

        Returns:
            pca (PCA): PCA model.
            X (np.array): Transformed data.
            components (np.array): Number of PCA components. If None, use the default value.
            explained_variance (np.array): Explained variance.
        """

        if self.raw_spikes_df is None:
            self.logger.error("No raw spikes to fit dimensionality reduction to.")
            raise SpikeSortingNotRun()

        self.logger.info(
            f"Fitting {self.dimred_method_spikes.method_name} with {self.dimred_method_spikes.n_components} components..."
        )
        spike_df = self.raw_spikes_df[
            self.raw_spikes_df["is_artifact"] == on_artifacts
        ].copy()
        spike_df = spike_df.groupby("Event_time")["Amplitude"].apply(list)
        spike_matrix = np.array(spike_df.tolist())  # (n_samples, n_spikes)

        if on_artifacts:
            if spike_matrix.shape[0] < 15:
                self.logger.warning(
                    "There are too few artifacts for dimensionality reduction."
                )
                self.dimred_method_artifacts.fit_data_df = None
                return
            self.dimred_method_artifacts.fit(spike_matrix)
        else:
            self.dimred_method_spikes.fit(spike_matrix)
            return self.dimred_method_spikes

    def _fit_clustering(self, on_artifacts=False):
        """Fits the clustering model to the PCA data."""
        if self.dimred_method_spikes.fit_data_df is None and not on_artifacts:
            self.logger.error("No data to fit clustering to.")
            raise SpikeSortingNotRun()
        if self.dimred_method_artifacts.fit_data_df is None and on_artifacts:
            self.logger.warning("No artifacts to run clustering on.")
            return
        if on_artifacts:
            if len(self.dimred_method_artifacts.fit_data_df) < 15:
                self.logger.warning(
                    "There are too few artifacts to properly cluster. All artifacts will be assigned to the same -2 cluster."
                )
                self._clustering_artifacts = self.dimred_method_artifacts
                self._clustering_artifacts.labels_ = np.full(
                    len(self.dimred_method_artifacts.fit_data_df), -2
                )
            else:
                self._clustering_artifacts = self._clustering_model_artifacts.fit(
                    self.dimred_method_artifacts.fit_data_df
                )
                # artifact clusters must be at -2 or lower
                self._clustering_model_artifacts.labels_ = (
                    self._clustering_artifacts.labels_ * -1
                ) - 3
            self.logger.debug(
                f"Found unique clusters for artifacts: {np.unique(self._clustering_artifacts.labels_)}"
            )
            return self._clustering_artifacts
        else:
            self._clustering = self._clustering_model.fit(
                self.dimred_method_spikes.fit_data_df
            )
            self.logger.debug(
                f"Found unique clusters for spikes: {np.unique(self._clustering.labels_)}"
            )
            return self._clustering

    def _add_clustering_to_dimred_df(self):
        """Adds the clustering results to the PCA DataFrame."""
        if self.dimred_method_spikes.fit_data_df is None or self._clustering is None:
            self.logger.error("No data to add clustering to, or clustering not run.")
            raise SpikeSortingNotRun()
        self.dimred_method_spikes.fit_data_df["Cluster"] = self._clustering.labels_
        if self.dimred_method_artifacts.fit_data_df is not None:
            self.dimred_method_artifacts.fit_data_df[
                "Cluster"
            ] = self._clustering_artifacts.labels_
        else:
            self.logger.warning(
                "Clustering labels could not be added to the artifacts.\n"
                "Plotting the latent space for artifacts will not be possible.\n"
                "(This is normal if the data has no or too few artifacts.)"
            )

    def _add_clustering_to_raw_df(self):
        """Adds the clustering results to the DataFrame."""
        if self.raw_spikes_df is None or self._clustering is None:
            self.logger.error("No data to add clustering to, or clustering not run.")
            raise SpikeSortingNotRun()
        raw_no_artifacts = self.raw_spikes_df[~self.raw_spikes_df["is_artifact"]].copy()
        raw_w_artifacts = self.raw_spikes_df[self.raw_spikes_df["is_artifact"]].copy()
        raw_no_artifacts.loc[:, "Cluster"] = np.repeat(
            self._clustering.labels_,
            raw_no_artifacts.shape[0] // self._clustering.labels_.shape[0],
        )
        if self._clustering_artifacts is not None:
            raw_w_artifacts.loc[:, "Cluster"] = np.repeat(
                self._clustering_artifacts.labels_,
                raw_w_artifacts.shape[0] // self._clustering_artifacts.labels_.shape[0],
            )
            self.raw_spikes_df = pd.concat(
                [raw_no_artifacts, raw_w_artifacts]
            ).sort_index()
        else:
            self.raw_spikes_df = (
                raw_no_artifacts.sort_index()
            )  # .groupby(by=["Event_time", "Spike_time"]).reset_index(drop=True)

    def _add_clustering_to_event_df(self):
        """Adds the clustering results to the DataFrame."""
        if self.processed_spike_events_df is None or self._clustering is None:
            self.logger.error("No data to add clustering to, or clustering not run.")
            raise SpikeSortingNotRun()
        processed_spikes_no_artifacts = self.processed_spike_events_df[
            ~self.processed_spike_events_df["is_artifact"].astype(bool)
        ].copy()
        processed_spikes_w_artifacts = self.processed_spike_events_df[
            self.processed_spike_events_df["is_artifact"].astype(bool)
        ].copy()
        processed_spikes_no_artifacts["Cluster"] = self._clustering.labels_
        if self._clustering_artifacts is not None:
            processed_spikes_w_artifacts["Cluster"] = self._clustering_artifacts.labels_
            self.processed_spike_events_df = pd.concat(
                [processed_spikes_no_artifacts, processed_spikes_w_artifacts]
            )
        else:
            self.processed_spike_events_df = processed_spikes_no_artifacts
        self.processed_spike_events_df.sort_index(
            inplace=True
        )  # .sort_values("Time", inplace=True)

    def _plot_3d_time(self, time_agg="min"):
        """Plots an animation of the 3D latent space over time."""
        # REQUIRES plotly, which is not installed by default
        try:
            import plotly.express as px
        except ImportError:
            raise ImportError(
                "Plotly is not installed. Please install it using 'pip install plotly',  \
                        or, in DataLore, by clicking on the 'Environment' button on the left and adding the 'plotly' package."
            )

        dimred_data = self.dimred_method_spikes.fit_data_df.copy()

        spike_events = self.processed_spike_events_df[["Time"]].copy()

        dimred_data = dimred_data.merge(spike_events, left_index=True, right_index=True)
        dimred_data["Time"] = dimred_data["Time"].dt.floor(time_agg)
        dimred_data["Time"] = dimred_data["Time"].apply(lambda x: x.timestamp())

        dimred_data = dimred_data.sort_values(by="Time")
        dimred_data["Frame"] = dimred_data["Time"].rank(method="dense").astype(int)

        cumulative_data = pd.concat(
            [
                dimred_data[dimred_data["Frame"] <= frame].assign(Frame=frame)
                for frame in dimred_data["Frame"].unique()
            ]
        )
        range_col = [cumulative_data["Time"].min(), cumulative_data["Time"].max()]
        fig = px.scatter_3d(
            cumulative_data,
            x="C1",
            y="C2",
            z="C3",
            color="Time",
            animation_frame="Frame",
            range_color=range_col,
            animation_group=cumulative_data.index,
            range_x=[cumulative_data["C1"].min(), cumulative_data["C1"].max()],
            range_y=[cumulative_data["C2"].min(), cumulative_data["C2"].max()],
            range_z=[cumulative_data["C3"].min(), cumulative_data["C3"].max()],
        )
        fig.show()

    @staticmethod
    def _plot_3d_cluster(data, show=True):
        # REQUIRES plotly, which is not installed by default
        try:
            import plotly.express as px

            if data is None:
                raise ValueError(
                    "No data to plot. Please run the spike sorting algorithm first."
                )
            data = data.copy()
            if "Cluster" in data.columns:
                data.Cluster = data.Cluster.astype(str)
                fig = px.scatter_3d(
                    data,
                    x="C1",
                    y="C2",
                    z="C3",
                    color="Cluster",
                    color_continuous_scale="Spectral",
                )
            else:
                fig = px.scatter_3d(data, x="C1", y="C2", z="C3")
            if show:
                fig.show()
            else:
                return fig
        except ImportError:
            raise ImportError(
                "Plotly is not installed. Please install it using 'pip install plotly',  \
                    or, in DataLore, by clicking on the 'Environment' button on the left and adding the 'plotly' package."
            )

    def _plot_raw_spikes(self, data, use_alpha_scaling=True, use_size_scaling=True):
        """Plot the raw spikes for a given cluster.

        Args:
            data (pd.DataFrame): DataFrame containing the raw spikes. Should be filtered prior to calling this function to avoid plotting all spikes.
            use_alpha_scaling (bool): If True, scales the alpha value based on the number of spikes. Default is True.
            use_size_scaling (bool): If True, scales the size of the spikes based on the number of spikes. Default is True.
        """
        # REQUIRES seaborn, which is not installed by default
        try:
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "Seaborn is not installed. Please install it using 'pip install seaborn',  \
                        or, in DataLore, by clicking on the 'Environment' button on the left and adding the 'seaborn' package."
            )
        if self.raw_spikes_df is None:
            self.logger.error("No raw spikes to plot.")
            raise SpikeSortingNotRun()
        if "Cluster" not in self.raw_spikes_df.columns:
            raise ValueError(
                "No clustering results found. Please run the spike sorting algorithm first."
            )
        if data.empty:
            self.logger.warning("Nothing to plot.")
            return
        plt.figure(figsize=(10, 6), dpi=200)
        clusters = data["Cluster"].unique().tolist()
        n_spikes = int(len(data) / len(data["Spike_time"].unique()))
        plt.title(
            f"Cluster {clusters} - electrode {self.electrode_id} over {self.time_before+self.time_after} ms"
            f"\nTotal : {n_spikes} spikes"
        )
        scaling = np.log10(n_spikes)
        if n_spikes > 1000:
            scaling *= 10
        alpha = min(1, max(0.0005, 0.5 / scaling)) if use_alpha_scaling else None
        size = min(1, max(0.001, 1 / scaling)) if use_size_scaling else None

        sns.lineplot(
            data=data,
            x="Spike_time",
            y="Amplitude",
            hue="Event_time",
            palette=["black"] * data["Event_time"].nunique(),
            alpha=alpha,
            size=size,
        )
        plt.legend().remove()
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude (uV)")
        plt.show()
