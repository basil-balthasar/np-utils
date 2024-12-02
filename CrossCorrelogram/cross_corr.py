import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

try:
    import numba as nb
except ImportError:
    raise ImportError("Please install 'numba' in your environment to use this module.")


class CrossCorrelogram:
    """
    A class to compute and plot cross-correlograms between two sets of event times.

    Attributes:
        deltat (int): Time window in milliseconds.
        bin_size (float): Bin size in milliseconds.
        normalize (bool): Normalize the cross-correlogram by the square root of the product of the number of events in each dataframe.

    Methods:
        prepare_dfs(dataframes): Format dataframes for cross-correlogram analysis.
        compute(df1, df2, start_time=None, end_time=None): Compute the cross-correlogram between the Time column of two dataframes.
        plot(cross_corr, from_="", to_=""): Plot the cross-correlogram.
    """

    def __init__(self, deltat: int, bin_size: float, normalize: bool = True):
        """
        Create a CrossCorrelogram object with the given time window and bin size.

        Args:
            deltat (int): Time window in milliseconds.
            bin_size (float): Bin size in milliseconds.
            normalize (bool, optional): Normalize the cross-correlogram by the square root of the product of the number of events in each dataframe.
        """
        self.deltat = deltat
        self.bin_size = bin_size
        self.normalize = normalize

    def prepare_dfs(self, dataframes):
        """
        Format dataframes for cross-correlogram analysis.

        Args:
            dataframes (list of pd.DataFrame): List of dataframes to be formatted.

        Returns:
            list of pd.DataFrame: List of formatted dataframes.
        """
        for df in dataframes:
            df = df.copy()
            df["Time"] = pd.to_datetime(df["Time"])
            df["channel"] = df["channel"].astype(int)
        return dataframes

    @staticmethod
    @nb.njit(parallel=True)
    def _cross_correlogram(events_i, events_j, deltat, bin_size, normalize=True):
        """
        Compute the cross-correlogram between two sets of event times.

        Args:
            events_i (np.ndarray): Array of event times from the first dataframe.
            events_j (np.ndarray): Array of event times from the second dataframe.
            deltat (int): Time window in milliseconds.
            bin_size (float): Bin size in milliseconds.
            normalize (bool): Normalize the cross-correlogram by the square root of the product of the number of events in each dataframe.

        Returns:
            np.ndarray: Cross-correlogram array.
        """
        n_i = len(events_i)
        n_j = len(events_j)
        n_bins = int(2 * deltat / bin_size)
        if n_bins % 2 == 0:
            n_bins += 1
        cross_corr = np.zeros(n_bins)

        # Implementation 1 (faster)
        for i in nb.prange(n_i):
            local_corr = np.zeros(n_bins)
            for j in range(n_j):
                delta = events_j[j] - events_i[i]
                if np.abs(delta) <= deltat:
                    bin_idx = min(int((delta + deltat) / bin_size), n_bins - 1)
                    local_corr[bin_idx] += 1
            cross_corr += local_corr

        # Implementation 2 (slower, likely due to np.where and np.floor + type conversion)
        # for i in nb.prange(n_i):
        #     indices = np.where(np.abs(events_j - events_i[i]) <= deltat)[0]
        #     lag = events_j[indices] - events_i[i]
        #     bin_ids = np.floor(lag / bin_size).astype(np.int32) + n_bins // 2
        #     cross_corr[bin_ids] += 1

        if normalize:
            cross_corr /= np.sqrt(n_i * n_j)

        return cross_corr

    def compute(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        start_time=None,
        end_time=None,
    ) -> np.ndarray:
        """
        Compute the cross-correlogram between the Time column of two dataframes.

        Args:
            df1 (pd.DataFrame): First dataframe containing event times.
            df2 (pd.DataFrame): Second dataframe containing event times.
            start_time (pd.Timestamp, optional): Start time for the analysis. If None, the minimum time from both dataframes is used.
            end_time (pd.Timestamp, optional): End time for the analysis. If None, the maximum time from both dataframes is used.

        Returns:
            np.ndarray: Cross-correlogram array.
        """
        if start_time is None:
            start_time = min(df1["Time"].min(), df2["Time"].min())
        if end_time is None:
            end_time = max(df1["Time"].max(), df2["Time"].max())

        df1 = df1.copy()
        df2 = df2.copy()

        df1["Time"] = (df1["Time"] - start_time).dt.total_seconds() * 1e3
        df2["Time"] = (df2["Time"] - start_time).dt.total_seconds() * 1e3

        events_i = df1["Time"].values
        events_j = df2["Time"].values

        return self._cross_correlogram(events_i, events_j, self.deltat, self.bin_size)

    def compute_from_db(
        self,
        fs_id,
        channel_from,
        channel_to,
        start_time,
        end_time,
        check_for_triggers=False,
        keep_dfs=True,
    ):
        """
        Compute the cross-correlogram between two channels from the database.

        Note : requires neuroplatform access to be used.

        Args:
            fs_id (str): ID of the fileset.
            channel_from (int): ID of the first channel.
            channel_to (int): ID of the second channel.
            start_time (pd.Timestamp): Start time for the analysis.
            end_time (pd.Timestamp): End time for the analysis.
            check_for_triggers (bool, optional): Check for triggers in the provided times for the channels. Defaults to False.
            keep_dfs (bool, optional): If True, returns the dataframes used for the analysis. Defaults to True.
        """
        try:
            from neuroplatform import Database
        except ImportError:
            raise ImportError("Neuroplatform access is required to use this function.")
        db = Database()

        df_all = db.get_spike_event(start=start_time, stop=end_time, fsname=fs_id)
        df_i = df_all[df_all["channel"] == channel_from]
        df_j = df_all[df_all["channel"] == channel_to]

        if check_for_triggers:
            trigs = db.get_all_triggers(start_time, end_time)
            if len(trigs) > 0:
                print(
                    f"Found {len(trigs)} triggers in the provided time range. Double-check whether there was stimulation on the provided channels, as this will alter the cross-correlogram."
                )

        cc = self.compute(df_i, df_j, start_time, end_time)
        if keep_dfs:
            return cc, df_i, df_j
        return cc

    def get_firing_rate(self, df):
        """
        Compute the firing rate of the two dataframes.

        Args:
            df
        """
        df = df.copy(deep=True)
        df.drop(columns=["Amplitude"], inplace=True)
        df.index = pd.DatetimeIndex(df["Time"])
        firing_rate = df.groupby("channel").resample("1s").size()
        firing_rate = firing_rate.reset_index(name="Firing rate")
        return firing_rate

    def plot_firing_rate(self, dfs, ax=None, start_time=None, end_time=None):
        """
        Plot the firing rate of the two dataframes.

        Args:
            dfs (list of pd.DataFrame): List of dataframes to be plotted.
            ax (plt.Axes, optional): Axes object to plot on. If None, a new figure is created.
            start_time (pd.Timestamp, optional): Start time for the analysis. If None, the minimum time from both dataframes is used.
            end_time (pd.Timestamp, optional): End time for the analysis. If None, the maximum time from both dataframes is used.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(20, 10))
        all_dfs = []
        for df in dfs:
            firing_rate = self.get_firing_rate(df)
            all_dfs.append(firing_rate)
        df_all = pd.concat(all_dfs)
        sns.lineplot(
            data=df_all,
            x="Time",
            y="Firing rate",
            hue="channel",
            ax=ax,
            alpha=0.7,
            palette="tab10",
        )
        ax.set_yscale("log")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title(f'Firing rate of channels {sorted(df_all["channel"].unique())}')
        ax.set_xlim(start_time, end_time)

    def plot(self, cross_corr: np.ndarray, from_: str = "", to_: str = "", ax=None):
        """
        Plot the cross-correlogram.

        Args:
            cross_corr (np.ndarray): Cross-correlogram array to be plotted.
            from_ (str, optional): Name of the first dataframe for the plot title.
            to_ (str, optional): Name of the second dataframe for the plot title.
            ax (plt.Axes, optional): Axes object to plot on. If None, a new figure is created.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        self._plot(cross_corr, ax, from_=from_, to_=to_)

    def plot_comparison(
        self,
        cross_corr1: np.ndarray,
        cross_corr2: np.ndarray,
        from_: str = "",
        to_: str = "",
        ax=None,
        label1=None,
        label2=None,
    ):
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
        self._plot(
            cross_corr1,
            ax,
            from_=from_,
            to_=to_,
            color="blue",
            alpha=0.5,
            label=label1,
        )
        self._plot(
            cross_corr2,
            ax,
            from_=from_,
            to_=to_,
            color="red",
            alpha=0.5,
            label=label2,
        )

    def _plot(
        self,
        cross_corr: np.ndarray,
        ax,
        from_: str = "",
        to_: str = "",
        color=None,
        alpha=1.0,
        label=None,
    ):
        ax.bar(
            np.arange(-self.deltat, self.deltat + self.bin_size, self.bin_size),
            cross_corr,
            width=self.bin_size,
            color=color,
            alpha=alpha,
            label=label,
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Count")
        title = "Cross-correlogram"
        if from_:
            title += f" from {from_}"
            if to_:
                title += f" to {to_}"
        ax.set_title(title)
        ax.axvline(
            x=0,
            color="black",
            linestyle="--",
            alpha=0.5,
        )

    def compute_from_dict(self, df_dict):
        """Compute all pairwise cross-correlograms from a dictionary of dataframes."""
        ccs = {}
        for channel, df in tqdm(df_dict.items(), desc="Computing cross-correlograms"):
            cross_corr = {}
            for other_channel, other_df in df_dict.items():
                if channel != other_channel:
                    cross_corr[other_channel] = self.compute(df, other_df)
            ccs[channel] = cross_corr

        return ccs

    def plot_from_dict(self, ccs, channels=None):
        """Plot all pairwise cross-correlograms from a dictionary of cross-correlograms."""
        for i, (channel, cross_corrs) in enumerate(ccs.items()):
            if channels is not None and i not in channels:
                continue
            n_cols = 4
            n_rows = int(np.ceil(len(cross_corrs) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
            axes = axes.flatten()
            for j, (other_channel, cross_corr) in enumerate(cross_corrs.items()):
                self._plot(
                    cross_corr,
                    from_=str(channel),
                    to_=str(other_channel),
                    ax=axes[j],
                )
            plt.tight_layout()
            plt.show()
