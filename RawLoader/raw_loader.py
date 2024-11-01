"""
Neuroplatform Utils - Utility functions for the Neuroplatform project.

This is intended to be a plug-and-play utility module for V1, without any changes to the main Neuroplatform codebase.

RawRecordingLoader : Raw recording loader utility functions, for h5f from Neuroplatform V2. Author : Cyril Achard, September 2024
"""

from typing import List
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import h5py
except ImportError:
    pass

### RawRecordingLoader


class RawRecordingLoader:
    try:
        import h5py
    except ImportError:
        raise ModuleNotFoundError(
            "h5py not found. Please install h5py and make sure to reload your kernel."
        )

    def __init__(
        self,
        hdf5_file: h5py.File,
        start: int | float = None,
        stop: int | float = None,
        electrodes: List[int] = None,
        sampling_freq: int = 3e4,
    ):
        """
        Initializes the RawRecordingLoader instance.

        Args:
            hdf5_file (h5py.File): The HDF5 file object containing the raw data.
            start (int | float, optional): Where to start loading data from. If float, it is interpreted as a time in seconds. Integers are interpreted as indices. Defaults to None.
            stop (int | float, optional): Where to stop loading data. If float, it is interpreted as a time in seconds. Integers are interpreted as indices. Defaults to None.
            electrodes (list, optional): List of electrode indices to load. Defaults to None.
            sampling_freq (int, optional): Sampling frequency of the data. Defaults to 30000.
        """
        self._electrodes = None
        self._start = None
        self._stop = None
        self._h5_entries = list(hdf5_file.keys())
        self._h5_electrodes = [
            entry for entry in self._h5_entries if entry.startswith("electrode_")
        ]
        self.has_triggers = "triggers" in self._h5_entries
        self._num_entries = None
        self._is_large = False
        self._mem_warn_limit = 4  # GB of memory above which a warning is issued
        ###
        self.file = hdf5_file
        """The HDF5 file object containing the raw data."""
        self.sampling_freq = sampling_freq
        """Sampling frequency of the data. Defaults to 30 kHz."""
        self.electrodes = electrodes
        """List of electrode indices to load. If None, all electrodes will be loaded."""

        if self.sampling_freq <= 0:
            raise ValueError("Sampling frequency must be a positive integer.")

        self._check_data()

        self.start = start
        self.stop = stop

        self._compute_mem()

    @property
    def start(self):
        return self._start if self._start is not None else 0

    @start.setter
    def start(self, value: int | float):
        if value is not None:
            self._start = self._convert_time_to_index(value, "Start")
        else:
            self._start = 0

    @property
    def stop(self):
        return self._stop if self._stop is not None else self._num_entries - 1

    @stop.setter
    def stop(self, value: int | float):
        if value is not None:
            self._stop = self._convert_time_to_index(value, "Stop")
        else:
            self._stop = self._num_entries - 1

    @property
    def electrodes(self):
        return (
            self._electrodes
            if self._electrodes is not None
            else [int(entry.split("_")[1]) for entry in self._h5_electrodes]
        )

    @electrodes.setter
    def electrodes(self, value: List[int]):
        if value is not None:
            if not all(isinstance(elec, int) for elec in value):
                raise ValueError("Electrode indices must be integers.")
            missing_electrodes = [
                elec for elec in value if f"electrode_{elec}" not in self._h5_electrodes
            ]
            if missing_electrodes:
                raise ValueError(
                    f"Electrode(s) {missing_electrodes} not found in the file."
                )
            self._electrodes = value
        else:
            self._electrodes = [
                int(entry.split("_")[1]) for entry in self._h5_electrodes
            ]

    def load(
        self,
        start: int | float = None,
        stop: int | float = None,
        electrodes: List[int] = None,
        include_triggers: bool = True,
        keep_recording_time: bool = True,
    ):
        """
        Loads the raw data from the HDF5 file and returns it as a DataFrame.

        Args:
            start (int | float, optional): Index to start loading data from, if not already set.. If float, it is interpreted as a time in seconds. Integers are interpreted as indices. Defaults to None.
            stop (int | float, optional): Index to stop loading data, if not already set.. If float, it is interpreted as a time in seconds. Integers are interpreted as indices. Defaults to None.
            electrodes (list, optional): List of electrode indices to load, if not already set. Defaults to None.
            include_triggers (bool, optional): Whether to include trigger data in the DataFrame. Defaults to True.
            keep_recording_time (bool, optional): If True, the recording time will be computed starting from 0 for the specified slice. If False, the original recording time will be preserved. Defaults to True.
        """
        if start is not None and stop is not None and start >= stop:
            raise ValueError("Start index must be less than stop index.")

        if stop is not None and start is not None and stop - start > self._num_entries:
            # should not happen since we check start and stop in the setter
            raise ValueError(
                "The difference between start and stop indices must be less than the total number of entries."
            )

        self.start = start
        self.stop = stop
        self.electrodes = electrodes

        if self._start is not None and self._stop is not None:
            print(
                f"Loading data for electrodes {self.electrodes} from index {self.start} ({self.start/self.sampling_freq:.2f}s) to index {self.stop} ({self.stop/self.sampling_freq:.2f}s)."
            )

        return self._load_df(
            load_triggers=include_triggers, keep_recording_time=keep_recording_time
        )

    @classmethod
    def from_path(
        cls,
        file_path: Path | str,
        start: int | float = None,
        stop: int | float = None,
        electrodes: List[int] = None,
        sampling_freq: int = 3e4,
    ):
        """
        Loads raw data from an HDF5 file and returns it as a DataFrame.

        Args:
            file_path (Path | str): Path to the HDF5 file containing the raw data.
            start (int | float, optional): Index to start loading data from. If float, it is interpreted as a time in seconds. Integers are interpreted as indices. Defaults to None.
            stop (int | float, optional): Index to stop loading data. If float, it is interpreted as a time in seconds. Integers are interpreted as indices. Defaults to None.
            electrodes (list, optional): List of electrode indices to load. Defaults to None (load all electrodes).
            sampling_freq (int, optional): Sampling frequency of the data. Defaults to 30000.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        elif not isinstance(file_path, Path):
            raise TypeError("File path must be a string or a Path object.")
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found.")
        if file_path.suffix != ".h5":
            raise ValueError(f"File must be an HDF5 file got {file_path.suffix}.")
        hdf5_file = h5py.File(file_path, "r")
        return cls(hdf5_file, start, stop, electrodes, sampling_freq)

    ###################################
    ###################################

    def _convert_time_to_index(self, value: int | float, name: str):
        if isinstance(value, float):
            print(f"Converting time {value}s to index.")
            value = int(value * self.sampling_freq)
        elif isinstance(value, int):
            pass
        elif value is not None:
            raise ValueError(
                f"{name} must be a positive integer (index) or a float (time in seconds)."
            )

        if value is not None and value < 0:
            raise ValueError("Time must be a positive integer.")
        if value is not None and value > self._num_entries:
            raise ValueError(
                f"{name} index must be less than the total number of entries. Max {self._num_entries}, got {value}."
            )

        return value

    def _check_data(self):
        if not self.electrodes:
            self.electrodes = [
                int(entry.split("_")[1]) for entry in self._h5_electrodes
            ]
        if not self._h5_electrodes:
            raise ValueError("No electrode data found in the file.")

        print(
            f"Found electrodes in file: {str([int(el.split('_')[1]) for el in self._h5_electrodes])}"
        )

        missing_electrodes = [
            elec
            for elec in self.electrodes
            if f"electrode_{elec}" not in self._h5_electrodes
        ]
        if missing_electrodes:
            raise ValueError(
                f"Electrode(s) {missing_electrodes} not found in the file."
            )

        el = self.file[self._h5_electrodes[0]]
        self._num_entries = int(
            el.shape[0] * el.shape[1]
        )  # full len, use -1 for slicing
        time_len, u = self._num_entries / self.sampling_freq, "s"
        if time_len >= 60:
            time_len, u = time_len / 60, "min"
            self._is_large = True
            if time_len >= 60:
                time_len, u = time_len / 60, "hr"
                if time_len >= 24:
                    time_len, u = time_len / 24, "days"
        print(f"Data length: {time_len:.2f} {u} ({self._num_entries} samples)")
        if self._is_large and (self.start is None and self.stop is None):
            print(
                "Consider using start and stop parameters to load a subset of the data."
            )

    def _compute_mem(self):
        sliced_len = self.stop - self.start + 1
        mem, u = sliced_len * 64, "bytes"  # consider each index as a float64
        kB = 1024
        if mem >= kB * kB:
            mem, u = mem / kB / kB, "MB"
            if mem >= kB:
                mem, u = mem / kB, "GB"
        print(
            f"Estimated maximum memory usage for loading a single electrode: {mem:.2f} {u}"
        )
        if u == "GB":
            if mem > self._mem_warn_limit and (
                self.start is None and self.stop is None
            ):
                print(
                    "Consider loading a subset of the data to avoid running out of memory."
                )
            elif mem > self._mem_warn_limit and (
                self.start is not None or self.stop is not None
            ):
                print(
                    "Consider loading a smaller subset of the data to avoid running out of memory."
                )

    @staticmethod
    def _load_dataset(
        dset: h5py.Dataset, start: int, stop: int, is_trigger: bool = False
    ):
        shape = dset.shape
        start_i = np.unravel_index(start, shape)
        stop_i = np.unravel_index(stop, shape)

        if start_i[0] == stop_i[0]:
            amplitude_data = dset[start_i[0], start_i[1] : stop_i[1]]
        else:
            data_segments = [dset[start_i[0], start_i[1] :]]
            data_segments.extend(dset[i, :] for i in range(start_i[0] + 1, stop_i[0]))
            data_segments.append(dset[stop_i[0], : stop_i[1] + 1])
            # Concatenate all segments into a single array
            amplitude_data = np.concatenate(data_segments)

        if is_trigger:
            # Convert the uint16 array to uint8 and then unpack the bits
            binary_arr = np.unpackbits(amplitude_data.view(np.uint8), bitorder="little")
            # Since uint16 is 2 bytes, we need to reshape the result to get the correct binary representation
            binary_arr = binary_arr.reshape(-1, 16)
            return binary_arr

        return amplitude_data

    @staticmethod
    def _find_trigger_values(row):
        active_trigs = np.where(row == 1)[0]
        if active_trigs.size == 0:
            return np.nan
        else:
            return [int(trig) for trig in active_trigs]

    def _load_triggers(self):
        print("Loading triggers...")
        trigs = self.file["triggers"]
        trigs = self._load_dataset(trigs, self.start, self.stop, is_trigger=True)
        trigs_value = np.apply_along_axis(self._find_trigger_values, 1, trigs)
        trigs_value = pd.DataFrame(trigs_value, columns=["trigger"])
        return trigs_value

    def _load_df(self, load_triggers=True, keep_recording_time=False):
        df = None
        print("Loading electrodes...")
        for elec in self.electrodes:
            dset_elec = self.file[f"electrode_{elec}"]
            amp = self._load_dataset(dset_elec, self.start, self.stop)
            if df is None:
                df = pd.DataFrame({"amplitude": amp, "electrode": elec})
            else:
                df = pd.concat(
                    [df, pd.DataFrame({"amplitude": amp, "electrode": elec})]
                )

        df = df.pivot(columns="electrode", values="amplitude")
        if self.has_triggers and load_triggers:
            trigs = self._load_triggers()
            df = pd.concat([df, trigs], axis=1)
        elif not self.has_triggers and load_triggers:
            print("No triggers found in the file, skipping.")

        if self.sampling_freq is not None:
            if keep_recording_time:
                df["Time"] = np.arange(self.start, self.stop + 1) / self.sampling_freq
            else:
                df["Time"] = df.index / self.sampling_freq
            df.set_index("Time", inplace=True, drop=True)
            df.sort_index(inplace=True)
        print("Done.")
        return df
