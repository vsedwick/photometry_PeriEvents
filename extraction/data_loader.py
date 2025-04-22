from pathlib import Path
import os
import numpy as np
import pandas as pd
from math import floor
import utils.time as ti


class TrialLoader:
    """
    Loads photometry and behavior data for *one* trial.

    Parameters
    ----------
    config : AnalysisParameters
    trial_id : str | None
        Folder name of the trial (batch mode). Optional when you
        supply explicit file paths.
    photometry_file : str | Path | None
    behavior_file   : str | Path | None
    """
    def __init__(
        self,
        config,
        trial_id: str | None = None,
        photometry_file: str | Path | None = None,
        behavior_file: str | Path | None = None,
    ):
        self.config = config

        # Resolve root directory for searches
        if photometry_file:
            self.photo_path = Path(photometry_file)
            self.trial_path = self.photo_path.parent
        else:
            self.photo_path = None

        if behavior_file:
            self.behav_path = Path(behavior_file)
            if not photometry_file:               # single behaviour file only
                self.trial_path = self.behav_path.parent
        else:
            self.behav_path = None

        # Batch‑mode fallback
        if trial_id and not hasattr(self, "trial_path"):
            self.trial_path = Path(config.dir.project_home) / str(trial_id)

        if not hasattr(self, "trial_path"):
            raise ValueError(
                "Either `trial_id` or at least one explicit file path must be provided."
            )

    # ---------- helpers -------------------------------------------------
    def _list_files(self):
        return os.listdir(self.trial_path)

    # ---------- photometry ---------------------------------------------
    def open_photometry(self):
        """
        Returns
        -------
        pd.DataFrame | None
        """
        # User gave explicit file → trust it
        if self.photo_path:
            return pd.read_csv(self.photo_path)

        # Otherwise search within folder
        for file in self._list_files():
            if file.lower().endswith(".csv"):
                return pd.read_csv(self.trial_path / file)
        return None

    # ---------- behaviour ----------------------------------------------
    def open_behavior(self):
        """
        Returns
        -------
        pd.DataFrame | None
        """
        # Explicit path provided
        if self.behav_path:
            return self._load_behavior_file(self.behav_path)

        # Otherwise search in folder
        for file in self._list_files():
            if (
                "Raw data" in file
                and file.lower().endswith(".xlsx")
                and "$" not in file
            ):
                return self._load_behavior_file(self.trial_path / file)
        return None

    # ---------- internals ----------------------------------------------
    def _load_behavior_file(self, path: Path):
        """
        Loads a Noldus‑style xlsx and auto‑detects the header row if necessary.
        """
        df = pd.read_excel(
            path,
            header=[self.config.behav.behavior_row - 1],
            skiprows=[self.config.behav.behavior_row],
        )
        # If first column is not "Trial time", try to auto‑locate the header
        if df.columns[0] != "Trial time":
            df = self._find_header_row(path)
        return df

    def _find_header_row(self, path: Path):
        """
        find the row that contains event names in a Noldus style xlsx
        """
        raw = pd.read_excel(path, header=None)
        for idx, row in raw.iterrows():
            if any(str(cell).startswith("Trial time") for cell in row):
                return pd.read_excel(path, header=[idx], skiprows=[idx + 1])
        # Fallback: return original raw dataframe (unlikely)
        return raw
    

class BehaviorProcessor:
    """
    Handles behavior data extraction, validation, and cropping.
    """
    def __init__(self, config):
        self.config = config
        self.behaviors = []
        self.start_list = []
        self.end_list = []

    def extract_behavior_columns(self, df):
        """
        Puts the event headers in a comprehensive list
        """
        self.behaviors = list(df.columns)[7:-1]  # Noldus format assumption
        return self.behaviors

    def validate_start_param(self): #NOTE add to report in main if false
        """
        Confirms start parameter designated in configuration file is present in event list
        """
        return self.config.behav.start_parameter in self.behaviors

    def get_start_end_times(self, df): 
        """
        Identifies the frame placement of the 'start' parameter.
        """
        start_col = df[self.config.behav.start_parameter].values
        start_idxs = [i for i, val in enumerate(start_col) if val == 1]
        if not start_idxs:
            return [0], [len(df)]
        if self.config.behav.controls and len(start_idxs) == 4:
            return [start_idxs[0], start_idxs[2]], [start_idxs[1], start_idxs[3]]
        return [start_idxs[0]], [start_idxs[-1]]

    def trim_beginning(self, behav_df, photo_trace):
        """"
        Trims the front of the behavior data frame to match the start of the photometry dataframe.
        This is because the is sometimes a lag between the start of the behavior video and the photometry machine.
        """
        behav_time = ti.get_time_array(behav_df, self.config.acq.behav_fps)
        photo_time = ti.get_time_array(photo_trace, self.config.acq.photo_fps)

        print("Behav array: ", behav_df)
        print("Photo Array: ", photo_time)
        if behav_time.max() + 1 > photo_time.max():
            diff = behav_time.max() - photo_time.max()
            cutit = floor(diff * self.config.acq.behav_fps)
            return behav_df[cutit:]
        return behav_df

class PhotometryProcessor:
    """
    Handles photometry signal extraction, cropping, and correction.
    """
    def __init__(self, config):
        self.config = config
        self.photo_470 = None
        self.photo_410 = None

    def split_led_signals(self, df):
        """
        Extracts the interwoven 470 and 410 LED signals.
        """
        roi_col = [c for c in df.columns if "Region" in c][self.config.acq.roi]
        led = df["LedState"].values
        trace = df[roi_col].values
        p470, p410 = [], []
        for i, val in enumerate(trace):
            if led[i] == 6:
                p470.append(val)
            else:
                p410.append(val)
        self.photo_470, self.photo_410 = np.array(p470), np.array(p410)
        self._equalize_lengths()
        return self.photo_470, self.photo_410

    def _equalize_lengths(self):
        """
        Ensures 470 and 410 are the same lengths.
        """
        min_len = min(len(self.photo_470), len(self.photo_410))
        self.photo_470 = self.photo_470[:min_len]
        self.photo_410 = self.photo_410[:min_len]

    def correct_led_swap(self): #NOTE add user input to main
        """
        Corrects the swapping of the LED in case of incorrect assignment.
        """
        norm_410 = (self.photo_410 - np.min(self.photo_410)) / (np.ptp(self.photo_410))
        norm_470 = (self.photo_470 - np.min(self.photo_470)) / (np.ptp(self.photo_470))
        snr_410 = np.mean(norm_410) / np.std(norm_410)
        snr_470 = np.mean(norm_470) / np.std(norm_470)
        if snr_410 > snr_470:
            self.photo_410, self.photo_470 = self.photo_470, self.photo_410

    def correct_fps(self, behav_df: pd.DataFrame):
        """
        Matches the photometry FPS to expected rate by interpolating if needed.

        Parameters
        ----------
        behav_df : pd.DataFrame
            The trimmed behavior data to determine matching time.
        """
        if len(self.photo_470) == 0 or len(self.photo_410) == 0:
            raise ValueError("Cannot correct_fps: photo_470 or photo_410 is empty.")

        behav_time = np.arange(len(behav_df)) / self.config.acq.behav_fps
        if behav_time.max() == 0:
            raise ValueError("Behavior time duration is zero — cannot adjust fps.")

        input_fps = float(len(self.photo_470) / behav_time.max())

        if round(input_fps, 1) != self.config.acq.photo_fps:
            print(f"Correcting frames per second. Current fps: {input_fps}")

            total_frames = int(behav_time.max() * self.config.acq.photo_fps)
            interp_index = np.linspace(0, len(self.photo_470) - 1, total_frames)

            self.photo_470 = np.interp(interp_index, np.arange(len(self.photo_470)), self.photo_470)
            self.photo_410 = np.interp(interp_index, np.arange(len(self.photo_410)), self.photo_410)

            output_fps = float(len(self.photo_470) / behav_time.max())
            print(f"Corrected frames per second: {output_fps}")


class TrialProcessor:
    """
    Coordinates loading, validation, and processing of one trial.
    """
    def __init__(self, config, trial_id):
        self.config = config
        self.trial_id = trial_id
        self.loader = TrialLoader(config, trial_id)
        self.photo_df = None
        self.behav_df = None
        self.behav_proc = BehaviorProcessor(config)
        self.photo_proc = PhotometryProcessor(config)
        self.start_times = None
        self.end_times = None

    def process(self):
        self.photo_df = self.loader.open_photometry()
        self.behav_df = self.loader.open_behavior()

        if self.photo_df is None or self.behav_df is None:
            print(f"Missing files for {self.trial_id}")
            return None

        behaviors = self.behav_proc.extract_behavior_columns(self.behav_df)
        if not self.behav_proc.validate_start_param():
            print(f"Invalid start parameter for {self.trial_id}")
            return None

        self.photo_proc.split_led_signals(self.photo_df)

        self.photo_proc.correct_led_swap()

        self.behav_df = self.behav_proc.trim_beginning(self.behav_df, self.photo_proc.photo_470)
        self.start_times, self.end_times = self.behav_proc.get_start_end_times(self.behav_df)

        self.crop_data(self.start_times[0], self.end_times[-1])

        self.photo_proc.correct_fps(self.behav_df)

        return self.photo_proc.photo_470, self.photo_proc.photo_410, behaviors, self.behav_df, self.start_times, self.end_times

    def crop_data(self, start_idx, end_idx):
        """
        Crops both photometry and behavior data based on configuration settings.
        """
        behav_fps = self.config.acq.behav_fps
        photo_fps = self.config.acq.photo_fps
        pconvert = photo_fps / behav_fps

        if self.config.acq.full_trace:
            crop_front_photo = int(self.config.acq.crop_front * photo_fps)
            crop_end_photo = int(self.config.acq.crop_end * photo_fps)
            crop_front_behav = int(self.config.acq.crop_front * behav_fps)
            crop_end_behav = int(self.config.acq.crop_end * behav_fps)

            self.photo_proc.photo_470 = self.photo_proc.photo_470[crop_front_photo : -crop_end_photo]
            self.photo_proc.photo_410 = self.photo_proc.photo_410[crop_front_photo : -crop_end_photo]
            self.behav_proc.behav_df = self.behav_proc.behav_df[crop_front_behav : -crop_end_behav].reset_index(drop=True)

        else:
            pre_photo = int(self.config.acq.time_from_start * photo_fps)
            post_photo = int(self.config.acq.time_from_end * photo_fps)
            pre_behav = int(self.config.acq.pre_s * behav_fps)
            post_behav = int(self.config.acq.post_s * behav_fps)

            start_photo = int(max(0, (start_idx * pconvert) - pre_photo))
            end_photo = int((end_idx * pconvert) + post_photo)

            start_behav = int(max(0, start_idx - pre_behav))
            end_behav = int(end_idx + post_behav)

            self.photo_proc.photo_470 = self.photo_proc.photo_470[start_photo:end_photo]
            self.photo_proc.photo_410 = self.photo_proc.photo_410[start_photo:end_photo]
            self.behav_proc.behav_df = self.behav_proc.behav_df[start_behav:end_behav].reset_index(drop=True)

        return self.photo_proc.photo_470, self.photo_proc.photo_410, self.behav_proc.behav_df