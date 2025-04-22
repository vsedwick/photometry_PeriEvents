import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import os


class TraceProcessor:
    """
    Processes photometry traces (470 and 410 nm) including fitting, normalization, and smoothing.
    """
    def __init__(self, photo_470, photo_410, fps):
        self.photo_470 = photo_470
        self.photo_410 = photo_410
        self.fps = fps
        self.time = np.round(np.linspace(0, len(photo_410) / fps, len(photo_410)), 2)
        self.fit = None
        self.dff = None
        self.zscored = None
        self.smoothed = None

    def fit_baseline(self):
        """Fits a double exponential baseline to the 410 signal."""
        try:
            p0 = [1, 1, 1, 0.1]
            params, _ = curve_fit(exp2, self.time, self.photo_410, p0, maxfev=10000)
            self.fit = exp2(self.time, *params)
        except Exception:
            self.fit = np.zeros_like(self.photo_410)

    def compute_dff(self):
        """Computes deltaF/F based on the fitted 410 baseline."""
        if self.fit is None:
            raise ValueError("Must call fit_baseline() before computing dF/F")
        self.dff = (self.photo_470 - self.fit) / self.fit

    def zscore(self):
        """Z-score normalizes the dF/F trace."""
        if self.dff is None:
            raise ValueError("Must compute dF/F before z-scoring")
        self.zscored = (self.dff - np.mean(self.dff)) / np.std(self.dff)

    def smooth(self, window_length=59, polyorder=3):
        """Applies Savitzky-Golay smoothing to the z-scored trace."""
        if self.zscored is None:
            raise ValueError("Must z-score before smoothing")
        if len(self.zscored) < window_length:
            self.smoothed = self.zscored
        else:
            self.smoothed = savgol_filter(self.zscored, window_length, polyorder)

    def save(self, output_dir, trace_name):
        """Saves the final smoothed trace to disk."""
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, trace_name), self.smoothed)


# Support functions

def get_time_array(signal_or_df, fps):
    length = len(signal_or_df)
    return np.round(np.linspace(0, length / fps, length), 2)

def get_frame_array(signal_or_df):
    length = len(signal_or_df)
    return np.arange(0, length)

def save_fitted_trace(output_dir, trace_name, trace_data):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, trace_name), trace_data)

def exp2(t, a, b, c, d):
    return a * np.exp(-b * t) + c * np.exp(-d * t)

def zscore(trace):
    return (trace - np.mean(trace)) / np.std(trace)

def calculate_snr(trace):
    return np.mean(trace) / np.std(trace)

def fit_exponential_baseline(signal, time):
    try:
        p0 = [1, 1, 1, 0.1]
        params, _ = curve_fit(exp2, time, signal, p0, maxfev=10000)
        return exp2(time, *params)
    except Exception:
        return np.zeros_like(signal)

def compute_dff(signal, fit):
    return (signal - fit) / fit

def smooth_trace(trace, window_length=59, polyorder=3):
    if len(trace) < window_length:
        return trace
    return savgol_filter(trace, window_length, polyorder)