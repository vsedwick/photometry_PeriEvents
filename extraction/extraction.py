import numpy as np


class ExtractbyEvent:
    """
    Extracts peri-event trace windows from continuous signals, including trace alignment,
    dual-trace handling, and basic aggregation.
    """
    def __init__(self, config):
        self.config = config

    def extract_aligned_traces(self, trace, event_frames):
        """
        Extracts peri-event windows from a trace based on event frame indices.

        Args:
            trace (np.ndarray): Input signal.
            event_frames (list): Frame indices to align to.

        Returns:
            list: List of extracted trace windows.
        """
        pre = int(self.config.acq.pre_s * self.config.acq.photo_fps)
        post = int(self.config.acq.post_s * self.config.acq.photo_fps)
        total_len = len(trace)

        windows = []
        for frame in event_frames:
            start = frame - pre
            stop = frame + post
            if start >= 0 and stop <= total_len:
                windows.append(trace[start:stop])
        return windows

    def extract_dual_traces(self, trace1, trace2, event_frames):
        """
        Extracts peri-event windows from two synchronized traces.

        Args:
            trace1 (np.ndarray): First signal.
            trace2 (np.ndarray): Second signal.
            event_frames (list): Frame indices to align to.

        Returns:
            list: List of (trace1_window, trace2_window) tuples.
        """
        pre = int(self.config.acq.pre_s * self.config.acq.photo_fps)
        post = int(self.config.acq.post_s * self.config.acq.photo_fps)
        total_len = min(len(trace1), len(trace2))

        pairs = []
        for frame in event_frames:
            start = frame - pre
            stop = frame + post
            if start >= 0 and stop <= total_len:
                pairs.append((trace1[start:stop], trace2[start:stop]))
        return pairs

    def extract_zone_windows(self, trace, zone_frames):
        """
        Extracts single-frame aligned zone windows.

        Args:
            trace (np.ndarray): Input signal.
            zone_frames (list): Frame indices for zone presence.

        Returns:
            list: List of trace values at those frames.
        """
        return [trace[f] for f in zone_frames if 0 <= f < len(trace)]

    def average_traces(self, trace_list):
        """
        Computes the average trace across a list of aligned traces.

        Args:
            trace_list (list): List of np.ndarrays.

        Returns:
            np.ndarray: Mean trace.
        """
        if not trace_list:
            return np.array([])
        return np.mean(np.stack(trace_list), axis=0)

    def handle_nan_events(self, trace, event_frames):
        """
        Removes events that would extract NaNs.

        Args:
            trace (np.ndarray): Input signal.
            event_frames (list): Event indices.

        Returns:
            list: Valid event indices.
        """
        pre = int(self.config.acq.pre_s * self.config.acq.photo_fps)
        post = int(self.config.acq.post_s * self.config.acq.photo_fps)
        total_len = len(trace)
        valid = []
        for frame in event_frames:
            start = frame - pre
            stop = frame + post
            if start >= 0 and stop <= total_len and not np.any(np.isnan(trace[start:stop])):
                valid.append(frame)
        return valid
