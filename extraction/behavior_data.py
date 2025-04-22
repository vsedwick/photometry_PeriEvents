import numpy as np
import pandas as pd


class BehaviorRestructurer:
    """
    Extracts and restructures behavior event timestamps, point events,
    applies grouping, duration filtering, and zone filtering.
    """
    def __init__(self, config):
        self.config = config

    def get_events(self, df, behaviors, start_frame):
        """
        Gets event onset and offset times for each behavior.

        Args:
            df (pd.DataFrame): Behavior data.
            behaviors (list): List of behaviors to score.
            start_frame (int): Offset frame to adjust time alignment.

        Returns:
            dict: Dictionary of {behavior: [onsets, offsets]} pairs.
        """
        results = {}
        for behav in behaviors:
            state = df[behav].values
            starts, stops = [], []
            for i in range(1, len(state)):
                if state[i] == 1 and state[i - 1] == 0:
                    starts.append(i - start_frame)
                elif state[i] == 0 and state[i - 1] == 1:
                    stops.append(i - start_frame)
            results[behav] = [starts, stops]
        return results

    def get_point_events(self, df, point_labels, start_frame):
        """
        Gets frame numbers of point events.

        Args:
            df (pd.DataFrame): Behavior data.
            point_labels (list): List of event labels.
            start_frame (int): Offset frame.

        Returns:
            dict: {event: [frames]}
        """
        result = {}
        for label in point_labels:
            state = df[label].values
            frames = [i - start_frame for i, val in enumerate(state) if val == 1]
            result[label] = frames
        return result

    def apply_min_duration(self, behavior_dict):
        """
        Filters out behavior events shorter than minimum duration.

        Args:
            behavior_dict (dict): {behavior: [starts, stops]}

        Returns:
            dict: Filtered behavior events.
        """
        min_dur = int(self.config.behav.min_duration * self.config.acq.behav_fps)
        filtered = {}
        for behav, (starts, stops) in behavior_dict.items():
            f_starts, f_stops = [], []
            for s, e in zip(starts, stops):
                if (e - s) >= min_dur:
                    f_starts.append(s)
                    f_stops.append(e)
            filtered[behav] = [f_starts, f_stops]
        return filtered

    def group_behaviors(self, behavior_dict):
        """
        Groups behaviors based on config-specified groupings.

        Args:
            behavior_dict (dict): {behavior: [starts, stops]}

        Returns:
            dict: {group_name: [starts, stops]}
        """
        grouped = {}
        for group, members in self.config.behav.Groups.items():
            g_starts, g_stops = [], []
            for m in members:
                if m in behavior_dict:
                    s, e = behavior_dict[m]
                    g_starts.extend(s)
                    g_stops.extend(e)
            grouped[group] = [sorted(g_starts), sorted(g_stops)]
        return grouped

    def add_zone_event(self, df, start_frame):
        """
        Extracts zone entry timestamps if zone data is provided.

        Args:
            df (pd.DataFrame): Behavior DataFrame.
            start_frame (int): Alignment offset.

        Returns:
            list: Frame indices where zone was active.
        """
        if not self.config.behav.add_zone:
            return []
        zone_state = df[self.config.behav.zone].values
        return [i - start_frame for i, val in enumerate(zone_state) if val == 1]
