import yaml
from pathlib import Path
import matplotlib.colors as mcolors


class DirectoryConfig:
    """
    Configuration for project directories.
    
    Attributes:
        project_home (Path): Path to the root project directory.
    """
    def __init__(self, data: dict):
        self.project_home = self._validate_path(data["project_home"])

    def _validate_path(self, path_str):
        """Ensures the directory path exists."""
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Directory '{path}' does not exist.")
        return path
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        pairs = [f"{k}={v!r}" for k, v in vars(self).items()]
        # Join with newline + indent
        body  = "\n  ".join(pairs)
        return f"<{self.__class__.__name__}(\n  {body}\n)>"



class AcquisitionConfig:
    """
    Configuration for photometry and video acquisition parameters.
    
    Attributes:
        silent_mode (bool): Whether to suppress prompts.
        full_trace (bool): Whether to analyze full trace.
        behav_fps (float): Behavior video frame rate.
        photo_fps (float): Photometry data frame rate.
        pre_s (int): Pre-event window in seconds.
        post_s (int): Post-event window in seconds.
        roi (int): Region of interest index.
        offset (int): ROI offset.
        crop_front (int): Front crop time (if full trace).
        crop_end (int): End crop time (if full trace).
        time_from_start (int): Start time offset (if not full trace).
        time_from_end (int): End time offset (if not full trace).
    """
    def __init__(self, data: dict):
        self.silent_mode = self._validate_bool(data["silent_mode"])
        self.full_trace = self._validate_bool(data["full_trace"])
        self.behav_fps = data["behaviorvideo_fps"]
        self.photo_fps = data["photometry_fps"]
        self.pre_s = data["peri-baseline_seconds"]
        self.post_s = data["peri-event_seconds"]
        self.roi = data["roi"]
        self.offset = data["offset"]
        if self.full_trace:
            self.crop_front = data["crop_front"]
            self.crop_end = data["crop_end"]
        else:
            self.time_from_start = data["time_from_start_s"]
            self.time_from_end = data["time_from_end_s"]

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        pairs = [f"{k}={v!r}" for k, v in vars(self).items()]
        # Join with newline + indent
        body  = "\n  ".join(pairs)
        return f"<{self.__class__.__name__}(\n  {body}\n)>"


    def _validate_bool(self, val):
        """Raises error if val is not boolean."""
        if not isinstance(val, bool):
            raise ValueError("Expected a boolean.")
        return val


class BehaviorConfig:
    """
    Configuration for behavioral scoring and grouping.
    
    Attributes:
        behavior_row (int): Excel row where behavior data starts.
        start_parameter (str): Column name used to detect start.
        min_duration (float): Minimum event duration.
        controls (bool): Whether trial includes controls.
        perievent_limit (float): Limit for peri-event windowing.
        point_events (list): List of point events to score.
        compile_behavior (bool): Whether to compile grouped events.
        Groups (dict): Grouped behavior mappings.
        add_zone (bool): Whether to include a custom zone.
        zone (list): List defining custom zone if applicable.
        events_to_score (list): Behaviors to extract.
    """
    def __init__(self, data: dict):
        self.behavior_row = data["behavior_row"]
        self.start_parameter = data["start_parameter"]
        self.min_duration = data["minimum_accepted_duration_s"]
        self.controls = self._validate_bool(data["use_control_trial"])
        self.perievent_limit = data["limit_perievent_extraction"]
        self.point_events = data["point_events"]
        self.compile_behavior = self._validate_bool(data["Compile_behaviors"])
        self.Groups = self._clean_dict(data.get("Behavior_Groupings", {})) if self.compile_behavior else None
        self.add_zone = self._validate_bool(data.get("Add_a_zone", False))
        self.zone = data.get("Zone") if self.add_zone else None
        self.events_to_score = [e for e in data["Behaviors_to_Score"] if e is not None] or None

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        pairs = [f"{k}={v!r}" for k, v in vars(self).items()]
        # Join with newline + indent
        body  = "\n  ".join(pairs)
        return f"<{self.__class__.__name__}(\n  {body}\n)>"


    def _validate_bool(self, val):
        """Ensures a boolean value."""
        if not isinstance(val, bool):
            raise ValueError("Expected a boolean.")
        return val

    def _clean_dict(self, d):
        """Removes empty or None entries in groupings."""
        return {k: [v for v in vals if v is not None] for k, vals in d.items() if vals}


class PlottingConfig:
    """
    Configuration for plotting. Only need preferred color.
    
    Attributes:
        color (str): Plot color.
    """
    def __init__(self, data: dict):
        self.color = self._validate_color(data["color"])

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        pairs = [f"{k}={v!r}" for k, v in vars(self).items()]
        # Join with newline + indent
        body  = "\n  ".join(pairs)
        return f"<{self.__class__.__name__}(\n  {body}\n)>"


    def _validate_color(self, color):
        """Validates if the provided color is known."""
        if color in mcolors.CSS4_COLORS or color in mcolors.TABLEAU_COLORS or mcolors.is_color_like(color):
            return color
        print("Invalid color, defaulting to 'blue'")
        return "blue"


class AnalysisParameters:
    """
    Container for the full set of analysis configuration parameters.
    
    Attributes:
        dir (DirectoryConfig): Directory paths.
        acq (AcquisitionConfig): Acquisition parameters.
        behav (BehaviorConfig): Behavioral scoring parameters.
        plot (PlottingConfig): Plot styling.
    """
    def __init__(self, config: dict):
        self.dir = DirectoryConfig(config["Directory_Information"])
        self.acq = AcquisitionConfig(config["Acquisition_Information"])
        self.behav = BehaviorConfig(config["Behavior_Parameters"])
        self.plot = PlottingConfig(config["Plotting_parameters"])

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        pairs = [f"{k}={v!r}" for k, v in vars(self).items()]
        # Join with newline + indent
        body  = "\n  ".join(pairs)
        return f"<{self.__class__.__name__}(\n  {body}\n)>"


    def validate(self):
        """Validates cross-field logic across all config sections."""
        # Check time-based params
        if self.acq.full_trace:
            assert self.acq.crop_end >= 0, "crop_end must be ≥ 0 when full_trace is True"
            assert self.acq.crop_front >= 0, "crop_front must be ≥ 0 when full_trace is True"
        else:
            assert self.acq.time_from_start >= 0, "time_from_start must be ≥ 0 when full_trace is False"
            assert self.acq.time_from_end >= 0, "time_from_end must be ≥ 0 when full_trace is False"

        # Check that event scoring is aligned with behavior scoring
        if self.behav.compile_behavior:
            assert self.behav.Groups is not None and isinstance(self.behav.Groups, dict), \
                "Behavior_Groupings must be defined and valid if Compile_behaviors is True"

        if self.behav.add_zone:
            assert self.behav.zone is not None, "Zone must be defined if Add_a_zone is True"

        # Basic sanity checks
        assert self.acq.photo_fps > 0, "Photometry FPS must be > 0"
        assert self.acq.behav_fps > 0, "Behavior video FPS must be > 0"
        assert self.acq.pre_s >= 0, "Pre-event seconds must be ≥ 0"
        assert self.acq.post_s >= 0, "Post-event seconds must be ≥ 0"

def load_config(config_path: str) -> dict:
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the config file.

    Returns:
        dict: Parsed YAML configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


