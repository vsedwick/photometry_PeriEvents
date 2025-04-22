import os
import sys
from config import load_config, AnalysisParameters
from trial_processing_refactor import TrialProcessor
from photo_signal import TraceProcessor, save_fitted_trace
from behavior_processing import BehaviorRestructurer
from event_extraction import ExtractbyEvent
from utils import make_folder
from pathlib import Path
import numpy as np


def save_variables(arrays, config, output_dir):
    """
    Saves arrays with consistent naming convention.

    Args:
        arrays (list of np.ndarray): Traces to save.
        config (AnalysisParameters): Configuration.
        output_dir (str): Directory to save to.
    """
    for i, array in enumerate(arrays):
        name = f"{config.trial_id}_{config.event_name}_{config.stage}_{i}.npy"
        save_fitted_trace(output_dir, name, array)


def main(configuration_file):
    config = AnalysisParameters(load_config(configuration_file))

    exclude_folder = ['Behaviors', 'Videos', 'Summary', 'Archive', 'Fittings']
    analysis_folders = os.listdir(config.dir.project_home)
    event_save_path = os.path.join(config.dir.project_home, "Behaviors")
    make_folder(event_save_path)
    skipped_trials = []

    for subject_trial_id in analysis_folders:
        if (not os.path.isdir(os.path.join(config.dir.project_home, subject_trial_id)) or 
            subject_trial_id in exclude_folder):
            continue

        config.trial_id = subject_trial_id
        processor = TrialProcessor(subject_trial_id, config)
        result = processor.process()

        if result is None:
            skipped_trials.append(subject_trial_id)
            continue

        photo_470, photo_410, behaviors, behav_df, start_times, end_times = result
        config.subject_path = processor.loader.trial_path

        trace_handler = TraceProcessor(photo_470, photo_410, config.acq.photo_fps)
        trace_handler.fit_baseline()
        trace_handler.compute_dff()
        trace_handler.zscore()
        trace_handler.smooth()

        trace_diary = {
            "zscored": trace_handler.zscored,
            "smoothed": trace_handler.smoothed,
            "dff": trace_handler.dff
        }

        behavior_handler = BehaviorRestructurer(config)
        event_dictionary = behavior_handler.get_events(behav_df, behaviors, start_frame=start_times[0])
        event_dictionary = behavior_handler.apply_min_duration(event_dictionary)

        if config.behav.compile_behavior:
            event_dictionary = behavior_handler.group_behaviors(event_dictionary)

        if config.behav.controls:
            stages = ['before', 'control', 'during']
        else:
            stages = ['before', 'during']

        for trace_name, trace in trace_diary.items():
            trace_path = make_folder(os.path.join(event_save_path, trace_name))

            for event_name, (starts, _) in event_dictionary.items():
                config.event_name = event_name
                event_path = make_folder(os.path.join(trace_path, event_name))

                for stage in stages:
                    config.stage = stage
                    stage_path = make_folder(os.path.join(event_path, stage))

                    try:
                        if stage == 'before':
                            time_markers = [0, start_times[0]]
                        elif stage == 'control':
                            time_markers = [start_times[0], end_times[0]]
                        elif stage == 'during' and config.behav.controls:
                            time_markers = [start_times[1], end_times[1]]
                        elif stage == 'during':
                            time_markers = [start_times[0], end_times[0]]
                    except IndexError:
                        print("Check 'use_control_trial' parameter in configuration file.")
                        continue

                    extraction = ExtractbyEvent(config)
                    valid_events = extraction.handle_nan_events(trace, starts)
                    extractor = ExtractbyEvent(config)
                    results = extractor.extract_aligned_traces(trace, valid_events)

                    # Save results per stage
                    save_variables(results, config, stage_path)

                # Save the full smoothed trace
                if trace_name == "smoothed":
                    out_path = make_folder(os.path.join(config.dir.project_home, "Fittings"))
                    save_fitted_trace(out_path, f"{subject_trial_id}_smoothed_trace", trace)

    print("\nExtraction complete.")
    print(f"Peri-events saved in: {event_save_path}")
    if skipped_trials:
        print("\nTrials not analyzed:", skipped_trials)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py path/to/config.yaml")
        sys.exit(1)

    main(sys.argv[1])
