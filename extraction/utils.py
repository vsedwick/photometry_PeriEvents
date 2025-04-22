import os
import re
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def make_folder(folder_path):
    """
    Creates a folder if it does not exist.

    Args:
        folder_path (str or Path): Path to the folder.
    """
    os.makedirs(folder_path, exist_ok=True)


def extract_trial_id_from_filename(filename):
    """
    Extracts a trial ID from a filename using a regex pattern.

    Args:
        filename (str): Filename string.

    Returns:
        str: Extracted trial ID or the original filename if no match.
    """
    match = re.search(r"(\d+[_-]?trial[_-]?\d+)", filename, re.IGNORECASE)
    return match.group(1) if match else filename


def get_csv_files(folder_path):
    """
    Returns list of CSV files in a folder.

    Args:
        folder_path (str): Path to the directory.

    Returns:
        list: List of CSV file paths.
    """
    return [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]


def flatten_nested_dict(d):
    """
    Flattens a nested dictionary of lists into a single list of values.

    Args:
        d (dict): Dictionary of lists.

    Returns:
        list: Flattened list of all values.
    """
    return [item for sublist in d.values() for item in sublist]


def safe_mean(trace_list):
    """
    Computes mean of a list of arrays, returns zeros if list is empty.

    Args:
        trace_list (list): List of np.ndarray.

    Returns:
        np.ndarray: Mean array or array of zeros.
    """
    if not trace_list:
        return np.zeros(1)
    return np.mean(np.stack(trace_list), axis=0)


def load_yaml_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str or Path): Path to config file.

    Returns:
        dict: Parsed YAML content.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def convert_frames_to_seconds(frames, fps):
    """
    Converts a list of frame indices to seconds.

    Args:
        frames (list): Frame indices.
        fps (float): Frame rate.

    Returns:
        list: Timestamps in seconds.
    """
    return [round(f / fps, 2) for f in frames]


def clean_column_names(df):
    """
    Cleans DataFrame column names (strip, lower, remove special characters).

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df.columns = [re.sub(r'[^\w\d_]+', '', col).strip().lower() for col in df.columns]
    return df
