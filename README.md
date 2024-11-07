
# Photometry Data Analysis and Visualization Repository

This repository contains Python scripts for analyzing and visualizing photometry data. The scripts are organized to streamline data processing, statistical summarization, and quick visualization of photometry signals, allowing researchers to efficiently derive insights from their data. Additional scripts are included to perform nested statistical tests, enabling a more robust and comprehensive analysis.

## Table of Contents
- [Installation](#installation)
- [Required Folder Structure](#required-folder-structure)
- [Key Features](#key-features)
- [Usage](#usage)

## Installation

To set up the environment for running these photometry data analysis scripts, start by installing the required dependencies. Ensure you have Python installed, then navigate to the repository directory and run:

```bash
pip install -r requirements.txt
```

This command installs all necessary packages, including:
- `pandas` (2.0.3) for data manipulation
- `numpy` (1.24.3) for numerical operations
- `matplotlib` (3.8.2) and `seaborn` (0.13.2) for data visualization
- `scipy` (1.11.1) and `statsmodels` (0.14.0) for statistical analysis
- `PyYAML` (6.0.1) for configuration management

These dependencies are essential for processing, summarizing, and visualizing photometry data and ensuring compatibility across the scripts.

## Required Folder Structure

For the scripts to function correctly, organize your data files as follows:

```
root/
¦   
+---Virgin Females                             # Defined name for analysis group
    ¦   
    +---1                                      # Sample ID 1
    ¦       1_split2022-09-20T14_28_47.CSV     # Photometry File
    ¦       Raw data-MeP-CRFR2photo_pupblock1-Trial 1 (1).xlsx   # Behavior File
    ¦       
    +---21                                     # Sample ID 21
    ¦       1_split2022-09-28T14_34_01.CSV     # Photometry File
    ¦       Raw data-Pup Block 2-Trial 1 (2).xlsx                 # Behavior File
    ¦               
    +---Archive                                # Folder to store samples you wish to ignore
    +---Videos                                 # Folder to store acquisition videos
```

### Folder Details
- **Group Folder** (e.g., "Virgin Females"): Each analysis group should have its own folder with a descriptive name.
- **Sample Folders** (e.g., "1", "21"): Within each group, create a subfolder for each sample. Name each folder by the sample ID.
- **File Naming**:
  - Place the photometry data file (`.CSV`) and the behavior data file (`.xlsx`) within each sample folder.
  - The code expects only one `.CSV` and one `.xlsx` file per sample folder to avoid conflicts.
  - Additional files or folders can exist in the sample folders, but they should not have a `.CSV` or `.xlsx` extension.
- **Optional Folders**:
  - **Archive**: Use this folder to store samples you wish to exclude from analysis.
  - **Videos**: Store acquisition videos here if they’re part of the analysis group.

## Key Features

This repository provides a full suite of tools for processing, summarizing, and visualizing photometry data, with each script designed to facilitate a specific step in the workflow:

- **Event-Based Signal Extraction and Processing**
  - `batch_photocode_v2.py` extracts specific portions of the photometry signal aligned with the onset of behavioral events, based on time intervals defined in `config.yaml`.
  - Fits, smooths, and normalizes the photometric signal, providing three trace options for analysis.
  - Outputs processed signal segments tailored for further statistical analysis, accommodating any number of events with customizable durations.

- **Comprehensive Summary Statistics**
  - `summarize_values_v2.py` calculates essential metrics:
    - Percentage change
    - Z-scores
    - Area under the curve (AUC)
  - Provides both individual and averaged summaries, offering a thorough quantitative overview.

- **Clear, Annotated Visualizations**
  - `quickplots-stats_v2.py` generates publication-ready plots with:
    - Automated metric labels
    - Statistical annotations to highlight significant findings
  - Enables easy, intuitive interpretation of complex data.

- **Advanced Statistical Analysis**
  - Additional scripts in the `stats` folder perform complex analyses, including:
    - **Nested statistical models** (e.g., mixed-effects models with fixed and random effects)
    - **Multi-comparison tests** for group comparisons
  - Custom color schemes and data compilation tools enhance flexibility, making it easy to tailor analyses to specific research questions.

These features together create an efficient, end-to-end workflow for photometry data, from raw processing to in-depth statistical analysis and visual presentation.
