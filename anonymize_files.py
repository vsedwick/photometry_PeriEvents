import os
from src.batch_photocode_v2 import make_folder
from pathlib import Path
import pandas as pd
import sys

project_home = r"C:\Users\sedwi\Desktop\Portfolio\Thesis_Research (python)\Photometry\example_data"

def search_and_destroy(keywords, df):
    
    if type(keywords) is list:
        location = df.apply(lambda row: row.astype(str).str.contains('|'.join(keywords)).any(), axis=1)
        return int(df[location].index[0])
    else:
        location = df.apply(lambda row: row.astype(str).str.contains(keywords).any(), axis=1)
        if location.any():
            row_index = int(df[location].index[0])
            col_index = df.loc[row_index].astype(str).tolist().index(keywords)

        return row_index, col_index

def main(project_home):
    assign_dict = {}
    #Begin iterating through analysis folders
    analysis_folders = os.listdir(project_home)
    for subject_trial_id in analysis_folders:
        #Exclude processed value or video folders that may be in the project home directory
        trial_id = subject_trial_id
        exclude_folder = ['Behaviors', 'Videos', 'Summary', 'Archive', 'Fittings']
        if not os.path.isdir(os.path.join(project_home, subject_trial_id)) or subject_trial_id in exclude_folder:
            continue
        else:
            subject_trial_path = make_folder(subject_trial_id, project_home)
            #Identify and load raw files
            files_to_load = os.listdir(subject_trial_path)

            if len(files_to_load)>= 1:
                for i in files_to_load:
                    if 'Raw data' in i and i.endswith('.xlsx') and '$' not in i:
                        file_path = Path(os.path.join(subject_trial_path, i))
                        print(trial_id, '   ', str(file_path).split('\\')[-1])
                        behav_raw = pd.read_excel(file_path, header = None)

            #Identify video file row and replace
            keywords_to_delete = ["Video file", "Video start time", "Reference time", "Experiment", "Start time"]
            for i in keywords_to_delete:
                row_index, col_index = search_and_destroy(i, behav_raw)
                behav_raw.iat[row_index, int(col_index) + 1] = ''

            behav_row = search_and_destroy(['Trial time', 'Recording time', 'X center', 'Y center', 'Area', 'Areachange'], behav_raw)
            col = 7
            counter = 1

            if len(assign_dict) == 0:
                while behav_raw.iat[behav_row, col] != "Result 1":
                    new_val = f"Event {counter}"
                    assign_dict.update({behav_raw.iat[behav_row, col] : new_val})
                    behav_raw.iat[behav_row, col] = new_val
                    col += 1
                    counter += 1
                behav_raw.to_excel(file_path, header = None, index = None)
            else:
                while behav_raw.iat[behav_row, col] != "Result 1":
                    if behav_raw.iat[behav_row, col] not in assign_dict:
                        max_val = len(assign_dict) + 1
                        new_val = f"Event {max_val}"
                        assign_dict.update({behav_raw.iat[behav_row, col] : new_val})
                    else:
                        new_val = assign_dict[behav_raw.iat[behav_row, col]]
                    behav_raw.iat[behav_row, col] = new_val
                    col += 1
                    counter += 1
                behav_raw.to_excel(file_path, header = None, index = None)

    print(assign_dict)
if __name__ == "__main__": 
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(project_home)
