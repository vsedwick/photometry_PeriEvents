
configuration_file = r'config.yaml'


from batch_photocode_v2 import make_folder, load_config, AnalysisParameters
import pandas as pd
import numpy as np
import os
import csv
from statistics import mean
import sys
from scipy import stats
import matplotlib.pyplot as plt

class EmptyDictionaries:
    def __init__(self):

        self.percentage_diary = {
            "title" : "Indv Percentage Change"
        }
        self.perc_avg={
            "title" : "Avg Percentage Change"
        }
        self.zscore_avg = {
            "title" : "Avg Zscore"
        }

        self.mag_auc={
            "title" : "Indv AUC Magnitude Change"
        }
        self.mag_auc_avg={
            "title" : "Avg AUC Magnitude Change"
        }
        self.auc_diary = {
            "title" : "Indv AUC"
        } 
        self.auc_avg={
            "title" : "Avg AUC"
        }
        self.raw_diary_i = {
            "title" : "Indv Raw Means"
        }
        self.raw_diary_avg = {
            "title" : "Avg Raw Means"
        }
        self.raw_diff_ind = {
            "title" : "Indv Raw Means Differences"
        }
        self.raw_diff_avg = {
            "title" : "Avg Raw Means Differences"
        }

        self.zscore_ind={
            "title" : "Indv Zscore"
        }
        self.behavior_list = []
    
    @staticmethod
    def load_full_dict(dicts, key, value, ids):

        base, event = value
        if "Indv" not in dicts["title"]: #check if exists
            dicts[f"Animal ID_{key}"] = list(dict.fromkeys(ids))
        else:
            dicts[f"Animal ID_{key}"] = ids

        dicts[f"{key}_baseline"] = base
        dicts[key] = event

        return dicts
    
    @staticmethod
    def load_dict_diffs( dicts, key, value, ids):

        if "Indv" not in dicts["title"]: #check if exists
            dicts[f"Animal ID_{key}"] = list(dict.fromkeys(ids))
        else:
            dicts[f"Animal ID_{key}"] = ids
        dicts[key] = value
      
        return dicts
    
class GrabEventValues:
    def __init__(self, config, data_list, event_path):
        
        self.config = config
        self.data_list = data_list
        self.event_path = event_path
        self.event_name = event_path.split('\\')[-1]
        self.stage = event_path.split('\\')[-2]
        self.id = None
        # VARIABLES TO SAVE FOR EACH BEHAVIOR; EMPTY WHEN NEXT BEHAVIOR FOLDER OPENS
        self.event_means = []  # Decide if I wanna append or extend
        self.baseline_means = []
        self.entrance_event = []; self.entrance_baseline = []
        self.id_list = []
        self.meansof_baseline = []; self.meansof_event = []
        self.peri_event_matrix = []; self.peri_baseline_matrix = []
        self.entrance_base_matrix = []; self.entrance_event_matrix = []
        self.avg_matrix_base=[]; self.avg_matrix_event=[]
        
    def match_by_id(self):
        for data in self.data_list:
            if 'event_means' in data:
                self.id = self._get_id(data)
                contents = self._openfile(data)
                self.event_means.extend(self._flatten(contents))

                self.id_list.extend([str(self.id) for _ in range(len(contents))])

                remaining_list = [i for i in self.data_list if i != data]

                for match in remaining_list:
                    id2 = self._get_id(match)
                    if self.id == id2:
                        if 'baseline_means' in match:
                            contents = self._openfile(match)
                            self.baseline_means.extend(self._flatten(contents))

                        if 'peri_event_matrix' in match:
                            contents = self._openfile(match)
                            self.avg_matrix_event.append(np.average(np.asarray(contents, dtype = float), axis = 0))
                            self.peri_event_matrix.extend(contents)

                        if 'peri_baseline_matrix' in match:
                            contents = self._openfile(match)
                            self.avg_matrix_base.append(np.average(np.asarray(contents, dtype = float), axis = 0)) #this needs to be cropped
                            self.peri_baseline_matrix.extend(contents)

                        if self.event_name != self.config.start_parameter:
                
                            if 'means_to_compare' in match:
                                contents = self._openfile(match)
                                self.meansof_event.extend(contents[1])
                                self.meansof_baseline.extend(contents[0])               
    @staticmethod
    def _flatten(x):
        list = [float(i) for ii in x for i in ii]
        return list
    
    @staticmethod
    def _get_id(file):
        return file.split('_')[-1].split('.')[0]
    
    def _openfile(self, file):
        data_path = os.path.join(self.event_path, file)
        return list(csv.reader(open(data_path)))

class ZscorenPlot:
    def __init__(self, config):
        self.config = config
        self.baseline_length = int(self.config.pre_s * self.config.photo_fps)
        self.event_length = int(self.config.post_s * self.config.photo_fps)
        self.zbase_avg = []
        self.zevent_avg = []
        self.colors = config.color
        self.event_name = config.event_name

        self.save_path = make_folder('zscore_figs', config.subject_path)

    @staticmethod
    def calculate_values(baseline_matrix, event_matrix, length):

        zscore_diary = []
        try:
            for base, event in zip(np.array(baseline_matrix, dtype = float), np.array(event_matrix, dtype = float)):
                means = np.average(base[-length : ])
                stdd = np.std(base[-length : ])

                full_array = np.concatenate((base[-length : ], event))
                #baseline part
                zscore = [(i - means) / stdd for i in full_array]
                # zscore.extend([(i - means) / stdd for i in event])
                zscore_diary.append(zscore)

        except IndexError:
            print("Likely a length mismatch, please check Pre_s and Post_s in configuration settings")
        except ValueError:
            print("Error")

        return zscore_diary
    
    def average_zscores(self, listOlists, length):
        # Check if elements in listOlists are iterable
        if not all(isinstance(i, (list, np.ndarray)) for i in listOlists):
            raise ValueError("Each element in listOlists must be a list or numpy array")
        
        self.zbase_avg = [np.mean(i[:length]) for i in listOlists]
        self.zevent_avg = [np.mean(i[length:]) for i in listOlists]

        return self.zbase_avg, self.zevent_avg
    
    def save_zscores(self, zscore_diary, id_list):
        #Get Average line for plotting
        averages=np.average(zscore_diary, axis=0)
        #Get sem for plotting
        sem=stats.sem(zscore_diary, axis=0)

        #x axis
        low = np.arange(-self.baseline_length, 0) / self.config.photo_fps
        high = np.arange(self.event_length ) / self.config.photo_fps 
        fp_times = np.concatenate((low, high)).tolist()

        #Save for future plots or self-plotting
        save_it = pd.DataFrame({"Zscore": np.asarray(averages), "SEM": np.asarray(sem), "Time": np.asarray(fp_times)})
        save_it.to_csv(fr'{self.save_path}\{self.event_name}_AVGvalues.csv', index=False, header=True)
        
        #Paired ttest
        base_avg, event_avg = self.average_zscores(zscore_diary, self.baseline_length)
        stat, p_val=stats.ttest_rel(base_avg, event_avg,)

        #For future calculations and plotting across groups
        diary = pd.DataFrame(zscore_diary).T  # Modified line        # sns.heatmap(zscore_diary, xticklabels=fp_times)
        diary.columns = id_list
        diary.to_csv(f'{self.save_path}\{self.event_name}_ALLTRIALS.csv', index=False, header=True)

        return save_it, diary, p_val
    
    def plot_zscores(self, zscore_diary, id_list):

        saved_avg, _, p_val = self.save_zscores(zscore_diary, id_list)

        zscore_fig = plt.figure()
        plt.plot(saved_avg["Time"], saved_avg["Zscore"], alpha=1, color = self.colors)
        plt.ylabel('Z-Score'); plt.xlabel('Time (s)')
        plt.xticks(np.arange(-self.baseline_length, self.event_length))
        plt.title(f"{self.event_name}_p={p_val:.3f}")  # fix y labels and tighten the graph
        plt.fill_between(saved_avg["Time"], saved_avg["Zscore"] - saved_avg["SEM"], saved_avg["Zscore"] + saved_avg["SEM"], 
                         alpha=0.1, color=self.colors)
        plt.axvline(x=0, color='k')

        zscore_fig.savefig(f'{self.save_path}\{self.event_name}_zscore_figs.tif')
        zscore_fig.savefig(f'{self.save_path}\{self.event_name}_zscore_figs.svg')

        plt.close()

def percent_change(baseline, event):
    return [(y - x)/x * 100 for x, y in zip(baseline, event)]

def auc(baseline_matrix, event_matrix):
    
    event = []
    base = []
    for b, e in zip(np.array(baseline_matrix, dtype = float), np.array(event_matrix, dtype = float)):
        event.append(np.abs(np.trapz(e)))
        base.append(np.abs(np.trapz(b)))

    return base, event

def load_raw_perc(dictionaries, event, unloader, config):
    raw_avg_base = np.asarray([mean(i[-(config.pre_s * config.photo_fps) :]) for i in unloader.avg_matrix_base])
    raw_avg_event = np.asarray([mean(i[config.pre_s * config.photo_fps :]) for i in unloader.avg_matrix_event])

    #Raw Values
    dictionaries.raw_diary_i = dictionaries.load_full_dict(dictionaries.raw_diary_i, event, [np.array(unloader.baseline_means, dtype = float), np.array(unloader.event_means, dtype = float)], unloader.id_list)
    dictionaries.raw_diff_ind = dictionaries.load_dict_diffs(dictionaries.raw_diff_ind, event, (np.array(unloader.event_means, dtype = float) - np.array(unloader.baseline_means, dtype = float)) , unloader.id_list )
    
    dictionaries.raw_diary_avg = dictionaries.load_full_dict(dictionaries.raw_diary_avg, event, [np.array(raw_avg_base, dtype = float), np.array(raw_avg_event, dtype = float)], unloader.id_list)
    dictionaries.raw_diff_avg = dictionaries.load_dict_diffs(dictionaries.raw_diff_avg, event, raw_avg_event - raw_avg_base, unloader.id_list )
    
    #Percentages
    dictionaries.percentage_diary = dictionaries.load_dict_diffs(dictionaries.percentage_diary, event, percent_change(unloader.baseline_means, unloader.event_means), unloader.id_list)
    dictionaries.perc_avg = dictionaries.load_dict_diffs(dictionaries.perc_avg, event, percent_change(raw_avg_base, raw_avg_event), unloader.id_list)

    return dictionaries
def load_zscore(dictionaries, event, unloader, zscore):
    #Zscores
    #Unequal length baseline and event (e.g. 5s baseline, 10s event)
    zscore_pre_post = zscore.calculate_values(unloader.peri_baseline_matrix, unloader.peri_event_matrix, zscore.baseline_length)
    zscore_by_trial = zscore.calculate_values(unloader.avg_matrix_base, unloader.avg_matrix_event, zscore.baseline_length)#NOTE for plotting
    
    zbase, zevent = zscore.average_zscores(zscore_pre_post, zscore.baseline_length)
    indv_diff = [ze - zb for zb, ze in zip(zbase, zevent)]
    dictionaries.zscore_ind = dictionaries.load_dict_diffs(dictionaries.zscore_ind, event, indv_diff, unloader.id_list) 
    
    zbase, zevent = zscore.average_zscores(zscore_by_trial, zscore.baseline_length)
    avg_diff = [ze - zb for zb, ze in zip(zbase, zevent)]                
    dictionaries.zscore_avg = dictionaries.load_dict_diffs(dictionaries.zscore_avg, event, avg_diff, unloader.id_list)

    return dictionaries

def load_auc(dictionaries, event, unloader):
    #AUC
    auc_base, auc_event = auc(unloader.peri_baseline_matrix, unloader.peri_event_matrix)
    dictionaries.auc_diary = dictionaries.load_full_dict(dictionaries.auc_diary, event, [auc_base, auc_event], unloader.id_list)

    avg_auc_base, avg_auc_event = auc(unloader.avg_matrix_base, unloader.avg_matrix_event)
    dictionaries.auc_avg = dictionaries.load_full_dict(dictionaries.auc_avg, event, [avg_auc_base, avg_auc_event], unloader.id_list)
    
    #AUC Mag change
    auc_mag_change = [((10/b) * e - 10) for b, e in zip(auc_base, auc_event)] 
    dictionaries.mag_auc = dictionaries.load_dict_diffs(dictionaries.mag_auc, event, auc_mag_change, unloader.id_list)

    avg_auc_mag_change = [((10/b) * e - 10) for b, e in zip(avg_auc_base, avg_auc_event)]
    dictionaries.mag_auc_avg = dictionaries.load_dict_diffs(dictionaries.mag_auc_avg, event, avg_auc_mag_change, unloader.id_list)

    return dictionaries

def write_to_excel(file_name, df, trace):
    try:
        with pd.ExcelWriter(file_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=trace, index = False)
    except FileNotFoundError:
        with pd.ExcelWriter(file_name, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=trace, index = False)

    
def save_dictionaries(dictionaries, stage, save_directory, trace):

    for attr, dictionary in dictionaries.__dict__.items():
        if "title" in dictionary:
            title = dictionary.pop("title")
            file_name = f"{title}_{stage}_.xlsx"

            df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in dictionary.items()]))
            if 'indv' in title.lower():
                ind_values_path = make_folder("Individual Values", save_directory)
                save_path = os.path.join(ind_values_path, file_name)
            if 'avg' in title.lower():
                avg_values_path = make_folder("Averages by Sample", save_directory)
                save_path = os.path.join(avg_values_path, file_name)
            write_to_excel(save_path, df, trace)
    return print(f"Sheet {trace} saved.")


def main(configuration_file):
    config = AnalysisParameters(load_config(configuration_file))

    config.color = config.iscolor(config.color)

    event_directory = os.path.join(config.project_home, "Behaviors")
    trace_list = os.listdir(event_directory)
    summary_directory = make_folder("Summary", config.project_home)

    for trace in trace_list:
        #join paths
        config.trace = trace
        trace_directory = os.path.join(event_directory, trace)
        stages = os.listdir(trace_directory)
        for stage in stages:
            stage_directory = os.path.join(trace_directory, stage)
            save_directory = make_folder(stage, summary_directory)
            config.subject_path = save_directory
            event_list = os.listdir(stage_directory)
            dictionaries = EmptyDictionaries()
            for event in event_list:
                event_path = os.path.join(stage_directory, event)
                data_list = os.listdir(event_path)
                if len(data_list) != 0:
                    unloader = GrabEventValues(config, data_list, event_path)
                    unloader.match_by_id()
                    config.event_name = event
                    config.trial_id = unloader.id
                    zscore = ZscorenPlot(config)
                    if trace == trace_list[1]:
                        zscore.plot_zscores(
                                zscore.calculate_values(unloader.peri_baseline_matrix, unloader.peri_event_matrix, zscore.baseline_length),
                                unloader.id_list)
                    #Load Dictionaries
                    load_raw_perc(dictionaries, event, unloader, config)
                    load_zscore(dictionaries, event, unloader, zscore)
                    load_auc(dictionaries, event, unloader)
            # print(dictionaries.raw_diary_avg)
            save_dictionaries(dictionaries, stage, save_directory, trace)
    print("Summarization Complete.")
    
if __name__ == "__main__": 
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(configuration_file)
