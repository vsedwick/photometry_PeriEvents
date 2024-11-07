import pandas as pd
import statsmodels.formula.api as smf
import os
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import ttest_ind
import functions.photo_nestedstats_funcs as photo
from functions.photo_nestedstats_funcs import make_folder


#per trace
# formula = 'Outcome ~ Fixed_Effect_1 + Fixed_Effect_2 + ... + (1 | Random_Effect)'
file = r"E:\Photometry-Fall2022\Final Analysis\REANALYSIS_zscore_nolimit\Virgin Males\IndV_ID_Virgin Males_during_Individual Z-score_summary.csv_compiled.xlsx"
# mouse = {'3': ['3', '23', '33'], '7':['7', '27', '37'], '10': ['10', '210', '310'], '11': ['11', '211', '311']}
mouse = {'3': ['3', '23', '33','43'], 
         '7':['7', '27', '37'], 
         '10': ['10', '210', '310'], 
         '11': ['11', '211', '311'], 
         '1': ['1', '21', '31'], 
         '6': ['6', '26', '36'], 
         '8': ['8', '28', '38'], 
         '16': ['156', '16', '216', '316', '416'],
         '5': ['5', '25', '35', '45', '55'], 
         '2': ['2'], 
         '12': ['12', '212', '312', '412','512'], 
         '4': ['4','24','34','44','54'], 
         '13': ['13','213','313','413'], 
         '82': ['82', '282', '382','482'], 
         '9': ['9', '29', '39','49']}

_2way_exclude = ["V. Males Attacking"]
do_2way = False
# mouse = {'1': ['1', '21', '31'], '6': ['6', '26', '36'], '8': ['8', '28', '38']}

def main(file, mouse, _2way_exclude):
    #TRACE IDs
    traces = {trace_id: mouse_id for mouse_id, trace_ids in mouse.items() for trace_id in trace_ids}
    
    file_folder = file.split('\\'); file_folder = '\\'.join(file_folder[0:-1])
    full_file = pd.ExcelFile(file)
    _ = make_folder('Statistics', file_folder)
    output = file.split('\\')
    output = output[-1].replace('.xlsx','')    
    save_folder = make_folder(output, _)

    for behavior in full_file.sheet_names:
        print(behavior.upper())
        df = pd.read_excel(file, sheet_name=behavior)
        if df.shape[0] <= 1:
            continue
        else:
            # Transform the data into a long format
            data_long = []
            data_long_noref = []
            data_noref_2pairs = []

            # Iterate over the columns by pairs
            for i in range(0, len(df.columns), 2):
                id_col = df.columns[i]  # ID column for the test
                test_col = df.columns[i + 1]  # Corresponding test measurement column
                
                # Extract test name by removing 'ID_' prefix
                test_name = id_col.replace('ID_', '')

                # Iterate through each row of these columns
                for _, row in df[[id_col, test_col]].dropna().iterrows():
                    trace_id = str(int(row[id_col]))  # Convert to string to match dictionary keys
                    mouse_id = traces.get(trace_id)  # Map trace ID to mouse ID
                    if mouse_id and test_name!='Reference':
                        data_noref_2pairs.append({
                            'Mouse_ID': mouse_id,
                            'Trace_ID': trace_id,
                            'Stimulus': test_name,
                            'mean_zF': row[test_col]
                        })     
                    if mouse_id and test_name != 'Reference':  # Ensure the trace ID was found in the mapping
                        data_long_noref.append({
                            'Mouse_ID': mouse_id,
                            'Trace_ID': trace_id,
                            'Stimulus': test_name,
                            'mean_zF': row[test_col]
                        })                   
                    if mouse_id and 'Reference' in df.columns:  # Ensure the trace ID was found in the mapping
                        data_long.append({
                            'Mouse_ID': mouse_id,
                            'Trace_ID': trace_id,
                            'Stimulus': test_name,
                            'mean_zF': row[test_col]
                        })
            # Transform the data into a long format
            # Convert the list to a DataFrame

            df_long = pd.DataFrame(data_long)
            df_long_noref = pd.DataFrame(data_long_noref)
            data_noref_2pairs = pd.DataFrame(data_noref_2pairs)
            # Corrected model formula: Just the fixed effect (intercept in this case)
            # # # Assuming 'Stimulus' is treated as a categorical variable and 'mean_zF' is your dependent variable
            # if 'Reference' in df.columns:
            #     model_formula = 'mean_zF ~ C(Stimulus, Treatment(reference = "Reference"))'
            #     model = smf.mixedlm(model_formula, df_long, groups=df_long['Mouse_ID'])
            #     result = model.fit()
            #     # print(result.pvalues)
            #     # Save to Excel
            #     save_name = os.path.join(save_folder, 'LMM_results.xlsx')
            #     photo.save_two(save_name, behavior, result)

            #     #MainEffects_LM
            #     results = photo.nested_anova(df_long_noref)
            #     save_name = os.path.join(save_folder, 'ANOVA_LM(LMM).xlsx')
            #     photo.save_results(results , behavior, save_name)

            # # Identify all unique stimulus groups
            # stimulus_groups = df_long_noref['Stimulus'].unique()
            # pairs = []
            # for i in stimulus_groups:
            #     for j in stimulus_groups: 
            #         if j != i and (j,i) not in pairs: 
            #             pairs.append((i,j))
            
            # if len(stimulus_groups)>1:
            #     if len(data_noref_2pairs['Stimulus'].unique())==2 and len(pairs) == 1:
            #         #Nested Unpaired T-test
            #         group1 = data_noref_2pairs[data_noref_2pairs['Stimulus'] == data_noref_2pairs['Stimulus'].unique()[0]]['mean_zF']
            #         group2 = data_noref_2pairs[data_noref_2pairs['Stimulus'] == data_noref_2pairs['Stimulus'].unique()[1]]['mean_zF']

            #         t_statistic, p_value = ttest_ind(group1, group2)
            #         tdf = pd.DataFrame({'Group': [str(stimulus_groups),str(stimulus_groups)],
            #             'Significance': [p_value, p_value],
            #             "T Stat": [t_statistic, t_statistic]})
            #         save_name = save_name = os.path.join(save_folder, 'Unpaired nested t-test.xlsx')
            #         photo.save_results(tdf, behavior, save_name)
                    
            #     if len(stimulus_groups)>2:
            #         # Convert the Tukey test results to a DataFrame
            #         # Perform the pairwise comparison (adjusting the method as necessary for your specific context)
            #         multicomp = MultiComparison(df_long_noref['mean_zF'], df_long_noref['Stimulus'])
            #         # Adjust this to 'Bonferroni' or another method if you prefer
            #         tukey_result = multicomp.tukeyhsd(alpha=0.05)
            #         tukey_df2 = tukey_result.summary()
            #         tukey_df2 = pd.DataFrame(tukey_df2.data[1:], columns=tukey_df2.data[0])
            #         tukey_df2 = photo.format_floats(tukey_df2, 6)
            #         save_name = os.path.join(save_folder, 'Nested Tukey_results.xlsx')
            #         photo.save_results(tukey_df2, behavior, save_name)
            #         if do_2way:
            #             if len(stimulus_groups)>3 and 'Pups' not in stimulus_groups:
            #                 #nested 2way
            #                 results, tukey = photo.nested_2way_bysex(df, traces, _2way_exclude)
            #                 # Save the results to an Excel file
            #                 save_name = os.path.join(save_folder, '2way Nested_Tukey_results.xlsx')
            #                 photo.save_results(tukey, behavior, save_name)
            #                 save_name = os.path.join(save_folder, 'Nested 2way ANOVA_results.xlsx')
            #                 photo.save_two(save_name, behavior, results)
                            
        photo.summarize_by_mice(df, mouse, save_folder, behavior, output) 

print('Statistical summaries will be found in the statistics folder')


if __name__ == "__main__":  # avoids running main if this file is imported
    main(file, mouse, _2way_exclude)