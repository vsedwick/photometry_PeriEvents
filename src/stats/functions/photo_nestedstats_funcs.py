import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.formula.api import mixedlm
import os
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from scipy.stats import ttest_ind
import numpy as np

def save_results(df, sheet_name, save_name):    
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    # Convert columns to float as needed and format to 6 decimal places
    for col in df.columns:
        try:
            df[col] = df[col].astype(float).map('{:,.6f}'.format)
        except ValueError:
            # This column cannot be converted to float, likely a non-numeric column, so ignore
            pass
    # Create a new Excel workbook or open an existing one
    if os.path.exists(save_name):
        with pd.ExcelWriter(save_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=f'{sheet_name}', index=True)
    else:
        with pd.ExcelWriter(save_name, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f'{sheet_name}', index=True)

def format_floats(df, decimal_places):
    for col in df.columns:
        if df[col].dtype.kind in 'fc':  # f for float, c for complex numbers
            df[col] = df[col].apply(lambda x: f'{x:.{decimal_places}f}')
    return df

def make_folder(new_folder, parent_directory):
    """
    Creates new folders and or sets the directory. 

    Args:
        parent_directory (str): The parent directory for the new folder.
        new_folder (str): The name of the new fodler to be created.

    Returns:
        full_path (str): The new directory where the folder was created
    
    Raises: 
        FileNotFoundError: If the specified parent directory does not exist.
        PermissionError: If the directory is not writable or the folder cannot be created.

    """

    mode = 0o666

    full_path = os.path.join(parent_directory, new_folder)

    #check if parent directory exists
    if not os.path.exists(parent_directory):
        raise FileNotFoundError(f"Parent directory '{parent_directory}' does not exist.")
    
    #checks if user has permission to write to that directory
    if not os.access(parent_directory, os.W_OK):
        raise PermissionError(f"Write permission denied for directory '{parent_directory}.")
    
    #Creates the folder if it doesnt exists
    if not os.path.exists(full_path):
        try:
            os.mkdir(full_path, mode)
        except OSError:
            raise PermissionError(f"Failed to create directory {full_path}. Check permissions: {OSError}")


    return full_path

def nested_anova(df_long):
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    # Step 2: Fit your models
    # Full model with the recoded 'Stimulus' (excluding reference category)
    full_model = smf.ols('mean_zF ~ C(Stimulus)', data=df_long).fit()

    # Reduced model without 'Stimulus_recode'
    reduced_model = smf.ols('mean_zF ~ 1', data=df_long).fit()

    # Step 3: Compare models with anova_lm
    return anova_lm(reduced_model, full_model)

def nested_2way_bysex(df, traces, _2way_exclude):
    # Initialize an empty DataFrame for the long-format data
    long_df = []
    # Iterate over the columns by pairs
    for i in range(0, len(df.columns), 2):
        id_col = df.columns[i]  # ID column for the test
        test_col = df.columns[i + 1]  # Corresponding test measurement column
        
        # Extract test name by removing 'ID_' prefix
        test_name = id_col.replace('ID_', '')
        if test_name in _2way_exclude or test_name == 'Reference':
            continue
        else:
            # Iterate through each row of these columns
            for _, row in df[[id_col, test_col]].dropna().iterrows():
                trace_id = str(int(row[id_col]))  # Convert to string to match dictionary keys
                mouse_id = traces.get(trace_id)  # Map trace ID to mouse ID
                groupings = test_name.split('.')
                condition = groupings[0]
                if ' male' in groupings[1].lower():
                    sex = 'Male'
                elif 'female' in groupings[1].lower():
                    sex = 'Female'
                if mouse_id and test_name!='Reference':
                    long_df.append(({
                        'Mouse_ID': mouse_id,
                        'Trace_ID': trace_id,
                        'Condition': condition,
                        'Sex': sex,
                        'mean_zF': row[test_col]
                    }))
    long_df = pd.DataFrame(long_df)

    model = mixedlm("mean_zF ~ Sex * Condition", data=long_df, groups=long_df["Mouse_ID"], re_formula="~Condition")
    result = model.fit()

    # Prepare data for Tukey HSD test - you might need to adjust this part
    # Here, I'm assuming you want to test the interaction between 'Sex' and 'Condition'
    long_df['Interaction'] = long_df['Sex'] + "_" + long_df['Condition']


    # Prepare for Tukey HSD test
    multicomp = MultiComparison(long_df['mean_zF'], long_df['Condition'] + "_" + long_df['Sex'])
    tukey_result = multicomp.tukeyhsd(alpha=0.05)
    tukey_df2 = pd.DataFrame(tukey_result.summary().data[1:], columns=tukey_result.summary().data[0])
    
    # Assuming `format_floats` is a function you've defined to format the float columns
    tukey_df2 = format_floats(tukey_df2, 6)  # You'll need to define this function
    
    return result, tukey_df2
    
def save_two(save_name, behavior, result):
    
    # Convert summary tables to DataFrames
    pd.set_option('display.float_format', lambda x: '%.6f' % x)
    df1 = result.summary().tables[0]
    if not isinstance(df1, pd.DataFrame):
        df1 = pd.DataFrame(df1)  # Convert to DataFrame if not already

    pval_save = result.pvalues.map(lambda x: f"{x:.6g}").to_frame('p-values')  # Formatting p-values
    # print(behavior, pval_save)
    df2 = result.summary().tables[1]
    if not isinstance(df2, pd.DataFrame):
        df2 = pd.DataFrame(df2)  # Convert to DataFrame if not already

   

    if os.path.exists(save_name):
        with pd.ExcelWriter(save_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            # Get the maximum row to avoid overwriting data in case of existing sheet
            if behavior in writer.book.sheetnames:
                startrow = writer.book[behavior].max_row
            else:
                startrow = 0

            # Write each DataFrame to the same sheet at different start rows
            df1.to_excel(writer, sheet_name=behavior, startrow=startrow, index=False)
            # pval_save.to_excel(writer, sheet_name=behavior, startrow=startrow + len(df1) + 2, index=False)
            pval_save.to_excel(writer, sheet_name=behavior, startrow=startrow + len(df1) + len(pval_save) + 4)
    else:
        with pd.ExcelWriter(save_name, mode='w', engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name = behavior, startrow = 0, index = False)
            # pval_save.to_excel(writer, sheet_name = behavior, startrow = 10, index = False)
            pval_save.to_excel(writer, sheet_name = behavior, startrow = 20)

def summarize_by_mice(df, mouse, save_folder, sheet_name, output):
    new_df = {}
    for i in range(0, len(df.columns), 2):
        id_col = df.columns[i]
        if 'Reference' in id_col:
            continue
        else:
            id_id = [str(int(i)) for i in df[id_col].dropna()]; unique_id = [*set(id_id)]
            test_col = df.columns[i + 1]
            test_name = id_col.replace('ID_', '')
            avg_values = []
            keys = []
            for key in mouse:
                new_idcol = [i in mouse[key] for i in id_id]
                test_col = df[test_name].dropna()
                new_testcol = test_col[new_idcol]
                if len(new_testcol) < 2:
                    continue
                else:
                    avg_values.append(np.average(new_testcol))
                    keys.append(key)

            new_df.update({
                f'ID_{test_name}': keys,
                f'{test_name}': avg_values
            })
    
    new_df = pd.DataFrame(padding(new_df))

    save_name = os.path.join(save_folder, f'BYMICE.xlsx')
    if os.path.exists(save_name):
        with pd.ExcelWriter(save_name, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
            new_df.to_excel(writer, sheet_name=sheet_name, startrow=0)
    else:
        with pd.ExcelWriter(save_name, mode='w', engine='openpyxl') as writer:
            new_df.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
def padding(data):
    # Find the maximum length of the lists in the dictionary
    max_length = max(len(lst) for lst in data.values())

    # Fill each list in the dictionary with NaNs to ensure they all have the same length
    for key in data.keys():
        data[key] += [np.nan] * (max_length - len(data[key]))

    return data