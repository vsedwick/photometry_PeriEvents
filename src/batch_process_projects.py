import batch_photocode_v2 as pc
import os
from ruamel.yaml import YAML

""" Allows user to process a folder of projects at once using the same configuration file.
"""

def update_config(configuration_params, config_file):
    """ Updates the 'project_home' parameter in the config to allow user to iterate through file folders
    """
    yaml = YAML()
    yaml.preserve_quotes = True  # Optional: Preserves quotes around values

    # Load the YAML file with comments
    with open(config_file, 'r') as file:
        config_data = yaml.load(file)
    
    # Update root-level configurations
    config_data['project_home'] = configuration_params.project_home  # Corrected typo from 'scoretpye'

    # Write the updated data back, preserving comments
    with open(config_file, 'w') as file:
        yaml.dump(config_data, file)

def main():
    rootdir = "E:\Photometry-Fall2022\Final Analysis\Mated Photometry (Males)"

    animals = os.listdir(rootdir)

    configuration_file = r'C:\Users\sedwi\Desktop\Portfolio\Thesis_Research (python)\Photometry\config.yaml'

    config = pc.AnalysisParameters(pc.load_config(configuration_file))
    for i in animals:
        assay_path = os.path.join(rootdir, i)
        assay = os.listdir(os.path.join(rootdir, i))
        for j in assay:
            trials = os.path.join(assay_path, j)
            config.project_home = trials

            update_config(config, configuration_file)

            pc.save_variables
            pc.main(trials)



main()
