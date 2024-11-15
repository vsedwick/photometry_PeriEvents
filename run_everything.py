from anonymize_files import main as anon
import os
# import batch_photocode_v2

parent_folder = r"C:\Users\sedwi\Downloads\photo_files_for_portfolio"

group_list = os.listdir(parent_folder)

for i in group_list:
    if i.endswith('.txt'):
        continue
    else:
        group_path = os.path.join(parent_folder, i)
        print(group_path)
        anon(group_path)