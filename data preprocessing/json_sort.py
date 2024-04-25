import json
import os
import natsort


path_dir = './annotation/Avenue'
file_list = os.listdir(path_dir)
file_list = natsort.natsorted(file_list)  # 정렬

for folder_idx in file_list:
    ori_path = './annotation/Avenue/' + folder_idx
    new_path = './annotation/Avenue/' + folder_idx[4:-5] + '.json'

    with open(ori_path, 'r') as f:
        fall_json = json.load(f)

    sort_json = natsort.natsorted(fall_json.items())  # 정렬


    with open(new_path, 'w') as write_f:
        json.dump(sort_json, write_f, indent=4)















    