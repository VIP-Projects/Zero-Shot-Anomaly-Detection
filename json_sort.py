import json
import os
import natsort

# # single
file_path = "./annotation/ori_image_caption.json"
new_file_path = "./annotation/image_caption.json"

with open(file_path, 'r') as f:
    fall_json = json.load(f)

sort_json = natsort.natsorted(fall_json.items())  # 정렬


with open(new_file_path, 'w') as write_f:
    json.dump(sort_json, write_f, indent=4)















    