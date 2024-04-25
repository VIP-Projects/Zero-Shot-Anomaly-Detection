import cv2
import os
import natsort
from tqdm import tqdm

# # single_video
# path = "/home/jykim/DataBase/Action/Abnomal/Avenue/Avenue_Dataset/testing_videos"
# save_path = "/home/jykim/DataBase/Action/Abnomal/Avenue/Avenue_Dataset/testing_videos"

# cap = cv2.VideoCapture(path + '.mp4')
# total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# for now_frame in tqdm(range(total_frame)):
#     success, image = cap.read()
#     if not success:
#         break
#     cv2.imwrite(save_path + f"/{now_frame:.0f}.jpg" , image)

# print("\n\nfinish! convert video to frame")



# multi_video
path_dir = '/home/jykim/DataBase/Action/Abnomal/Avenue/Avenue_Dataset/testing_videos'
file_list = os.listdir(path_dir)
file_list = natsort.natsorted(file_list)  # 정렬

for folder_idx in file_list:
    path = "/home/jykim/DataBase/Action/Abnomal/Avenue/Avenue_Dataset/testing_videos/" + folder_idx[:-4]
    save_path = "/home/jykim/DataBase/Action/Abnomal/Avenue/Avenue_Dataset/testing_images/" + folder_idx[:-4]
    
    cap = cv2.VideoCapture(path + '.avi')
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for now_frame in tqdm(range(total_frame)):
        success, image = cap.read()
        if not success:
            break
        cv2.imwrite(save_path + f"/{now_frame:.0f}.jpg" , image)

print("\n\nfinish! convert video to frame")

