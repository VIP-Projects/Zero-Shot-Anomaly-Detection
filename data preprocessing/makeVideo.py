import glob
import cv2
import os
import re
import natsort
import numpy as np
from tqdm import tqdm

video_list = [24]
# 방법2

for folder_idx in video_list:
    if folder_idx < 10:
        path = "./result/fall/real_fall/fall00" + str(folder_idx) + "/"  # image path
        save_path = './result/fall/real_fall/fall00' + str(folder_idx) + '.mp4'
    else:
        path = "./result/fall/real_fall/fall0" + str(folder_idx) + "/"  # image path
        save_path = './result/fall/real_fall/fall0' + str(folder_idx) + '.mp4'

    paths = [os.path.join(path , i) for i in os.listdir(path) if re.search(".png$", i )]   
    paths = natsort.natsorted(paths)  # 이미지 정렬

    fps = 15  # fps: 1초당 보여주는 화면의 장수(15장으로 설정)

    frame_array = []
    for idx , path in enumerate(paths) : 
        if (idx % 2 == 0) | (idx % 5 == 0) :
            continue
        img = cv2.imread(path)
        height, width, layers = img.shape
        size = (width,height)
        frame_array.append(img)
    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()