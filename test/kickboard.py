import os
import json
import numpy as np
import cv2
import re
import locale
import natsort
import matplotlib.pyplot as plt
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate, bbox_intersection

HOME = "/home/jykim/JY_project/Few_Shot/"
CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")  # set model configuration file path
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"  # set model weight file ath
WEIGHTS_PATH = os.path.join(HOME, "GroundingDINO/weights", WEIGHTS_NAME)
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
xyxy = []

def compute_intersection(bbox1, bbox2):
    # 각 바운딩 박스의 좌표 추출
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 교차하는 영역 계산
    x1_intersection = max(x1_1, x1_2)
    y1_intersection = max(y1_1, y1_2)
    x2_intersection = min(x2_1, x2_2)
    y2_intersection = min(y2_1, y2_2)

    # 교차 영역의 폭과 높이 계산
    width = x2_intersection - x1_intersection
    height = y2_intersection - y1_intersection

    intersection_area = width * height
    # print("교차 영역 좌표 (x1, y1, x2, y2):", (x1_intersection, y1_intersection, x2_intersection, y2_intersection))
    # print("교차 영역의 넓이:", intersection_area)

    # 교차 영역의 좌표와 크기를 출력
    if intersection_area > 1000:
        return True   # 교차 영역 존재
    else:
        return False  # 교차 영역 없음

def inference(img, prompt, box_threshold=0.35, text_threshold=0.25):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(Image.fromarray(img), None)

    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    bbox_xyxy = bbox_intersection(image_source=img, boxes=boxes)
    numpy2List = bbox_xyxy.tolist()
    xyxy.append(numpy2List)
    annotated_frame = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)

    return annotated_frame

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"

locale.getpreferredencoding = getpreferredencoding


# image caption 
# file_path = "/home/jykim/JY_project/Few_Shot/BLIP/annotation/kick/new_kick.json"

# with open(file_path, 'r') as f:
#     kick_json = json.load(f)

# image load
path = "./dataset/kickboard/"
paths = [os.path.join(path , i) for i in os.listdir(path) if re.search(".jpg$", i )]
paths = natsort.natsorted(paths)

# Metrics
kick_ture = []
kick_prediction = []

for folder_idx in range(9, 13):
    riding_check = False
    # image caption 
    if folder_idx < 10:
        image_path = "./dataset/real_kick/img/kick00" + str(folder_idx) + "/"  # image_path
    else:
        image_path = "./dataset/real_kick/img/kick0" + str(folder_idx) + "/"  # image_path

    print(image_path)
    # image load
    paths = [os.path.join(image_path , i) for i in os.listdir(image_path) if re.search(".jpg$", i )]  # images list
    paths = natsort.natsorted(paths)
    kick_cnt = 0
    for idx, img_path in enumerate(paths):  
        img = cv2.imread(paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # kick detection(BLIP + DINO + HOI)
        TEXT_PROMPT1 = "kickboard"
        TEXT_PROMPT2 = "man"
        TEXT_PROMPT3 = "helmet"
        TEXT_PROMPT4 = "riding"

        try:
            result_img1 = inference(img, TEXT_PROMPT1)     
            result_img2 = inference(result_img1, TEXT_PROMPT2)
            result_img3 = inference(result_img2, TEXT_PROMPT3)
            result_img3 = cv2.cvtColor(result_img3, cv2.COLOR_BGR2RGB)
            result_img3 = result_img3[:, :, ::-1]

            # # safety equipment check
            # try:
            #     result_img3, xyxy3 = inference(result_img2, TEXT_PROMPT3)
            #     kick_prediction.append(1)
            # except:
            #     result_img3 = cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB)

            # riding check
            riding_check = compute_intersection(xyxy[0][0], xyxy[1][0])  
            if riding_check:
                # print("riding")
                result_img4 = inference(result_img3, TEXT_PROMPT4)
                result_img4 = cv2.cvtColor(result_img4, cv2.COLOR_BGR2RGB)
                result_img4 = result_img4[:, :, ::-1]
            else:
                # print("no riding")
                result_img4 = cv2.cvtColor(result_img3, cv2.COLOR_BGR2RGB)
            result_img4 = result_img4[:, :, ::-1]

        except:
            # print("no inference")
            result_img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kick_prediction.append(0)
        

        plt.figure(figsize=(8, 8))
        plt.imshow(result_img4[:, :, ::-1])
        plt.axis("off")
        if folder_idx < 10:
            if idx < 9:
                save_img = "./result/kick/real_kick/kick00" + str(folder_idx) + "/kick00" + str(idx+1) + ".png"
            elif 8 < idx < 99:
                save_img = "./result/kick/real_kick/kick00" + str(folder_idx) + "/kick0" + str(idx+1) + ".png"
            else:
                save_img = "./result/kick/real_kick/kick00" + str(folder_idx) + "/kick" + str(idx+1) + ".png"  
        else:
            if idx < 9:
                save_img = "./result/kick/real_kick/kick0" + str(folder_idx) + "/kick00" + str(idx+1) + ".png"
            elif 8 < idx < 99:
                save_img = "./result/kick/real_kick/kick0" + str(folder_idx) + "/kick0" + str(idx+1) + ".png"
            else:
                save_img = "./result/kick/real_kick/kick0" + str(folder_idx) + "/kick" + str(idx+1) + ".png"  
        plt.savefig(save_img, dpi=300)
        xyxy = []


print(kick_ture)
print(kick_prediction)
print("done.")




