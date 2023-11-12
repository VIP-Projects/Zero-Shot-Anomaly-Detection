import os
import json
import numpy as np
import cv2
import re
import locale
import natsort
import matplotlib.pyplot as plt
from PIL import Image
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate, bbox_intersection

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
file_path = "./annotation/image_caption.json"

with open(file_path, 'r') as f:
    fall_json = json.load(f)

# image load
path = "./dataset/real_Fall/img/fall024/"
paths = [os.path.join(path , i) for i in os.listdir(path) if re.search(".jpg$", i )]
paths = natsort.natsorted(paths)

# abnormal detection
for idx, img_path in enumerate(paths):
    img = cv2.imread(paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    caption = fall_json[idx][1]
    print("img")
    
    # fall detection
    if "falling" in caption or "laying" in caption or "lying" in caption or "sleeping" in caption:
        TEXT_PROMPT1 = "man"
        TEXT_PROMPT2 = "falling"
        TEXT_PROMPT3 = "floor" 
        try:
            process_img1 = inference(img, TEXT_PROMPT1)
            process_img2 = inference(img, TEXT_PROMPT2)     
            result_img = inference(process_img2, TEXT_PROMPT3)
            result_img = result_img[:, :, ::-1]
        except:
            result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # safety equipment detection
    elif "kick" in caption:
        TEXT_PROMPT1 = "kickboard"
        TEXT_PROMPT2 = "man"
        TEXT_PROMPT3 = "helmet"
        TEXT_PROMPT4 = "riding"

        try:
            process_img1 = inference(img, TEXT_PROMPT1)     
            process_img2 = inference(process_img1, TEXT_PROMPT2)
            process_img3 = inference(process_img2, TEXT_PROMPT3)
            process_img3 = cv2.cvtColor(process_img3, cv2.COLOR_BGR2RGB)
            process_img3 = process_img3[:, :, ::-1]

            # interaction check
            riding_check = compute_intersection(xyxy[0][0], xyxy[1][0])  
            if riding_check:
                # print("riding")
                result_img = inference(process_img3, TEXT_PROMPT4)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                result_img = result_img[:, :, ::-1]
            else:
                # print("no riding")
                result_img = cv2.cvtColor(process_img3, cv2.COLOR_BGR2RGB)
            result_img = result_img[:, :, ::-1]

        except:
            # print("no inference")
            result_img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            kick_prediction.append(0)

    # fire detection
    elif "fire" in caption:
        TEXT_PROMPT1 = "fire" 
        TEXT_PROMPT2 = "man"

        try:
            process_img1= inference(img, TEXT_PROMPT1)     
            result_img= inference(process_img1, TEXT_PROMPT2)
            result_img = result_img[:, :, ::-1]
        except:
            result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # knife detection
    elif "knife" in caption or "holding" in caption:
        TEXT_PROMPT1 = "knife"
        TEXT_PROMPT2 = "man"

        try:
            process_img1= inference(img, TEXT_PROMPT1)
            process_img1 = cv2.cvtColor(process_img1, cv2.COLOR_BGR2RGB)  
            result_img = inference(process_img1, TEXT_PROMPT2)   
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  
            result_img = result_img[:, :, ::-1]
        except:
            result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # industry detection1
    elif "scratch" in caption:
        TEXT_PROMPT1 = "scratch"

        try:
            result_img = inference(img, TEXT_PROMPT1)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_img = result_img[:, :, ::-1]
        except:
            result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # industry detection2
    elif "hole" in caption:
        TEXT_PROMPT1 = "hole"

        try:
            result_img = inference(img, TEXT_PROMPT1)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_img = result_img[:, :, ::-1]
        except:
            result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    plt.figure(figsize=(8, 8))
    plt.imshow(result_img[:, :, ::-1])
    plt.axis("off")
    if idx < 9:
        save_img = "./result/result00" + str(idx+1) + ".png"
    elif 8 < idx < 99:
        save_img = "./result/result0" + str(idx+1) + ".png"
    else:
        save_img = "./result/result" + str(idx+1) + ".png"  

    plt.savefig(save_img, dpi=300)
    xyxy = []

print("done.")