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
from groundingdino.util.inference import load_model, load_image, predict, annotate

HOME = "/home/jykim/JY_project/Few_Shot/"
CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")  # set model configuration file path
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"  # set model weight file ath
WEIGHTS_PATH = os.path.join(HOME, "GroundingDINO/weights", WEIGHTS_NAME)
model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# def compute_intersection(bbox1, bbo2):
#     # 절대 좌표로 변환
#     bbox = (bbox * image_size).int()
#     bbox = (bbox * image_size).int()

#     # 교차 부분 좌표 범위 계산
#     left_x = max(bbox1[0], bbox2[0])
#     left_y = max(bbox1[1], bbox2[1])
#     right_x = min(bbox1[2], bbox2[2])
#     right_y = min(bbox1[3], bbox2[3])

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

    annotated_frame, xyxy = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)

    return annotated_frame, xyxy

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


image_path = "./dataset/real_kick/img/kick001/32.jpg"  # image_path


print(image_path)
# image load
kick_cnt = 0
xyxy1, xyxy2, xyxy3 = 0, 0, 0
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# kick detection(BLIP + DINO + HOI)
TEXT_PROMPT1 = "kickboard"
TEXT_PROMPT2 = "man"
TEXT_PROMPT3 = "helmet"
# TEXT_PROMPT4 = "riding"

try:
    result_img1, xyxy1 = inference(img, TEXT_PROMPT1)     
    result_img2, xyxy2 = inference(result_img1, TEXT_PROMPT2)
    result_img3, xyxy3 = inference(result_img2, TEXT_PROMPT3)
    # result_img3 = result_img3[:, :, ::-1]
    # result_img4, bbox4 = inference(result_img3, TEXT_PROMPT4)
    # try: # 안전 장비 있는 경우
    #     result_img3 = inference(result_img2, TEXT_PROMPT3)
    #     kick_prediction.append(1)
    # except: # 안전 장비 없는 경우
    #     result_img3 = cv2.cvtColor(result_img2, cv2.COLOR_BGR2RGB)
    result_img3 = cv2.cvtColor(result_img3, cv2.COLOR_BGR2RGB)
    result_img3 = result_img3[:, :, ::-1]
    
except:
    result_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kick_prediction.append(0)
print("kick_bbox1: {}/{}\nman_bbox2: {}/{}\nhel_bbox3: {}/{}\n".format(bbox1, xyxy1, bbox2, xyxy2, bbox3, xyxy3))
print("--------------")
plt.figure(figsize=(8, 8))
plt.imshow(result_img3[:, :, ::-1])
plt.axis("off")

save_img = "./result/kick/real_kick/kick001/kick032.png" 
plt.savefig(save_img, dpi=300)


# print(kick_ture)
# print(kick_prediction)
print("done.")




