import os
import json
import numpy as np
import cv2
import re
import locale
import matplotlib.pyplot as plt
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate

HOME = "/home/jykim/JY_project/Few_Shot/"
CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join(HOME, "GroundingDINO/weights", WEIGHTS_NAME)
model = load_model(CONFIG_PATH, WEIGHTS_PATH)


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

    annotated_frame = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)

    return annotated_frame

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"

locale.getpreferredencoding = getpreferredencoding


fall_prediction = []
count_list = []

# # single video check
# cap = cv2.VideoCapture("./dataset/Multiple_Fall/chute02/cam6.avi")
# fall_cnt = 0

# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter('./result/multiple_fall/chute02/cam6.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# while cap.isOpened():
#     ret, img = cap.read()
#     if not ret:
#         break
#     TEXT_PROMPT = "falling"
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     try:
#         result_img = inference(img, TEXT_PROMPT)
#         fall_cnt += 1
#     except:
#         try:
#             TEXT_PROMPT = "frame"
#             result_img = inference(img, TEXT_PROMPT)
#         except:
#             TEXT_PROMPT = "table"
#             result_img = inference(img, TEXT_PROMPT)
    

#     out.write(result_img)

# out.release()
# print(fall_cnt)
# print("done.")



#####################
# multiple video check
for i in range(5, 25):  # 이따가는 5, 25
    for j in range(1, 9):
        if i < 10:
            video_dir = './dataset/Multiple_Fall/chute0' + str(i) + '/cam' + str(j) + '.avi'
            result_dir = './result/multiple_fall/chute0' + str(i) + '/cam' + str(j) + '.mp4'
        else:
            video_dir = './dataset/Multiple_Fall/chute' + str(i) + '/cam' + str(j) + '.avi'
            result_dir = './result/multiple_fall/chute' + str(i) + '/cam' + str(j) + '.mp4'

        print(video_dir)
        fall_cnt = 0
        cap = cv2.VideoCapture(video_dir)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(result_dir, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break
            TEXT_PROMPT = "falling"
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            try:
                result_img = inference(img, TEXT_PROMPT)
                fall_cnt += 1
            except:
                try:
                    TEXT_PROMPT = "frame"
                    result_img = inference(img, TEXT_PROMPT)
                except:
                    TEXT_PROMPT = "table"
                    result_img = inference(img, TEXT_PROMPT)
            

            out.write(result_img)

        count_list.append(fall_cnt)
        if fall_cnt > 80:  # fall
            fall_prediction.append(1)
        else:  # not fall
            fall_prediction.append(0)

print("fall_prediction: ")    
print(fall_prediction)
print("count_list: ")  
print(count_list)
print("done.")