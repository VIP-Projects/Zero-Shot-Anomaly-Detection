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

# Metrics
fall_prediction = []
count_list = []

for folder_idx in range(1, 41):
    # image caption 
    if folder_idx < 10:
        json_path = "/home/jykim/JY_project/Few_Shot/BLIP/annotation/fall/ur/adl_0" + str(folder_idx) + ".json"  # json_file_path
        image_path = "./dataset/UR_Fall/img/adl/adl_0" + str(folder_idx) + "/"  # image_folder_path
    else:
        json_path = "/home/jykim/JY_project/Few_Shot/BLIP/annotation/fall/ur/adl_" + str(folder_idx) + ".json"  # json_file_path
        image_path = "./dataset/UR_Fall/img/adl/adl_" + str(folder_idx) + "/"  # image path

    with open(json_path, 'r') as f:
        fall_json = json.load(f)

    print(image_path)
    # image load
    paths = [os.path.join(image_path , i) for i in os.listdir(image_path) if re.search(".jpg$", i )]  # images list
    paths.sort()
    fall_cnt = 0

    # grounding dino
    for idx, img_path in enumerate(paths):  
        img = cv2.imread(paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        caption = fall_json[idx][1]
    
        # # detection(just DINO)
        # TEXT_PROMPT = "falling"
        # try:
        #     result_img= inference(img, TEXT_PROMPT)
        #     fall_cnt += 1
        # except:
        #     result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # # fall detection(BLIP + DINO)
        # if "falling" in caption or "laying" in caption or "lying" in caption or "sleeping" in caption:
        #     TEXT_PROMPT = "falling"
        #     try:
        #         result_img= inference(img, TEXT_PROMPT)
        #         fall_cnt += 1
        #     except:
        #         result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # else:
        #     TEXT_PROMPT = fall_json[idx][1]
        #     result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # fall detection(BLIP + DINO + HOI)
            if "falling" in caption or "laying" in caption or "lying" in caption or "sleeping" in caption or "man" in caption:
                TEXT_PROMPT1 = "falling"
                TEXT_PROMPT2 = "floor" 
                TEXT_PROMPT3 = "man"
                try:
                    result_img1= inference(img, TEXT_PROMPT1)
                    result_img2= inference(result_img1, TEXT_PROMPT2)          
                    result_img3= inference(img, TEXT_PROMPT3)
                    fall_cnt += 1
                except:
                    result_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                TEXT_PROMPT = fall_json[idx][1]
                result_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # #result visual
        # plt.figure(figsize=(8, 8))
        # plt.imshow(result_img3[:, :, ::-1])
        # plt.axis("off")
        # if folder_idx < 10:
        #     if idx < 9:
        #         save_img = "./result/ur_fall/fall_0" + str(folder_idx) + "/" + "fall_00" + str(idx+1) + ".png"
        #     elif 8 < idx < 99:
        #         save_img = "./result/ur_fall/fall_0" + str(folder_idx) + "/" + "fall_0" + str(idx+1) + ".png"
        #     else:
        #         save_img = "./result/ur_fall/fall_0" + str(folder_idx) + "/" + "fall_" + str(idx+1) + ".png"  
        # else:
        #     if idx < 9:
        #         save_img = "./result/ur_fall/fall_" + str(folder_idx) + "/" + "fall_00" + str(idx+1) + ".png"
        #     elif 8 < idx < 99:
        #         save_img = "./result/ur_fall/fall_" + str(folder_idx) + "/" + "fall_0" + str(idx+1) + ".png"
        #     else:
        #         save_img = "./result/ur_fall/fall_" + str(folder_idx) + "/" + "fall_" + str(idx+1) + ".png"    
        # plt.savefig(save_img, dpi=300)

    count_list.append(fall_cnt)
    # fall인지 판별
    if fall_cnt > 10:  # fall
        fall_prediction.append(1)
    else:  # not fall
        fall_prediction.append(0)


print("fall_prediction: ")   
print(fall_prediction)
print("count_list: ")  
print(count_list)
print("done.")
