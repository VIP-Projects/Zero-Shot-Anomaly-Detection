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


# image caption 
file_path = "/home/jykim/JY_project/Few_Shot/BLIP/annotation/fall/kaggle/new_kaggle_fall.json"

with open(file_path, 'r') as f:
    fall_json = json.load(f)

# image load
# path = "./dataset/kaggle_Fall/images/fall/"
path = "./result/real/"
paths = [os.path.join(path , i) for i in os.listdir(path) if re.search(".jpg$", i )]
paths = natsort.natsorted(paths)

# Metrics
fall_true = []
fall_prediction = []


# # grounding dino
# for idx, img_path in enumerate(paths):  
#     img = cv2.imread(paths[idx])
#     print(paths[idx])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     caption = fall_json[idx][1]
#     answer = fall_json[idx][0]
    
#     # fall GT
#     if "not" in answer:
#         fall_true.append(0)
#     else:
#         fall_true.append(1)

#     # fall detection(BLIP + DINO)
#     if "falling" in caption or "laying" in caption or "lying" in caption or "sleeping" in caption:
#         TEXT_PROMPT = "falling"
#         try:
#             result_img= inference(img, TEXT_PROMPT)
#             fall_prediction.append(1)
#         except:
#             result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             fall_prediction.append(0)
#     else:
#         TEXT_PROMPT = fall_json[idx][1]
#         result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         fall_prediction.append(0)


#     # # detection(just DINO)
#     # TEXT_PROMPT = "falling"
#     # try:
#     #     result_img= inference(img, TEXT_PROMPT)
#     #     fall_prediction.append(1)
#     # except:
#     #     result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     fall_prediction.append(0)


#     #result visual
#     plt.figure(figsize=(8, 8))
#     plt.imshow(result_img[:, :, ::-1])
#     plt.axis("off")
#     if idx < 9:
#         save_img = "//home/jykim/JY_project/Few_Shot/GroundingDINO/result/kaggle_train/hoi/fall00" + str(idx+1) + ".png"
#     elif 8 < idx < 99:
#         save_img = "/home/jykim/JY_project/Few_Shot/GroundingDINO/result/kaggle_train/hoi/fall0" + str(idx+1) + ".png"
#     else:
#         save_img = "/home/jykim/JY_project/Few_Shot/GroundingDINO/result/kaggle_train/hoi/fall" + str(idx+1) + ".png"  
#     plt.savefig(save_img)


# ------------------------------------
# # dino single test
# # image caption 
# img_num = 43
# img = cv2.imread(paths[img_num])
# print(paths[img_num])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# caption = fall_json[img_num][1]
# answer = fall_json[img_num][0]

# # fall GT
# if "not" in answer:
#     fall_true.append(0)
# else:
#     fall_true.append(1)

# if fall_json[img_num][1] in "falling" or "laying" or "lying" or "sleeping":
#     TEXT_PROMPT1 = "man"
#     TEXT_PROMPT2 = "stair, floor"
#     TEXT_PROMPT3 = "falling"
#     try:
#         result_img1= inference(img, TEXT_PROMPT1)
#         result_img2= inference(img, TEXT_PROMPT2)
#         result_img3= inference(img, TEXT_PROMPT3)
#         fall_prediction.append(1)
#     except:
#         result_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         result_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         result_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         fall_prediction.append(0)
# else:
#     TEXT_PROMPT = fall_json[img_num][1]
#     result_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     result_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     fall_prediction.append(0)


# # 여러 이미지 한번에
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 3, 1)
# plt.imshow(result_img1[:, :, ::-1])
# plt.axis("off")
# plt.subplot(1, 3, 2)
# plt.imshow(result_img2[:, :, ::-1]) 
# plt.axis("off")
# plt.subplot(1, 3, 3)
# plt.imshow(result_img3[:, :, ::-1])
# plt.axis("off")
# plt.show()
# save_img = "/home/jykim/JY_project/Few_Shot/GroundingDINO/result/kaggle_train/hoi2/fall02.png"    
# plt.savefig(save_img)


# # ------------------
# grounding dino HOI
for idx, img_path in enumerate(paths):  
    img = cv2.imread(paths[idx])
    print(paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    caption = fall_json[idx][1]
    answer = fall_json[idx][0]
    
    # fall GT
    if "not" in answer:
        fall_true.append(0)
    else:
        fall_true.append(1)

    TEXT_PROMPT1 = "man"
    TEXT_PROMPT2 = "falling" 
    TEXT_PROMPT3 = "floor"

    try:
        result_img1= inference(img, TEXT_PROMPT1)
        result_img2= inference(img, TEXT_PROMPT2)     
        result_img3= inference(result_img2, TEXT_PROMPT3)
        result_img3 = result_img3[:, :, ::-1]
        fall_prediction.append(1)
    except:
        result_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fall_prediction.append(0)
    # # fall detection(BLIP + DINO + HOI)
    # if "falling" in caption or "laying" in caption or "lying" in caption or "sleeping" in caption:
    #     TEXT_PROMPT1 = "man"
    #     TEXT_PROMPT2 = "falling" 
    #     TEXT_PROMPT3 = "floor"

    #     try:
    #         result_img1= inference(img, TEXT_PROMPT1)
    #         result_img2= inference(img, TEXT_PROMPT2)     
    #         result_img3= inference(result_img2, TEXT_PROMPT3)
    #         result_img3 = result_img3[:, :, ::-1]
    #         fall_prediction.append(1)
    #     except:
    #         result_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         fall_prediction.append(0)
    # else:
    #     TEXT_PROMPT = fall_json[idx][1]
    #     result_img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     fall_prediction.append(0)
    plt.figure(figsize=(8, 8))
    plt.imshow(result_img3[:, :, ::-1])
    plt.axis("off")
    save_img = "./result/kaggle_train/" + str(idx+1) + ".png"  

    # #resulgure(figsize=(8, 8))
    # plt.imshow(result_img3[:, :, ::-1])
    # plt.axis("off")
    # if idx < 9:
    #     save_img = "./result/real/fall00" + str(idx+1) + ".png"
    # elif 8 < idx < 99:
    #     save_img = "./result/real/fall0" + str(idx+1) + ".png"
    # else:
    #     save_img = "./result/real/fall" + str(idx+1) + ".png"  
    plt.savefig(save_img)


print(fall_true)
print(fall_prediction)
print("done.")




