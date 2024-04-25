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
file_path = "/home/jykim/JY_project/Few_Shot/BLIP/annotation/fire/new_fire.json"

with open(file_path, 'r') as f:
    fall_json = json.load(f)

# image load
path = "./dataset/fire/fire_images/"
paths = [os.path.join(path , i) for i in os.listdir(path) if re.search(".png$", i )]
paths = natsort.natsorted(paths)

# Metrics
fire_ture = []
fire_prediction = []


for idx, img_path in enumerate(paths):  
    img = cv2.imread(paths[idx])
    print(paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    caption = fall_json[idx][1]
    answer = fall_json[idx][0]


    # fall detection(BLIP + DINO + HOI)
    if "fire" in caption:
        TEXT_PROMPT1 = "fire" 
        TEXT_PROMPT2 = "man"

        try:
            result_img1= inference(img, TEXT_PROMPT1)     
            result_img2= inference(result_img1, TEXT_PROMPT2)
            result_img2 = result_img2[:, :, ::-1]
            fire_prediction.append(1)
        except:
            result_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fire_prediction.append(0)
    else:
        result_img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fire_prediction.append(0)


    #resulgure(figsize=(8, 8))
    plt.imshow(result_img2[:, :, ::-1])
    plt.axis("off")
    if idx < 9:
        save_img = "./result/fire/fire00" + str(idx+1) + ".png"
    elif 8 < idx < 99:
        save_img = "./result/fire/fire0" + str(idx+1) + ".png"
    else:
        save_img = "./result/fire/fire" + str(idx+1) + ".png"  
    plt.savefig(save_img, dpi=300)


print(fire_ture)
print(fire_prediction)
print("done.")




