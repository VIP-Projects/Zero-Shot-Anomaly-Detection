import cv2
from tqdm import tqdm

# single_video
path = "./real_fall/video"
save_path = "./real_fall/img"

cap = cv2.VideoCapture(path + '.mp4')
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for now_frame in tqdm(range(total_frame)):
    success, image = cap.read()
    if not success:
        break
    cv2.imwrite(save_path + f"/{now_frame:.0f}.jpg" , image)

print("\n\nfinish! convert video to frame")

