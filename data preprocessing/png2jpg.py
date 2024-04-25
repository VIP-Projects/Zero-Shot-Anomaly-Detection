import os
import re
from PIL import Image

# for fold_idx in range(1, 10):
#     paths = []
#     if fold_idx < 10:
#         img_path = "./UR_Fall/img/adl/adl-0" + str(fold_idx) + "-cam0-rgb/"
#         move_file = "./UR_Fall/img/adl/adl_0" + str(fold_idx)
#     else:
#         img_path = "./UR_Fall/img/adl/adl-" + str(fold_idx) + "-cam0-rgb/"
#         move_file = "./UR_Fall/img/adl/adl_" + str(fold_idx)

#     paths = [os.path.join(img_path , i ) for i in os.listdir(img_path) if re.search(".png$", i )]
#     paths.sort()

#     for idx, im_path in enumerate(paths):
#         im = Image.open(im_path).convert('RGB')
#         if idx < 9:
#             save_file = move_file + '/adl-0' + str(fold_idx) + '-cam0-rgb-00' + str(idx+1) + '.jpg'
#         elif 8 < idx < 99:
#             save_file = move_file + '/adl-0' + str(fold_idx) + '-cam0-rgb-0' + str(idx+1) + '.jpg'
#         else:
#             save_file = move_file + '/adl-0' + str(fold_idx) + '-cam0-rgb-' + str(idx+1) + '.jpg'
#         im.save(save_file, 'jpeg')


# # single
im_path = "./density/density_2.png"
save_file = "./density/density (40).jpg"

im = Image.open(im_path).convert('RGB')
im.save(save_file, 'jpeg')