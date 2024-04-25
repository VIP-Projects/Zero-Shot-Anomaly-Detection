import os

for i in range(1, 22):
    if i < 10:
        fold_name = "/home/jykim/DataBase/Action/Abnomal/Avenue/Avenue_Dataset/testing_images/0" + str(i)
    else:
        fold_name = "/home/jykim/DataBase/Action/Abnomal/Avenue/Avenue_Dataset/testing_images/" + str(i)
    os.mkdir(fold_name)