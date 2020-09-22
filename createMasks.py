import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import random

def cropWindow(img, w, h, center_x, center_y):
    tmp_img = img

    p1_x = center_x - int(w/2)
    p1_y = center_y - int(h/2)
    p2_x = center_x + int(w/2)
    p2_y = center_y + int(h/2)

    cropped_img = tmp_img[p1_y:p2_y, p1_x:p2_x]
    return cropped_img


img_path = "templates/16mm_test.png"
mask_path = "./film_types/" # 16mm_mask_prep_v5.png
dst_path = "./masks_3classes/"

mask_files = os.listdir(mask_path)
cnt = 0;
for i, mask in enumerate(mask_files):
    print(mask)
    mask_filepath = mask_path + "/" + mask;

    #dst_path = "./16mm_masks_full_scale/"
    #random_x_range = [510, 680]
    #random_y_range = [360, 545]
    #crop_w = 960
    #crop_h = 720

    if(i == 0):
        crop_w = 960  #840
        crop_h = 720  #630
        random_x_range = [540, 680]  # [580, 600]
        random_y_range = [370, 545]  # [420, 490]
        class_indices = i+1
    elif(i == 1):
        crop_w = 960  # 840
        crop_h = 720  # 630
        random_x_range = [580, 620]  # [580, 600]
        random_y_range = [430, 480]  # [430, 480]
        class_indices = i+1


    nSamples = 50

    img = cv2.imread(img_path)
    img = img[200:380, 200:440]
    mask = cv2.imread(mask_filepath)
    print(mask.shape)
    print(img.shape)

    #print(mask[2:5])


    #mask_resized = cv2.resize(mask, (int(mask.shape[1] / 3), int(mask.shape[0] / 3)))
    img_resized = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    #print(mask_resized.shape)
    print(img.shape)
    #print(img_resized.shape)

    #random.seed(5)
    for i in range(0, nSamples):
        cnt = cnt + 1

        rand_x = random.randint(random_x_range[0], random_x_range[1])
        rand_y = random.randint(random_y_range[0], random_y_range[1])
        #print(rand_x)
        #print(rand_y)

        alpha = 0.5

        src1 = img_resized.copy()
        src2 = mask.copy()
        output = mask.copy()
        output = cropWindow(mask, w=crop_w, h=crop_h, center_x=rand_x, center_y=rand_y)   #780, 1150   700, 1000



        #print(output.shape)
        #output = src1 | src2
        #cv2.rectangle(src1, (0, 0), (960, 720), (0, 0, 255), -1)
        #cv2.putText(overlay, "PyImageSearch: alpha={}".format(alpha), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.addWeighted(src2, alpha, src1, 1 - alpha, 0, output)
        #print(output.shape)

        output = cv2.resize(output, (960, 720))
        idx = np.where(output == 255)
        output[output == 255] = class_indices
        print(output.shape)
        print(np.unique(output))

        #output = cv2.resize(output, (int(output.shape[1] / 2), int(output.shape[0] / 2)))

        cv2.imwrite(dst_path + "/mask_" + str(cnt) + ".png", output)
        #cv2.imshow("Output", output)
        #cv2.waitKey(0)