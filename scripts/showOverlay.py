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


img_path = "../templates/EF-NS_001_OeFM_3_2.png"
mask_path = "../templates/16mm_mask_prep_v5.png"

#dst_path = "./16mm_masks_full_scale/"
#random_x_range = [510, 680]
#random_y_range = [360, 545]
#crop_w = 960
#crop_h = 720

dst_path = "./16mm_masks_small_scale/"
random_x_range = [580, 600]
random_y_range = [420, 490]
crop_w = 840
crop_h = 630

nSamples = 50

img = cv2.imread(img_path)
#img = img[200:380, 200:440]
mask = cv2.imread(mask_path)
print(mask.shape)
print(img.shape)


img_resized = cv2.resize(img, (mask.shape[1], mask.shape[0]))

alpha = 0.5

src1 = img_resized.copy()
src2 = mask.copy()
output = mask.copy()
#output = cropWindow(mask, w=crop_w, h=crop_h, center_x=rand_x, center_y=rand_y)

#print(output.shape)
#output = src1 | src2
#cv2.rectangle(src1, (0, 0), (960, 720), (0, 0, 255), -1)
#cv2.putText(overlay, "PyImageSearch: alpha={}".format(alpha), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
cv2.addWeighted(src2, alpha, src1, 1 - alpha, 0, output)
print(output.shape)

output = cv2.resize(output, (960, 720))
print(output.shape)

cv2.imshow("Output", output)
cv2.waitKey(0)