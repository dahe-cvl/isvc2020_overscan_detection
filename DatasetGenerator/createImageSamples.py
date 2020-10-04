import numpy as np
import cv2
#from matplotlib import pyplot as plt
import os
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets


#####################################################
####           CONFIGURATION SECTION             ####
#####################################################
dst_path = "/data/share/frame_border_detection_db_v5/rgb_3class/"
src_path = "/data/share/frame_border_detection_db_v3/ms_coco_images_v3/"
mask_path = "./masks_3classes/"
#####################################################
'''
#####################################################
####           CONFIGURATION SECTION             ####
#####################################################
dst_path = "/caa/Projects02/vhh/public/frame_border_detection_db/"
src_path = "/caa/Projects02/vhh/public/frame_border_detection_db/ms_coco_images/"
mask_path = "./16mm_masks_full_scale"
#####################################################

#####################################################
####           CONFIGURATION SECTION             ####
#####################################################
dst_path = "./frame_border_detection_db/"
src_path = "./images/"
mask_path = "./16mm_masks_full_scale"
#####################################################
'''

sample_dir = dst_path + "/samples"
label_dir = dst_path + "/labels"

print("create folder structure ... ")
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

image_files_list = os.listdir(src_path)
#mask_files_list.sort()
mask_files_list = os.listdir(mask_path)
#mask_files_list.sort()

number_of_masks = len(mask_files_list)
number_of_images = len(image_files_list)

random.seed(50)
for i in range(0, number_of_images):
    # get random mask
    rand_idx = random.randint(0, number_of_masks - 1)
    #print(rand_idx)
    tmp_mask_path = mask_path + "/" + mask_files_list[rand_idx]
    mask = cv2.imread(tmp_mask_path)
    mask_dim = mask.shape
    #print(mask_dim)

    # get sample image
    tmp_img_path = src_path + "/" + image_files_list[i]
    img = cv2.imread(tmp_img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    #img_dim = img.shape
    #print(img_dim)
    img = cv2.resize(img, (mask_dim[1], mask_dim[0]))
    #img_dim = img.shape
    #print(img_dim)

    mask_orig = np.copy(mask)
    mask_tmp = np.copy(mask)

    mask[mask_tmp == 1] = 255
    mask[mask_tmp == 2] = 255

    # merge mask with sample image
    sample_result = img | mask
    print("------------------")
    print(tmp_img_path)
    print(sample_result.shape)
    print(np.unique(sample_result))
    print(sample_result.dtype)
    # save sample pair to folder
    cv2.imwrite(sample_dir + "/s_" + str("{:04d}".format(i)) + ".png", sample_result)
    cv2.imwrite(label_dir + "/l_" + str("{:04d}".format(i)) + ".png", mask_orig)

    #cv2.imshow("Output", sample_result)
    #cv2.imshow("Output", img)
    #cv2.waitKey(0)