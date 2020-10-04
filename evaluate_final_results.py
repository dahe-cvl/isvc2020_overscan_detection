import numpy as np
from metrics import *
import os
import cv2
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True #Let TeX do the typsetting


def calculateMetrics(mask_gt, mask_pred, n_classes, tOverall_acc_sum, tAvg_per_class_acc_sum, tAvg_jacc_sum, tAvg_dice_sum):
    #################################
    ## evaluate
    #################################
    mask_gt = torch.tensor(mask_gt.reshape(mask_gt.shape[2], mask_gt.shape[0], mask_gt.shape[1]))
    mask_pred = torch.tensor(mask_pred.reshape(mask_pred.shape[2], mask_pred.shape[0], mask_pred.shape[1]))

    overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(mask_gt, mask_pred, n_classes)
    tOverall_acc_sum += overall_acc
    tAvg_per_class_acc_sum += avg_per_class_acc
    tAvg_jacc_sum += avg_jacc
    tAvg_dice_sum += avg_dice
    return tOverall_acc_sum, tAvg_per_class_acc_sum, tAvg_jacc_sum, tAvg_dice_sum


def evaluate_final_results():
    root_dir = "./20200417_0032_gray_fcn_resnet101_ExpNum_1_1_cross_entropy_lf"

    mask_gt_path = root_dir + "msk_gt\\"
    mask_pred_path = root_dir + "msk_pred\\"
    mask_pred_gmm_path = root_dir + "gmm_msk_pred\\"
    n_classes = 2

    mask_gt_list = os.listdir(mask_gt_path)
    mask_pred_list = os.listdir(mask_pred_path)
    mask_pred_gmm_list = os.listdir(mask_pred_gmm_path)

    #overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(mask_gt, mask_pred, n_classes)

    tOverall_acc_sum1 = 0
    tAvg_per_class_acc_sum1 = 0
    tAvg_jacc_sum1 = 0
    tAvg_dice_sum1 = 0

    tOverall_acc_sum2 = 0
    tAvg_per_class_acc_sum2 = 0
    tAvg_jacc_sum2 = 0
    tAvg_dice_sum2 = 0
    for i in range(0, len(mask_gt_list)):
        #print(mask_pred_list[i])

        msk_gt = cv2.imread(mask_gt_path + mask_gt_list[i])
        msk_pred = cv2.imread(mask_pred_path + mask_pred_list[i])
        msk_pred_gmm = cv2.imread(mask_pred_gmm_path + mask_pred_gmm_list[i])

        msk_gt[msk_gt == 1] = 1
        msk_gt[msk_gt == 2] = 1
        msk_pred[msk_pred == 255] = 1
        msk_pred_gmm[msk_pred_gmm == 255] = 1

        tOverall_acc_sum1, tAvg_per_class_acc_sum1, tAvg_jacc_sum1, tAvg_dice_sum1 = calculateMetrics(msk_gt,
                                                                                                      msk_pred,
                                                                                                      n_classes,
                                                                                                      tOverall_acc_sum1,
                                                                                                      tAvg_per_class_acc_sum1,
                                                                                                      tAvg_jacc_sum1,
                                                                                                      tAvg_dice_sum1)

        tOverall_acc_sum2, tAvg_per_class_acc_sum2, tAvg_jacc_sum2, tAvg_dice_sum2 = calculateMetrics(msk_gt,
                                                                                                  msk_pred_gmm,
                                                                                                  n_classes,
                                                                                                  tOverall_acc_sum2,
                                                                                                  tAvg_per_class_acc_sum2,
                                                                                                  tAvg_jacc_sum2,
                                                                                                  tAvg_dice_sum2)

    print("#############################################################")
    print("without GMM")
    print("tOverall_acc_sum: " + str(tOverall_acc_sum1 / 200))
    print("tAvg_per_class_acc_sum: " + str(tAvg_per_class_acc_sum1 / 200))
    print("tAvg_jacc_sum: " + str(tAvg_jacc_sum1 / 200))
    print("tAvg_dice_sum: " + str(tAvg_dice_sum1 / 200))

    print("#############################################################")
    print("with GMM")
    print("tOverall_acc_sum: " + str(tOverall_acc_sum2 / 200))
    print("tAvg_per_class_acc_sum: " + str(tAvg_per_class_acc_sum2 / 200))
    print("tAvg_jacc_sum: " + str(tAvg_jacc_sum2 / 200))
    print("tAvg_dice_sum: " + str(tAvg_dice_sum2 / 200))

evaluate_final_results()
