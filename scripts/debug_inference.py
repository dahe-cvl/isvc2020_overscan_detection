import numpy as np
import cv2
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True #Let TeX do the typsetting

from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.mixture import GaussianMixture

def applyGMM(img):

    hist, bin_edges = np.histogram(img, bins=256)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    classif = GaussianMixture(n_components=1, covariance_type='tied', tol=1e-3,
                 reg_covar=1e-6, max_iter=800, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10)
    classif.fit(img.reshape((img.size, 1)))
    threshold = np.mean(classif.means_)

    binary_img = img.copy()
    binary_img[binary_img > threshold] = 255
    binary_img[binary_img <= threshold] = 0
    #print(binary_img)
    #print(img.dtype)

    '''
    plt.figure()
    plt.subplot(131)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(132)
    plt.plot(bin_centers, hist, lw=2)
    plt.axvline(threshold, color='r', ls='--', lw=2)
    plt.yticks([])
    plt.subplot(133)
    plt.imshow(binary_img.astype('uint8'), cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.subplots_adjust(wspace=0.02, hspace=0.3, top=1, bottom=0.1, left=0, right=1)
    plt.show()
    #plt.savefig("gmm_hist" + ".pdf", dpi=300)
    

    plt.figure()
    plt.imshow(img.astype('uint8'), cmap=plt.cm.gray, interpolation='nearest')
    plt.savefig("img" + ".pdf", dpi=300)
     '''
    plt.figure()
    plt.plot(bin_centers, hist, lw=2)
    plt.axvline(threshold, color='r', ls='--', lw=2)
    plt.show()

    plt.figure()
    plt.plot(bin_centers, hist, lw=2)
    plt.axvline(threshold, color='r', ls='--', lw=2)
    plt.savefig("gmm_hist" + ".png", dpi=300)
    #exit()

    return binary_img


def applyTSNE(mask_prep):
    print(mask_prep.shape)
    print(mask_prep[:, :, 1].shape)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(mask_prep[:, :, 1])
    print(tsne_results.shape)


    #plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
    #            #palette=plt.color_palette("hls", 10),
    #            alpha=0.3
    #)
    #plt.show()

    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        palette=sns.color_palette("hls", 10),
        legend="full",
        alpha=0.3
    )
    plt.show()

def calculateMetrics(mask_gt, mask_pred, n_classes, tOverall_acc_sum, tAvg_per_class_acc_sum, tAvg_jacc_sum, tAvg_dice_sum):
    #################################
    ## evaluate
    #################################
    #print(mask_gt.shape)
    #print(mask_pred.shape)

    mask_gt = torch.tensor(mask_gt.reshape(mask_gt.shape[2], mask_gt.shape[0], mask_gt.shape[1]))
    mask_pred = torch.tensor(mask_pred.reshape(mask_pred.shape[2], mask_pred.shape[0], mask_pred.shape[1]))
    #print(mask_gt.size())
    #print(mask_pred.size())



    #mask_pred = mask_pred.detach().cpu()
    #mask_gt = mask_gt.detach().cpu()
    overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(mask_gt, mask_pred, n_classes)
    tOverall_acc_sum += overall_acc
    tAvg_per_class_acc_sum += avg_per_class_acc
    tAvg_jacc_sum += avg_jacc
    tAvg_dice_sum += avg_dice
    return tOverall_acc_sum, tAvg_per_class_acc_sum, tAvg_jacc_sum, tAvg_dice_sum

from metrics import *
import os
def evaluate():
    #root_dir = "C:\\Users\\dhelm\\Documents\\20200418_0401_gray_fcn_mobilenet_ExpNum_1_1_cross_entropy_lf\\"
    #root_dir = "C:\\Users\\dhelm\\Documents\\20200417_0032_gray_fcn_resnet101_ExpNum_1_1_cross_entropy_lf\\"
    #root_dir = "C:\\Users\\dhelm\\Documents\\20200418_1049_gray_fcn_squeezenet_ExpNum_1_1_cross_entropy_lf\\"
    #root_dir = "C:\\Users\\dhelm\\Documents\\20200416_0645_gray_deeplabv3_vgg16_ExpNum_1_1_cross_entropy_lf\\"

    #root_dir = "C:\\Users\\dhelm\\Documents\\20200417_2117_gray_fcn_mobilenet_ExpNum_1_1_cross_entropy_hf\\"
    #root_dir = "C:\\Users\\dhelm\\Documents\\20200416_1310_gray_fcn_resnet101_ExpNum_1_1_cross_entropy_hf\\"
    #root_dir = "C:\\Users\\dhelm\\Documents\\20200418_0414_gray_fcn_squeezenet_ExpNum_1_1_cross_entropy_hf\\"
    root_dir = "C:\\Users\\dhelm\\Documents\\20200416_0512_gray_deeplabv3_vgg16_ExpNum_1_1_cross_entropy_hf\\"

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

        #print(np.unique(msk_gt))
        #print(np.unique(msk_pred))
        #print(np.unique(msk_pred_gmm))
        #continue

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
    print("tOverall_acc_sum: " + str(tOverall_acc_sum1 / 200))
    print("tAvg_per_class_acc_sum: " + str(tAvg_per_class_acc_sum1 / 200))
    print("tAvg_jacc_sum: " + str(tAvg_jacc_sum1 / 200))
    print("tAvg_dice_sum: " + str(tAvg_dice_sum1 / 200))

    print("#############################################################")
    print("tOverall_acc_sum: " + str(tOverall_acc_sum2 / 200))
    print("tAvg_per_class_acc_sum: " + str(tAvg_per_class_acc_sum2 / 200))
    print("tAvg_jacc_sum: " + str(tAvg_jacc_sum2 / 200))
    print("tAvg_dice_sum: " + str(tAvg_dice_sum2 / 200))

#evaluate()

'''
root_dir = "./templates/test1/"

img = root_dir + "1.png"
mask_pred = root_dir + "3.png"
mask_gt = root_dir + "2.png"

input_img = cv2.imread(img)

m_p = cv2.imread(mask_pred)
m_p = m_p[:,:, 1]
m_gt = cv2.imread(mask_gt)

print(m_p.shape)

n_components, components, bbs, centroids = cv2.connectedComponentsWithStats(m_p, 4, cv2.CV_16U)
print(bbs[1:])

bbs = bbs[1:]

margin_x = 0
margin_y = 0

tmp = np.zeros((input_img.shape[0], input_img.shape[1], 1))

for i, bb in enumerate(bbs):
    x = bb[0]
    y = bb[1]
    w = bb[2]
    h = bb[3]

    p1_x = x - margin_x
    if (p1_x <= 0): p1_x = 0
    p1_y = y - margin_y
    if (p1_y <= 0): p1_y = 0
    p2_x = x + w + margin_x
    if (p2_x >= input_img.shape[1]): p2_x = input_img.shape[1]
    p2_y = y + h + margin_y
    if (p2_y >= input_img.shape[0]): p2_y = input_img.shape[0]

    img_crop = input_img[p1_y:p2_y, p1_x:p2_x]
    mask_crop = m_p[p1_y:p2_y, p1_x:p2_x]



    #cv2.imshow("img_crop", img_crop)
    #cv2.waitKey(0)
    #cv2.imshow("masked_data", tmp)
    #cv2.waitKey(0)

    #tmp_img = np.zeros((input_img.shape))
    #tmp_img[y:y+h, x:x+w] = img_crop
    #print(tmp_img.shape)
    #cv2.imshow("output", tmp_img)
    #cv2.waitKey(0)
    #exit()

    bin_img_crop = applyGMM(img_crop)
    tmp[p1_y:p2_y, p1_x:p2_x] = bin_img_crop[:, :, 1:2]
    #applyTSNE(img_crop)

    m_p_C = cv2.applyColorMap(mask_crop, cv2.COLORMAP_JET)
    stacked = np.hstack((img_crop, m_p_C))

    alpha = 0.5
    src1 = img_crop.copy()
    src2 = m_p_C.copy()
    output = m_p_C.copy()
    cv2.addWeighted(src2, alpha, src1, 1 - alpha, 0, output)
    output = cv2.resize(output, (output.shape[1]*1, output.shape[0]*1))
    print(output.shape)

    #cv2.imwrite(root_dir + "/img_crop" + str(i+1) + ".png", img_crop)
    #cv2.imwrite(root_dir + "/output" + str(i+1) + ".png", output)
    #cv2.imwrite(root_dir + "/gmm_output" + str(i + 1) + ".png", bin_img_crop)

#cv2.imwrite(root_dir + "/final_output" + str(i + 1) + ".png", tmp)
cv2.imshow("output", tmp)
cv2.waitKey(0)
'''


