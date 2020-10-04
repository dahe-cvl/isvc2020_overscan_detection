import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from datetime import datetime
import json
import argparse
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from models import *
from metrics import *
from utils import *

def drawRectangle(img, w, h, center_x, center_y):
    tmp_img = img

    p1_x = center_x - int(w/2)
    p1_y = center_y - int(h/2)
    p2_x = center_x + int(w/2)
    p2_y = center_y + int(h/2)

    cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 5)

    cropped_img = tmp_img[p1_y:p2_y, p1_x:p2_x]
    return cropped_img

def drawRectangleWithScaleFactor(img, h, scale_factor, center_x, center_y):
    tmp_img = img

    w = int(h * scale_factor)
    p1_x = center_x - int(w/2)
    p1_y = center_y - int(h/2)
    p2_x = center_x + int(w/2)
    p2_y = center_y + int(h/2)

    cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 5)

    cropped_img = tmp_img[p1_y:p2_y, p1_x:p2_x]
    return cropped_img


def drawLine(img, start_point, end_point):
    tmp_img = img
    #p1_x = center_x - int(w/2)
    #p1_y = center_y - int(h/2)
    #p2_x = center_x + int(w/2)
    #p2_y = center_y + int(h/2)
    #cv2.rectangle(img, (p1_x, p1_y), (p2_x, p2_y), (0, 0, 255), 5)
    cv2.line(img, start_point, end_point, (255, 0, 0), thickness=2)
    #cropped_img = tmp_img[p1_y:p2_y, p1_x:p2_x]
    #return cropped_img


def generateMaskOfFrame(frame=None, pre_trained_model=None):
    mask_pred = None
    print("not implemented yet")

    label_orig_pil = frame.convert("RGB")
    label_orig_np = np.array(label_orig_pil)

    return mask_pred


def reelTypeClassifier(mask):
    class_name = "None"
    print("not implemented yet")

    return class_name


def cropWindowCalculator(frame=None, mask=None, class_name=None, scale_factor=1):
    crop_window_box = None
    print("not implemented yet")
    return crop_window_box


def run():
    pre_trained_model = ""

    vid_name = "asdfasdfadsf"

    # open a video

    # iterate over all frames
    frame = np.ones((224, 224, 3))

    # convert frame to pil image
    frame_pil = Image.fromarray(frame)

    # SegNet - generate mask
    mask_pred = generateMaskOfFrame(frame=frame_pil, pre_trained_model=pre_trained_model)

    # post-processing

    # Reel-Type-Classifier
    class_name = reelTypeClassifier(mask_pred)

    # Crop-Window-Calculator
    crop_window_box = cropWindowCalculator(frame=None, mask=None, class_name=None, scale_factor=1)

    # apply cw box to frame

    # Save Final Video


def run_inference_on_folder():
    #################################################################################
    #################################################################################
    ##                       CONFIGURATION SECTION                                 ##
    #################################################################################
    db_path = "/data/share/datasets/frame_border_detection_db_v6/db_test_all_v3/"
    threshold = 0.8
    TENSORBOARD_IMG_LOG = False
    SAVE_FLAG = True
    CWC_FLAG = False
    TIME_LOG = False
    ACTIVATE_GMM_FLAG = True
    ACTIVATE_GPU_FLAG = True
    MODE_CORNER_POINTS = "inner_points"  # inner_points  OR centroids
    FILM_SCALE_FACTOR = 1.37

    # 20200929 - final - Low-level features
    exp_root_path = "/data/share/datasets/frame_border_detection_db_v6/results/experiments_20200929_lower_features/"
    #expName = "20200418_1049_gray_fcn_squeezenet_ExpNum_1_1_cross_entropy_lf/"
    #expName = "20200416_0645_gray_deeplabv3_vgg16_ExpNum_1_1_cross_entropy_lf/"
    expName = "20200417_0032_gray_fcn_resnet101_ExpNum_1_1_cross_entropy_lf/"
    #expName = "20200417_1413_gray_deeplabv3_mobilenet_ExpNum_1_1_cross_entropy_lf/"
    # 20200929 - final - high-level features
    #exp_root_path = "/data/share/datasets/frame_border_detection_db_v6/results/experiments_20200929_higher_features/"
    #expName = "20200418_0414_gray_fcn_squeezenet_ExpNum_1_1_cross_entropy_hf/"
    #expName = "20200416_0512_gray_deeplabv3_vgg16_ExpNum_1_1_cross_entropy_hf/"
    #expName = "20200416_1310_gray_fcn_resnet101_ExpNum_1_1_cross_entropy_hf/"
    #expName = "20200417_0900_gray_deeplabv3_mobilenet_ExpNum_1_1_cross_entropy_hf/"
    #################################################################################
    #################################################################################

    expFolder = str(exp_root_path) + str(expName)
    with open(expFolder + "/experiment_notes.json", 'r') as json_file:
        param_dict = json.load(json_file)

    batch_size = param_dict['batch_size']
    classes = param_dict['classes']
    expNet = param_dict['expNet']
    expType = param_dict['expType']
    dim = param_dict['resized_dim']
    activate_lower_features = param_dict['activate_lower_features']

    print("\n")
    print("########################################")
    print("Summary ")
    print("########################################")
    print("batch_size: " + str(batch_size))
    print("classes: " + str(classes))
    print("db_path: " + str(db_path))
    print("expNet: " + str(expNet))
    print("expType: " + str(expType))
    print("dim: " + str(dim))
    print("activate_lower_features: " + str(activate_lower_features))
    print("########################################")
    print("\n")

    samples_path = db_path + "/samples/"
    labels_path = db_path + "/labels/"

    writer = SummaryWriter(log_dir="./runs/" + "test_" + str(expName))

    # Whether to train on a gpu
    train_on_gpu = ACTIVATE_GPU_FLAG
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    model, features = loadModel(model_arch=expNet,
                                classes=classes,
                                pre_trained_path=expFolder + "best_model.pth",
                                lower_features=activate_lower_features)

    if train_on_gpu:
        model = model.to('cuda')
    else:
        model = model.to('cpu')

    if multi_gpu:
        model = nn.DataParallel(model)

    imgs_list = os.listdir(samples_path)
    labels_list = os.listdir(labels_path)
    imgs_list.sort()
    labels_list.sort()

    frm_based_results_list = []
    pred_class_label_list = []
    gt_class_label_list = []

    ################
    # load dataset
    ################
    for i in range(0, len(imgs_list)):
        if (TIME_LOG == True):
            start1 = datetime.now()

        vid_name = imgs_list[i]
        print("(" + str(i+1) + "|" + str(len(imgs_list)) + ") process " + str(vid_name) + " ... ")
        input_orig_pil = Image.open(samples_path + "/" + imgs_list[i])
        label_orig_pil = Image.open(labels_path + "/" + labels_list[i])
        label_orig_pil = label_orig_pil.convert("RGB")
        label_orig_np = np.array(label_orig_pil)

        sample = (input_orig_pil, label_orig_pil)

        transform_model_input = transforms.Compose([
            ToGrayScale(),
            Resize(dim),
            Normalize(mean=[110.0057 / 255.0,
                           110.0057 / 255.0,
                           110.0057 / 255.0],
                      std=[62.1277 / 255.0,
                           62.1277 / 255.0,
                           62.1277 / 255.0]),
        ])

        transform_mask = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((720, 960), interpolation=Image.NEAREST)
        ])

        model.eval()
        with torch.no_grad():
            ############################################
            # prepare input image for pre_trained model
            ############################################
            input_tensor = transform_model_input(sample)
            in_img = input_tensor[0]
            gt_msk = input_tensor[1]

            in_img = in_img.unsqueeze(0)
            input_batch = in_img

            ## Convert torch tensor to Variable
            inputs = Variable(input_batch)

            # If we have GPU, shift the data to GPU
            #CUDA = torch.cuda.is_available()
            if train_on_gpu:
                inputs = inputs.cuda()
            else:
                inputs = inputs.cpu()

            #############################
            # generated mask
            #############################
            outputs = model(inputs)['out']
            outputs = torch.sigmoid(outputs)
            #print(outputs.dtype)
            outputs[outputs >= threshold] = 1
            outputs[outputs <= threshold] = 0
            outputs = torch.argmax(outputs, dim=1)
            mask_pred = outputs

            final_mask = mask_pred
            final_mask = final_mask[0].detach().cpu().float()

            if (TIME_LOG == True):
                stop1 = datetime.now()

            #############################
            # classify mask
            #############################

            ## prepare ground truth mask
            label_orig = np.array(label_orig_pil).transpose(0, 1, 2)
            label_orig_np_bin = cv2.cvtColor(label_orig, cv2.COLOR_RGB2GRAY)
            output_ccl = cv2.connectedComponentsWithStats(label_orig_np_bin, 4, cv2.CV_16U)
            num_labels = output_ccl[0]

            number_of_holes = num_labels - 1
            if (number_of_holes == 2):
                gt_class_label = 2
            elif(number_of_holes == 4):
                gt_class_label = 1

            unique_class_indices = torch.unique(final_mask)
            idx = torch.max(unique_class_indices)

            #############################
            # transform generated mask
            #############################
            msk_pil = transform_mask(final_mask)
            msk_pil = msk_pil.convert("RGB")
            msk = np.array(msk_pil).transpose(0, 1, 2)
            msk_np = np.array(msk_pil).transpose(2, 0, 1)

            #############################
            # create heatmap
            #############################
            heatmap = cv2.applyColorMap(np.array(msk_pil), cv2.COLORMAP_JET)
            heatmap_np = np.array(heatmap).transpose(2, 0, 1)
            heatmap_pil = Image.fromarray(heatmap)

            #############################
            # transform input image
            #############################
            input_image = np.array(input_orig_pil).transpose(0, 1, 2)
            input_image_np = np.array(input_image).transpose(2, 0, 1)
            input_image_pil = Image.fromarray(input_image)

            #############################
            # connected-component-labeling
            #############################
            msk_np_bin = cv2.cvtColor(msk, cv2.COLOR_RGB2GRAY)
            output_ccl = cv2.connectedComponentsWithStats(msk_np_bin, 4, cv2.CV_16U)
            num_labels = output_ccl[0]
            labels = output_ccl[1]
            stats = output_ccl[2]
            centroids = output_ccl[3]
            print(num_labels)
            print(stats)
            print(centroids)

            #############################
            # cleanup ccl results
            #############################
            idx = np.where(stats[:, 4:] <= 500)[0]
            stats_n = np.delete(stats, idx, axis=0)
            centroids_n = np.delete(centroids, idx, axis=0)
            idx = np.where(stats[:, 4:] >= 30000)[0]
            stats = np.delete(stats_n, idx, axis=0)
            centroids = np.delete(centroids_n, idx, axis=0)
            print("-----------------")
            print(len(stats))
            print(centroids)
            print(stats)

            ########################################
            # calculate 3x3 grid of original mask
            ########################################
            print(msk_np.shape)

            h = msk_np.shape[1]
            w = msk_np.shape[2]

            th1_x = int(w/3)
            th2_x = int(w/3) * 2
            th1_y = int(h/3)
            th2_y = int(h/3) * 2
            print(th1_x)
            print(th2_x)
            print(th1_y)
            print(th2_y)

            mask_center_x = int(msk_np.shape[2] / 2)
            mask_center_y = int(msk_np.shape[1] / 2)
            mask_center_point = (mask_center_x, mask_center_y)

            ########################################
            # register region position (quadrant)
            ########################################
            centroids_final = []
            for p in centroids:
                # print(p)
                p_x = p[0]
                p_y = p[1]

                x = p_x
                y = p_y
                if (x <= th1_x and y <= th1_y):
                    pos = 1
                elif (x > th1_x and x <= th2_x and y <= th1_y):
                    pos = 2
                elif (x > th2_x and y <= th1_y):
                    pos = 3
                elif (x > th2_x and y <= th2_y and y > th1_y):
                    pos = 4
                elif (x > th1_x and x <= th2_x and y <= th2_y and y > th1_y):
                    pos = 5
                elif (x <= th1_x and y <= th2_y and y > th1_y):
                    pos = 6
                elif (x <= th1_x and y > th2_y):
                    pos = 7
                elif (x > th1_x and x <= th2_x and y > th2_y):
                    pos = 8
                elif (x > th2_x and y > th2_y):
                    pos = 9

                centroids_final.append([p_x, p_y, pos])
            centroids_final_np = np.array(centroids_final)
            print(np.array(stats))
            print(np.array(centroids_final))
            print("-----------------")

            ##############################
            ## classify predicted masks
            ##############################
            hole_cnt = 0
            final_hole_list = []
            for i, e in enumerate(centroids_final):
                p_x = e[0]
                p_y = e[1]
                pos = e[2]

                if(pos == 1):
                    hole_cnt = hole_cnt + 1
                    final_hole_list.append(stats[i])
                elif (pos == 3):
                    hole_cnt = hole_cnt + 1
                    final_hole_list.append(stats[i])
                elif (pos == 9):
                    hole_cnt = hole_cnt + 1
                    final_hole_list.append(stats[i])
                elif (pos == 7):
                    hole_cnt = hole_cnt + 1
                    final_hole_list.append(stats[i])
            final_holes_np = np.array(final_hole_list)

            print(hole_cnt)
            print(final_holes_np)

            class_name = "nan"
            number_of_holes_pred = 0
            if (hole_cnt == 4):
                class_name = "16mm"
                pred_class_label = 1
                number_of_holes_pred = 4
            elif (hole_cnt == 2):
                class_name = "9.5mm"
                pred_class_label = 2
                number_of_holes_pred = 2
            elif (idx < 0 or idx > 2 or idx == 0):
                print("class indices undefined")
                pred_class_label = -1
                number_of_holes_pred = -1
            
            pred_class_label_list.append(pred_class_label)
            gt_class_label_list.append(gt_class_label)
            ''''''
            #############################
            # post process mask - fine segmentation
            #############################
            if (ACTIVATE_GMM_FLAG == True):
                from FineSegmentation import FineSegmentation

                fine_seg_instance = FineSegmentation()

                # img_np = np.squeeze(in_img.detach().cpu().numpy())
                input_image_np = input_image_np.transpose(1, 2, 0)
                print(input_image_np.shape)

                margin_x = 0
                margin_y = 0
                final_gmm_mask = np.zeros((input_image_np.shape[0], input_image_np.shape[1], 1))

                print("########################")
                print(len(stats))
                print(len(stats_n))
                print("########################")

                for s, bb in enumerate(stats):
                    print("-----------------")
                    print(bb)
                    x = bb[0]
                    y = bb[1]
                    w = bb[2]
                    h = bb[3]

                    p1_x = x - margin_x
                    if (p1_x <= 0): p1_x = 0
                    p1_y = y - margin_y
                    if (p1_y <= 0): p1_y = 0
                    p2_x = x + w + margin_x
                    if (p2_x >= input_image_np.shape[1]): p2_x = input_image_np.shape[1]
                    p2_y = y + h + margin_y
                    if (p2_y >= input_image_np.shape[0]): p2_y = input_image_np.shape[0]

                    print(p1_x)
                    print(p1_y)
                    print(p2_x)
                    print(p2_y)
                    img_crop = input_image_np[p1_y:p2_y, p1_x:p2_x]
                    mask_crop = msk_np[p1_y:p2_y, p1_x:p2_x]

                    print(input_image_np.shape)
                    print(img_crop.shape)

                    if (TIME_LOG == True):
                        start2 = datetime.now()

                    bin_img_crop = fine_seg_instance.apply_gmm(img_crop)
                    final_gmm_mask[p1_y:p2_y, p1_x:p2_x] = bin_img_crop[:, :, 1:2]
                    print(final_gmm_mask.shape)

                    if (TIME_LOG == True):
                        stop2 = datetime.now()

            corner_points_list = []
            centroids_points_list = []
            line_coordinates_final = []
            line_coordinates = []
            center_point = -1
            max_height = -1

            if (CWC_FLAG == True):
                if(class_name == "16mm"):
                    print(class_name)

                    if(hole_cnt == 4):
                        if (len(final_holes_np) == 4 ):
                            idx = np.argsort(centroids_final_np, axis=0)[:, :1].flatten()
                            centroids_final_np = centroids_final_np[idx]
                            stats = stats[idx]

                            ########################################
                            # center of original mask
                            ########################################
                            #print(msk_np.shape)
                            mask_center_x = int( msk_np.shape[2] / 2 )
                            mask_center_y = int( msk_np.shape[1] / 2 )
                            mask_center_point = (mask_center_x, mask_center_y)

                            ########################################
                            # register region position (quadrant)
                            ########################################
                            centroids_final = []
                            for p in centroids_final_np:
                                #print(p)
                                p_x = p[0]
                                p_y = p[1]
                                c_x = mask_center_point[0]
                                c_y = mask_center_point[1]

                                pos = -1
                                if(p_x <= c_x and p_y <= c_y):
                                    #print("pos 1: upperleft")
                                    pos = 1
                                elif (p_x <= c_x and p_y > c_y):
                                    #print("pos 4: upperright")
                                    pos = 4
                                elif (p_x > c_x and p_y <= c_y):
                                    #print("pos 2: lowerleft")
                                    pos = 2
                                elif (p_x > c_x and p_y > c_y):
                                    #print("pos 3: lowerright")
                                    pos = 3
                                centroids_final.append([p_x, p_y, pos])

                            print(np.array(stats))
                            print(np.array(centroids_final))

                            ################################
                            # calculate corner points
                            ################################
                            x14 = -1
                            x23 = -1
                            y12 = -1
                            y43 = -1

                            if (MODE_CORNER_POINTS == "centroids"):
                                ################################
                                # prepare centroids
                                ################################
                                centroids_final_np = np.array(centroids_final)

                                pos = 1
                                idx = np.where(centroids_final_np[:, 2:].astype('int') == pos)[0]
                                x1, y1 = np.squeeze(centroids_final_np[idx])[:2]
                                centroids_points_list.append((x1, y1))

                                pos = 2
                                idx = np.where(centroids_final_np[:, 2:].astype('int') == pos)[0]
                                x2, y2 = np.squeeze(centroids_final_np[idx])[:2]
                                centroids_points_list.append((x2, y2))

                                pos = 3
                                idx = np.where(centroids_final_np[:, 2:].astype('int') == pos)[0]
                                x3, y3 = np.squeeze(centroids_final_np[idx])[:2]
                                centroids_points_list.append((x3, y3))

                                pos = 4
                                idx = np.where(centroids_final_np[:, 2:].astype('int') == pos)[0]
                                x4, y4 = np.squeeze(centroids_final_np[idx])[:2]
                                centroids_points_list.append((x4, y4))

                                # calculate mean values
                                x14 = int((x1 + x4) / 2)
                                x23 = int((x2 + x3) / 2)
                                y12 = int((y1 + y2) / 2)
                                y43 = int((y4 + y3) / 2)

                                corner_points_list.append((x14, y12))
                                corner_points_list.append((x23, y12))
                                corner_points_list.append((x14, y43))
                                corner_points_list.append((x23, y43))

                                line_coordinates_final.append([(x23, y12), (x14, y12)])
                                line_coordinates_final.append([(x23, y12), (x23, y43)])
                                line_coordinates_final.append([(x23, y43), (x14, y43)])
                                line_coordinates_final.append([(x14, y43), (x14, y12)])
                                line_coordinates_final.append([(x14, y12), (x23, y43)])
                                line_coordinates_final.append([(x23, y12), (x14, y43)])

                                #print(np.array(corner_points_list))

                            elif (MODE_CORNER_POINTS == "inner_points"):
                                #############################
                                # calculate inner borders
                                #############################
                                inner_point_list = []
                                for k, bb in enumerate(final_holes_np):
                                    bb_x = bb[0]
                                    bb_y = bb[1]
                                    bb_w = bb[2]
                                    bb_h = bb[3]
                                    hole_pos = centroids_final[k][2]

                                    x = -1
                                    y = -1
                                    if (hole_pos == 1):
                                        x = bb_x + bb_w
                                        y = bb_y + bb_h
                                    elif (hole_pos ==2):
                                        x = bb_x
                                        y = bb_y + bb_h
                                    elif (hole_pos == 3):
                                        x = bb_x
                                        y = bb_y
                                    elif (hole_pos == 4):
                                        x = bb_x + bb_w
                                        y = bb_y
                                    p = (x, y)
                                    inner_point_list.append([p, hole_pos])
                                #print(np.array(inner_point_list))

                                inner_points_np = np.array(inner_point_list)
                                print(inner_points_np)

                                line_coordinates = []
                                x1, y1 = (-1, -1)
                                x2, y2 = (-1, -1)
                                x3, y3 = (-1, -1)
                                x4, y4 = (-1, -1)

                                pos = 1
                                idx = np.where(inner_points_np[:, 1:] == pos)[0]
                                if (len(idx) > 0):
                                    x1, y1 = np.squeeze(inner_points_np[idx])[0]

                                pos = 2
                                idx = np.where(inner_points_np[:, 1:] == pos)[0]
                                if(len(idx) > 0):
                                    x2, y2 = np.squeeze(inner_points_np[idx])[0]

                                # get values of pos 3 and pos 4
                                pos = 3
                                idx = np.where(inner_points_np[:, 1:] == pos)[0]
                                if (len(idx) > 0):
                                    x3, y3 = np.squeeze(inner_points_np[idx])[0]

                                pos = 4
                                idx = np.where(inner_points_np[:, 1:] == pos)[0]
                                if (len(idx) > 0):
                                    x4, y4 = np.squeeze(inner_points_np[idx])[0]

                                #line_coordinates.append([(x2, y2), (x2, y2)])
                                #line_coordinates.append([(x3, y3), (x4, y4)])

                                # create final borders
                                if(x1 != -1 and x2 != -1 and x3 != -1 and x4 != -1 and y1 != -1 and y2 != -1 and y3 != -1 and y4 != -1):
                                    x14 = int((x1 + x4) / 2)
                                    x23 = int((x2 + x3) / 2)
                                    y12 = int((y1 + y2) / 2)
                                    y43 = int((y4 + y3) / 2)

                                    corner_points_list.append((x14, y12))
                                    corner_points_list.append((x23, y12))
                                    corner_points_list.append((x14, y43))
                                    corner_points_list.append((x23, y43))

                                    line_coordinates_final.append([(x23, y12), (x14, y12)])
                                    line_coordinates_final.append([(x23, y12), (x23, y43)])
                                    line_coordinates_final.append([(x23, y43), (x14, y43)])
                                    line_coordinates_final.append([(x14, y43), (x14, y12)])
                                    line_coordinates_final.append([(x14, y12), (x23, y43)])
                                    line_coordinates_final.append([(x23, y12), (x14, y43)])


                            if (x1 != -1 and x2 != -1 and x3 != -1 and x4 != -1
                                    and y1 != -1 and y2 != -1 and y3 != -1 and y4 != -1):
                                #############################
                                # calculate center of crop
                                #############################
                                center_x = x14 + int(abs(x14 - x23) / 2)
                                center_y = y12 + int(abs(y12 - y43) / 2)
                                center_point = (center_x, center_y)

                                #############################
                                # set max height of crop
                                #############################
                                max_height = int(abs(y12 - y43))

                elif (class_name == "9.5mm"):
                    print("CROP WINDWO 9.5mm")

                    if (num_labels >= 3):
                        #############################
                        # cleanup ccl results
                        #############################
                        idx = np.where(stats[:, 4:] <= 500)[0]
                        stats_n = np.delete(stats, idx, axis=0)
                        # labels_n = np.delete(labels, idx, axis=0)
                        centroids_n = np.delete(centroids, idx, axis=0)
                        idx = np.where(stats[:, 4:] >= 20000)[0]
                        stats = np.delete(stats_n, idx, axis=0)
                        # labels = np.delete(labels_n, idx, axis=0)
                        centroids = np.delete(centroids_n, idx, axis=0)
                        #print(centroids)

                        if (len(centroids) == 2):
                            idx = np.argsort(centroids, axis=0)[:, :1].flatten()
                            centroids = centroids[idx]
                            stats = stats[idx]

                            ########################################
                            # center of original mask
                            ########################################

                            mask_center_x = int(msk_np.shape[2] / 2)
                            mask_center_y = int(msk_np.shape[1] / 2)
                            mask_center_point = (mask_center_x, mask_center_y)

                            ########################################
                            # register region position (quadrant)
                            ########################################
                            centroids_final = []
                            for p in centroids:
                                #print(p)
                                p_x = p[0]
                                p_y = p[1]
                                c_x = mask_center_point[0]
                                c_y = mask_center_point[1]

                                pos = -1
                                if (p_y <= c_y):
                                    #print("pos 1: top")
                                    pos = 1
                                elif (p_y > c_y):
                                    #print("pos 2: down")
                                    pos = 2
                                centroids_final.append([p_x, p_y, pos])

                            ################################
                            # calculate corner points
                            ################################
                            x1 = -1
                            x2 = -1
                            y1 = -1
                            y2 = -1

                            if (MODE_CORNER_POINTS == "centroids"):
                                ################################
                                # prepare centroids
                                ################################
                                centroids_final_np = np.array(centroids_final)

                                pos = 1
                                idx = np.where(centroids_final_np[:, 2:].astype('int') == pos)[0]
                                x1, y1 = np.squeeze(centroids_final_np[idx])[:2]
                                centroids_points_list.append((x1, y1))

                                pos = 2
                                idx = np.where(centroids_final_np[:, 2:].astype('int') == pos)[0]
                                x2, y2 = np.squeeze(centroids_final_np[idx])[:2]
                                centroids_points_list.append((x2, y2))

                                # calculate mean values
                                x12 = int((x1 + x2) / 2)
                                y1 = int(y1)
                                y2 = int(y2)


                                corner_points_list.append((x12, y1))
                                corner_points_list.append((x12, y2))

                                line_coordinates_final.append([(x12, y1), (x12, y2)])
                                line_coordinates_final.append([(0, y1), (960, y1)])
                                line_coordinates_final.append([(0, y2), (960, y2)])

                            elif (MODE_CORNER_POINTS == "inner_points"):
                                #############################
                                # calculate inner borders
                                #############################

                                inner_point_list = []
                                for k, bb in enumerate(stats):
                                    bb_x = bb[0]
                                    bb_y = bb[1]
                                    bb_w = bb[2]
                                    bb_h = bb[3]
                                    hole_pos = centroids_final[k][2]

                                    x = -1
                                    y = -1
                                    if (hole_pos == 1):
                                        x = bb_x + int(bb_w/2)
                                        y = bb_y + bb_h
                                    elif (hole_pos == 2):
                                        x = bb_x + int(bb_w/2)
                                        y = bb_y

                                    p = (x, y)
                                    inner_point_list.append([p, hole_pos])

                                inner_points_np = np.array(inner_point_list)
                                # print(inner_points_np)

                                line_coordinates = []
                                pos = 1
                                idx = np.where(inner_points_np[:, 1:] == pos)[0]
                                x1, y1 = np.squeeze(inner_points_np[idx])[0]

                                pos = 2
                                idx = np.where(inner_points_np[:, 1:] == pos)[0]
                                x2, y2 = np.squeeze(inner_points_np[idx])[0]

                                line_coordinates.append([(0, y1), (960, y1)])
                                line_coordinates.append([(0, y2), (960, y2)])
                                line_coordinates.append([(x1, y1), (x2, y2)])

                                # create final borders
                                x12 = int((x1 + x2) / 2)

                                corner_points_list.append((x12, y1))
                                corner_points_list.append((x12, y2))

                                line_coordinates_final.append([(x12, y1), (x12, y2)])

                            #############################
                            # calculate center of crop
                            #############################
                            center_x = x12
                            center_y = y1 + int(abs(y1 - y2) / 2)
                            center_point = (center_x, center_y)
                            #print(center_point)

                            #############################
                            # set max height of crop
                            #############################
                            max_height = int(abs(y1 - y2))

            ########################
            ## visualization
            ########################
            tmp = input_image.copy()

            ########################
            # draw line segments
            ########################
            for p in line_coordinates_final:
                drawLine(tmp, p[0], p[1])

            for p in line_coordinates:
                drawLine(tmp, p[0], p[1])

            ########################
            # draw corner points
            ########################
            for x, y in corner_points_list:
                cv2.circle(tmp, (x, y), 5, (255, 255, 0), -1)

            if (center_point != -1):
                cv2.circle(tmp, center_point, 5, (0, 0, 255), -1)

            ########################
            # draw final crop window
            ########################
            if (max_height != -1):
                drawRectangleWithScaleFactor(tmp, max_height, FILM_SCALE_FACTOR, center_point[0], center_point[1])
            res = tmp.transpose(2, 0, 1)

            ########################
            # calculate overlay
            ########################
            overlay_pil = Image.blend(input_image_pil, heatmap_pil, 0.8)
            overlay_np = np.array(overlay_pil).transpose(2, 0, 1)

            ########################
            # plot on tensorboard
            ########################
            if(TENSORBOARD_IMG_LOG == True):
                label_orig_np = label_orig_np.transpose(2, 0, 1)
                label_orig_np[label_orig_np == 1] = 255
                label_orig_np[label_orig_np == 2] = 255
                writer.add_image("input", input_image_np, 0)
                writer.add_image("ground truth mask", label_orig_np, 0)
                writer.add_image("predicted mask", msk_np, 0)
                writer.add_image("heatmap", heatmap_np, 0)
                writer.add_image("overlay", overlay_np, 0)
                writer.add_image("result", res, 0)
            ''''''
            ########################
            # save to disc
            ########################
            if(SAVE_FLAG == True):
                print("<<<<<<<<CREATE DEBUG PKG >>>>>>>")

                save_path = "./" + str(expName[:-1]) + "/"
                createFolder(save_path)
                createFolder(save_path + "/imgs/")
                createFolder(save_path + "/msk_gt/")
                createFolder(save_path + "/msk_pred/")
                createFolder(save_path + "/gmm_msk_pred/")

                cv_format = (1, 2, 0)
                #cv_format = (0, 1, 2)

                image_list = []
                input_image_np = input_image_np.transpose((0, 1, 2))
                label_orig_np = label_orig_np.transpose((0, 1, 2))
                msk_np = msk_np.transpose(cv_format)
                heatmap_np = heatmap_np.transpose(cv_format)
                overlay_np = overlay_np.transpose(cv_format)
                res = res.transpose(cv_format)

                print(input_image_np.shape)
                print(label_orig_np.shape)
                print(msk_np.shape)
                #exit()

                image_list.append(input_image_np)
                image_list.append(label_orig_np)
                image_list.append(msk_np)
                #image_list.append(heatmap_np)
                #image_list.append(overlay_np)
                #image_list.append(res)

                print(vid_name)
                print(i)
                cv2.imwrite(save_path + "/imgs/" + "img_" + str(vid_name) + ".png", input_image_np)
                cv2.imwrite(save_path + "/msk_gt/" + "gt_" + str(vid_name) + ".png", label_orig_np)
                cv2.imwrite(save_path + "/msk_pred/" + "pred_" + str(vid_name) + ".png", msk_np)
                cv2.imwrite(save_path + "/gmm_msk_pred/" + "gmm_pred_" + str(vid_name) + ".png", final_gmm_mask)

            ###############################
            # calculate pixel accuracy
            ###############################

            msk_pil = msk_pil.convert("L")
            msk_np = np.expand_dims(np.array(msk_pil), axis=0)
            msk_tensor = torch.Tensor(msk_np).long()
            msk_tensor[msk_tensor == 255] = 1
            msk_tensor[msk_tensor == 254] = 2
            #print(torch.unique(msk_tensor))

            label_orig_pil = label_orig_pil.convert("L")
            label_orig_np = np.expand_dims(np.array(label_orig_pil), axis=0)
            label_orig_np[label_orig_np == 255] = 1
            label_orig_np[label_orig_np == 254] = 2
            label_orig_tensor = torch.Tensor(label_orig_np).long()
            #print(torch.unique(label_orig_tensor))

            overall_acc, avg_per_class_acc, avg_jacc, avg_dice = eval_metrics(label_orig_tensor, msk_tensor, len(classes))

            frm_based_results_list.append([vid_name,
                                           np.round(overall_acc.data.numpy().astype('float'), 4),
                                           np.round(avg_per_class_acc.data.numpy().astype('float'), 4),
                                           np.round(avg_jacc.data.numpy().astype('float'), 4),
                                           np.round(avg_dice.data.numpy().astype('float'), 4)])
            print("-----------------------")
            print('overall_acc: %.4f' % overall_acc)
            print('avg_per_class_acc: %.4f' % avg_per_class_acc)
            print('avg_jacc: %.4f' % avg_jacc)
            print('avg_dice: %.4f' % avg_dice)

    frm_based_results_np = np.array(frm_based_results_list)
    overall_scores = np.mean(frm_based_results_np[:, 1:].astype('float32'), axis=0)

    if (TIME_LOG == True):
        time_diff1 = stop1 - start1
        time_diff2 = stop2 - start2
        print("\n#####################################")
        print("time_diff(segnet):" + str(time_diff1))
        print("time_diff(gmm):" + str(time_diff2))
        print("time_diff(all):" + str(time_diff1 + time_diff2))
        print("#####################################")

    print("-----------------------")
    print('overall_acc: %.4f' % overall_scores[0])
    print('avg_per_class_acc: %.4f' % overall_scores[1])
    print('avg_jacc: %.4f' % overall_scores[2])
    print('avg_dice: %.4f' % overall_scores[3])


    # calculate final class scores
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(gt_class_label_list, pred_class_label_list)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(gt_class_label_list, pred_class_label_list, average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(gt_class_label_list, pred_class_label_list, average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(gt_class_label_list, pred_class_label_list, average='weighted')
    print('F1 score: %f' % f1)
    # confusion matrix
    matrix = confusion_matrix(gt_class_label_list, pred_class_label_list, labels=[0, 1, 2])
    print(matrix)
    writer.close()

def main():
    #run()
    run_inference_on_folder()

main()
