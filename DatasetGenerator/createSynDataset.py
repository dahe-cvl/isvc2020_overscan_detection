import cv2
import numpy as np

#####################################################
####           CONFIGURATION SECTION             ####
#####################################################
#dst_path = "/data/share/frame_border_detection_db_v5/rgb_3class/"
#src_path = "/data/share/frame_border_detection_db_v3/ms_coco_images_v3/"
#mask_path = "./masks_3classes/"

#reel_type = "16mm"   # 16mm OR 9.5mm

db_set = "train"
dst_path = "/data/share/frame_border_detection_db_v6/rgb_2class_large/" + str(db_set) + "/"
src_path = "/data/share/frame_border_detection_db_v6/ms_coco_images_v3/" + str(db_set) + "/"
deform_flag = True
n_classes = 3
image_size = (720, 960, 1)

#####################################################


def  rounded_rectangle(src, top_left, bottom_right, radius=1.0, color=[], thickness=1, line_type=cv2.LINE_AA):

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = (bottom_right[0], bottom_right[1])
    p4 = (top_left[0], bottom_right[1])

    #height = abs(bottom_right[0] - top_left[1])
    #if radius > 1:
    #    radius = 1

    #corner_radius = int(radius * (height/2))
    corner_radius = radius

    if thickness < 0:

        #big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
        [top_left_main_rect, bottom_right_main_rect],
        [top_left_rect_left, bottom_right_rect_left],
        [top_left_rect_right, bottom_right_rect_right]]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

    return src

import random
import os

def generateMask(reel_type="16mm", n_classes=3):

    if(reel_type == "16mm"):
        ###################################################
        ### raster settings
        ###################################################
        raster_scale_factor = 1.37
        raster_pos = (0, 0)  # -150 to 130  -95 to 115
        raster_random_range = ([-150, 130], [-95, 115])
        distance_h = 600
        distance_w = int(distance_h * raster_scale_factor)

        ###################################################
        ### hole settings
        ###################################################
        hole_scale_factor = 1.56
        hole_h = 100
        hole_w = int(hole_h * hole_scale_factor)

        if (n_classes == 3):
            color = [2, 2, 2]
        elif (n_classes == 2):
            color = [1, 1, 1]

        corner_radius = 20

        rand_x = random.randint(raster_random_range[0][0], raster_random_range[0][1])
        rand_y = random.randint(raster_random_range[1][0], raster_random_range[1][1])
        raster_pos = (rand_x, rand_y)

        hole_list = []
        top_left = raster_pos #(10, 10)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        top_left = (raster_pos[0], raster_pos[1] + distance_h)   # (10, 500)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        top_left = (raster_pos[0] + distance_w, raster_pos[1]) # (500, 10)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        top_left = (raster_pos[0] + distance_w, raster_pos[1] + distance_h)  #(500, 500)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        if(corner_radius > int(hole_h/2)):
            print("radius too high")
            exit()

        mask = np.zeros(image_size)
        mask[:] = 0

        for hole in hole_list:
            top_left, bottom_right = hole
            mask = rounded_rectangle(mask, top_left, bottom_right, color=color, radius=corner_radius, thickness=-1)

    elif (reel_type == "95mm"):
        ###################################################
        ### raster settings
        ###################################################
        raster_scale_factor = 1.307
        raster_pos = (380, -15)  # -150 to 130  -75 to -15
        raster_random_range = ([370, 390], [-75, -15])
        distance_h = 730   # -715   790

        ###################################################
        ### hole settings
        ###################################################
        hole_scale_factor = 3.0
        hole_h = 80
        hole_w = int(hole_h * hole_scale_factor)

        if(n_classes == 3):
            color = [2, 2, 2]
        elif(n_classes == 2):
            color = [1, 1, 1]

        corner_radius = 20

        #
        # a ... width of frame window [mm]
        # b ... height of frame window [mm]
        # x ... width of hole [mm]
        # y ... height of hole [mm]
        # D ... centered distance between holes (y axis) [mm]
        # c ... distance between hole to frame border [mm]
        #
        #
        #
        # x = 2
        # y = 0.8
        # x_px = 200
        #
        # a = 8.5
        # b = 6.5
        # D = 7.54
        # c = 0.12
        #
        # factor = int( (x_px / x) * y)
        #
        # a_px = factor * a
        # b_px = factor * b
        # c_px = factor * c
        # D_px = factor * D
        #

        rand_x = random.randint(raster_random_range[0][0], raster_random_range[0][1])
        rand_y = random.randint(raster_random_range[1][0], raster_random_range[1][1])
        raster_pos = (rand_x, rand_y)

        hole_list = []
        top_left = raster_pos  # (10, 10)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        top_left = (raster_pos[0], raster_pos[1] + distance_h)  # (10, 500)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        if (corner_radius > int(hole_h / 2)):
            print("radius too high")
            exit()

        mask = np.zeros(image_size)
        mask[:] = 0

        for hole in hole_list:
            top_left, bottom_right = hole
            mask = rounded_rectangle(mask, top_left, bottom_right, color=color, radius=corner_radius, thickness=-1)


    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    #exit()

    return mask

def generateMaskWithFrameWindow(reel_type="16mm"):

    if(reel_type == "16mm"):
        ###################################################
        ### raster settings
        ###################################################
        raster_scale_factor = 1.37
        raster_pos = (0, 0)  # -150 to 130  -95 to 115
        raster_random_range = ([-150, 130], [-95, 115])
        distance_h = 600
        distance_w = int(distance_h * raster_scale_factor)

        ###################################################
        ### hole settings
        ###################################################
        hole_scale_factor = 1.56
        hole_h = 100
        hole_w = int(hole_h * hole_scale_factor)
        color = [1, 1, 1]
        corner_radius = 20

        rand_x = random.randint(raster_random_range[0][0], raster_random_range[0][1])
        rand_y = random.randint(raster_random_range[1][0], raster_random_range[1][1])
        raster_pos = (rand_x, rand_y)

        hole_list = []
        top_left = raster_pos #(10, 10)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        top_left = (raster_pos[0], raster_pos[1] + distance_h)   # (10, 500)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        top_left = (raster_pos[0] + distance_w, raster_pos[1]) # (500, 10)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        top_left = (raster_pos[0] + distance_w, raster_pos[1] + distance_h)  #(500, 500)
        bottom_right = (top_left[0] + hole_w, top_left[1] + hole_h)
        hole_list.append([top_left, bottom_right])

        if(corner_radius > int(hole_h/2)):
            print("radius too high")
            exit()

        mask = np.zeros(image_size)
        mask[:] = 0

        for hole in hole_list:
            top_left, bottom_right = hole
            mask = rounded_rectangle(mask, top_left, bottom_right, color=color, radius=corner_radius, thickness=-1)

    elif (reel_type == "95mm"):
        ###################################################
        ### settings
        ###################################################
        #
        # a ... width of frame window [mm]
        # b ... height of frame window [mm]
        # x ... width of hole [mm]
        # y ... height of hole [mm]
        # D ... centered distance between holes (y axis) [mm]
        # c ... distance between hole to frame border [mm]
        #
        #
        ###################################################

        ###################################################
        ### hole
        ###################################################
        color = [2, 2, 2]
        corner_radius = 20

        # measures
        x_px = 200
        x = 2.0
        y = x / 2.5  # factor 2.5 --> 2mm / 0.8mm scale of hole

        ###################################################
        ### frame window
        ###################################################
        a = 8.5
        b = 6.5
        D = 7.54
        c = 0.12
        frame_radius = 5

        # position
        raster_random_range = ([285, 440], [-70, -12])
        x1 = 440
        y1 = -50
        ###################################################

        rand_x = random.randint(raster_random_range[0][0], raster_random_range[0][1])
        rand_y = random.randint(raster_random_range[1][0], raster_random_range[1][1])
        x1, y1 = (rand_x, rand_y)

        factor = int((x_px / x) * y)
        a_px = factor * a
        b_px = factor * b
        c_px = factor * c
        D_px = factor * D
        y_px = factor * y

        hole1_pos = (x1, y1)
        x_frm = int((x1 + x_px/2) - a_px/2)
        y_frm = int(y1 + y_px + c_px)
        frame_pos = (x_frm, y_frm)

        x2 = x1
        y2 = int(y1 + y_px + 2*c_px + b_px)
        hole2_pos = (x2, y2)

        hole_list = []
        frame_list = []
        ## rect hole 1
        hole1_rect = [(hole1_pos[0], hole1_pos[1]), (int(hole1_pos[0] + x_px), int(hole1_pos[1] + y_px))]
        ## rect frame window
        frame_rect = [(frame_pos[0], frame_pos[1]), (int(frame_pos[0] + a_px), int(frame_pos[1] + b_px))]
        ## rect hole 2
        hole2_rect = [(hole2_pos[0], hole2_pos[1]), (int(hole2_pos[0] + x_px), int(hole2_pos[1] + y_px))]

        hole_list.append(hole1_rect)
        hole_list.append(hole2_rect)
        frame_list.append(frame_rect)

        mask = np.zeros(image_size)
        mask[:] = 0

        for hole in hole_list:
            top_left, bottom_right = hole
            mask = rounded_rectangle(mask, top_left, bottom_right, color=color, radius=corner_radius, thickness=-1)

        for frm in frame_list:
            top_left, bottom_right = frm
            mask = rounded_rectangle(mask, top_left, bottom_right, color=color, radius=frame_radius, thickness=-1)

    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    exit()

    return mask

def generateImageWithHoles(src_image, mask):
    src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    src_image_gray = np.expand_dims(src_image_gray, axis=2).astype('uint8')
    print(src_image_gray.shape)
    print(mask.shape)
    mask = mask.astype('uint8')

    src_image_gray[mask == 1] = 255
    src_image_gray[mask == 2] = 255
    res = cv2.cvtColor(src_image_gray, cv2.COLOR_GRAY2RGB)

    cv2.imwrite("../templates/sample_result_paper.png", res)
    cv2.imshow('sample_result', res)
    cv2.waitKey(0)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    return res


def applyDeformation(src_image, blur_flag=False, brightness_flag=False, gamma_correction_flag=False, hole_deformation_flag=False):
    result_image = src_image.copy()

    if(blur_flag == True):
        kernel_size = (7, 7)
        sigmaX = 5.5
        sigmaY = 8.5
        result_image = cv2.GaussianBlur(result_image, kernel_size, sigmaX=sigmaX, sigmaY=sigmaY)
    if (brightness_flag == True):
        alpha = 1.0   # [1.0 - 3.0]
        beta = 50.0    # [0 - 100]

        for y in range(result_image.shape[0]):
            for x in range(result_image.shape[1]):
                for c in range(result_image.shape[2]):
                    result_image[y, x, c] = np.clip(alpha * result_image[y, x, c] + beta, 0, 255)

    if (gamma_correction_flag == True):
        gamma = random.uniform(0.2, 2.5)
        #gamma = 2.2  # 0.5 - 2.2
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        result_image = cv2.LUT(result_image, lookUpTable)

    if (hole_deformation_flag == True):
        print("NOT IMPLEMENTED YET")

    return result_image;

def run():
    #db_set = "train"
    #dst_path = "/data/share/frame_border_detection_db_v6/rgb_3class_large/"
    #src_path = "/data/share/frame_border_detection_db_v6/ms_coco_images_v3/" + str(db_set) + "/"
    #deform_flag = True
    #n_classes = 3

    sample_dir = dst_path + "/samples"
    label_dir = dst_path + "/labels"

    print("create folder structure ... ")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    image_files_list = os.listdir(src_path)
    # mask_files_list.sort()

    reel_type_list = ["16mm", "95mm"]
    t = "16mm"
    for i, file in enumerate(image_files_list):
        #for t in reel_type_list:
        src_image_file = src_path + "/" + file
        src_image = cv2.imread(src_image_file)
        src_image = cv2.resize(src_image, (image_size[1], image_size[0]))

        mask = generateMask(reel_type=t, n_classes=n_classes)
        res = generateImageWithHoles(src_image, mask);

        if(deform_flag == True):
            res = applyDeformation(res, blur_flag=True,
                                       brightness_flag=False,
                                       gamma_correction_flag=True,
                                       hole_deformation_flag=False
                                       );


        sample = (res, mask)

        cv2.imwrite(sample_dir + "/s_" + str("{:05d}".format(i)) + ".png", sample[0])
        cv2.imwrite(label_dir + "/l_" + str("{:05d}".format(i)) + ".png", sample[1])

        if(i == int(len(image_files_list) / 2)):
            t = "95mm"

def run_debug1():
    img_path = "../templates/000000575823.jpg"
    src_image = cv2.imread(img_path)
    src_image = cv2.resize(src_image, (image_size[1], image_size[0]))

    mask = generateMask(reel_type="16mm")
    res = generateImageWithHoles(src_image, mask);

    sample = (res, mask)

def run_debug2():
    img_path = "../templates/16mm_test.png"
    src_image = cv2.imread(img_path)

    mask = generateMaskWithFrameWindow(reel_type="16mm")
    #res = generateImageWithHoles(src_image, mask);
    #sample = (res, mask)

def run_debug3():
    img_path = "s_4997.png"
    src_image = cv2.imread(img_path)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

    res_img = applyDeformation(src_image, blur_flag=True,
                               brightness_flag=False,
                               gamma_correction_flag=True,
                               hole_deformation_flag=False
                               );
    cv2.imshow("blurred_frame", np.hstack((src_image, res_img)))
    cv2.waitKey(0)

def applyMask(source_img, mask):
    fine_mask = mask.copy()
    input_img = source_img.copy()

    kernel = np.ones((25, 25), np.uint8)
    prep_mask = cv2.dilate(fine_mask, kernel, 25)
    #cv2.imshow("inputs", np.hstack((input_img, fine_mask, prep_mask)))
    #cv2.waitKey(0)

    res = input_img.copy()
    res[prep_mask == 0] = 0
    #cv2.imshow("inputs", np.hstack((input_img, fine_mask, res)))
    #cv2.waitKey(0)


    return res

from matplotlib import pyplot as plt
def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

import skimage.segmentation as seg
import skimage.color as color
def applyfelzenszwab(source_img, mask):
    fine_mask = mask.copy()
    input_img = source_img.copy()

    cv2.imshow("inputs", np.hstack((input_img, fine_mask)))
    cv2.waitKey(0)

    image_felzenszwalb = seg.felzenszwalb(mask)
    image_felzenszwalb_colored = color.label2rgb(image_felzenszwalb, mask, kind='avg')

    print(image_felzenszwalb.shape)
    print(image_felzenszwalb[:100])
    image_show(image_felzenszwalb)
    image_show(image_felzenszwalb_colored)

    #plt.imshow(image_felzenszwalb)
    plt.show()

    #cv2.imshow("adsf", image_felzenszwalb); #, cmap='gray'
    #cv2.waitKey(0)


    return fine_mask

from sklearn.manifold import TSNE
import seaborn as sns
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


def run_debug4():
    img_path = "1.png"
    mask_path = "2.png"
    pred_path = "3.png"
    dim = (480, 360)



    src_image = cv2.imread(img_path)
    src_image = cv2.resize(src_image, dim)
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, dim)
    pred_mask = cv2.imread(pred_path)
    pred_mask = cv2.resize(pred_mask, dim)

    prep_mask = applyMask(src_image, pred_mask)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharped_mask = cv2.filter2D(prep_mask, -1, kernel)

    #cv2.imshow("inputs", np.hstack((src_image, pred_mask, prep_mask, sharped_mask)))
    #cv2.waitKey(0)

    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    histogram_img = cv2.calcHist(mask, [0], None, [256], [0, 256])
    cv2.imshow("hist", histogram_img)
    cv2.waitKey(0)

    applyTSNE(prep_mask)

def run_debug5():
    import cv2
    import numpy as np

    def get8n(x, y, shape):
        out = []
        maxx = shape[1] - 1
        maxy = shape[0] - 1

        # top left
        outx = min(max(x - 1, 0), maxx)
        outy = min(max(y - 1, 0), maxy)
        out.append((outx, outy))

        # top center
        outx = x
        outy = min(max(y - 1, 0), maxy)
        out.append((outx, outy))

        # top right
        outx = min(max(x + 1, 0), maxx)
        outy = min(max(y - 1, 0), maxy)
        out.append((outx, outy))

        # left
        outx = min(max(x - 1, 0), maxx)
        outy = y
        out.append((outx, outy))

        # right
        outx = min(max(x + 1, 0), maxx)
        outy = y
        out.append((outx, outy))

        # bottom left
        outx = min(max(x - 1, 0), maxx)
        outy = min(max(y + 1, 0), maxy)
        out.append((outx, outy))

        # bottom center
        outx = x
        outy = min(max(y + 1, 0), maxy)
        out.append((outx, outy))

        # bottom right
        outx = min(max(x + 1, 0), maxx)
        outy = min(max(y + 1, 0), maxy)
        out.append((outx, outy))

        return out

    def region_growing(img, seed):
        list = []
        outimg = np.zeros_like(img)
        list.append((seed[0], seed[1]))
        processed = []
        while (len(list) > 0):
            pix = list[0]
            outimg[pix[0], pix[1]] = 255
            for coord in get8n(pix[0], pix[1], img.shape):
                print(img[coord[0], coord[1]])
                if img[coord[0], coord[1]] > 200:
                    outimg[coord[0], coord[1]] = 255
                    if not coord in processed:
                        list.append(coord)
                    processed.append(coord)
            list.pop(0)
            # cv2.imshow("progress",outimg)
            # cv2.waitKey(1)
        return outimg

    def on_mouse(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Seed: ' + str(x) + ', ' + str(y), img[y, x])
            clicks.append((y, x))

    clicks = []
    img = cv2.imread('1.png', 0)
    #ret, img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    cv2.namedWindow('Input')
    cv2.setMouseCallback('Input', on_mouse, 0, )
    cv2.imshow('Input', img)
    cv2.waitKey()
    seed = clicks[-1]
    out = region_growing(img, seed)
    cv2.imshow('Region Growing', out)
    cv2.waitKey()
    cv2.destroyAllWindows()

    #applyfelzenszwab(src_image, sharped_mask)



#run()
run_debug1()
#run_debug5()