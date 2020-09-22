import numpy as np
import cv2
import os


def readSamples(db_path, image_size):
    files = []
    print(db_path)
    # r=root, d=directories, f = files
    for r, d, f in os.walk(db_path):
        for file in f:
            if '.png' in file:
                files.append(os.path.join(r, file))
    all_samples_r = []
    all_samples_g = []
    all_samples_b = []
    for f in files:
        # print(f)

        # read images
        frame = cv2.imread(f);

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        frame_np = np.array(frame)

        # resize image
        dim = (image_size, image_size)
        frame_resized = cv2.resize(frame_np, dim, interpolation=cv2.INTER_AREA)
        # print(frame_resized.shape)

        # split image
        b, g, r = cv2.split(frame_resized)

        all_samples_r.append(r)
        all_samples_g.append(g)
        all_samples_b.append(b)

        print("--------------------------------------------------")
        print("process frame: " + str(f))

    all_samples_r_np = np.array(all_samples_r)
    all_samples_g_np = np.array(all_samples_g)
    all_samples_b_np = np.array(all_samples_b)
    print(all_samples_r_np.shape)
    print(all_samples_g_np.shape)
    print(all_samples_b_np.shape)

    return all_samples_r_np, all_samples_g_np, all_samples_b_np


def checkStatistics(zero_centered_r_np, zero_centered_g_np, zero_centered_b_np, normalized_r_np, normalized_g_np, normalized_b_np):
    # calculate zero-centered frames
    print(np.mean(zero_centered_r_np))
    print(np.mean(zero_centered_g_np))
    print(np.mean(zero_centered_b_np))

    # calculate standard deviation for each color channel
    print(np.std(normalized_r_np))
    print(np.std(normalized_g_np))
    print(np.std(normalized_b_np))

def calculateSTD(all_samples_r_np, all_samples_g_np, all_samples_b_np ):
    print("calculate standard deviation of zero-centered frames ... ")
    std_r = np.std(all_samples_r_np)
    std_g = np.std(all_samples_g_np)
    std_b = np.std(all_samples_b_np)
    print(std_r)
    print(std_g)
    print(std_b)
    return std_r, std_g, std_b

def calculateMean(all_samples_r_np, all_samples_g_np, all_samples_b_np):
    print("calculate mean value for each color channel ... ")
    mean_r = np.mean(all_samples_r_np)
    mean_g = np.mean(all_samples_g_np)
    mean_b = np.mean(all_samples_b_np)
    print(mean_r)
    print(mean_g)
    print(mean_b)
    return mean_r, mean_g, mean_b

    #print("calculate mean image for each color channel ... ")
    #mean_r = np.mean(all_samples_r_np, axis=0);
    #mean_g = np.mean(all_samples_g_np, axis=0);
    #mean_b = np.mean(all_samples_b_np, axis=0);
    #print(mean_r.shape)
    #print(mean_g.shape)
    #print(mean_b.shape)

    #print("merge color channels to one mean image ... ")
    #mean_frame = cv2.merge((mean_b, mean_g, mean_r));
    #print(mean_frame.shape)

    #print("save image ... ")
    #cv2.imwrite(dst_path + "/mean_frame_" + str(image_size) + ".jpg", mean_frame)

def saveStatistics(dst_path, image_size, mean_r, mean_g, mean_b, std_r, std_g, std_b):
    print("save statistics to file ... ")

    fp = open(dst_path + "statistics_" + str(image_size) + "x" + str(image_size) + ".txt", 'w')
    fp.write("image_size:" + str(image_size) + "\n")
    fp.write("mean_r = " + str(mean_r.round(5)) + "\n")
    fp.write("mean_g = " + str(mean_g.round(5)) + "\n")
    fp.write("mean_b = " + str(mean_b.round(5)) + "\n")
    fp.write("std_r = " + str(std_r.round(5)) + "\n")
    fp.write("std_g = " + str(std_g.round(5)) + "\n")
    fp.write("std_b = " + str(std_b.round(5)) + "\n")

def loadStatistics(statistics_filepath):
    print("save statistics to file ... ")

    fp = open(statistics_filepath, 'r')
    lines = fp.readlines()
    print(lines)
    image_size = int(lines[0].split(':')[1])

    mean_r = float(lines[1].split(' = ')[1])
    mean_g = float(lines[2].split(' = ')[1])
    mean_b = float(lines[3].split(' = ')[1])

    std_r = float(lines[4].split(' = ')[1])
    std_g = float(lines[5].split(' = ')[1])
    std_b = float(lines[6].split(' = ')[1])

    return image_size, mean_r, mean_g, mean_b, std_r, std_g, std_b


def main():
    print("prepare keras database")

    ############################################################################
    ## CONFIGURATION
    ############################################################################
    db_path = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/generated/stc/20191203/db_v7/train/"
    dst_path = "/caa/Projects02/vhh/private/database_nobackup/VHH_datasets/generated/stc/20191203/db_v7/"
    image_size = 128
    ############################################################################

    print("get all samples...")
    all_samples_r_np, all_samples_g_np, all_samples_b_np = readSamples(db_path, image_size)

    ACTIVE_FLAG = True
    if(ACTIVE_FLAG == True):
        mean_r, mean_g, mean_b = calculateMean(all_samples_r_np, all_samples_g_np, all_samples_b_np)
        std_r, std_g, std_b = calculateSTD(all_samples_r_np, all_samples_g_np, all_samples_b_np)

        # save statiscits
        saveStatistics(dst_path, image_size, mean_r, mean_g, mean_b, std_r, std_g, std_b)
    elif (ACTIVE_FLAG == False):
        image_size, mean_r, mean_g, mean_b, std_r, std_g, std_b = loadStatistics(dst_path + "/statistics_" + str(image_size) + "x"+ str(image_size) + ".txt")

        # zero-centering
        zero_centered_r_np = all_samples_r_np - mean_r
        zero_centered_g_np = all_samples_g_np - mean_g
        zero_centered_b_np = all_samples_b_np - mean_b

        # normalization
        normalized_r_np = zero_centered_r_np / std_r
        normalized_g_np = zero_centered_g_np / std_g
        normalized_b_np = zero_centered_b_np / std_b

        checkStatistics(zero_centered_r_np, zero_centered_g_np, zero_centered_b_np, normalized_r_np, normalized_g_np,
                        normalized_b_np)


if(__name__== "__main__"):
  main()