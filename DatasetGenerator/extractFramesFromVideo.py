import os
import cv2
import glob

#####################################################################
##           CONFIGURATION 
#####################################################################
film_reel_type = "16mm"
vPath = "/home/dhelm/VHH_Develop/pycharm_fbd/orig_videos/" + str(film_reel_type) + "/"
dstPath = "/home/dhelm/VHH_Develop/pycharm_fbd/orig_videos/testset_" + str(film_reel_type) + "/samples/"
TIME_STEP = 2
#####################################################################

def getFrame(cap, sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    ret, frame = cap.read()
    return ret, frame

#####################################################################
##           MAIN PART
#####################################################################
# filename list
# filename_list = os.listdir(vPath);
os.chdir(vPath)
filename_list = glob.glob("*.mp4")
print(filename_list)

# step through all videos
for filename in filename_list:
    vidName = filename.split('.')[0]

    # open video
    cap = cv2.VideoCapture(vPath + vidName + ".mp4")
    print(vidName)

    # extract video information
    frame_rate_orig = cap.get(cv2.CAP_PROP_FPS)
    number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frm_pos = 0
    sec = round(frm_pos / frame_rate_orig, 2)
    time_step = TIME_STEP
    length_of_video = round(number_of_frames / frame_rate_orig, 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frm_pos)

    # extract n frames per second
    while (cap.isOpened()):
        sec = sec + time_step
        sec = round(sec, 2)
        ret, frame = getFrame(cap, sec)

        frm_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #ret, frame = cap.read();

        if (ret == True):
            print("save " + str(vidName) + ": " + str(frm_pos))
            fName = str(vidName) + "_" + str(int(frm_pos))
            cv2.imwrite(dstPath + "/" + fName + ".png", frame)
        else:
            break

    cap.release()

