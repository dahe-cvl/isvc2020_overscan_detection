import json
import os
import cv2
import numpy as np

db_path = "C:\\Users\\dhelm\\Documents\\testset_16mm\\"
labels_path = db_path + "\\" + "labels\\"
img_path = db_path + "\\" + "samples\\"
dst_path = db_path + "\\" + "masks\\"
NUM_REGIONS = 4
CLASS_IDX = 1

file_list = os.listdir(labels_path)
#file_list = [file_list[0]]

for file in file_list:
    if(file.split('.')[-1] != 'json'):
        continue

    with open(labels_path + file) as f:
      data = json.load(f)

    aid = data['asset']['id']
    name = data['asset']['name']
    path = data['asset']['path']
    dim = (data['asset']['size']['height'], data['asset']['size']['width'])

    print("---------------------------")
    print("aid: " + str(aid))
    print("name: " + str(name))
    print("path: " + str(path))
    print("dim: " + str(dim))



    # Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
    regions_list = data['regions']

    #print(data['asset'].keys())
    #print(len(regions_list))

    if(len(regions_list) != NUM_REGIONS):
        continue

    regions_final_list = []
    for region in regions_list:
        #print(region)
        bb = region['boundingBox']
        points_list = region['points']

        points_l = []
        for point in points_list:
            p_x = int(round(point['x']))
            p_y = int(round(point['y']))
            p = (p_x, p_y)
            points_l.append(p)

        bb_h = float(round(bb['height']))
        bb_w = float(round(bb['width']))
        bb_x = float(round(bb['left']))
        bb_y = float(round(bb['top']))

        regions_final_list.append([bb_x, bb_y, bb_w, bb_h, points_l])

        #print(bb_x)
        #print(bb_y)
        #print(bb_h)
        #print(bb_w)

    #print(np.array(regions_final_list))
    #print(img_path + "\\" + name)

    img = cv2.imread(img_path + "" + name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_gray = np.expand_dims(img_gray, axis=2)
    mask = np.zeros((img_gray.shape[0], img_gray.shape[1], 1))

    #print(mask.shape)

    for region in regions_final_list:
        bb_x, bb_y, bb_w, bb_h, points_l = region
        '''
        for x, y in points_l:
            print("---")
            print(x)
            print(y)
            if(x >= mask.shape[0]): x = mask.shape[0]-2
            if(y >= mask.shape[1]): y = mask.shape[1]-2
            mask[int(x), int(y)] = 255
        '''
        pts = np.array(points_l).astype('int32')
        #pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        #print(pts)
        #print(pts.shape)

        #mask = cv2.polylines(mask, [pts], isClosed=True, color=(255, 0, 0), thickness=1)
        mask = cv2.fillPoly(mask, [pts], color=(CLASS_IDX, CLASS_IDX, CLASS_IDX))

    mask = mask.astype('int32')
    print(dst_path + "\\" + str(name.split('.')[0]) + "_mask.png")
    cv2.imwrite(dst_path + "\\" + str(name.split('.')[0]) + "_mask.png", mask)

    #cv2.imshow("asdf", mask)
    #cv2.waitKey()



