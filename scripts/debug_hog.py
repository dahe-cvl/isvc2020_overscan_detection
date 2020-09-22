import numpy as np
import cv2
from matplotlib import pyplot as plt

img_src = "../templates/EF-NS_001_OeFM_481_part1.png"
img = cv2.imread(img_src)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv2.imshow("HOG Image", img)
cv2.waitKey()

exit()




import cv2
import numpy as np
''''''
import argparse
max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
def CannyThreshold(val):
    low_threshold = val
    img_blur = cv2.blur(src_gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv2.imshow(window_name, dst)

img_src = "../templates/EF-NS_001_OeFM_481.png"
src = cv2.imread(img_src)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
cv2.namedWindow(window_name)
cv2.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv2.waitKey()

exit()


from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
import argparse
import cv2
from matplotlib import pyplot as plt


img_src = "../templates/EF-NS_001_OeFM_481_part1.png"
img = cv2.imread(img_src)
print(img.shape)

(H, hogImage) = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(4, 4), transform_sqrt=True, block_norm="L2",
                            visualize=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
print(hogImage.shape)

cv2.imshow("HOG Image", hogImage)
cv2.waitKey()


from sklearn.manifold import TSNE
import seaborn as sns
def applyTSNE(mask_prep):
    print(mask_prep.shape)
    print(mask_prep[:, :, 1].shape)

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(mask_prep[:, :, 1])
    print(tsne_results)


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

applyTSNE(img)