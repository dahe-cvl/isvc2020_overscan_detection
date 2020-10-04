import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
plt.rcParams['text.usetex'] = True #Let TeX do the typsetting


class FineSegmentation(object):
    def __init__(self):
        print("create instance of finesegmentation class")
        self.verbose = 0

    def apply_gmm(self, img):
        hist, bin_edges = np.histogram(img, bins=256)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        classif = GaussianMixture(n_components=1, covariance_type='tied', tol=1e-3,
                     reg_covar=1e-6, max_iter=800, n_init=1, init_params='kmeans',
                     weights_init=None, means_init=None, precisions_init=None,
                     random_state=None, warm_start=False,
                     verbose=self.verbose, verbose_interval=10)
        classif.fit(img.reshape((img.size, 1)))
        threshold = np.mean(classif.means_)

        binary_img = img.copy()
        binary_img[binary_img > threshold] = 255
        binary_img[binary_img <= threshold] = 0

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
         
        plt.figure()
        plt.plot(bin_centers, hist, lw=2)
        plt.axvline(threshold, color='r', ls='--', lw=2)
        plt.show()
    
        plt.figure()
        plt.plot(bin_centers, hist, lw=2)
        plt.axvline(threshold, color='r', ls='--', lw=2)
        plt.savefig("gmm_hist" + ".png", dpi=300)
        #exit()
        '''
        return binary_img
