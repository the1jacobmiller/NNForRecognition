import numpy as np
import matplotlib.pyplot as plt

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

buffer = 5

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
    # this can be 10 to 15 lines of code using skimage functions
    rows,cols = image.shape[:2]
    min_width = cols/64.
    min_height = rows/64.

    # estimate noise
    sigma_est = skimage.restoration.estimate_sigma(image, multichannel=True, average_sigmas=True)

    # denoise
    image = skimage.restoration.denoise_tv_chambolle(image, weight=sigma_est, multichannel=True)

    # grayscale
    grayscale = skimage.color.rgb2gray(image)

    # threshold image
    thresh = skimage.filters.threshold_yen(grayscale)
    binary = grayscale <= thresh
    bw = grayscale >= thresh

    # morphology
    binary = skimage.morphology.opening(binary)

    # find connected groups of pixels
    connected,num_labels = skimage.measure.label(binary,background=0,return_num=True,connectivity=2)

    # draw bbox around connected components
    bboxes = []
    for group in range(1,num_labels):
        minr = max(np.min(np.argwhere(connected==group)[:,0])-buffer,0)
        maxr = min(np.max(np.argwhere(connected==group)[:,0])+buffer,rows)
        minc = max(np.min(np.argwhere(connected==group)[:,1])-buffer,0)
        maxc = min(np.max(np.argwhere(connected==group)[:,1])+buffer,cols)

        w = maxc-minc
        h = maxr-minr
        if w>min_width and h>min_height:
            bboxes.append((minr, minc, maxr, maxc))

    return bboxes, bw
