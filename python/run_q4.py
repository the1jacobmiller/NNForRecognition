import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    print('--------------------')
    center_xs = []
    center_ys = []
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        center_x = (maxc+minc)/2.
        center_y = (maxr+minr)/2.
        center_xs.append(center_x)
        center_ys.append(center_y)

    max_distance = (1./12.)*im1.shape[0]
    lines = [] # indices of bboxes for each line
    clusters = [] # center y values in each line cluster
    for i in range(len(center_ys)):
        y = center_ys[i]
        added = False
        distance_to_cluster = np.zeros((len(clusters,)))
        for j in range(len(clusters)):
            cluster = np.asarray(clusters[j])
            mean_cluster = np.mean(cluster)
            distance_to_cluster[j] = abs(y-mean_cluster)
        if distance_to_cluster.shape[0] > 0:
            index = np.argmin(distance_to_cluster)
            if distance_to_cluster[index] < max_distance:
                lines[index].append(i)
                clusters[index].append(y)
                added = True
        if not added:
            new_line = [i]
            new_cluster = [y]
            lines.append(new_line)
            clusters.append(new_cluster)
    print('Found',len(clusters),'lines')

    # sort by line
    means = np.zeros((len(clusters,)))
    for i in range(len(clusters)):
        cluster = np.asarray(clusters[i])
        means[i] = np.mean(cluster)
    sorted_lines = []
    for i in range(len(means)):
        line_index = np.argmin(means)
        means[line_index] = np.inf
        sorted_lines.append(lines[line_index])

    # sort left to right
    lines = []
    for line in sorted_lines:
        sorted_line = []
        xs = np.zeros((len(line),))
        for i in range(len(line)):
            letter_index = line[i]
            xs[i] = center_xs[letter_index]
        for i in range(len(xs)):
            line_index = np.argmin(xs)
            xs[line_index] = np.inf
            sorted_line.append(line[line_index])
        lines.append(sorted_line)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    crops = np.zeros((len(bboxes),1024))
    crop_count = 0
    for line in lines:
        for index in line:
            minr, minc, maxr, maxc = bboxes[index]

            # square crop
            rect = None
            if maxr-minr > maxc-minc:
                s = maxr-minr
                middle_c = (minc+maxc)/2
                rect = im1[int(minr):int(maxr),int(max(middle_c-s/2,0)):int(min(middle_c+s/2,im1.shape[1]))]
            else:
                s = maxc-minc
                middle_r = (minr+maxr)/2
                rect = im1[int(max(middle_r-s/2,0)):int(min(middle_r+s/2,im1.shape[0])),int(minc):int(maxc)]
            # convert to grayscale
            rect = skimage.color.rgb2gray(rect)

            # sharpen edges
            # rect = skimage.filters.unsharp_mask(rect, radius=1, amount=1)

            # threshold image to enhance letters
            thresh = skimage.filters.threshold_yen(rect)
            rect = rect >= thresh

            # blur image
            rect = skimage.filters.gaussian(rect, sigma=1.5)

            # resize image
            rect = skimage.transform.resize(rect,(32,32))
            # plt.imshow(rect,cmap=plt.cm.gray);plt.show()
            crop = (rect.T).flatten()
            crops[crop_count,:] = crop
            crop_count += 1

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    text = ''
    for crop in crops:
        h1 = forward(crop,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        l = letters[np.argmax(probs)]
        text = text + str(l)

    print(text)
