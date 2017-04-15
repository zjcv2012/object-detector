# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
# To read file names
import argparse as ap
import glob
import os
from skimage.color import rgb2gray
from config import *
import numpy as np

if __name__ == "__main__":
    # Argument Parser
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive test images",
            required=True)
    parser.add_argument('-n', "--negpath", help="Path to negative test images",
            required=True)
    args = vars(parser.parse_args())

    pos_im_path = args["pospath"]
    neg_im_path = args["negpath"]

    # Load the classifier
    clf = joblib.load(model_path)

    print "Testing our classifier on positive images"
    pos_im_test=glob.glob(os.path.join(pos_im_path, "*"))
    pos_im_num=len(pos_im_test)
    clf_suc_num=0

    for im_path in pos_im_test:
        im = imread(im_path, as_grey=True)
        im = rgb2gray(im)
        fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        #fd=hog(im)
        temp = np.array(fd).reshape((1, -1))
        pred = clf.predict(temp)
        #print pred
        if pred[0]==1:
            clf_suc_num=clf_suc_num+1
            #print(clf_suc_num)


    print(clf_suc_num/(float)(pos_im_num))

    print "Testing our classifier on negative images"
    neg_im_test=glob.glob(os.path.join(neg_im_path, "*"))
    neg_im_num=len(neg_im_test)
    clf_suc_num=0

    for im_path in neg_im_test:
        im = imread(im_path, as_grey=True)
        im = rgb2gray(im)
        fd = hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
        #fd=hog(im)
        temp = np.array(fd).reshape((1, -1))
        pred = clf.predict(temp)
        if pred[0]==0:
            clf_suc_num=clf_suc_num+1

    print(clf_suc_num/(float)(neg_im_num))