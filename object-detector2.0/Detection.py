import numpy as np
import cv2
import matplotlib
import human_detection
matplotlib.use('Agg')

def hsv_thresholding(img_RGB):
    bounding_boxes_B = np.array([])
    object = False
    if bounding_boxes_B.shape[0] != 0:
        object = True
    return bounding_boxes_B, object
def Integrate(img_th, img_RGB):
    img_th = cv2.cvtColor(img_th, cv2.COLOR_RGB2GRAY)
    bounding_boxes_RGB, bounding_boxes_th = human_detection.human_detection(img_RGB, img_th)
    human = False
    if bounding_boxes_th.shape[0] != 0:
        human = True
    return bounding_boxes_th, bounding_boxes_RGB, human
