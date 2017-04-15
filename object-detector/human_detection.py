import thermal_detection
import rgb_detection

import cv2
import numpy as np
import copy

#Perform 
from sklearn.externals import joblib
from skimage.feature import hog
from config import *


def overlap_ratio(rect1,rect2):
	overlap_rect=rect1.copy()
	overlap_rect[0]=max(rect1[0],rect2[0])
	overlap_rect[1]=max(rect1[1],rect2[1])
	overlap_rect[2]=min(rect1[0]+rect1[2],rect2[0]+rect2[2])-overlap_rect[0]
	overlap_rect[3]=min(rect1[1]+rect1[3],rect2[1]+rect2[3])-overlap_rect[1]
	
	overlap_ratio=float(overlap_rect[2]*overlap_rect[3])/max(rect1[2]*rect1[3],rect2[2]*rect2[3])
	return overlap_ratio

def fuse_candidate(rgb_candidate,thermal_candidate):
	x_drift=10
	y_drift=20
	im_rows=480
	im_cols=640

	overlap_thre=0.4

#Map candidate from RGB to thermal
	rtot_candidate=[]
	for i in range(len(rgb_candidate)):
		rtot_temp=rgb_candidate[i].copy()
		rtot_temp[0]=max(0,rgb_candidate[i][0]-x_drift)
		rtot_temp[2]=max(0,rgb_candidate[i][0]+rgb_candidate[i][2]-x_drift)-rtot_temp[0]
		rtot_temp[1]=max(0,rgb_candidate[i][1]-y_drift)
		rtot_temp[3]=max(0,rgb_candidate[i][1]+rgb_candidate[i][3]-y_drift)-rtot_temp[1]
		rtot_candidate.append(rtot_temp)
	
#Map candidate from thermal to RGB
	ttor_candidate=[]
	for i in range(len(thermal_candidate)):
		ttor_temp=thermal_candidate[i].copy()
		ttor_temp[0]=min(im_cols-1,thermal_candidate[i][0]+x_drift)
		ttor_temp[2]=min(im_cols-1,thermal_candidate[i][0]+thermal_candidate[i][2]+x_drift)-ttor_temp[0]
		ttor_temp[1]=min(im_rows-1,thermal_candidate[i][1]+y_drift)
		ttor_temp[3]=min(im_rows-1,thermal_candidate[i][1]+thermal_candidate[i][3]+y_drift)-ttor_temp[1]
		ttor_candidate.append(ttor_temp)

#Try to match candidates from rgb and those from thermal
	rgb_fused_candidate=[]
	thermal_fused_candidate=[]
	flag=0 #check whether correspondonce is founded
	indice=[] #store the indices of matched candidate

	for i in range(len(rgb_candidate)):
		for j in range(len(thermal_candidate)):
			if (overlap_ratio(rtot_candidate[i],thermal_candidate[j])>=overlap_thre):
				rgb_fused_candidate.append(rgb_candidate[i])
				thermal_fused_candidate.append(thermal_candidate[j])
				indice.append(j)
				flag=1
				break

		if(flag==0):
			rgb_fused_candidate.append(rgb_candidate[i])
			thermal_fused_candidate.append(rtot_candidate[i])

		flag=0

	temp=np.zeros(len(thermal_candidate))
	temp[indice]=1
	non_matched_indice=np.where(temp==0)[0]
	for i in non_matched_indice:
		thermal_fused_candidate.append(thermal_candidate[i])
		rgb_fused_candidate.append(ttor_candidate[i])

	return rgb_fused_candidate,thermal_fused_candidate


#Integrated human detection algorithm
def human_detection(rgb_img,thermal_img):
	if(len(rgb_img.shape)==3):
		rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

	rgb_img=rgb_detection.reshape_rgb(rgb_img)

#Get candidates in RGB image
	rgb_extraction=rgb_detection.edge_extraction(rgb_img)
	rgb_candidate=rgb_detection.find_candidate(rgb_extraction)

#Get candidates in thermal image
	thermal_extraction=thermal_detection.edge_extraction(thermal_img)
	thermal_candidate=thermal_detection.find_candidate(thermal_extraction)

#Fuse the candidate
	rgb_fused_candidate,thermal_fused_candidate=fuse_candidate(rgb_candidate,thermal_candidate)

#Find the detected human
	rgb_human=[]
	thermal_human=[]
	thermal_clf = joblib.load(model_path_thermal)
	rgb_clf = joblib.load(model_path_rgb)

	for i in range(len(rgb_fused_candidate)):
		rgb_crop=rgb_img[rgb_fused_candidate[i][1]:rgb_fused_candidate[i][1]+rgb_fused_candidate[i][3],
		rgb_fused_candidate[i][0]:rgb_fused_candidate[i][0]+rgb_fused_candidate[i][2]]
		rgb_crop=cv2.resize(rgb_crop,(24,40))


		thermal_crop=thermal_img[thermal_fused_candidate[i][1]:thermal_fused_candidate[i][1]+thermal_fused_candidate[i][3],
		thermal_fused_candidate[i][0]:thermal_fused_candidate[i][0]+thermal_fused_candidate[i][2]]
		thermal_crop=cv2.resize(thermal_crop,(32,64))

		if rgb_detection.candidate_classification(rgb_crop,rgb_clf)==True and thermal_detection.candidate_classification(thermal_crop,thermal_clf)==True:
			rgb_human.append(rgb_fused_candidate[i])
			thermal_human.append(thermal_fused_candidate[i])


	return np.array(rgb_human),np.array(thermal_human)

if __name__ == "__main__":
	rgb_img = cv2.imread('C:/Users/zjcv2/Desktop/test_object_detector/rgb_4.bmp', cv2.IMREAD_COLOR)
	thermal_img = cv2.imread('C:/Users/zjcv2/Desktop/test_object_detector/ir_4.bmp', cv2.IMREAD_GRAYSCALE)
	rgb_human,thermal_human=human_detection(rgb_img,thermal_img)



