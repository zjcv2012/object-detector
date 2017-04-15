import cv2
import numpy as np

#Perform 
from sklearn.externals import joblib
from skimage.feature import hog
from config import *


def edge_extraction(img):
	margin=5

	img_blur = cv2.GaussianBlur(img,(5,5),0)
	#Extract Edge
	canny = cv2.Canny(img_blur, 100, 100)
	rows, cols = img.shape
	canny[:margin,:]=0
	canny[rows-margin:,:]=0
	canny[:,:margin]=0
	canny[:,cols-margin:]=0

	#Morphological Transformations
	kernel = np.ones((5,5), np.uint8)
	#img_erosion = cv2.erode(img, kernel, iterations=1)
	img_dilation = cv2.dilate(canny, kernel, iterations=1)

	#Fill the holes
	img2,contour,hier = cv2.findContours(img_dilation,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contour:
	    cv2.drawContours(img_dilation,[cnt],0,255,-1)
	closing = cv2.morphologyEx(img_dilation, cv2.MORPH_CLOSE, kernel)

	return closing


def find_candidate(img):
	num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)
	human_candidate=[]
	bound_low=25
	bound_high=60
	ratio=0.6

	for i in range(stats.shape[0]):
		if stats[i,2]<stats[i,3] and bound_low<=stats[i,2]<=bound_high:

			if  stats[i,2]/ratio<=stats[i,3]:
				img_temp=np.round(stats[i,3]*ratio)
				stats[i,0]=max(0,stats[i,0]-(img_temp-stats[i,2])/2)
				stats[i,2]=min(img_temp,img.shape[1]-stats[i,0]-1)
			else:
				img_temp=np.round(stats[i,2]/ratio)
				stats[i,1]=max(0,stats[i,1]-(img_temp-stats[i,3])/2)
				stats[i,3]=min(img_temp,img.shape[0]-stats[i,1]-1)


			human_candidate.append(stats[i,:4])

	return human_candidate


def candidate_classification(img,clf):
	fd = hog(img, orientations, pixels_per_cell, cells_per_block, visualize, normalize)
	temp = np.array(fd).reshape((1, -1))
	pred = clf.predict(temp)

	return pred

#This function is for transforming rgb image so that it has the same resolution as the thermal image
#From 1920*1080 -> 640*480
def reshape_rgb(img):
	img_temp=cv2.resize(img, (0,0), fx=float(4)/9, fy=float(4)/9)
	img_reshape=img_temp[:,106:746]
	return img_reshape 




if __name__ == "__main__":
	img = cv2.imread('C:/Users/zjcv2/Desktop/test_object_detector/rgb_2.bmp', cv2.IMREAD_GRAYSCALE)
	img=reshape_rgb(img)
	# Load the classifier
	clf = joblib.load(model_path_rgb)

	img_extraction=edge_extraction(img)
	candidate=find_candidate(img_extraction)
	img_temp=img.copy()

	for i in range(len(candidate)):
		# img_temp=cv2.rectangle(img_temp,(candidate[i][0],candidate[i][1]),(candidate[i][0]+candidate[i][2],candidate[i][1]+candidate[i][3]),
		# 	(0,255,0),1)
		img_crop=img[candidate[i][1]:candidate[i][1]+candidate[i][3],candidate[i][0]:candidate[i][0]+candidate[i][2]]
		img_crop=cv2.resize(img_crop,(24,40))
		if candidate_classification(img_crop,clf)==True:
			#print("true")
			img_temp=cv2.rectangle(img_temp,(candidate[i][0],candidate[i][1]),(candidate[i][0]+candidate[i][2],candidate[i][1]+candidate[i][3]),
			(0,0,0),1)
		else:
			#print("false")
			img_temp=cv2.rectangle(img_temp,(candidate[i][0],candidate[i][1]),(candidate[i][0]+candidate[i][2],candidate[i][1]+candidate[i][3]),
			(255,255,255),1)

	cv2.imshow('Extraction',img_extraction)
	cv2.imshow('Closing', img_temp)
	cv2.waitKey(0)
	cv2.destroyAllWindows()