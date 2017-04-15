import pylab
import numpy as np
import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt
import colorsys
import datetime
import time
from datetime import timedelta
import Detection
matplotlib.use('Agg')


def getOriginaltime(RGB_videoname, th_videoname):
    start = '_'
    end = '_'
    s = RGB_videoname
    answer = (s.split(start))[1].split(end)[0]
    hour = int(answer[0:2])
    mini = int(answer[2:4])
    sec = int(answer[4:6])
    time_ori = datetime.datetime(2017, 4, 26, hour, mini, sec)
    return time_ori

def getTimestr(time_):
    if time_.microsecond == 0:
        string_millisec = str(time_.microsecond) + "00000"
    else:
        string_millisec = str(time_.microsecond)
    time_str = str(time_.hour) + ":" + str(time_.minute) + ":" + str(time_.second) + "." + string_millisec
    return time_str

def readVideo(RGB_videoname, th_videoname, outputFolder):
    time_ori = getOriginaltime(RGB_videoname, th_videoname)
    # thermal processing
    frame_IR = imageio.get_reader(th_videoname, 'ffmpeg')
    IR_nframes = frame_IR._meta['nframes']
    # RGB processing
    frame_RGB = imageio.get_reader(RGB_videoname, 'ffmpeg')
    RGB_nframes = frame_RGB._meta['nframes']
    for th_num in range(1195, 1197):
        sec_add = th_num / frame_IR._meta['fps']
        #sec_add = 13.265
        RGB_num = int((float(th_num) / IR_nframes) * RGB_nframes)
        image_IR = frame_IR.get_data(th_num) #480*640*3
        image_RGB = frame_RGB.get_data(RGB_num)#1080*1920*3
        # plt.imshow(image_RGB)
        # plt.show()
        img_temp = image_RGB.copy()
        bounding_boxes_th, bounding_boxes_RGB, human = Detection.Integrate(image_IR, image_RGB)
        bounding_boxes_B, object = Detection.hsv_thresholding(image_RGB)
        time_final = time_ori + datetime.timedelta(hours=0, minutes=0, seconds=sec_add)
        time_final_string = getTimestr(time_final)

        if human is True:
            ########################################################
            #############################thermal#####################################
            pixel_T = np.zeros((bounding_boxes_th.shape[0],2)) #n*2
            for i in range(bounding_boxes_th.shape[0]):
                # img_temp = cv2.rectangle(img_temp, (bounding_boxes_th[i][0], bounding_boxes_th[i][1]),
                #                          (bounding_boxes_th[i][0] + bounding_boxes_th[i][2],
                #                           bounding_boxes_th[i][1] + bounding_boxes_th[i][3]),
                #                          (255, 0, 0), 1)
                # plt.imshow(img_temp)
                # plt.show()
                pixel_T[i, :] = np.array((bounding_boxes_th[i, 2] / 2 + bounding_boxes_th[i, 0] , bounding_boxes_th[i, 3] + bounding_boxes_th[i, 1]))
                text_file_th.write("%s %s %s %s" % (time_final_string, str(pixel_T[i, 0]), str(pixel_T[i, 1]), "H\n" ))#thermal folder + txt'H'
            #plt.savefig()
            ########################################################
            #################################RGB#####################################
            pixel_R = np.zeros((bounding_boxes_RGB.shape[0], 2))  # n*2
            for i in range(bounding_boxes_RGB.shape[0]):
                bounding_boxes_RGB[i][0] += 106
                bounding_boxes_RGB *= 1080 / 480
                pixel_R[i, :] = np.array((bounding_boxes_RGB[i, 2] / 2 + bounding_boxes_RGB[i, 0] , bounding_boxes_RGB[i, 3] + bounding_boxes_RGB[i, 1]))
                text_file_RGB.write("%s %s %s %s" % (time_final_string, str(pixel_R[i, 0]), str(pixel_R[i, 1]), "H\n" ))#RGB folder + txt'H'
                # img_temp = cv2.rectangle(img_temp, (bounding_boxes_RGB[i][0], bounding_boxes_RGB[i][1]),
                #                          (bounding_boxes_RGB[i][0] + bounding_boxes_RGB[i][2],
                #                           bounding_boxes_RGB[i][1] + bounding_boxes_RGB[i][3]),
                #                          (255, 0, 0), 1)
                # plt.imshow(img_temp)
                # plt.show()
            #plt.savefig()
                
        if object is True:
            pixel_R_B = np.zeros((bounding_boxes_B.shape[0], 2)) # n*2
            for i in range(bounding_boxes_B.shape[0]):
                pixel_R_B[i, :] = np.array((bounding_boxes_B[i, 2] / 2 + bounding_boxes_B[i, 0] , bounding_boxes_B[i, 3] / 2 + bounding_boxes_B[i, 1]))
                text_file_th.write("%s %s %s %s" % (time_final_string, str(pixel_R_B[i, 0]), str(pixel_R_B[i, 1]), "H\n" ))#RGB folder + txt'B'

            # plt.savefig()


if __name__ == '__main__':

    # plt.imshow(image_first)
    # plt.show()
    text_file_th = open("./Output/thermal/Output.txt", "w")
    text_file_RGB = open("./Output/RGB/Output.txt", "w")
    readVideo('20170404_112346_VIS.mov', '20170404_112346_IR.mov', './Output')
    text_file_th.close()
    text_file_RGB.close()