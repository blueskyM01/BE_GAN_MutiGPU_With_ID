import cv2
import numpy as np
import os
from utils import *
import tensorflow as tf

# data_dir = '/media/yang/F/DataSet/Face'
# data_set_name = 'CASIA-WebFace'
# label_dir = '/media/yang/F/DataSet/Face/Label'
# label_name = 'pair_FGLFW.txt'
#
# aaaa = m4_face_label_maker(os.path.join(data_dir,data_set_name),'/media/yang/F/1.txt')
#
# ad = np.loadtxt('/media/yang/F/1.txt',dtype=str)
# print(ad.shape)





dad = cv2.imread('/media/yang/F/DataSet/Face/CASIA-WebFace/0000045/002.jpg')
# cv2.imshow('ll',dad)
# cv2.waitKey(0)
print(dad.shape)