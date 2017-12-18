# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:30:05 2017

@author: dharr
"""

import cv2 
import dlib 
import numpy as np 
from util import *
import matplotlib.pyplot as plt

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
video1=cv2.VideoCapture('MarquesBrownlee.mp4')
tf,frameV1=video1.read()
[h1,w1,c1]=frameV1.shape
frameV1=cv2.resize(frameV1,(w1,h1))

#feature detection and extraction
DETECTOR = dlib.get_frontal_face_detector()
PREDICTOR=dlib.shape_predictor(PREDICTOR_PATH)
detectedPts1 = DETECTOR(frameV1,1)
pointsF1=[]
for k,d in enumerate(detectedPts1):
    shape=PREDICTOR(frameV1,d)
    for i in range(68):
        pointsF1.append((shape.part(i).x,shape.part(i).y))

points_arr = np.asarray(pointsF1,dtype=None,order=None)
img_size=frameV1.shape
offset = 10
#rect=(0,0,max(points_arr[:,1]),max(points_arr[:,0])+offset)
rect=(0,0,img_size[1],img_size[0])
tri = calculateDelaunayTriangles(tuple(rect), pointsF1)

plt.axis("off")
plt.imshow(cv2.cvtColor(frameV1, cv2.COLOR_BGR2RGB))
#plt.scatter(points_arr[:,0],points_arr[:,1],marker = ".", c = "y")
plt.show()





