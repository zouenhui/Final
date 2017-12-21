# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:21:00 2017

@author: 
"""

import cv2
import dlib 
import numpy as np 
from util import *


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
global frameV1
global v2Frame
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
video1=cv2.VideoCapture('MarquesBrownlee.mp4')
video2=cv2.VideoCapture('MrRobot.mp4')
tf,frameV1=video1.read()
tf,frameV2=video2.read()
oldFrameV1Gray=cv2.cvtColor(frameV1, cv2.COLOR_BGR2GRAY)
oldFrameV2Gray=cv2.cvtColor(frameV2, cv2.COLOR_BGR2GRAY)
frameV1=cv2.imread('Adnan.jpg')
[h1,w1,c1]=frameV1.shape
[h2,w2,c2]=frameV2.shape
if h1>h2:
    height=h2
else:
    height=h1
if w1>w2:
    width=w2
else:
    width=w1
frameV1=cv2.resize(frameV1,(width,height))
frameV2=cv2.resize(frameV2,(width,height))
fps1=video1.get(cv2.CAP_PROP_FPS)
fps2=video1.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vOut1=cv2.VideoWriter('output5.mp4',cv2.VideoWriter_fourcc(*'MP4V'),fps1,(width,height))
vOut2=cv2.VideoWriter('output2.mp4',cv2.VideoWriter_fourcc(*'MP4V'),fps2,(width,height))

#feature detection and extraction
DETECTOR=dlib.get_frontal_face_detector()
PREDICTOR=dlib.shape_predictor(PREDICTOR_PATH)
detectedPts1=DETECTOR(frameV1,1)
detectedPts2=DETECTOR(frameV2,2)
pointsF1=[]
pointsF2=[]
frameNum=0
for k,d in enumerate(detectedPts2):
    shape=PREDICTOR(frameV2,d)
    for i in range(68):
        pointsF2.append((shape.part(i).x,shape.part(i).y))
for k,d in enumerate(detectedPts1):
    shape=PREDICTOR(frameV1,d)
    for i in range(68):
        pointsF1.append((shape.part(i).x,shape.part(i).y))
old2FeaturePoints=np.array(pointsF2)
old2FeaturePoints=np.float32(old2FeaturePoints.reshape(-1,1,2))
while True:
    tf2,v2Frame=video2.read()
    frameNum=frameNum+1
    print frameNum
    if tf2==True:
        v2FrameGray=cv2.cvtColor(v2Frame, cv2.COLOR_BGR2GRAY)
        v2Frame=cv2.resize(v2Frame,(width,height))
        warped_img1=np.copy(v2Frame)
        detections=DETECTOR(v2Frame,2)
        for k,d in enumerate(detections):
            shape=PREDICTOR(v2Frame,d)
            points2=[]
            for i in range(68):
                points2.append((shape.part(i).x,shape.part(i).y))
    #        convex hull
            p1, st, err = cv2.calcOpticalFlowPyrLK(oldFrameV2Gray, v2FrameGray,old2FeaturePoints , None, **lk_params)
            estimated=p1.reshape(-1,2)
            detected=np.array(points2)
            if len(detected)==0:
                total=estimated
            else:
                total=estimated*0.5+detected*0.5
            hull_1=[]
            hull_2=[]
            hull_ind=cv2.convexHull(total.astype(np.int), returnPoints = False)
            old2FeaturePoints=np.float32(total.reshape(-1,1,2))
            oldFrameV2Gray=v2FrameGray
            for i in xrange(0,len(hull_ind)):
                
                hull_1.append(pointsF1[hull_ind[i,0]])
                hull_2.append(points2[hull_ind[i,0]])
            img2_size=v2Frame.shape
            rect=(0,0,img2_size[1],img2_size[0])
            d_tri=calculateDelaunayTriangles(rect, points2)
            
            if len(d_tri)==0:
                quit()
            for i in xrange(0,len(d_tri)):
                tri_1=[]
                tri_2=[]
                
                for j in xrange(0,3):
                    tri_1.append(pointsF1[d_tri[i][j]])
                    tri_2.append(points2[d_tri[i][j]])
                warpTriangle(frameV1,warped_img1,tri_1,tri_2)
            hull8U=[]
            for i in xrange(0,len(hull_2)):
                hull8U.append((hull_2[i][0],hull_2[i][1]))
            mask=np.ones(v2Frame.shape,dtype=v2Frame.dtype)
            cv2.fillConvexPoly(mask,np.int32(hull8U),(255,255,255))
            r=cv2.boundingRect(np.float32(hull_2))
            center=((r[0]+int(r[2]/2),r[1]+int(r[3]/2)))
            output=cv2.seamlessClone(np.uint8(warped_img1),v2Frame,mask,center,cv2.NORMAL_CLONE)
            vFrame2=output
        vOut1.write(output.astype('uint8'))
#        cv2.imshow("current frame",output)
#        cv2.waitKey(0)
#        cv2.destroyWindow("current frame")
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    else:
        break
vOut1.release()
video2.release()
cv2.imshow("Image0", frameV1)
cv2.waitKey(0)
cv2.destroyWindow("Image0")
