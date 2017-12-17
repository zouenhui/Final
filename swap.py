# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 16:21:00 2017
@author: zoue
"""

import cv2
import dlib 
import numpy as np 
from util import *


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
#(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
global frameV1
global v2Frame
video1=cv2.VideoCapture('MarquesBrownlee.mp4')
video2=cv2.VideoCapture('TheMartian.mp4')
tf,frameV1=video1.read()
tf,frameV2=video2.read()
[h1,w1,c1]=frameV1.shape
[h2,w2,c2]=frameV2.shape
fps1=video1.get(cv2.CAP_PROP_FPS)
fps2=video1.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vOut1=cv2.VideoWriter('easy1_motioncomp.avi',fourcc, fps1, (w1, h1))
vOut2=cv2.VideoWriter('easy1_motioncomp.avi',fourcc, fps2, (w1, h1))

#feature detection and extraction
DETECTOR=dlib.get_frontal_face_detector()
PREDICTOR=dlib.shape_predictor(PREDICTOR_PATH)
detectedPts1=DETECTOR(frameV1,1)
detectedPts2=DETECTOR(frameV2,1)
pointsF1=[]
pointsF2=[]
for k,d in enumerate(detectedPts1):
    shape=PREDICTOR(frameV1,d)
    for i in range(68):
        pointsF1.append((shape.part(i).x,shape.part(i).y))
while True:
    tf2,v2Frame=video2.read()
    if tf2==True:
        print tf2
        warped_img1=np.copy(v2Frame)
        detections=DETECTOR(v2Frame,2)
        for k,d in enumerate(detections):
            shape=PREDICTOR(v2Frame,d)
            points2=[]
            for i in range(68):
                points2.append((shape.part(i).x,shape.part(i).y))
    #        convex hull
            hull_1=[]
            hull_2=[]
            hull_ind=cv2.convexHull(np.array(points2), returnPoints = False)
            
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
            output=cv2.seamlessClone(np.uint8(warped_img1),v2Frame,mask,center,cv2.MIXED_CLONE)
            vFrame2=output
        cv2.imshow("Face swapped" ,output)
        vOut1.write(output)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    else:
        break
cv2.imshow("Image0", frameV1)
cv2.waitKey(0)
cv2.destroyWindow("Image0")