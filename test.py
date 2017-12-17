# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 15:30:05 2017

@author: dharr
"""

import numpy as np
import cv2 


img = cv2.imread('pre.jpg',0)
equ = cv2.equalizeHist(img)
equ2 = cv2.equalizeHist(equ)
res = np.hstack((img,equ)) #stacking images side-by-side
res2 = np.hstack((equ,equ2))
cv2.imshow('Image',res)
cv2.imshow('Image 2',res2)