# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:25:22 2017

@author: dharr
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


#Sample code for CLAHE:        # Contrast Limited Adaptive Histogram Equalization

lab = cv2.cvtColor(cv2.imread('img.png'), cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
clahe_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
plt.figure(1)
plt.axis("off")
plt.imshow(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB))
plt.figure(2)
plt.axis("off")
plt.imshow(cv2.cvtColor(cv2.imread('img.png'), cv2.COLOR_BGR2RGB))
plt.show()