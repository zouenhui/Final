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
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

