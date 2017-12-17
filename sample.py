# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:21:19 2017

@author: zoue
"""

#import cv2
#import dlib
#import numpy
#import sys
#
#PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
#SCALE_FACTOR = 1
#FEATHER_AMOUNT = 11
#
#FACE_POINTS = list(range(17, 68))
#MOUTH_POINTS = list(range(48, 61))
#RIGHT_BROW_POINTS = list(range(17, 22))
#LEFT_BROW_POINTS = list(range(22, 27))
#RIGHT_EYE_POINTS = list(range(36, 42))
#LEFT_EYE_POINTS = list(range(42, 48))
#NOSE_POINTS = list(range(27, 35))
#JAW_POINTS = list(range(0, 17))
## Points used to line up the images.
#ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
#RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + JAW_POINTS)
## Points from the second image to overlay on the first. The convex hull of each
## element will be overlaid.
#OVERLAY_POINTS = [LEFT_EYE_POINTS + RIGHT_EYE_POINTS + NOSE_POINTS + MOUTH_POINTS]
#
## Amount of blur to use during colour correction, as a fraction of the
## pupillary distance.
#COLOUR_CORRECT_BLUR_FRAC = 0.6
#
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor(PREDICTOR_PATH)
#
#class TooManyFaces(Exception):
#    pass
#
#class NoFaces(Exception):
#    pass
#
#def get_landmarks(im):
#    rects = cascade.detectMultiScale(im, 1.3,5)
#    #if len(rects) > 1:
#    #    raise TooManyFaces
#    if len(rects) == 0:
#        raise NoFaces
#    print len(rects)
#    x,y,w,h =rects[0]
#    rect=dlib.rectangle(x,y,x+w,y+h)
#    return numpy.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
#
#def get_landmarks2(im,rects):
#
#    if len(rects) > 2:
#        raise TooManyFaces
#    if len(rects) == 0:
#        raise NoFaces
#    landmarks=[]
#    landmarks.append(numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]))
#    landmarks.append(numpy.matrix([[p.x, p.y] for p in predictor(im, rects[1]).parts()]))
#
#    return landmarks
#
#
#def annotate_landmarks(im, landmarks):
#    im = im.copy()
#    for idx, point in enumerate(landmarks):
#        pos = (point[0, 0], point[0, 1])
#        cv2.putText(im, str(idx), pos,
#                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
#                    fontScale=0.4,
#                    color=(0, 0, 255))
#        cv2.circle(im, pos, 3, color=(0, 255, 255))
#    return im
#
#def draw_convex_hull(im, points, color):
#    points = cv2.convexHull(points)
#    cv2.fillConvexPoly(im, points, color=color)
#
#def get_face_mask(im, landmarks):
#    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
#
#    for group in OVERLAY_POINTS:
#        draw_convex_hull(im,
#                         landmarks[group],
#                         color=1)
#
#    im = numpy.array([im, im, im]).transpose((1, 2, 0))
#
#    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
#    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
#
#    return im
#    
#def transformation_from_points(points1, points2):
#    """
#    Return an affine transformation [s * R | T] such that:
#        sum ||s*R*p1,i + T - p2,i||^2
#    is minimized.
#    """
#    # Solve the procrustes problem by subtracting centroids, scaling by the
#    # standard deviation, and then using the SVD to calculate the rotation. See
#    # the following for more details:
#    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
#
#    points1 = points1.astype(numpy.float64)
#    points2 = points2.astype(numpy.float64)
#
#    c1 = numpy.mean(points1, axis=0)
#    c2 = numpy.mean(points2, axis=0)
#    points1 -= c1
#    points2 -= c2
#
#    s1 = numpy.std(points1)
#    s2 = numpy.std(points2)
#    points1 /= s1
#    points2 /= s2
#
#    U, S, Vt = numpy.linalg.svd(points1.T * points2)
#
#    # The R we seek is in fact the transpose of the one given by U * Vt. This
#    # is because the above formulation assumes the matrix goes on the right
#    # (with row vectors) where as our solution requires the matrix to be on the
#    # left (with column vectors).
#    R = (U * Vt).T
#
#    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
#                                       c2.T - (s2 / s1) * R * c1.T)),
#                         numpy.matrix([0., 0., 1.])])
#
#def read_im_and_landmarks(fname):
#
#    im = cv2.imread(fname)
#    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
#                         im.shape[0] * SCALE_FACTOR))
#    rects = detector(im, 1)
#    s = get_landmarks2(im,rects)
#
#    return im, s
#
#def warp_im(im, M, dshape):
#    output_im = numpy.zeros(dshape, dtype=im.dtype)
#    cv2.warpAffine(im,
#                   M[:2],
#                   (dshape[1], dshape[0]),
#                   dst=output_im,
#                   borderMode=cv2.BORDER_TRANSPARENT,
#                   flags=cv2.WARP_INVERSE_MAP)
#    return output_im
#
#def correct_colours(im1, im2, landmarks1):
#    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
#                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
#                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
#    blur_amount = int(blur_amount)
#    if blur_amount % 2 == 0:
#        blur_amount += 1
#    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
#    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
#
#    # Avoid divide-by-zero errors.
#    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
#
#    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
#                                                im2_blur.astype(numpy.float64))
#
#
#im, lndmL = read_im_and_landmarks('together.jpg')
#l1=lndmL[0]
#l2=lndmL[1]
#res=im.copy()
#M1 = transformation_from_points(l1[ALIGN_POINTS],
#                               l2[ALIGN_POINTS])
#M2 = transformation_from_points(l2[ALIGN_POINTS],
#                               l1[ALIGN_POINTS])
#
#mask1 = get_face_mask(im, l1)
#mask2 = get_face_mask(im, l2)
#
#
#warped_mask = warp_im(mask2, M1, im.shape)
#combined_mask = numpy.max([get_face_mask(im, l1), warped_mask],axis=0)
#warped_im2 = warp_im(im, M1, im.shape)
#warped_corrected_im2 = correct_colours(im, warped_im2, l1)
#
#res = im * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
#
#warped_mask = warp_im(mask1, M2, im.shape)
#combined_mask = numpy.max([get_face_mask(im, l2), warped_mask],axis=0)
#warped_im2 = warp_im(im, M2, im.shape)
#warped_corrected_im2 = correct_colours(im, warped_im2, l2)
#res = res * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
#
#cv2.imwrite('result.jpg',res)

import cv2
import numpy as np
import random
 
# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
 
# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color, -1, cv2.LINE_AA, 0 )
 
 
# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
 
 
# Draw voronoi diagram
def draw_voronoi(img, subdiv) :
 
    ( facets, centers) = subdiv.getVoronoiFacetList([])
 
    for i in xrange(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
         
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
 
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), -1, cv2.LINE_AA, 0)
 
 
if __name__ == '__main__':
 
    # Define window names
    win_delaunay = "Delaunay Triangulation"
    win_voronoi = "Voronoi Diagram"
 
    # Turn on animation while drawing triangles
    animate = False
     
    # Define colors for drawing.
    delaunay_color = (255,255,255)
    points_color = (0, 0, 255)
 
    # Read in the image.
    img = cv2.imread("MLBSmall.png");
     
    # Keep a copy around
    img_orig = img.copy();
     
    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])
     
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);
 
    # Create an array of points.
    points = [];
     
    # Read in the points from a text file
    with open("Points.txt") as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
 
    # Insert points into subdiv
    for p in points :
        subdiv.insert(p)
         
        # Show animation
        if animate :
            img_copy = img_orig.copy()
            # Draw delaunay triangles
            draw_delaunay( img_copy, subdiv, (255, 255, 255) );
            cv2.imshow(win_delaunay, img_copy)
            cv2.waitKey(100)
 
    # Draw delaunay triangles
    draw_delaunay( img, subdiv, (255, 255, 255) );
 
    # Draw points
    for p in points :
        draw_point(img, p, (0,0,255))
 
    # Allocate space for Voronoi Diagram
    img_voronoi = np.zeros(img.shape, dtype = img.dtype)
 
    # Draw Voronoi diagram
    draw_voronoi(img_voronoi,subdiv)
 
    # Show results
    cv2.imshow(win_delaunay,img)
    cv2.imshow(win_voronoi,img_voronoi)
    cv2.waitKey(0)
