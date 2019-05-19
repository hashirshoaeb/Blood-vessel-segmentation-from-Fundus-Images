import cv2 as cv
import numpy as np
import math as m
import functions as f
import operationfunctions as of

# array1 = cv.imread("IM000001/IM000001--vessels.jpg", 0)
# ret, tarray = cv.threshold(array1, 120, 255, cv.THRESH_BINARY)

path = [["IM000001", "IM000001.JPG", "IM000001--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"],
        ["IM000004", "IM000004.JPG", "IM000004--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"],
        ["IM000023", "IM000023.JPG", "IM000023--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"],
        ["IM000024", "IM000024.JPG", "IM000024--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"],
        ["IM000135", "IM000135.JPG", "IM000135--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"],
        ["IM000136", "IM000136.JPG", "IM000136--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"],
        ["IM000167", "IM000167.JPG", "IM000167--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"],
        ["IM000168", "IM000168.JPG", "IM000168--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"],
        ["IM000189", "IM000189.JPG", "IM000189--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"],
        ["IM000209", "IM000209.JPG", "IM000209--vessels.jpg", "Morphology.jpg", "ConComp.jpg", "Adaptive.jpg", "Canny.jpg", "Gabor.jpg", "RegGrowing.jpg", "ColorMorphology.jpg", "ColorConComp.jpg", "ColorAdaptive.jpg", "ColorCanny.jpg", "ColorGabor.jpg", "ColorRegGrowing.jpg"]]




kernel = np.ones((6, 6), np.uint8)
for p in path:
    imageArray = cv.imread(p[0]+"/"+p[1], 0)
    vesselimage = cv.imread(p[0]+"/"+p[2],0)
    ret, vesselimage = cv.threshold(vesselimage, 245, 255, cv.THRESH_BINARY)

    # Morphology
    resMorph = of.morphology(imageArray, kernel)
    cv.imwrite(p[0]+"/"+p[3], resMorph)
    out = of.performancePrams(resMorph, vesselimage)
    cv.imwrite(p[0]+"/"+p[9], cv.applyColorMap(out, cv.COLORMAP_HSV))

    # Canny
    resCanny = of.canny(imageArray)
    cv.imwrite(p[0]+"/"+p[6], resCanny)
    out = of.performancePrams(resCanny, vesselimage)
    cv.imwrite(p[0]+"/"+p[12], cv.applyColorMap(out, cv.COLORMAP_HSV))

    # Adaptive thresholding
    resAdaptive = cv.adaptiveThreshold(imageArray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,51,4)
    cv.imwrite(p[0] + "/" + p[5], resAdaptive)
    out = of.performancePrams(resAdaptive, vesselimage)
    cv.imwrite(p[0] + "/" + p[11], cv.applyColorMap(out, cv.COLORMAP_HSV))










# img1 = cv.imread("IM000001/IM000001.JPG",0)
# img1 = cv.blur(img1, (5,5))
# img1 = cv.Laplacian(img1,)
# img1 = cv.Canny(img1,30,30, L2gradient = False)
# kernel = np.ones((3, 3), np.uint8)
# for i in range(0, 1):
#     array = cv.morphologyEx(array, cv.MORPH_DILATE, kernel)
#     array = cv.morphologyEx(array, cv.MORPH_ERODE, kernel)
# img1 = cv.blur(img1, (5,5))
# cv.imwrite("2CannyEdgeBlur.jpg", img1)




# /////   ---- Extra ----   /////
# kernel = cv.getStructuringElement(cv.MORPH_ERODE,(17,17))
# print(kernel)
# wow = array[:]
# for i in range(0,10):
#     wow = cv.erode(wow,kernel)
#     wow = cv.dilate(wow,kernel)
# # wow = cv.dilate(cv.erode(cv.erode(cv.dilate(cv.erode(cv.erode(cv.dilate(cv.erode(cv.erode(array,kernel),kernel),kernel),kernel),kernel),kernel),kernel),kernel),kernel)
# # wow = cv.erode(array,kernel)
# array = wow - array
# row , col = np.shape(array)
# for i in range(0,row):
#     for j in range(0,col):
#         if array[i][j] < 20:
#             array[i][j] = 255
# array = cv.erode(array,kernel)
# array = cv.dilate(array,kernel)
# cv.imwrite("hahaha.JPG",array)

# parray = f.paddingAdder(array, 1)
# carray = f.medianconvolution(kernel,parray)
# print(np.max(array), np.min(array))
# row, col = np.shape(array)
# minn, maxx = np.min(array),np.max(array)
#
# for i in range(0,row):
#     for j in range(0,col):
#         array[i][j] = f.normalization(array[i][j],minn, maxx)
# kernel = np.ones((7,7), np.uint8)
# >>> cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
# array([[0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0],
#        [1, 1, 1, 1, 1],
#        [0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0]], dtype=uint8)
# kernel = np.ones((7, 7), np.uint8)
#
# gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)
#
# img = 255 - gradient
# print("min and max in img", np.min(img), np.max(img))
# closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
# closing = cv.morphologyEx(closing, cv.MORPH_CLOSE, kernel)
# dilation = cv.dilate(closing, kernel, iterations=1)
#
# ret, dilation = cv.threshold(dilation, 245, 255, cv.THRESH_BINARY)
#
# cv.imwrite("hth.JPG", dilation)
# arr = cv.imread("IM000001/IM000001.JPG",0)
# print(arr)
# r = [[255, 0, 255],
#      [255, 0, 0],
#      [0, 255, 0]]
# g = [[255, 0, 255],
#      [255, 0, 255],
#      [255, 255, 0]]
# b = [[0, 255, 0],
#      [0, 255, 0],
#      [0, 255, 0]]
#
# warr = np.dstack((arr,255-arr,255-arr))
# cv.imwrite("nice.jpg", cv.applyColorMap(warr,cv.COLORMAP_HSV))


# /////     ---- LINKS ----    //////
# https://datascience.stackexchange.com/questions/30589/how-to-interpret-fpr-and-tpr-in-roc-curve
# https://www.sciencedirect.com/science/article/pii/S2210832718301546
# https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/MARBLE/low/edges/canny.htm
# https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6
# https://www.numpy.org/devdocs/reference/generated/numpy.dstack.html#numpy.dstack




# SAMEEEEEEN
# def gabor_fn(sigma, theta, Lambda, psi, gamma):
#     sigma_x = sigma
#     sigma_y = float(sigma) / gamma
#
#     # Bounding box
#     nstds = 3 # Number of standard deviation sigma
#     xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
#     xmax = np.ceil(max(1, xmax))
#     ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
#     ymax = np.ceil(max(1, ymax))
#     xmin = -xmax
#     ymin = -ymax
#     (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
#
#     # Rotation
#     x_theta = x * np.cos(theta) + y * np.sin(theta)
#     y_theta = -x * np.sin(theta) + y * np.cos(theta)
#
#     gb = np.exp(-.5 * (x_theta * 2 / sigma_x * 2 + y_theta * 2 / sigma_y * 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
#     return gb