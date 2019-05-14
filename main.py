import cv2 as cv
import numpy as np
import math as m
import functions as f

array = cv.imread("IM000001/IM000001.JPG",0)
cv.imwrite("haha.JPG",array)
array1 = cv.imread("IM000001/IM000001--vessels.jpg")
# kernel = np.ones((7,7), np.uint8)
# >>> cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
# array([[0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0],
#        [1, 1, 1, 1, 1],
#        [0, 0, 1, 0, 0],
#        [0, 0, 1, 0, 0]], dtype=uint8)


# Using Median Filter
filter = np.array([[1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]])
parray = f.paddingAdder(array, 1)
carray = f.medianconvolution(filter,parray)

cv.imwrite("hahaha.JPG",carray)



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