import cv2 as cv
import numpy as np
import math as m
import functions as f


def morphology(image, kernel):
    array = image[:]
    print("Start Morphological Operation")
    array = cv.morphologyEx(array, cv.MORPH_GRADIENT, kernel)
    for i in range(0, 3):
        array = cv.morphologyEx(array, cv.MORPH_DILATE, kernel)
        array = cv.morphologyEx(array, cv.MORPH_ERODE, kernel)

    print("normalization max min", np.max(array), np.min(array))

    row, col = np.shape(array)
    minn, maxx = np.min(array), np.max(array)
    for i in range(0, row):
        for j in range(0, col):
            array[i][j] = f.normalization(array[i][j], minn, maxx)
            array[i][j] = f.normalization(array[i][j], 255, 0)

    # ret, tarray = cv.threshold(array,250,255,cv.THRESH_BINARY)
    # todo: Adaptive thresholding
    return array
