import cv2 as cv
import numpy as np
import math as m
import functions as f



def morphology(image, kernel):
    array = image[:]
    print("Start Morphological Operation")
    array = cv.morphologyEx(array, cv.MORPH_GRADIENT, kernel)
    for i in range(0, 2):
        array = cv.morphologyEx(array, cv.MORPH_DILATE, kernel)
        array = cv.morphologyEx(array, cv.MORPH_ERODE, kernel)
    kernel1 = np.ones((3, 3), np.uint8)
    array = cv.morphologyEx(array, cv.MORPH_ERODE, kernel1)

    # print("normalization max min", np.max(array), np.min(array))
    # row, col = np.shape(array)
    # minn, maxx = np.min(array), np.max(array)
    # for i in range(0, row):
    #     for j in range(0, col):
    #         array[i][j] = f.normalization(array[i][j], minn, maxx)
    #         array[i][j] = f.normalization(array[i][j], 255, 0)
    array = 255 - array
    ret, tarray = cv.threshold(array,242,255,cv.THRESH_BINARY)
    # todo: Adaptive thresholding
    return tarray



# yourimage     givenImage      label       number
# 0             0               TN          2*0 + 0 = 0
# 0             1               FN          2*0 + 1 = 1
# 1             0               FP          2*1 + 0 = 2
# 1             1               TP          2*1 + 1 = 3
# where 0 is background 1 is foreground

# yourimage     givenImage      label       number
# 255           255             TN          2*255 + 255 = 765 = 253 + 3 = 0(as ans will not exceed 255)
# 255           0               FN          2*255 + 0 = 510 = 254 + 3 = 1
# 0             255             FP          2*0 + 255 = 255 = 255 + 3 = 2
# 0             0               TP          2*0 + 0 = 0 = 0 + 3 = 3
# where 255 is background 0 is foreground
def performancePrams(yourImage, givenImage):
    output = (((yourImage * 2) + givenImage) + 3)*63
    unique_elements, counts = np.unique(output, return_counts=True)
    print(unique_elements, counts)
    TP, TN, FN, FP = counts
    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    diceCoff = 2*TP / (FN + (2*TP) + FP)
    print([TPR, FPR, diceCoff])
    return output



# Parameters
# image	8-bit input image.
# edges	output edge map; single channels 8-bit image, which has the same size as image .
# threshold1	first threshold for the hysteresis procedure.
# threshold2	second threshold for the hysteresis procedure.
# apertureSize	aperture size for the Sobel operator.
# L2gradient	a flag, indicating whether a more accurate L2 norm =(dI/dx)2+(dI/dy)2√ should be used to calculate
# the image gradient magnitude ( L2gradient=true ),
# or whether the default L1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
def canny(image):
    array = image[:]
    print("Start Canny Operation")
    # array = cv.blur(array, (5, 5))
    # array = cv.Canny(array, 1, 20, L2gradient=False)
    array = cv.Canny(array,30,30, L2gradient = False)
    array = 255 - array
    return array