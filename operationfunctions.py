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

    ret, tarray = cv.threshold(array,220,255,cv.THRESH_BINARY)
    # todo: Adaptive thresholding
    return tarray



# yourimage     givenImage      label       number
# 0             0               TN          2*0 + 0 = 0
# 0             1               FN          2*0 + 1 = 1
# 1             0               FP          2*1 + 0 = 2
# 1             1               TP          2*1 + 1 = 3
# where 0 is background 1 is foreground

# yourimage     givenImage      label       number
# 255           255             TN          2*255 + 255 = 765 = 253 (as ans will not exceed 255)
# 255           0               FN          2*255 + 0 = 510 = 254
# 0             255             FP          2*0 + 255 = 255 = 255
# 0             0               TP          2*0 + 0 = 0 = 0
# where 255 is background 0 is foreground
def performancePrams(yourImage, givenImage):
    output = (yourImage * 2) + givenImage
    unique_elements, counts = np.unique(output, return_counts=True)
    print(unique_elements, counts)
    TP, TN, FN, FP = counts
    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    print([TPR, FPR])
    return [TPR, FPR]