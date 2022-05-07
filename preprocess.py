# IMAGE PREPROCESSING FUNCTION (https://stackoverflow.com/questions/29313667/how-do-i-remove-the-background-from-this-kind-of-image)
import cv2

# GLOBALS

# Data Directory
import numpy as np
from matplotlib import pyplot as plt

data_dir = 'DatasetTest1/Segmented/Train'

# Image Processing Parameters
BLUR = 21
CANNY_THRESH_1 = 0
CANNY_THRESH_2 = 500
MASK_DILATE_ITER = 15
MASK_ERODE_ITER = 15
MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format


def preprocess(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], 255)

    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)

    # -- Create final image ---------------------------------------------------------------
    img[mask <= 100] = 0
    return img


filename = data_dir + '/GoodSeed/Segmented_1front_S3.jpg'
img = preprocess(filename)

plt.figure()
plt.imshow(img)
plt.show()
