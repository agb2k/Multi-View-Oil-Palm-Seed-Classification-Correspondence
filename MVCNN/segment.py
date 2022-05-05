from os import getcwd

import cv2
import os
from PIL import Image

trainSet_path = '../DatasetTest1/Train'
testSet_path = '../DatasetTest1/Test'

list_good_train_seeds = sorted(os.listdir(trainSet_path + '/GoodSeed'))
list_bad_train_seeds = sorted(os.listdir(trainSet_path + '/BadSeed'))

list_good_test_seeds = sorted(os.listdir(testSet_path + '/GoodSeed'))
list_bad_test_seeds = sorted(os.listdir(testSet_path + '/BadSeed'))


def seed_segment(image_path, filename):
    # reading the image
    # img_list = []
    image = cv2.imread(image_path)
    # print(img.shape)
    y = image.shape[0] // 11
    x = image.shape[1] // 6
    h = image.shape[0] // 11 * 7
    w = int(image.shape[1] // 3 * 2)
    image = image.copy()[y:y + h, x:x + w]

    # print(image.shape)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray",img_gray)
    im_remove_noise = cv2.fastNlMeansDenoising(img_gray, 8, 8, 7, 21)
    edged = cv2.Canny(im_remove_noise, 130, 240)

    # cv2.imshow("edged",edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # plt.imshow(closed)

    min_threshold_area = 500  # threshold area
    max_threshold_area = 50000

    # finding_contours
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    copy = image.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        # if max_threshold_area > area > min_threshold_area:
        peri = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(copy, [approx], -1, (0, 255, 0), 1)
        # print(area)

    # cv2.imshow("edge",image)
    # cv2.waitKey(0)
    seed_boxes = []
    idx = 0


    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if max_threshold_area > area > min_threshold_area:
            cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
            seed_boxes.append(((x, y), (x + w, y + h)))

        if w > 50 and h > 50:
            idx += 1
            new_img = image[y:y + h, x:x + w]
            # img_list.append(new_img)
            cv2.imwrite('Image Segmented_' + str(idx) + filename, new_img)
    # cv2.imshow("t", new_img)

    # cv2.imshow("boxed", image)
    # cv2.waitKey(0) & 0xFF
    # cv2.destroyAllWindows()

    # return image, seed_boxes




for filename in list_good_train_seeds:

    impath = os.path.join(trainSet_path, 'GoodSeed/', filename)
    seed_segment(impath, filename)
