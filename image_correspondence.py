##### Change to your directory location in line 68, 88, 109, 130, 153, 172, 189, 210, 231 and 254.

import os
import numpy as np
import cv2  

def extract_sift(img1,img2):

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    good = []

    # Initialize lists
    list_kp1 = []
    list_kp2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            good.append(m)

            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = kp1[m.queryIdx].pt
            (x2, y2) = kp2[m.trainIdx].pt

            # Append to each list
            list_kp1.append((x1, y1))
            list_kp2.append((x2, y2))
            
    draw_params = dict( matchesMask = matchesMask,flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    print("Total number of matches points: ")
    print(len(good))
    print("Coordinates of each matches points in Image 1: ")
    print(list_kp1)
    print("Coordinates of each matches points in Image 2: ")
    print(list_kp2)
    return (img3)



########## WHILE RUNNING THE PROGRAM, PLEASE CLOSE THE PREVIOUS OUPUT IMAGE TO SEE NEXT OUTPUT IMAGE. ##########



#### GOOD SEEDS 

##### CHANGE TO YOUR GOOD SEEDS FILE LOCATION #####
os.chdir("...\\Multiview_jpg\\Good seeds")
os.mkdir('Good seeds Correspondence') 

# Image correspondence between TOP view and LEFT view of each sets of good seeds
i = 1
while i < 11 :

    # Convert original image to grayscale
    img1 = cv2.imread("top_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY)# queryImage

    # Rotate the vertical image to horizontal
    if(img1.shape[0]>4000):
        img1 = cv2.rotate(img1,cv2.ROTATE_90_CLOCKWISE)
        
    img2 = cv2.imread("left_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY) # trainImage
    print("___________________________________________________________________________________________________")
    print("Image correspondence of seeds between "+"top_S" + str(i) + " view and " + "left_S" + str(i) + " view of Good Seeds")
    x = extract_sift(img1,img2)

    ##### CHANGE TO YOUR THE LOCATION THAT YOU WANT TO SAVE ALL IMAGES #####
    path = "...\\Multiview_jpg\\Good seeds\\Good seeds Correspondence"
    cv2.imwrite(os.path.join(path, "top_S" + str(i) + ".jpg VS " + "left_S" + str(i) + ".jpg of Good Seeds.jpg"), x)
    cv2.namedWindow("top_S" + str(i) + ".jpg VS " + "left_S" + str(i) + ".jpg of Good Seeds", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("top_S" + str(i) + ".jpg VS " + "left_S" + str(i) + ".jpg of Good Seeds", 2016, 1134)
    cv2.imshow("top_S" + str(i) + ".jpg VS " + "left_S" + str(i) + ".jpg of Good Seeds",x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1

# Image correspondence between TOP view and RIGHT view of each sets of good seeds
i = 1
while i < 11 :
    img1 = cv2.imread("top_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY)# queryImage
    if(img1.shape[0]>4000):
        img1 = cv2.rotate(img1,cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.imread("right_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY) # trainImage
    print("____________________________________________________________________________________________________")
    print("Image correspondence of seeds between "+"top_S" + str(i) + " view and " + "right_S" + str(i) + " view of Good Seeds")
    x = extract_sift(img1,img2)

    ##### CHANGE TO YOUR THE LOCATION THAT YOU WANT TO SAVE ALL IMAGES #####
    path = "...\\Multiview_jpg\\Good seeds\\Good seeds Correspondence"
    cv2.imwrite(os.path.join(path, "top_S" + str(i) + ".jpg VS " + "right_S" + str(i) + ".jpg of Good Seeds.jpg"), x)
    cv2.namedWindow("top_S" + str(i) + ".jpg VS " + "right_S" + str(i) + ".jpg of Good Seeds", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("top_S" + str(i) + ".jpg VS " + "right_S" + str(i) + ".jpg of Good Seeds", 2016, 1134)
    cv2.imshow("top_S" + str(i) + ".jpg VS " + "right_S" + str(i) + ".jpg of Good Seeds",x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1

# Image correspondence between TOP view and REAR view of each sets of good seeds
i = 1
while i < 11 :
    img1 = cv2.imread("top_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY)# queryImage
    if(img1.shape[0]>4000):
        img1 = cv2.rotate(img1,cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.imread("rear_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY) # trainImage
    print("____________________________________________________________________________________________________")
    print("Image correspondence of seeds between "+"top_S" + str(i) + " view and " + "rear_S" + str(i) + " view of Good Seeds")
    x = extract_sift(img1,img2)

    ##### CHANGE TO YOUR THE LOCATION THAT YOU WANT TO SAVE ALL IMAGES #####
    path = "...\\Multiview_jpg\\Good seeds\\Good seeds Correspondence"
    cv2.imwrite(os.path.join(path,"top_S" + str(i) + ".jpg VS " + "rear_S" + str(i) + ".jpg of Good Seeds.jpg"), x)
    cv2.namedWindow("top_S" + str(i) + ".jpg VS " + "rear_S" + str(i) + ".jpg of Good Seeds", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("top_S" + str(i) + ".jpg VS " + "rear_S" + str(i) + ".jpg of Good Seeds", 2016, 1134)
    cv2.imshow("top_S" + str(i) + ".jpg VS " + "rear_S" + str(i) + ".jpg of Good Seeds",x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1

# Image correspondence between TOP view and FRONT view of each sets of good seeds
i = 1
while i < 11 :
    img1 = cv2.imread("top_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY)# queryImage
    if(img1.shape[0]>4000):
        img1 = cv2.rotate(img1,cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.imread("front_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY) # trainImage
    if(img2.shape[0]>4000):
        img2 = cv2.rotate(img2,cv2.ROTATE_90_CLOCKWISE)
    print("____________________________________________________________________________________________________")
    print("Image correspondence of seeds between "+"top_S" + str(i) + " view and " + "front_S" + str(i) + " view of Good Seeds")
    x = extract_sift(img1,img2)

    ##### CHANGE TO YOUR THE LOCATION THAT YOU WANT TO SAVE ALL IMAGES #####
    path = "...\\Multiview_jpg\\Good seeds\\Good seeds Correspondence"
    cv2.imwrite(os.path.join(path,"top_S" + str(i) + ".jpg VS " + "front_S" + str(i) + ".jpg of Good Seeds.jpg"), x)
    cv2.namedWindow("top_S" + str(i) + ".jpg VS " + "front_S" + str(i) + ".jpg of Good Seeds", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("top_S" + str(i) + ".jpg VS " + "front_S" + str(i) + ".jpg of Good Seeds", 2016, 1134)
    cv2.imshow("top_S" + str(i) + ".jpg VS " + "front_S" + str(i) + ".jpg of Good Seeds",x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1








#### BAD SEEDS 

##### CHANGE TO YOUR BAD SEEDS FILE LOCATION #####
os.chdir("...\\Multiview_jpg\\Multiview_jpg\\Bad seeds")
os.mkdir('Bad seeds Correspondence') 



# Image correspondence between TOP view and LEFT view of each sets of bad seeds
i = 1
while i < 13 :
    img1 = cv2.imread("top_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY)# queryImage
    if(img1.shape[0]>4000):
        img1 = cv2.rotate(img1,cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.imread("left_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY) # trainImage
    print("___________________________________________________________________________________________________")
    print("Image correspondence of seeds between "+"top_S" + str(i) + " view and " + "left_S" + str(i) + " view of Bad Seeds")
    x = extract_sift(img1,img2)

    ##### CHANGE TO YOUR THE LOCATION THAT YOU WANT TO SAVE ALL IMAGES #####
    path = "...\\Multiview_jpg\\Bad seeds\\Bad seeds Correspondence"
    cv2.imwrite(os.path.join(path,"top_S" + str(i) + ".jpg VS " + "left_S" + str(i) + ".jpg of Bad Seeds.jpg"), x)
    cv2.namedWindow("top_S" + str(i) + ".jpg VS " + "left_S" + str(i) + ".jpg of Bad Seeds", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("top_S" + str(i) + ".jpg VS " + "left_S" + str(i) + ".jpg of Bad Seeds", 2016, 1134)
    cv2.imshow("top_S" + str(i) + ".jpg VS " + "left_S" + str(i) + ".jpg of Bad Seeds",x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1

# Image correspondence between TOP view and RIGHT view of each sets of bad seeds
i = 1
while i < 13 :
    img1 = cv2.imread("top_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY)# queryImage
    if(img1.shape[0]>4000):
        img1 = cv2.rotate(img1,cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.imread("right_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY) # trainImage
    print("____________________________________________________________________________________________________")
    print("Image correspondence of seeds between "+"top_S" + str(i) + " view and " + "right_S" + str(i) + " view of Bad Seeds")
    x = extract_sift(img1,img2)

    ##### CHANGE TO YOUR THE LOCATION THAT YOU WANT TO SAVE ALL IMAGES #####
    path = "...\\Multiview_jpg\\Bad seeds\\Bad seeds Correspondence"
    cv2.imwrite(os.path.join(path,"top_S" + str(i) + ".jpg VS " + "right_S" + str(i) + ".jpg of Bad Seeds.jpg"), x)
    cv2.namedWindow("top_S" + str(i) + ".jpg VS " + "right_S" + str(i) + ".jpg of Bad Seeds", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("top_S" + str(i) + ".jpg VS " + "right_S" + str(i) + ".jpg of Bad Seeds", 2016, 1134)
    cv2.imshow("top_S" + str(i) + ".jpg VS " + "right_S" + str(i) + ".jpg of Bad Seeds",x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1

# Image correspondence between TOP view and REAR view of each sets of bad seeds
i = 1
while i < 13 :
    img1 = cv2.imread("top_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY)# queryImage
    if(img1.shape[0]>4000):
        img1 = cv2.rotate(img1,cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.imread("rear_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY) # trainImage
    print("____________________________________________________________________________________________________")
    print("Image correspondence of seeds between "+"top_S" + str(i) + " view and " + "rear_S" + str(i) + " view of Bad Seeds")
    x = extract_sift(img1,img2)

    ##### CHANGE TO YOUR THE LOCATION THAT YOU WANT TO SAVE ALL IMAGES #####
    path = "...\\Multiview_jpg\\Bad seeds\\Bad seeds Correspondence"
    cv2.imwrite(os.path.join(path,"top_S" + str(i) + ".jpg VS " + "rear_S" + str(i) + ".jpg of Bad Seeds.jpg"), x)
    cv2.namedWindow("top_S" + str(i) + ".jpg VS " + "rear_S" + str(i) + ".jpg of Bad Seeds", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("top_S" + str(i) + ".jpg VS " + "rear_S" + str(i) + ".jpg of Bad Seeds", 2016, 1134)
    cv2.imshow("top_S" + str(i) + ".jpg VS " + "rear_S" + str(i) + ".jpg of Bad Seeds",x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1

# Image correspondence between TOP view and FRONT view of each sets of bad seeds
i = 1
while i < 13 :
    img1 = cv2.imread("top_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY)# queryImage
    if(img1.shape[0]>4000):
        img1 = cv2.rotate(img1,cv2.ROTATE_90_CLOCKWISE)
    img2 = cv2.imread("front_S" + str(i) + ".jpg", cv2.COLOR_BGR2GRAY) # trainImage
    if(img2.shape[0]>4000):
        img2 = cv2.rotate(img2,cv2.ROTATE_90_CLOCKWISE)
    print("____________________________________________________________________________________________________")
    print("Image correspondence of seeds between "+"top_S" + str(i) + " view and " + "front_S" + str(i) + " view of Bad Seeds")
    x = extract_sift(img1,img2)
    
    ##### CHANGE TO YOUR THE LOCATION THAT YOU WANT TO SAVE ALL IMAGES #####
    path = "...\\Multiview_jpg\\Bad seeds\\Bad seeds Correspondence"
    cv2.imwrite(os.path.join(path,"top_S" + str(i) + ".jpg VS " + "front_S" + str(i) + ".jpg of Bad Seeds.jpg"), x)
    cv2.namedWindow("top_S" + str(i) + ".jpg VS " + "front_S" + str(i) + ".jpg of Bad Seeds", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("top_S" + str(i) + ".jpg VS " + "front_S" + str(i) + ".jpg of Bad Seeds", 2016, 1134)
    cv2.imshow("top_S" + str(i) + ".jpg VS " + "front_S" + str(i) + ".jpg of Bad Seeds",x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    i += 1