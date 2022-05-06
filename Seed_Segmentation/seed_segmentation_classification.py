import cv2
import numpy as np
import os
import csv


'''
#Source, reference, and train image are used interchangeably 
#Destenation, target, and query image are used interchangeably 
'''


def seed_segment(image, path_save, Set, view):
    
    if type(image) == str:
        image = cv2.imread(image)

    print(image.shape)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_remove_noise = cv2.fastNlMeansDenoising(img_gray, 8, 8, 7, 21)
    edged = cv2.Canny(im_remove_noise, 130, 240)

    # cv2.imshow("edged",edged)
    # cv2.waitKey(0) 

    # applying closing function
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    min_threshold_area = 500 # threshold area
    max_threshold_area = 50000

    # finding_contours
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    copy = image.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        if max_threshold_area > area > min_threshold_area:
            peri = 0.1 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            cv2.drawContours(copy, [approx], -1, (0, 255, 0), 1)
    # print(area)

    # cv2.imshow("edge",image)
    # cv2.waitKey(0) 
    
    
    # Extract the rectangular area around each seed
    seed_boxes = []
    
    idx = 0
    seed_idx = 0 
    # copy = image.copy()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        if max_threshold_area > area > min_threshold_area and w>50 and h>50 and w<450 and h<600:
            cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
            seed_boxes.append(((x, y), (x + w, y + h)))
            print(seed_boxes[seed_idx])
            seed_idx += 1 
            idx+=1 
            new_img=image[y:y+h,x:x+w]
            path = os.path.join(path_save,"Set "+ str(Set) + " - " + view + " Segmented Seed " + str(idx) + '.png')
            cv2.imwrite(path, new_img)
            print(path)
    # Uncomment to save every seed individually
     


    # 		cv2.imshow("t",new_img)
    cv2.namedWindow("boxed", cv2.WINDOW_NORMAL)
    cv2.imshow("boxed",copy)
    cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    
    return copy, seed_boxes


'''
    Running the script for correspondence and segmentation 
'''


'''
os.chdir(r"F:/CompV/Multiview_jpg/Good seeds")
if not os.path.exists('Good seeds Correspondence - set 2'):
    os.mkdir('Good seeds Correspondence - set 2')

path_ref = 'top_S2.jpg'
path_trgt = 'left_S2.jpg'
path_save = r"F:/CompV/Multiview_jpg/Good seeds/" + "Good seeds Correspondence - set 2" 
trgt = "left"


find_correspondence(path_ref, path_trgt, path_save ,trgt)

'''

'''
# Select the images path based on the images dir

os.chdir(r"F:/CompV/Multiview_jpg/Good seeds")


j=1
while j <11:
    i=1
    while i < 5:
        print("Matching Set" + str(j) +" in respect to Top view")
        path_ref = ("top_S" + str(j) + ".jpg") 
        if not os.path.exists('Good seeds Correspondence - set' + str(j)):
            os.mkdir('Good seeds Correspondence - set' + str(j))
        path_save = r"F:/CompV/Multiview_jpg/Good seeds/" + "Good seeds Correspondence - set" + str(j)
        if i == 1:
            print("matching left to top")
            trgt = "left"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
        elif i == 2:
            print("matching right to top")
            trgt = "right"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
        elif i==3:
            print("matching rear to top")
            trgt = "rear"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
            
        elif i == 4:
            print("matching front to top")
            trgt = "front"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
        else:
            print("Something went wrong")
            
        
        i += 1
    print("check")
    j += 1

'''

'''
os.chdir(r"F:/CompV/Multiview_jpg/Good seeds")
if not os.path.exists('Good seeds - set 9'):
    os.mkdir('Good seeds - set 9')

path_image = 'left_S9.jpg'
view = "left"
Set = "9"

path_save = "F:\\CompV\\Multiview_jpg\\Good seeds\\" + "Good seeds - set 9" 


image_src_box, seed_src_boxes = seed_segment(path_image, path_save, Set, view)

'''


os.chdir(r"F:/CompV/Multiview_jpg/Good seeds")


j=9
while j <11:
    i=1
    while i < 6:
        print("Segmenting...")
        if not os.path.exists('Good seeds - set ' + str(j)):
            os.mkdir('Good seeds - set ' + str(j))
        path_save = "F:\\CompV\\Multiview_jpg\\Good seeds\\" + "Good seeds - set " + str(j) 
            
        if i == 1:
            print("Segmenting left")
            trgt = "left"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            seed_segment(path_trgt, path_save, str(j), trgt)
            
        elif i == 2:
            print("Segmenting right")
            trgt = "right"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            seed_segment(path_trgt, path_save, str(j), trgt)
            
        elif i==3:
            print("Segmenting rear")
            trgt = "rear"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            seed_segment(path_trgt, path_save, str(j), trgt)
            
        elif i == 4:
            print("segmenting front")
            trgt = "front"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            seed_segment(path_trgt, path_save, str(j), trgt)
            
        elif i == 5:
            print("segmenting top")
            trgt = "top"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            seed_segment(path_trgt, path_save, str(j), trgt)
            
        i += 1
    print("check")
    j += 1