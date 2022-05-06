import cv2
import numpy as np
import os


'''
#Source, reference, and train image are used interchangeably 
#Destenation, target, and query image are used interchangeably 
'''

'''
    To run the script:
        1. change the directory to the Multiview_jpg folder
        2. uncomment the wanted sets loop
        3. Run the script
    The outputs are the segmented seeds corresponded from the top view to the target view

'''


# Cropping function in case it is needed

def image_crop(image_path):
    img = cv2.imread(image_path)
    
    #rotate the image in case it is in portrait
    if(img.shape[0]>4000):
       img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
       
    print(img.shape)
    y = 0
    x = int(img.shape[1]*0.2)
    h = img.shape[0]
    w = int(img.shape[1]*0.60)
    img = img.copy()[y:y+h, x:x+w]
    return img

def seed_segment(image):
    
    if type(image) == str:
        image = cv2.imread(image)

    print(image.shape)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_remove_noise = cv2.fastNlMeansDenoising(img_gray, 8, 8, 7, 21)
    edged = cv2.Canny(im_remove_noise, 130, 240)


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

    
    # Extract the rectangular area around each seed
    seed_boxes = []
    
    # idx = 0
    # copy = image.copy()
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        
        if max_threshold_area > area > min_threshold_area:
            cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
            seed_boxes.append(((x, y), (x + w, y + h)))
            
    # Uncomment to save every seed individually
     
    # 	if w>50 and h>50: 
    # 		idx+=1 
    # 		new_img=image[y:y+h,x:x+w]
    # 		#cv2.imwrite('Image Segmented_'+ str(idx) + '.png', new_img)
    # 		cv2.imshow("t",new_img)


    print(seed_boxes)
    return copy, seed_boxes


def extract_sift(image):
    print('Extracting SIFT features points for {}'.format(image))
    if type(image) == str:
        image = cv2.imread(image)
    # convert to greyscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)

    # draw the detected key points
    sift_image = cv2.drawKeypoints(img_gray, keypoints, img_gray)


    return image, keypoints, descriptors




def find_correspondence(path_ref, path_trgt,path_save,trgt):
    
    # cropping the images to the seeds view only
    path_ref = image_crop(path_ref)
    path_trgt = image_crop(path_trgt)
    
    # Extract sift features for different views
    image_src, keypoints_src, descriptors_src = extract_sift(path_ref)

    image_dst, keypoints_dst, descriptors_dst = extract_sift(path_trgt)


    # Segment the source and target images
    
    image_src_box, seed_src_boxes = seed_segment(path_ref)

    image_dst_box, seed_dst_boxes = seed_segment(path_trgt)

    print("Matching the keypoints using FLANN...")
    # Using the FLANN Matcher
    MIN_MATCH_COUNT = 10

    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=1000)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_src, descriptors_dst, k=2)

    pts_src = []
    pts_dst = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.73 * n.distance:
            pts_dst.append(keypoints_dst[m.trainIdx].pt)
            pts_src.append(keypoints_src[m.queryIdx].pt)

    pts_src = np.int32(pts_src)
    pts_dst = np.int32(pts_dst)
    print("computing the homography for the images...")
    Matrix, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)  # Matrix is the homography

    # We select only inlier points
    pts_src = pts_src[mask.ravel() == 1]
    pts_dst = pts_dst[mask.ravel() == 1]

    def drawlines(img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r, c = img1.shape[:2]

        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2

    # Find epilines corresponding to points in target
    lines1 = cv2.computeCorrespondEpilines(pts_dst.reshape(-1, 1, 2), 2, Matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(image_src, image_dst, lines1, pts_src, pts_dst)

    # Find epilines corresponding to points in source image

    lines2 = cv2.computeCorrespondEpilines(pts_src.reshape(-1, 1, 2), 1, Matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(image_dst, image_src, lines2, pts_dst, pts_src)



    src_list = pts_src
    dst_list = pts_dst
    
    #Each array in source list is matched to the array with the same index in target list  
    
    print("Source inlier points: " + str(src_list))
    print("Target inlier points: " + str(dst_list))

    
    src_box = []

    # check if the source good keypoints lay inside one of the seed boxes
    correct_points = 0
    idx = 0
    # loop through the boxes in the source image
    for (top, bottom) in seed_src_boxes:
        src_box = []
        for j in range(len(src_list)):

            # check if the keypoint lays inside the box of the source image
            if src_list[j][0] > top[0] - 30 and src_list[j][0] < bottom[0] + 30:
                
                if src_list[j][1] > top[1] + 30 and src_list[j][1] < bottom[1] + 30:
                    
                    correct_points = j      # save the index of the keypoint
                    src_box = (top, bottom)  # source seed boxes of the matched features

        if src_box:


            idx += 1  # track the segmented seeds
            new_img_src = image_src[src_box[0][1]:src_box[1][1], src_box[0][0]:src_box[1][0]] #crop the seed based on the detection box size


            # Display image
            # cv2.namedWindow("new_img_src", cv2.WINDOW_NORMAL)
            # cv2.imshow("new_img_src",new_img_src)
            # k = cv2.waitKey(0) & 0xFF




            # loop through the boxes in the target image
            for (top, bottom) in seed_dst_boxes:
                dst_box = []  # target seed boxes of the matched features

                # check if the keypoint lays inside the box of the target image
                if dst_list[correct_points][0] > top[0] - 30 and dst_list[correct_points][0] < bottom[0] + 30:
                    
                    if dst_list[correct_points][1] > top[1] - 30 and dst_list[correct_points][1] < bottom[1] + 30:
                        
                        dst_box = (top, bottom) #Save the target seed boxes

                # if the destination boxes exist
                if dst_box:


                    new_img_dst = image_dst[dst_box[0][1]:dst_box[1][1], dst_box[0][0]:dst_box[1][0]] #crop the seed based on the detection box size
                    print(new_img_dst.shape)
                    
                    print("Seed " + str(idx) + " has a correspondence")
                    
                    
                    
                    # the output is both corresponding seeds
                    #the source seed refers to the top view seeds
                    #the target seed refers to the current target seed
                    
                    
                    # the output name is in the following order:
                        # "the current target seed view" + - +  'Source or target Seed Segmented' + "seed number" + .png
                        
                        
                    cv2.imwrite(os.path.join(path_save,trgt +"-" + 'Source Seed Segmented_' + str(idx) + '.png'), new_img_src)
                    cv2.imwrite(os.path.join(path_save,trgt +"-" +'Target Seed Segmented_' + str(idx) + '.png'), new_img_dst)


                    
                    # Display image
                    # cv2.namedWindow("new_img_dst", cv2.WINDOW_NORMAL)
                    # cv2.imshow("new_img_dst",new_img_dst)
                    # k = cv2.waitKey(0) & 0xFF



'''
    Running the script for correspondence and segmentation 
'''

# You could run the correspondence detection and segmentation for one or more sets

# Uncomment the required correspondence set


'''
ONE PAIR

os.chdir(r"F:/CompV/Multiview_jpg/Good seeds")
if not os.path.exists('Good seeds Correspondence - set 2'):
    os.mkdir('Good seeds Correspondence - set 2')

path_ref = 'top_S2.jpg'
path_trgt = 'left_S2.jpg'
path_save = r"F:/CompV/Multiview_jpg/Good seeds/" + "Good seeds Correspondence - set 2" 
trgt = "left"


find_correspondence(path_ref, path_trgt, path_save ,trgt)

'''



#all sets from the good seeds

''' 
GOOD SETS
# Select the images path based on the images dir
os.chdir(r"F:/CompV/Multiview_jpg/Good seeds")


        
#loop through the good sets
j=1
while j <11:
    i=1
    #loop through the povs

    while i < 5:
        print("Matching Set" + str(j) +" in respect to Top view")
        path_ref = ("top_S" + str(j) + ".jpg") 
        if not os.path.exists('Good seeds Correspondence - set' + str(j)):
            os.mkdir('Good seeds Correspondence - set' + str(j))
        path_save = r"F:/CompV/Multiview_jpg/Good seeds/" + "Good seeds Correspondence - set" + str(j)
        #matching left to top and segmenting the matches
        if i == 1:
            print("matching left to top")
            trgt = "left"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
        
        #matching right to top and segmenting the matches     
        elif i == 2:
            print("matching right to top")
            trgt = "right"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
            
        #matching rear to top and segmenting the matches     
        elif i==3:
            print("matching rear to top")
            trgt = "rear"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
            
        #matching front to top and segmenting the matches     
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

# Select the images path based on the images dir
os.chdir(r"F:/CompV/Multiview_jpg/Bad seeds")


        
#loop through the bad sets
j=1
while j <13:
    i=1
    #loop through the povs

    while i < 5:
        print("Matching Set" + str(j) +" in respect to Top view")
        path_ref = ("top_S" + str(j) + ".jpg") 
        if not os.path.exists('Bad seeds Correspondence - set' + str(j)):
            os.mkdir('Bad seeds Correspondence - set' + str(j))
        path_save = r"F:/CompV/Multiview_jpg/Bad seeds/" + "Bad seeds Correspondence - set" + str(j)
        #matching left to top and segmenting the matches
        if i == 1:
            print("matching left to top")
            trgt = "left"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
        
        #matching right to top and segmenting the matches     
        elif i == 2:
            print("matching right to top")
            trgt = "right"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
            
        #matching rear to top and segmenting the matches     
        elif i==3:
            print("matching rear to top")
            trgt = "rear"
            path_trgt = (trgt + "_S" + str(j) + ".jpg") 
            find_correspondence(path_ref, path_trgt,path_save,trgt)
            
        #matching front to top and segmenting the matches     
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


