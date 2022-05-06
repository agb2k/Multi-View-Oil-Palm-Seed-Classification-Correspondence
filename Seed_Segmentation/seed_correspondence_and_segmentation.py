import cv2 
import numpy as np
import os

'''
#Source, reference, and train image are used interchangeably 
#Destenation, target, and query image are used interchangeably 
'''

# Cropping function in case it is needed

# def image_crop(image_path):
#     img = cv2.imread(image_path)
#     # print(img.shape)
#     # y = 0
#     # x = img.shape[1]//6
#     # h = img.shape[0]
#     # w = int(img.shape[1]//3 * 2)
#     # img = img.copy()[y:y+h, x:x+w]
#     return img

def seed_segment(image):

    image = cv2.imread(image)

    print(image.shape)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_remove_noise= cv2.fastNlMeansDenoising(img_gray, 8, 8, 7, 21) 
    edged = cv2.Canny(im_remove_noise,130,240) 
    
    
    # cv2.imshow("edged",edged)
    # cv2.waitKey(0) 
    
    #applying closing function  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8)) 
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    

    min_threshold_area = 500     #threshold area  
    max_threshold_area = 50000
    
    
    #finding_contours
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    copy = image.copy()
    for c in cnts:
        area = cv2.contourArea(c)
        peri = 0.1*cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(copy, [approx], -1, (0, 255, 0), 1)
        #print(area)
    
    
     
     
    # cv2.imshow("edge",image)
    # cv2.waitKey(0) 
    
    seed_boxes = []
    
    #idx = 0 
    #copy = image.copy()
    for c in cnts: 
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if max_threshold_area > area > min_threshold_area:
            cv2.rectangle(copy,(x, y), (x + w, y + h),(0,0,255),3)
            seed_boxes.append(((x, y), (x + w, y + h)))
    
    # Uncomment to save every seed individually     
    # 	if w>50 and h>50: 
    # 		idx+=1 
    # 		new_img=image[y:y+h,x:x+w]
    # 		#cv2.imwrite('Image Segmented_'+ str(idx) + '.png', new_img)
    # 		cv2.imshow("t",new_img)

    # cv2.imshow("boxed",image)
    # cv2.waitKey(0) & 0xFF
    # cv2.destroyAllWindows()
    
    return  copy, seed_boxes

def extract_sift(image):
    
  print('Extracting SIFT features points for {}'.format(image))
  image = cv2.imread(image)
  # convert to greyscale
  img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  sift = cv2.SIFT_create()

  # detect features from the image
  keypoints, descriptors = sift.detectAndCompute(img_gray, None)
  
  # draw the detected key points
  sift_image = cv2.drawKeypoints(img_gray, keypoints, img_gray)
  
  # show the image
  #print('SIFT features for {}'.format(image))
  # cv2.namedWindow("sift", cv2.WINDOW_NORMAL)
  # cv2.resizeWindow("sift", 1200, 800)
  # cv2.imshow("sift",sift_image)
  # k = cv2.waitKey(0) & 0xFF
  # if k == 27:         # wait for ESC key to exit
  #   cv2.destroyAllWindows()
  
  # # save the image
   # image_name = imagepath.rsplit('/',1)[1]
   # cv2.imwrite('sift_' + image_name, sift_image)
  
  
  return image, keypoints, descriptors





# croping the images to the seeds view only
# top_cropped_image = image_crop(path_ref)
# right_cropped_image = image_crop(path_trgt)

def find_correspondance(path_ref, path_trgt):
    # Extract sift features for different views
    image_src, keypoints_src, descriptors_src = extract_sift(path_ref)
    
    # cv2.namedWindow("src", cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow("src", 900, 400)
    # cv2.imshow("src",image_srcc)
    
    image_dst, keypoints_dst, descriptors_dst = extract_sift(path_trgt)
    
    # cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
    # # cv2.resizeWindow("dst", 900, 400)
    # cv2.imshow("dst",image_dst)
    
    # k = cv2.waitKey(0) & 0xFF
    # if k == 27:         # wait for ESC key to exit
    #     cv2.destroyAllWindows()
        
    image_src_box, seed_src_boxes = seed_segment(path_ref)
    
    image_dst_box, seed_dst_boxes = seed_segment(path_trgt)
    
    print("Matching the keypoints using FLANN...")
    # Using the FLANN Matcher
    MIN_MATCH_COUNT = 10
    
    FLANN_INDEX_KDTREE = 1
    
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 1000)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_src, descriptors_dst, k=2)
    
    
    
    pts_src = []
    pts_dst = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.73*n.distance:
            pts_dst.append(keypoints_dst[m.trainIdx].pt)
            pts_src.append(keypoints_src[m.queryIdx].pt)
    
    pts_src = np.int32(pts_src)
    pts_dst = np.int32(pts_dst)
    print("computing the homography for the images..." )
    Matrix, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0) # Matrix is the homography
    
    # We select only inlier points
    pts_src = pts_src[mask.ravel()==1]
    pts_dst = pts_dst[mask.ravel()==1]
            

    def drawlines(img1,img2,lines,pts1,pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r,c = img1.shape[:2]

        for r,pt1,pt2 in zip(lines,pts1,pts2):
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
        return img1,img2
    
    # Find epilines corresponding to points in target
    lines1 = cv2.computeCorrespondEpilines(pts_dst.reshape(-1,1,2), 2,Matrix)
    lines1 = lines1.reshape(-1,3)
    img5, img6 = drawlines(image_src, image_dst, lines1, pts_src, pts_dst)
    
    # Find epilines corresponding to points in source image

    lines2 = cv2.computeCorrespondEpilines(pts_src.reshape(-1,1,2), 1,Matrix)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(image_dst, image_src, lines2, pts_dst, pts_src)
    
    # show the matched images
    # cv2.namedWindow("Matched", cv2.WINDOW_NORMAL)
    # #cv2.resizeWindow("Matched", 1200, 800)
    # cv2.imshow("Matched",img3)
    # cv2.namedWindow("Matched2", cv2.WINDOW_NORMAL)
    # cv2.imshow("Matched2",img4)
    
    # k = cv2.waitKey(0) & 0xFF
    # if k == 27:         # wait for ESC key to exit
    #     cv2.destroyAllWindows()
    
    
    src_list = pts_src
    dst_list = pts_dst
    
    print(src_list)
    print(dst_list)
    
    src_box = []
    
    # check if the source good keypoints lay inside one of the seed boxes
    correct_points = 0
    idx = 0
    # loop through the boxes in the source image
    for (top,bottom)  in seed_src_boxes:
        src_box = []
        for j in range(len(src_list)):
            
            #check if the keypoint lays inside the box of the source image
            if src_list[j][0]>top[0]-30 and src_list[j][0]<bottom[0]+30: 
                if src_list[j][1]>top[1]+30 and src_list[j][1]<bottom[1]+30:
                    correct_points= j #save the index of the keypoint
                    src_box = (top,bottom) #source seed boxes of the matched features
        
        if src_box:
            #print(src_box)
    
            idx+=1 #track the segmented seeds
            new_img_src=image_src[src_box[0][1]:src_box[1][1], src_box[0][0]:src_box[1][0]]
            print(new_img_src.shape)
            
            # create a new folder for each correspndance set
            # dir = os.path.join("C:\\","temp","python")
            # if not os.path.exists(dir):
            #     os.mkdir(dir)
            
            
            cv2.imwrite('Source Seed Segmented_'+ str(idx) + '.png', new_img_src)
            
            
            #display image
            # cv2.namedWindow("new_img_src", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("new_img_src", 1200, 800)
            # cv2.imshow("new_img_src",new_img_src)
            # k = cv2.waitKey(0) & 0xFF
            # if k == 27:         # wait for ESC key to exit
            #     cv2.destroyAllWindows()
                
            
            # loop through the boxes in the target image
            for (top,bottom)  in seed_dst_boxes:
                dst_box = [] #target seed boxes of the matched features
                
                #check if the keypoint lays inside the box of the target image
                if dst_list[correct_points][0]>top[0]-30 and dst_list[correct_points][0]<bottom[0]+30:
                    if dst_list[correct_points][1]>top[1]-30 and dst_list[correct_points][1]<bottom[1]+30:
                        dst_box = (top,bottom)
                
                #if the destination boxes exist                 
                if dst_box:
                    #print(dst_box)
                    
                    new_img_dst=image_dst[dst_box[0][1]:dst_box[1][1], dst_box[0][0]:dst_box[1][0]]
                    print(new_img_dst.shape)
                    

                    cv2.imwrite('Target Seed Segmented_'+ str(idx) + '.png', new_img_dst)
                
                    #display image
                    # cv2.namedWindow("new_img_dst", cv2.WINDOW_NORMAL)
                    # cv2.resizeWindow("new_img_dst", 1200, 800)
                    # cv2.imshow("new_img_dst",new_img_dst)
                    # k = cv2.waitKey(0) & 0xFF
                    # if k == 27:         # wait for ESC key to exit
                    #     cv2.destroyAllWindows()


'''
    Running the script for correspondance and segmentation 
'''

# Select the images path based on the images dir

path_ref = "...\\Multiview_jpg\\Bad seeds\\top_S4.jpg"# queryImage
path_trgt ="...\\Multiview_jpg\\Bad seeds\\left_S4.jpg"# trainImage

find_correspondance(path_ref, path_trgt)

