import cv2 
import math
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt


path_top = 'Bad seeds/Set2/top_S2.jpg'
path_right = 'Bad seeds/Set2/right_S2.jpg'

def image_crop(image_path):
    
    img = cv2.imread(image_path)
    # print(img.shape)
    y = img.shape[0]//11
    x = img.shape[1]//6
    h = img.shape[0]//9 * 7
    w = int(img.shape[1]//3 * 2)
    image_crop = img.copy()[y:y+h, x:x+w]
    
    return image_crop

def seed_segment(image):

    image = cv2.imread(image)

    print(image.shape)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_remove_noise= cv2.fastNlMeansDenoising(img_gray, 8, 8, 7, 21) 
    edged = cv2.Canny(im_remove_noise,130,240) 
    
    
    # cv2.imshow("edged",edged)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    
    
    #applying closing function  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8)) 
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # plt.imshow(closed)
    
    min_threshold_area = 500     #threshold area  
    max_threshold_area = 50000
    
    
    #finding_contours
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    
    copy = image.copy()
    for c in cnts:
    		area = cv2.contourArea(c)
    	#if max_threshold_area > area > min_threshold_area:
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
  #print('Extracting SIFT features points for {}'.format(image))
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



# Extract sift features for different views
# Set top view as destination or target where the source image will be aligned or warped to the target image
# This is because the top view has much wider field of view since it is 50cm above the platform


top_cropped_image = image_crop(path_top)
right_cropped_image = image_crop(path_right)



image_src, keypoints_src, descriptors_src = extract_sift(path_top)

# cv2.namedWindow("src", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("src", 900, 400)
# cv2.imshow("src",image_src)

image_dst, keypoints_dst, descriptors_dst = extract_sift(path_right)

# cv2.namedWindow("dst", cv2.WINDOW_NORMAL)
# # cv2.resizeWindow("dst", 900, 400)
# cv2.imshow("dst",image_dst)

k = cv2.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
    
image_src_box, seed_src_boxes = seed_segment(path_top)

image_dst_box, seed_dst_boxes = seed_segment(path_right)



MIN_MATCH_COUNT = 10

FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
# We align the front view to the top view, i.e., front view is the source and the top view is the target/desination
# You may also swap the two and see how the performance is
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
#F, mask = cv2.findFundamentalMat(pts_src,pts_dst,cv2.FM_LMEDS)
Matrix, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0) # Matrix is the homography

# We select only inlier points
pts_src = pts_src[mask.ravel()==1]
pts_dst = pts_dst[mask.ravel()==1]
        
# store all the good matches as per Lowe's ratio test.
# good = []

# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    #img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1,img2

# Find epilines corresponding to points in right image (second image, or target/destination) and
# drawing its lines on left image (first image, or source image)
lines1 = cv2.computeCorrespondEpilines(pts_dst.reshape(-1,1,2), 2,Matrix)
lines1 = lines1.reshape(-1,3)
img5, img6 = drawlines(image_src, image_dst, lines1, pts_src, pts_dst)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts_src.reshape(-1,1,2), 1,Matrix)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(image_dst, image_src, lines2, pts_dst, pts_src)

# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()

cv2.namedWindow("Matched", cv2.WINDOW_NORMAL)


#cv2.resizeWindow("Matched", 1200, 800)
cv2.imshow("Matched",img3)
cv2.namedWindow("Matched2", cv2.WINDOW_NORMAL)
cv2.imshow("Matched2",img4)

k = cv2.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

# if len(good)>MIN_MATCH_COUNT:
#     src_pts = np.float32([ keypoints_src[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
#     dst_pts = np.float32([ keypoints_dst[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
#     Matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # Matrix is the homography
#     print(Matrix)
    
#     matchesMask = mask.ravel().tolist()
#     h,w = image_src.shape[:2]
#     pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#     dst = cv2.perspectiveTransform(pts, Matrix)

#     homography = cv2.polylines(image_src, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
#     # cv2.namedWindow("homography", cv2.WINDOW_NORMAL)
#     # cv2.resizeWindow("homography", 1200, 800)
#     # cv2.imshow("homography",homography)
#     # k = cv2.waitKey(0) & 0xFF
#     # if k == 27:         # wait for ESC key to exit
#     #     cv2.destroyAllWindows()
# else:
#     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
#     matchesMask = None



# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)
# image_out = cv2.drawMatches(image_src, keypoints_src, image_dst, keypoints_dst, good, None, **draw_params)
# cv2.namedWindow("Matched", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Matched", 1200, 800)
# cv2.imshow("Matched",image_out)
# # k = cv2.waitKey(0) & 0xFF
# # if k == 27:         # wait for ESC key to exit
# #     cv2.destroyAllWindows()

src_list = pts_src
dst_list = pts_dst

print(src_list)
print(dst_list)

src_box = []

# check if the source good keypoints lay inside one of the seed boxes
correct_points = 0
idx = 0
for (top,bottom)  in seed_src_boxes:
    src_box = []
    for j in range(len(src_list)):
        #print(src_list[j][0][0])
        if src_list[j][0]>top[0]-30 and src_list[j][0]<bottom[0]+30:
            print("check")
            if src_list[j][1]>top[1]+30 and src_list[j][1]<bottom[1]+30:
                print("check 2")
                correct_points= j
                src_box = (top,bottom)
    
    if src_box:
        print(src_box)

        idx+=1
        new_img_src=image_src[src_box[0][1]:src_box[1][1], src_box[0][0]:src_box[1][0]]
        print(new_img_src.shape)
        #cv2.imwrite('Image Segmented_'+ str(idx) + '.png', new_img)
    
        cv2.namedWindow("new_img_src", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("new_img_src", 1200, 800)
        cv2.imshow("new_img_src",new_img_src)
        # k = cv2.waitKey(0) & 0xFF
        # if k == 27:         # wait for ESC key to exit
        #     cv2.destroyAllWindows()
            
        
        for (top,bottom)  in seed_dst_boxes:
            dst_box = []
            if dst_list[correct_points][0]>top[0]-30 and dst_list[correct_points][0]<bottom[0]+30:
                print("check")
                if dst_list[correct_points][1]>top[1]-30 and dst_list[correct_points][1]<bottom[1]+30:
                    print("check 2")
                    
                    dst_box = (top,bottom)
            
            if dst_box:
                print(dst_box)

                # idx+=1
                new_img_dst=image_dst[dst_box[0][1]:dst_box[1][1], dst_box[0][0]:dst_box[1][0]]
                print(new_img_dst.shape)
                #cv2.imwrite('Image Segmented_'+ str(idx) + '.png', new_img)
            
                cv2.namedWindow("new_img_dst", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("new_img_dst", 1200, 800)
                cv2.imshow("new_img_dst",new_img_dst)
                k = cv2.waitKey(0) & 0xFF
                if k == 27:         # wait for ESC key to exit
                    cv2.destroyAllWindows()

#TODO test other povs
#TODO save the segmented seeds
