# Computer Vision Group 4 Coursework
- Sample seed detector: "seed_detection_fastercnn" 
- Open source implementation of SIFT: PythonSIFT (Go through README to figure out usage)
- Pretrained model for sample seed detector has been removed as it's too large for GitHub
- Use "Dataset" folder as dataset for project
- The "DatasetTest" folder was created for the transfer learning experiments

# README

All python codes include image_correspondence.py, seed_correspondence_and_segmentation.py, 
## seed_segmentaion
#### To run this code, change the directory path to the seeds folders. The code containes three loops for single image segmenting, all good seeds, and all bad seeds images.
#### Line 187 is the path of the images directory and 216 is the target file that saves the segmented images path_save

## image_correspondence.py

##### To run the code, change the file location in lines 68, 88, 109, 130, 153, 172, 189, 210, 231 and 254 to your directory location where the ZIP file you have downloaded.

##### While running the program, each output result will show at the terminal before the output images are displayed, thus, please close the previous output images to see the next output images.

##### All output images will be saved to created folders 'Good seeds Correspondence' and 'Bad seeds Correspondence' respectively.


## seed_correspondence_and_segmentation.py

##### To run the script:
        1. change the directory to the Multiview_jpg folder for three paths: chdir('..'), outpath, and path_save 
        2. uncomment the wanted sets loop from line 290
        3. Run the script
##### The outputs are the segmented seeds corresponded from the top view to the target view
    
## MVCNN.py
##### To run the code, change the file locations in line 28, 29, 58, 70, 138, 140, 150. During training, pre-processed segmented images are taken as an input. The segmentation was
##### Done using the seed_correspondence_and_segmentation.py code. After training the model, it will automatically test the model using the assigned test images
