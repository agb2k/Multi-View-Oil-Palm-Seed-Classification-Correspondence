# Computer Vision Group 4 Coursework
- Sample seed detector: "seed_detection_fastercnn" 
- Open source implementation of SIFT: PythonSIFT (Go through README to figure out usage)
- Pretrained model for sample seed detector has been removed as it's too large for GitHub
- Use "Dataset" folder as dataset for project
- The "DatasetTest" folder was created for the transfer learning experiments

# README

All python codes include image_correspondence.py, seed_correspondence_and_segmentation.py, 


## image_correspondence.py

##### To run the code, change the file location in lines 68, 88, 109, 130, 153, 172, 189, 210, 231 and 254 to your directory location where the ZIP file you have downloaded.

##### While running the program, each output result will show at the terminal before the output images are displayed, thus, please close the previous output images to see the next output images.

##### All output images will be saved to created folders 'Good seeds Correspondence' and 'Bad seeds Correspondence' respectively.


## seed_correspondence_and_segmentation.py

##### To run the script:
        1. change the directory to the Multiview_jpg folder
        2. uncomment the wanted sets loop from line 290
        3. Run the script
##### The outputs are the segmented seeds corresponded from the top view to the target view
    
