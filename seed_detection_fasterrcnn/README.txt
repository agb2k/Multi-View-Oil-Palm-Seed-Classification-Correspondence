There are two files:

1. detect_predict_image.py 
   It performs the individual seed detection for a given input image that contains multiple seeds. 
   It overlays the detected bounding boxes as well as the label (i.e., good or bad) on the orginal image and saves the overlayed image.
   It also saves all the coordinates of all the bounding boxes and their labels along with the image file path in a csv file named "bbox_record.csv".
   Both output files are saved in a output path the user specifies. The default folder would be the current working directory.

2. fasterrcnn_resnet50_fpn_pretrained_detection_True_pretrained_backbone_True_trainable_backbone_layers_5_01-06 06_22_27_best_epoch_16.pth
   This is the trained seed detection model based on faster rcnn. You must place the file in the same directory where you would put the file detect_predict_image.py

To use the detection file, there are three inputs at the command line:

   --imageinputroot 
   It specifies the path of the input folder where your input images are placed. 
   If you do not specify the imageinputroot, it will use the current working directory by default. However, you must make sure your images are placed in the default directory.

   --outputroot
   It specifies the path of the output folder where you would like the output files to be.
   If you do not specify the outputroot, it will use the current working directory by default.

   --confidence
   It specifies the confidence threshold below which the bounding boxes will be filtered out. It expects a float type input with the range between 0 and 1.
   If you do not specify the confidence value, it will use 0.5 by default.

Example usage:

detect_predict_image.py --imageinputroot "C:\Users\IYL\Documents\Documents_X230\COLLABORATORS\AAR\Dataset\Multiview_jpg\Bad seeds\Set1" --outputroot "multiview_output" --confidence 0.4

detect_predict_image.py