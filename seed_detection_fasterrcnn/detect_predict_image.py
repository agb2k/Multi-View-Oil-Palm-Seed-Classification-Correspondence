import torch
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import csv
import cv2

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import argparse
import glob

# Define constants
CLASSES = ['BACKGROUND', 'BAD', 'GOOD']
COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # blue for background, red for bad, green for good

# Define a list of pretrained detection models from torchvision.models.detection
TORCHVISION_DETECTION_MODELS = [
    torchvision.models.detection.fasterrcnn_resnet50_fpn,
    torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    torchvision.models.detection.retinanet_resnet50_fpn,
    torchvision.models.detection.ssd300_vgg16,
    torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    torchvision.models.detection.maskrcnn_resnet50_fpn,
    torchvision.models.detection.keypointrcnn_resnet50_fpn
]

TORCHVISION_DETECTION_MODELS_NAMES = [
    'fasterrcnn_resnet50_fpn',
    'fasterrcnn_mobilenet_v3_large_fpn',
    'fasterrcnn_mobilenet_v3_large_320_fpn',
    'retinanet_resnet50_fpn',
    'ssd300_vgg16',
    'ssdlite320_mobilenet_v3_large',
    'maskrcnn_resnet50_fpn',
    'keypointrcnn_resnet50_fpn'
]

CSV_SEED_INDIVIDUAL_HEADER = ["file_name", "x_min", "y_min", "x_max", "y_max", "bbox_label"]


# create a detection model
def get_detection_model(model_name,  # model name from torchvision_detection_models
                        pretrained=False,  # if True, returns a model pre-trained on COCOtrain2017
                        progress=True,  # If True, displays a progress bar of the download to stderr
                        num_classes=3,  # number of output classes including the background,
                                        # it is 3 in the case of good/bad seed detection
                        pretrained_backbone=True,  # if True, returns a model with backbone pretrained on imagenet
                        trainable_backbone_layers=None,  # number of trainable (not frozen) resnet layers starting from
                                                      # final block. Valid values are between 0 and 6, with 6 meaning
                                                      # all backbone layers are trainable.
                        **kwargs
                        ):
    for md_funciton in TORCHVISION_DETECTION_MODELS:
        if model_name in str(md_funciton):
            model = md_funciton(pretrained=pretrained,
                                progress=progress,
                                pretrained_backbone=pretrained_backbone,
                                trainable_backbone_layers=trainable_backbone_layers)
            break

    if 'fasterrcnn' in model_name:
        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        raise Exception('Model not implemented yet!')

    return model

# Please make sure you keep the following Argument class!!!
class Arguments:
    detection_model_name = TORCHVISION_DETECTION_MODELS_NAMES[0]
    pretrain_detection_model = True
    pretrain_detection_backbone = True
    trainable_backbone_layers = 0
    gamma = 0.9
    learning_rate = 0.005
    #train_trans = train_transforms_for_detection
    train_trans = [
                  #'resize',
                  #'vflip',
                  #'hflip',
                  #'rotate'
                  #'vflip+hflip'
                  #'rotate+vflip',
                  #'rotate+hflip',
                  #'rotate+vflip+hflip'
                  ]  # if True, then define train_transforms_for_detection inside the Dataset and does transform based on the image size of a particular sample
    test_trans = [
                  #'resize',
                  #'vflip',
                  #'hflip',
                  #'rotate'
                  #'vflip+hflip'
                  #'rotate+vflip',
                  #'rotate+hflip',
                  #'rotate+vflip+hflip'
                  ]  # if True, then define train_transforms_for_detection inside the Dataset and does transform based on the image size of a particular sample
    num_epochs_not_improved = 20
    output_dir = '/content/drive/My Drive/FYP/models'


# Define detection and prediction function for a new image based on a trained model
def detection_prediction(image_path, 
                         model_path, 
                         outpath, 
                         device=None, 
                         confidence_threshold=0.5):
    """
    Based on the saved model name to create an empty model
    Load the model state_dict
    Load the image and prepare it with the correct data type to be processed by the model
    Use the model make a prediction
    Display the detected objects and their scores
    """
    #---------------------------------------------------------------------------
    # load the image data from the image_path given
    if not os.path.exists(image_path):
        raise Exception('The image specified does not exist!')
    if '/' in image_path:
        image_name = image_path.rsplit('/', 1)[-1]  # obtain the image name for later use
    elif '\\' in image_path:
        image_name = image_path.rsplit('\\', 1)[-1]  # obtain the image name for later use

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # Make a copy of the image to overlay seed bounding boxes
    image_with_bboxes = image.copy()
    bound_padding = 5

    if not torch.is_tensor(image) and isinstance(image, np.ndarray):
        # convert seed_img to tensor
        image = transforms.ToTensor()(image).float()
        if len(image.shape) == 2:
            # convert (H,W) to (C,H,W)
            image = image.unsqueeze(0)
    else:
        raise Exception('The input image for classification must be either a Tensor or an Numpy array!')
    image = image.to(device)

    # create output path if it does not exist, to store images overlayed with bounding boxes
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    #---------------------------------------------------------------------------
    # create the model
    for model_name in TORCHVISION_DETECTION_MODELS_NAMES:
        if model_name in model_path:
            # create the model, then break from the for_loop
            model = get_detection_model(model_name=model_name)
            break
    # load model state_dict from the model_path provided
    if torch.cuda.is_available():
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    # move model to the right device
    model.to(device)

    #---------------------------------------------------------------------------
    # make a prediction. Note the input to model should be a list of images
    model.eval()
    output = model([image])
    detections = output[0]  
    # since there is only one output corresponding to one input image
    # run non maximum suppression
    valid_indices = torchvision.ops.nms(detections['boxes'], detections['scores'], iou_threshold=0.5)
    bboxes = detections['boxes'][valid_indices, :]
    scores = detections['scores'][valid_indices]
    labels = detections['labels'][valid_indices]

    #---------------------------------------------------------------------------
    # Create the CSV file if it does not exist
    outcsv = os.path.join(outpath, 'bbox_record.csv')
    if not os.path.exists(outcsv):
        with open(outcsv, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(CSV_SEED_INDIVIDUAL_HEADER)

    rows = []
    # loop over the detections
    for i in range(0, len(bboxes)):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = scores[i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > confidence_threshold:
            # extract the index of the class label from the
            # detections, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(labels[i])
            box = bboxes[i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # draw the bounding box and label on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(image_with_bboxes, (startX, startY), (endX, endY), 
                          COLORS[idx], 5)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image_with_bboxes, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 2)
            # append the bounding box in the row of a csv file
            rows.append([image_path, startX, startY, endX, endY, CLASSES[idx]])

    # save the image overlayed with bounding boxes to the output path
    print('writing the output image...')
    cv2.imwrite(os.path.join(outpath, 'bbox_' + image_name), cv2.cvtColor(image_with_bboxes, cv2.COLOR_RGB2BGR))

    # save the bounding boxes in a csv file
    print('Saving the bounding boxes to csv file...')
    with open(outcsv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)



# define the main function for detection
if __name__ == '__main__':
    print('start...')
    
    #Input
    # Construct a parser
    parser = argparse.ArgumentParser()
    # Specify the data root directory or take the default from const
    parser.add_argument('--imageinputroot', nargs='?', const=os.getcwd(), default=os.getcwd())
    # Specify the output root directory or take the default from const
    parser.add_argument('--outputroot', nargs='?', const=os.getcwd(), default=os.getcwd())
    # Specify the confidence score for filtering the low confidence detection
    parser.add_argument('--confidence', nargs='?', type=float, const=0.5, default=0.5)
    args = parser.parse_args()

    image_folder_path = args.imageinputroot
    outpath = args.outputroot

    #Trained on 6th Jan 2022
    trained_model = 'fasterrcnn_resnet50_fpn_pretrained_detection_True_pretrained_backbone_True_trainable_backbone_layers_5_01-06 06_22_27_best_epoch_16.pth'
    model_path = os.path.join(os.getcwd(), trained_model)

    # use cuda if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set the confidence threshold for classification
    confidence_threshold = args.confidence

    #image_folder_path = '/content/drive/My Drive/FYP/dataset/multi_1'
    #image_folder_path = '/content/drive/My Drive/FYP/dataset/multiview_seed'

    path = os.path.join(image_folder_path, '**/*.jpg')
    print(path)
    for imgfile in glob.iglob(path, recursive=True):
        # predict for the input image
        detection_prediction(imgfile, 
                            model_path, 
                            outpath, 
                            device=device, 
                            confidence_threshold=confidence_threshold)

    print('End...')
