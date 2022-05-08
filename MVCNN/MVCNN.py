import copy
import csv
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_sample_weight
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torch.autograd import Variable

print(torch.cuda.is_available())

# Change boolean to determine whether you should train or not
trainBool = True

# Base path for images
trainSet_path = '../DatasetTest1/Segmented/Train'
testSet_path = '../DatasetTest1/Test'

# List of training and test images
list_good_train_seeds = sorted(os.listdir(trainSet_path + '/GoodSeed'))
list_bad_train_seeds = sorted(os.listdir(trainSet_path + '/BadSeed'))
list_good_test_seeds = sorted(os.listdir(testSet_path + '/GoodSeed'))
list_bad_test_seeds = sorted(os.listdir(testSet_path + '/BadSeed'))


# Compute weight based on classes
def compute_sample_weight(class_weights, y):
    assert isinstance(class_weights, dict)
    result = np.array([class_weights[i] for i in y])
    return result


# Class Weights used in the weighted accuracy
class_weights = {
    0: 0.5,
    1: 0.5,
}

# Class labels to indices
class_id_map = {'Bad Seed': 0, 'Good Seed': 1}

# Creating CSV files
with open('../CSV/testData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "label"])

    # create test data with the first 200 images of good seeds
    for filename in list_good_test_seeds:
        writer.writerow([filename, 1])

    # create test data with the first 200 images of bad seeds
    for filename in list_bad_test_seeds:
        writer.writerow([filename, 0])

with open('../CSV/trainingData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "label"])

    # create training data with the rest of the images of good seeds
    for filename in list_good_train_seeds:
        writer.writerow([filename, 1])

    # create training data with the rest of the images of bad seeds
    for filename in list_bad_train_seeds:
        writer.writerow([filename, 0])


# Dataset class
class OilPalmSeedsDataset(Dataset):
    def __init__(self, df, base_path, transform=None, target_transform=None):
        self.df = df
        self.base_path = base_path
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx, 1]

        # The training images are stored in two separate folders
        if label == 1:
            img_path = os.path.join(self.base_path, 'GoodSeed/', self.df.iloc[idx, 0])
        elif label == 0:
            img_path = os.path.join(self.base_path, 'BadSeed/', self.df.iloc[idx, 0])

        image = Image.open(img_path).convert('RGB')

        # Transformations done on the image
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {"image": image, "label": label}
        return sample


# Transformations done on the images
transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose(
    [
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

    ])

# load the training csv file in terms of annotations to dataframe and
# randomly split it to training and validation sets respectively
trainvaldf = pd.read_csv("../CSV/trainingdata.csv")
traindf, valdf = np.split(trainvaldf.sample(frac=1, random_state=42), [int(.8 * len(trainvaldf))])
base_path = '../DatasetTest1/Segmented/Train/'

# Create training and validation dataset with OilPalmSeedsDataset
train_dataset = OilPalmSeedsDataset(traindf, base_path=base_path, transform=transform)
val_dataset = OilPalmSeedsDataset(valdf, base_path=base_path, transform=test_transform)

print('training set', len(train_dataset))
print('val set', len(val_dataset))

# load the testing csv file as dataframe
testdf = pd.read_csv("../CSV/testData.csv")
test_dataset = OilPalmSeedsDataset(testdf, base_path=base_path, transform=transforms.Resize([224, 224]))
print('test set', len(test_dataset))

# Loads datasets into dataloader
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)


# Depp Learning Model
class MVCNN(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(MVCNN, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        # using the initial features from resnet
        fc_in_features = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Sequential(
            # Mainly changed the paramters of the functions here
            # nn.Flatten(),
            nn.Linear(fc_in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
            # Added a logSoftmax for the NLLLoss function
            nn.LogSoftmax(dim=1)
            # nn.Dropout(),
            # nn.Linear(fc_in_features, 2048),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(2048, 2048),
            # nn.ReLU(inplace=True),
            # nn.Linear(2048, num_classes)
        )

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)
        view_features = []
        # pooling
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch)

        pooled_views, _ = torch.max(torch.stack(view_features), 0)
        outputs = self.classifier(pooled_views)
        return outputs


# initializing the model with the number of classes we have
model = MVCNN(num_classes=2, pretrained=True)
# print(model)
# exit()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

if trainBool:
    # Training with Validation
    data_loaders = {'train': train_dataloader, 'val': val_dataloader}
    data_lengths = {'train': len(train_dataset), 'val': len(val_dataset)}


    # Training function
    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
        # To calculate the time taken for training
        start = time.time()

        val_acc_history = []

        # save best model weights for the fine tuning part
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        # training loop
        for epoch in range(1, num_epochs + 1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)
            val_running_loss = 0.0
            val_running_corrects = 0
            running_loss = 0.0
            running_corrects = 0

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                all_preds = []
                all_labels = []
                if phase == 'train':
                    model.train()  # Set model to training mode
                # Iterate over data.
                for data in dataloaders[phase]:
                    inputs = data['image']
                    labels = data['label']
                    inputs, labels = Variable(inputs), Variable(labels)
                    inputs = inputs.type(torch.FloatTensor).to(device)
                    labels = labels.type(torch.LongTensor).to(device)
                    inputs = torch.unsqueeze(inputs, 1)
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        preds = preds.type(torch.FloatTensor).to(device)
                        loss = criterion(input=outputs, target=labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    # Validation part without grad
                    if phase == 'val':
                        model.eval()
                        with torch.no_grad():
                            outputs = model(inputs)
                            maxi, preds = torch.max(outputs, 1)
                            preds = preds.type(torch.FloatTensor).to(device)
                            val_loss = criterion(input=maxi, target=labels)
                            val_acc = torch.sum(preds == labels.data)

                        # statistics
                    if phase == 'val':
                        val_running_loss += val_loss.item()
                        val_running_corrects += val_acc
                        all_preds.append(preds)
                        all_labels.append(labels)
                    else:
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        epoch_loss = running_loss / len(dataloaders[phase])
                        epoch_acc = running_corrects / len(dataloaders[phase])
                        all_preds.append(preds)
                        all_labels.append(labels)
                    if phase == 'val':
                        epoch_loss = val_running_loss / len(dataloaders[phase])
                        epoch_acc = val_running_corrects / len(dataloaders[phase])

                all_labels = torch.cat(all_labels, 0)
                all_preds = torch.cat(all_preds, 0)
                epoch_weighted_acc = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(),
                                                    sample_weight=compute_sample_weight(class_weights,
                                                                                        all_labels.cpu().numpy()))

                print('{} Loss: {:.4f} - Acc: {:.4f} - Weighted Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                    epoch_weighted_acc))

                # save the best weights
                if phase == 'val' and epoch_weighted_acc > best_acc:
                    best_acc = epoch_weighted_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_weighted_acc)

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history


    for param in model.features.parameters():
        param.requires_grad = False

    # initial call to train the classifier
    model.to(device)
    EPOCHS = 50
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.01)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model, val_acc_history = train_model(model=model, dataloaders=data_loaders, criterion=criterion,
                                         optimizer=optimizer, num_epochs=EPOCHS)

    for param in model.parameters():
        param.requires_grad = True

    # Second Call for fine-tuning of the entire network
    EPOCHS = 50
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # We use a smaller learning rate
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    model, val_acc_history = train_model(model=model, dataloaders=data_loaders, criterion=criterion,
                                         optimizer=optimizer,
                                         num_epochs=EPOCHS)
    # saving the model
    torch.save(model.state_dict(), '../Models/mvcnn2.pt')
else:
    # Load existing model
    loaded_dict = torch.load("../Models/mvcnn2.pt", map_location=torch.device('cpu'))
    model.load_state_dict(loaded_dict)
    model.eval()


# Function to get predictions of each seed image
def mvcnn_pred(seed_name, data_dir, model, device):
    transform = test_transform
    seed_fnames = glob.glob(data_dir + f'/* {seed_name}.png')
    seed = torch.stack([transform(Image.open(fname).convert('RGB')) for fname in seed_fnames]).unsqueeze(0)
    seed = seed.to(device)
    pred = torch.nn.functional.softmax(model(seed), dim=1)
    pred = pred.argmax()
    pred = pred.item()
    return pred, {v: k for k, v in class_id_map.items()}[pred]


# Get new predictions from segmented seed images using a seed name
seed_name = 'Seed 2'
print(mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Good seeds - set 10', model, device))

# Iterate through each folder of segmented seeds to find correct and incorrectly classified seeds
correct = 0
incorrect = 0
truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0
for files in glob.iglob(f'../Seed_Segmentation_Classification/Good seeds - set 9'):
    for n in range(1, 9):
        seed_name = f"Seed {n}"
        print(mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Good seeds - set 10', model, device))
        if mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Good seeds - set 9', model, device)[0] == 1:
            correct = correct + 1
            truePos = truePos + 1
        else:
            incorrect = incorrect + 1
            falsePos = falsePos + 1

for files in glob.iglob(f'../Seed_Segmentation_Classification/Good seeds - set 10'):
    for n in range(1, 9):
        seed_name = f"Seed {n}"
        print(mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Good seeds - set 10', model, device))
        if mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Good seeds - set 10', model, device)[0] == 1:
            correct = correct + 1
            truePos = truePos + 1
        else:
            incorrect = incorrect + 1
            falsePos = falsePos + 1

for files in glob.iglob(f'../Seed_Segmentation_Classification/Bad seeds - set 10'):
    for n in range(1, 11):
        seed_name = f"Seed {n}"
        print(mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Bad seeds - set 10', model, device))
        if mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Bad seeds - set 10', model, device)[0] == 0:
            correct = correct + 1
            trueNeg = trueNeg + 1
        else:
            incorrect = incorrect + 1
            falseNeg = falseNeg + 1

for files in glob.iglob(f'../Seed_Segmentation_Classification/Bad seeds - set 11'):
    for n in range(1, 10):
        seed_name = f"Seed {n}"
        print(mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Bad seeds - set 11', model, device))
        if mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Bad seeds - set 11', model, device)[0] == 0:
            correct = correct + 1
            trueNeg = trueNeg + 1
        else:
            incorrect = incorrect + 1
            falseNeg = falseNeg + 1

for files in glob.iglob(f'../Seed_Segmentation_Classification/Bad seeds - set 12'):
    for n in range(1, 12):
        seed_name = f"Seed {n}"
        print(mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Bad seeds - set 12', model, device))
        if mvcnn_pred(seed_name, '../Seed_Segmentation_Classification/Bad seeds - set 12', model, device)[0] == 0:
            correct = correct + 1
            trueNeg = trueNeg + 1
        else:
            incorrect = incorrect + 1
            falseNeg = falseNeg + 1

print(f"Number of correctly classified seeds: {correct}")
print(f"Number of incorrectly classified seeds: {incorrect}")
print(f"Accuracy: {correct / (correct + incorrect)}")

print(trueNeg)
print(truePos)
print(falseNeg)
print(falsePos)

precision = truePos / (truePos + trueNeg)
print(f"Precision: {precision}")

recall = truePos / (truePos + falseNeg)
print(f"Recall: {recall}")

f1 = 2*((precision * recall) / (precision + recall))
print(f"F1: {f1}")



