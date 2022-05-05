import copy
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time

import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_sample_weight
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Lambda
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import models

from torchvision.io import read_image

trainSet_path = '../DatasetTest1/Train'
testSet_path = '../DatasetTest1/Test'

list_good_train_seeds = sorted(os.listdir(trainSet_path + '/GoodSeed'))
list_bad_train_seeds = sorted(os.listdir(trainSet_path + '/BadSeed'))

list_good_test_seeds = sorted(os.listdir(testSet_path + '/GoodSeed'))
list_bad_test_seeds = sorted(os.listdir(testSet_path + '/BadSeed'))


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
class_id_map = {'Bad Seed': 0,
                'Good Seed': 1}

h = nn.Sigmoid()

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

        # images = np.array(seed_segment(img_path), dtype="object")

        # for image in images:
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {"image": image, "label": label}
        return sample


# Note that the following transforms are not necessarily applicable to the raw images in our oil palm seeds dataset
# You will have to do something more appropriate and relevant
transform = transforms.Compose(
    [
        # transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor()

    ])

test_transform = transforms.Compose(
    [
        # transforms.ToPILImage(),
        transforms.Resize([224, 224]),
        transforms.ToTensor()

    ])

# load the training csv file in terms of annotations to dataframe and randomly split it to training and validation sets respectively
trainvaldf = pd.read_csv("../CSV/trainingdata.csv")
traindf, valdf = np.split(trainvaldf.sample(frac=1, random_state=42), [int(.8 * len(trainvaldf))])
base_path = '../DatasetTest1/Train/'

# Create training and validation dataset with OilPalmSeedsDataset
train_dataset = OilPalmSeedsDataset(traindf, base_path=base_path, transform=transform)
val_dataset = OilPalmSeedsDataset(valdf, base_path=base_path, transform=transform)

print('training set', len(train_dataset))
print('val set', len(val_dataset))

# load the testing csv file as dataframe
testdf = pd.read_csv("../CSV/testData.csv")
test_dataset = OilPalmSeedsDataset(testdf, base_path=base_path, transform=transforms.Resize([224, 224]))
print('test set', len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)


class MVCNN(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(MVCNN, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        fc_in_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_in_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)
        view_features = []
        for view_batch in inputs:
            view_batch = self.features(view_batch)
            view_batch = view_batch.view(view_batch.shape[0], view_batch.shape[1:].numel())
            view_features.append(view_batch)

        pooled_views, _ = torch.max(torch.stack(view_features), 0)
        outputs = self.classifier(pooled_views)
        return outputs


model = MVCNN(num_classes=2, pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

from torch.autograd import Variable

# Training with Validation
n_epochs = 10  # just to test the code
data_loaders = {'train': train_dataloader, 'val': val_dataloader}
data_lengths = {'train': len(train_dataset), 'val': len(val_dataset)}


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            # Iterate over data.
            for data in dataloaders[phase]:
                inputs = data['image']
                labels = data['label']
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.type(torch.FloatTensor).to(device)
                labels = labels.type(torch.FloatTensor).to(device)
                inputs = torch.unsqueeze(inputs, 1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    print(outputs)
                    exit()
                    _, preds = torch.max(outputs, 1)
                    # labels = torch.Tensor([0, 1]).to(torch.long).to(device)
                    # labels = torch.unsqueeze(labels, 2)
                    loss = criterion(input=h(_), target=labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.append(preds)
                all_labels.append(labels)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase])
            all_labels = torch.cat(all_labels, 0)
            all_preds = torch.cat(all_preds, 0)
            epoch_weighted_acc = accuracy_score(all_labels.cpu().numpy(), all_preds.detach().numpy(),
                                                sample_weight=compute_sample_weight(class_weights,
                                                                                    all_labels.cpu().numpy()))
            # epoch_weighted_acc = accuracy_score(all_labels.detach().numpy(), all_preds.detach().numpy(),
            #                                     sample_weight=compute_sample_weight(class_weights,
            #                                                                         all_labels.detach().numpy()))
            print('{} Loss: {:.4f} - Acc: {:.4f} - Weighted Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc,
                                                                                epoch_weighted_acc))

            # deep copy the model
            if phase == 'val' and epoch_weighted_acc > best_acc:
                best_acc = epoch_weighted_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_weighted_acc)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


for param in model.features.parameters():
    param.requires_grad = False

model.to(device)
EPOCHS = 30
weight = torch.tensor([0.5, 0.5]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0005)

model, val_acc_history = train_model(model=model, dataloaders=data_loaders, criterion=criterion,
                                     optimizer=optimizer, num_epochs=EPOCHS)

for param in model.parameters():
    param.requires_grad = True

EPOCHS = 30
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)  # We use a smaller learning rate

model, val_acc_history = train_model(model=model, dataloaders=data_loaders, criterion=criterion, optimizer=optimizer,
                                     num_epochs=EPOCHS)
