import csv

import numpy as np
import torch
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

import os
import pandas as pd
from torchvision.io import read_image
import torch.optim as optim

# List of file names
trainSet_path = 'DatasetTest1/Train'
testSet_path = 'DatasetTest1/Test'

list_good_train_seeds = sorted(os.listdir(trainSet_path + '/GoodSeed/'))
list_bad_train_seeds = sorted(os.listdir(trainSet_path + '/BadSeed'))

list_good_test_seeds = sorted(os.listdir(testSet_path + '/GoodSeed'))
list_bad_test_seeds = sorted(os.listdir(testSet_path + '/BadSeed'))

# Creating CSV files
with open('CSV/testData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "label"])

    # create test data with the first 200 images of good seeds
    for filename in list_good_test_seeds:
        writer.writerow([filename, 1])

    # create test data with the first 200 images of bad seeds
    for filename in list_bad_test_seeds:
        writer.writerow([filename, 0])

with open('CSV/trainingData.csv', 'w', newline='') as file:
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
        image = read_image(img_path)
        # print(image.shape)
        # image = transforms.ToTensor() # convert ndarray to tensor
        # label = transforms.ToTensor()

        # if any transformation is needed, e.g., to resize the image
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        sample = {"image": image, "label": label}
        return sample


# Note that the following transforms are not necessarily applicable to the raw images in our oil palm seeds dataset
# You will have to do something more appropriate and relevant
transform = transforms.Compose(
    [transforms.Resize([256, 256]),
     transforms.RandomCrop(224),
     transforms.RandomHorizontalFlip(),
     # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # mean, variance
     # transforms.ToTensor()
     ])

test_transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     # transforms.RandomCrop(224),
     # transforms.RandomHorizontalFlip(),
     # transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # mean, variance
     # transforms.ToTensor()
     ])

# load the training csv file in terms of annotations to dataframe and randomly split it to training and validation sets respectively
trainvaldf = pd.read_csv("CSV/trainingdata.csv")
traindf, valdf = np.split(trainvaldf.sample(frac=1, random_state=42), [int(.8 * len(trainvaldf))])
base_path = 'DatasetTest1/Train'

# Create training and validation dataset with OilPalmSeedsDataset
train_dataset = OilPalmSeedsDataset(traindf, base_path=base_path, transform=transform)
val_dataset = OilPalmSeedsDataset(valdf, base_path=base_path, transform=transform)

print('training set', len(train_dataset))
print('val set', len(val_dataset))

# load the testing csv file as dataframe
testdf = pd.read_csv("CSV/testData.csv")
test_dataset = OilPalmSeedsDataset(testdf, base_path=base_path, transform=transforms.Resize([224, 224]))
print('test set', len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=0)


class myCNN(nn.Module):
    def __init__(self, input_img_size):
        super(myCNN, self).__init__()
        self.input_img_size = input_img_size  # the input size of a single image (C,H,W)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        # we use the maxpool multiple times, but define it once
        self.pool = nn.MaxPool2d(2, 2)
        # in_channels = 6 because self.conv1 output 6 channel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # The input number to the first fully connected layer is a bit tricky
        # We can run through a dummy forward pass to obtain the input number to the first fully connected layer
        dummydata = torch.rand(1, *self.input_img_size)
        dummydata = self.pool(F.relu(self.conv1(dummydata)))
        dummydata = self.pool(F.relu(self.conv2(dummydata)))
        self.num_features_2fc = dummydata.shape[0] * dummydata.shape[1] * dummydata.shape[2] * dummydata.shape[
            3]  # there are clever ways but I don't recall them and need to search up
        self.fc1 = nn.Linear(in_features=self.num_features_2fc, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_features_2fc)
        # x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # no activation on final layer
        return F.log_softmax(x)


model = myCNN(train_dataset[0]['image'].shape)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))
model.to(device)


def initialize_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_uniform_(model.weight.data, nonlinearity='relu')
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight.data, 1)
        nn.init.constant_(model.bias.data, 0)
    elif isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight.data)
        nn.init.constant_(model.bias.data, 0)


model.apply(initialize_weights)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

data_lengths = {'train': len(train_dataset), 'val': len(val_dataset)}
print(data_lengths)

from torch.autograd import Variable

# Training with Validation
n_epochs = 10  # just to test the code
data_loaders = {'train': train_dataloader, 'val': val_dataloader}
data_lengths = {'train': len(train_dataset), 'val': len(val_dataset)}

for epoch in range(n_epochs):
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            # optimizer = scheduler(optimizer, epoch)
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0

        # Iterate over data.
        for data in data_loaders[phase]:

            # get the input images and their corresponding labels
            images = data['image']
            labels = data['label']

            # wrap them in a torch Variable
            images, labels = Variable(images), Variable(labels)

            # convert variables to floats for regression loss
            labels = labels.type(torch.LongTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)

            # forward pass to get outputs
            output_labels = model(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_labels, labels)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward + optimize only if in training phase
            if phase == 'train':
                loss.backward()
                # update the weights
                optimizer.step()

            # print loss statistics
            running_loss += loss.item()

        epoch_loss = running_loss / data_lengths[phase]
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))

for param in model.state_dict():
    print(param, "\t", model.state_dict()[param].size())

fname = 'Models/myModel.pkl'
torch.save(model.state_dict(), fname)

loaded_dict = torch.load(fname)
model.load_state_dict(loaded_dict)
model.eval()

total = 0  # keeps track of how many images we have processed
correct = 0  # keeps track of how many correct images our model predicts
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        images = data['image']
        labels = data['label']

        # wrap them in a torch Variable
        images, labels = Variable(images), Variable(labels)

        # convert variables to floats for regression loss
        labels = labels.type(torch.LongTensor).to(device)
        images = images.type(torch.FloatTensor).to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size()[0]
        correct += (predicted == labels).sum().item()

print("Accuracy: ", correct / total)
