import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time
from PIL import Image
import cv2
import csv
import copy

from collections import OrderedDict
from scipy import spatial
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import models
import torchvision.transforms as transforms

import pickle
from joblib import dump, load

import warnings

trainSet_path = '../DatasetTest1/Train/'
testSet_path = '../DatasetTest1/Test/'

list_good_train_seeds = sorted(os.listdir(trainSet_path + 'GoodSeed/'))
list_bad_train_seeds = sorted(os.listdir(trainSet_path + 'BadSeed/'))

list_good_test_seeds = sorted(os.listdir(testSet_path + 'GoodSeed/'))
list_bad_test_seeds = sorted(os.listdir(testSet_path + 'BadSeed/'))

# Creating CSV files
with open('../CSV/MVCNN/testData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "seed_no", "side", "label"])

    # create test data with the first 200 images of good seeds
    for filename in list_good_test_seeds:
        seedNum = filename.split('_')[1].strip('.jpg')
        seedSide = filename.split('_')[0]
        writer.writerow([filename, seedNum, seedSide, 1])

    # create test data with the first 200 images of bad seeds
    for filename in list_bad_test_seeds:
        seedNum = filename.split('_')[1].strip('.jpg')
        seedSide = filename.split('_')[0]
        writer.writerow([filename, seedNum, seedSide, 0])

with open('../CSV/MVCNN/trainingData.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "seed_no", "side", "label"])

    # create training data with the rest of the images of good seeds
    for filename in list_good_train_seeds:
        seedNum = filename.split('_')[1].strip('.jpg')
        seedSide = filename.split('_')[0]
        writer.writerow([filename, seedNum, seedSide, 1])

    # create training data with the rest of the images of bad seeds
    for filename in list_bad_train_seeds:
        seedNum = filename.split('_')[1].strip('.jpg')
        seedSide = filename.split('_')[0]
        writer.writerow([filename, seedNum, seedSide, 0])


class SeedDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.nb_views = 5
        self.seed_label_map = self._get_seed_label_map(root + '/trainingData.csv')
        self.seed_names = self._get_seed_names(root)
        self.label_encoder = {'Good': 0,
                              'Bad': 1
                              }

    def _get_seed_names(self, root):
        seed_names = [fname.split('_')[1] for fname in os.listdir(root) if fname.endswith('.jpg')]
        seed_names = list(seed_names)
        return seed_names

    def _get_seed_label_map(self, filename):
        reader = csv.DictReader(open(filename))
        seed_label_map = {}
        for row in reader:
            seed_label_map[row['seed_no']] = row['label']
        return seed_label_map

    def __len__(self):
        return len(self.seed_names)

    def _transform(self, image):
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        return transform(image)

    def __getitem__(self, index):
        seed_name = self.seed_names[index]

        # Get Images of the Seed
        seed_fnames = glob.glob(self.root + f'/*_{seed_name}.jpg')
        print(seed_fnames)
        seed = torch.stack([self._transform(Image.open(fname).convert('RGB')) for fname in seed_fnames])
        label = self.seed_label_map[seed_name]
        return seed, label


# CREATE DATASET
root = '../CSV/MVCNN'
dataset = SeedDataset(root)
print(len(dataset))
