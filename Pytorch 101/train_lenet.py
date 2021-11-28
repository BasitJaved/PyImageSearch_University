# set matplotlib so figures can be saved in background
import matplotlib
matplotlib.use('Agg')

from Preprocessing.lenet import LeNet
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import time

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, required=True, help='Path to output trained model')
ap.add_argument('-p', '--plot', type=str, required=True, help='Path to output loss/accuracy plot')
args = vars(ap.parse_args())

# define training hyperparameters
Init_lr = 1e-3
batch_size = 64
epochs = 10

# define train and validation splits
train_split = 0.75
val_split = 1 - train_split

# set device we will be using to train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load KMNIST dataset
print('[INFO] Loading KMNIST dataset...')
train_data = KMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = KMNIST(root='data', train=False, download=True, transform=ToTensor())

# Generate Train/Validation split
print('[INFO] Generating train/validation split...')
num_train_samples = int(len(train_data)*train_split)
num_val_samples = int(len(train_data)*val_split)
(train_data, val_data) = random_split(train_data, [num_train_samples, num_val_samples],
                                      generator=torch.Generator().manual_seed(42))

# initialize train, validation and test data loaders
train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_data_loader = DataLoader(val_data, batch_size=batch_size)
test_data_loader = DataLoader(test_data, batch_size=batch_size)

# calculate steps per epochs for training and validation set
train_steps = len(train_data_loader)//batch_size
val_steps = len(val_data_loader)//batch_size

# initialize LeNet Model
print('[INFO] Initializing LeNet Model...')
model = LeNet(num_channels=1, classes=len(train_data.dataset.classes)).to(device)

# initialize optimizer and loss function
opt = Adam(model.parameters(), lr = Init_lr)
loss_fn = nn.NLLLoss()