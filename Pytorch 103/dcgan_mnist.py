from Preprocessing.dcgan import Generator, Discriminator
from sklearn.utils import shuffle
from imutils import build_montages
from torch.optim import Adam
from torch.nn import BCELoss
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import argparse
import torch
import cv2 as cv
import os

# custom weight initialization on generator and discriminator
def weight_init(model):

    #get class name
    class_name = model.__class__.__name__

    # check if class name contains word conv
    if class_name.find('conv') !=-1:
        # initialize the weight from normal distribution
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    # otherwise if the name contains word 'batch_norm'
    elif class_name.find('batch_norm') !=-1:
        # initialize the weights from normal distribution and set the bias to 0
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True, help='Path to output directory')
ap.add_argument('-e', '--epochs', type=int, default=20, help='No. of Epochs to train for')
ap.add_argument('-b', '--batch_size', type=int, default=128, help='Batch Size for training')
args = vars(ap.parse_args())

# store epochs and batch size in convenience veriable
num_epochs = args['epochs']
batch_size = args['batch_size']

# set the device we will be using
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define data transforms
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

# load MNIST dataset and stack training and testing data so we have additional training data
print('[INFO] loading MNIST dataset...')
train_data = MNIST(root='data', train=True, download=True, transforms=data_transforms)
test_data = MNIST(root='data', train=False, download=True, transforms=data_transforms)
data = torch.utils.data.ConcatDataset((train_data, test_data))

# initialize our data loader
data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)

# calculate steps per Epochs
steps_per_epochs = len(data_loader.dataset)//batch_size

# build the generator initialize its weights and flush it into current device
print('[INFO] building generator...')
gen = Generator(input_dim=100, output_dim=512, output_channels=1)
gen.apply(weight_init)
gen.to(device)

# build the Discriminator initialize its weights and flush it into current device
print('[INFO] building generator...')
dis = Discriminator(depth=1)
dis.apply(weight_init)
dis.to(device)

# initialize optimizer for both generator and discriminator
gen_optim = Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002/num_epochs)
dis_optim = Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002/num_epochs)

# initialize BCE loss function
criterion = BCELoss()

# randomly generate some benchmark noise so we can consistantly visualize how the generative modeling is learning
print('[INFO] starting training...')
benchmark_noise = torch.randn(256, 100, 1, 1, device=device)

# define real and fake label values
real_label = 1
fake_label = 0

# loop over the epochs
for epochs in range(num_epochs):

    # show epoch information and compute number of batches per epochs
    print(f'[INFO] starting epoch {epochs+1} of {num_epochs}')