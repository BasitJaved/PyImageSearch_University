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
train_data = MNIST(root='data', train=True, download=True, transform=data_transforms)
test_data = MNIST(root='data', train=False, download=True, transform=data_transforms)
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

    # initiate current epoch loss for generator and discriminator
    epoch_loss_g = 0
    epoch_loss_d = 0

    for x in data_loader:
        # zero out discriminator gradient
        dis.zero_grad()

        # grab the images and send them to device
        images = x[0]
        images = images.to(device)

        # get the batch size and create a labels tensor
        bs = images.size(0)
        labels = torch.full((bs,), real_label, dtype=torch.float, device=device)

        # forward pass through discriminator
        output = dis(images).view(-1)

        # calculate loss on all-real batch
        error_real = criterion(output, labels)

        # calculate gradient by performing a backward pass
        error_real.backward()

        # randomly generate noise for the generator to predict on
        noise = torch.randn(bs, 100, 1, 1, device=device)

        # generate a fake image batch using generator
        fake = gen(noise)
        labels.fill_(fake_label)

        # perform a forward pass through discriminator using fake data
        output = dis(fake.detach()).view(-1)
        error_fake = criterion(output, labels)

        # calculate gradient by performing a backward pass
        error_fake.backward()

        # compute the error for discriminator and update it
        error_d = error_real + error_fake
        dis_optim.step()

        # set all the generator gradient to zero
        gen.zero_grad()

        # update the labels as fake labels are real for generator and perform a forward pass of fake data through
        # discriminator
        labels.fill_(real_label)
        output = dis(fake).view(-1)

        # calculate generator's loss based on output from discriminator and calculate gradients for generator
        error_g = criterion(output, labels)
        error_g.backward()

        # update the generator
        gen_optim.step()

        # add the current iteration loss of discriminator and generator
        epoch_loss_d += error_d
        epoch_loss_g += error_g

        # display training information to disk
        print(f'[INFO] Generator loss: {epoch_loss_g/steps_per_epochs:.4f}, '
              f'Discriminator loss: {epoch_loss_d/steps_per_epochs:.4f}')

        # check to see if we should visualize the output of generator model on our benchmark data
        if (epochs+1)%2 == 0:
            # Set the generator in evaluation mode, make prediction on benchmark noise, scale it back to
            # range [0, 255], and generate the montage
            gen.eval()
            images = gen(benchmark_noise)
            images = images.detach().cpu().numpy().transpose((0,2,3,1))
            images = ((images*127.5)+127.5).astype('uint8')
            images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28, 28), (16, 16))[0]

            # build the output path and write the visualization to disk
            p = os.path.join(args['output'], f'output_{str(epochs+1).zfill(4)}.png')
            cv.imwrite(p, vis)

            # set the generator to training mode
            gen.train()