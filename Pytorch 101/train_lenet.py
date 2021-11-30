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
train_steps = len(train_data_loader.dataset)//batch_size
val_steps = len(val_data_loader.dataset)//batch_size

# initialize LeNet Model
print('[INFO] Initializing LeNet Model...')
model = LeNet(num_channels=1, classes=len(train_data.dataset.classes)).to(device)

# initialize optimizer and loss function
opt = Adam(model.parameters(), lr = Init_lr)
loss_fn = nn.NLLLoss()        # log_soft_max + nn.NLLLoss() = categorical cross entropy

# initialize a dictionary to store training history
H = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# measure how log training is going to take
print('[INFO] training network...')
start_time = time.time()

# loop over the epochs
for e in range (0, epochs):
    # set model in training mode
    model.train()

    # initialize total training and validation loss
    total_train_loss = 0
    total_val_loss = 0

    # initialize number of correct predictions in training and validation step
    train_correct = 0
    val_correct = 0

    # loop over the training set
    for (x, y) in train_data_loader:
        # send input to device
        (x, y) = (x.to(device), y.to(device))

        # perform forward pass and calculate training loss
        pred = model(x)
        loss = loss_fn(pred, y)

        # zero out the gradient, perform backpropagation step and update weights
        opt.zero_grad()
        loss.backward()
        opt.step()

        # add the loss to total training loss so far and calculate the number of correct predictions
        total_train_loss += loss
        train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # swtich off auto grad for evaluation
    with torch.no_grad():
        # set model in evaluation mode
        model.eval()

        # loop over the validation set
        for (x, y) in val_data_loader:
            # send input to device
            (x, y) = (x.to(device), y.to(device))

            # make predictions and calculate validation loss
            pred = model(x)
            total_val_loss += loss_fn(pred, y)

            # calculate number of correct predictions
            val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # calculate average training and validation loss
    avg_train_loss = total_train_loss/train_steps
    avg_val_loss = total_val_loss/val_steps

    # calculate the training and validation accuracy
    train_correct = train_correct/len(train_data_loader.dataset)
    val_correct = val_correct/len(val_data_loader.dataset)

    # update training history
    H['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    H['train_acc'].append(train_correct)
    H['val_loss'].append(avg_val_loss.cpu().detach().numpy())
    H['val_acc'].append(val_correct)

    # print model training and validation information
    print(f'[INFO] Epoch: {e+1}/{epochs}')
    print(f'Train loss: {avg_train_loss}, Train Accuracy: {train_correct}')
    print(f'Val loss: {avg_val_loss}, Val Accuracy: {val_correct}')

# finish measuring how long training took
end_time = time.time()
print(f'[INFO] Total time taken to train model: {end_time-start_time}s')

# We can now evaluate network on test set
print('[INFO] evaluating network...')

# turn off auto grad for testing evaluation
with torch.no_grad():
    # set model in evaluation mode
    model.eval()

    # initialize a list to store predictions
    preds = []

    # loop over the test set
    for (x, y) in test_data_loader:
        # send input to device
        x = x.to(device)

        # make predictions and add them to list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

# generate a clssification report
print(classification_report(test_data.targets.cpu().numpy(), np.array(preds), target_names=test_data.classes))

# plot training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(H['train_loss'], label='train_loss')
plt.plot(H['val_loss'], label='val_loss')
plt.plot(H['train_acc'], label='train_acc')
plt.plot(H['val_acc'], label='val_acc')
plt.title('Training loss and accuracy on dataset')
plt.xlabel('Epoch#')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig(args['plot'])

# serialize model to disk
torch.save(model, args['model'])
