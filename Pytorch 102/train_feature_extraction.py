from Preprocessing import config
from Preprocessing import create_data_loader
from imutils import paths
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

# define augmentation piplines
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor()
    transforms.Normalize(mean=config.mean, std=config.std)
])

val_transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor()
    transforms.Normalize(mean=config.mean, std=config.std)
])

# create data loader
(train_ds, train_loader) = create_data_loader.get_data_loader(config.train_path, transforms=train_transform,
                                                              batch_size=config.feature_extraction_batch_size)
(val_ds, val_loader) = create_data_loader.get_data_loader(config.val_path, transforms=val_transform,
                                                              batch_size=config.feature_extraction_batch_size,
                                                          shuffle=False)

# load up resnet50 model
model = resnet50(pretrained=True)

# since we are using Resnet50 model as a feature extractor we set its parameters to non-trainable
for param in model.parameters():
    param.requires_grad = False

# append a new classification top to our feature extractor and pop it on to the current device
model_output_feats = model.fc.in_features
model.fc = nn.Linear(model_output_feats, len(train_ds.classes))
model = model.to(config.device)

# initialize loss function and classifier
loss_func = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.fc.parameters(), lr = config.lr)

# calculate steps per epoch for training and validation set
train_step = len(train_ds//config.feature_extraction_batch_size)
val_step = len(val_ds//config.feature_extraction_batch_size)

# initialize a dictionary to store training history
H = {'training_loss': [],
     'train_acc' : [],
     'val_loss' : [],
     'val_acc' : []}

# loop over epochs
print('[INFO] training the network ...')
start_time = time.time()
for e in tqdm(range(config.epochs)):

    # set model to training mode
    model.train()

    # initialize total training and validation loss
    total_train_loss = 0
    total_val_loss = 0

    # initialize number of correct predictions in training and validation set
    train_correct = 0
    val_correct = 0

    # loop over the training set
    for (i, (x, y)) in enumerate(train_loader):
        # send inout to device
        (x, y) = (x.to(config.device), y.to(config.device))

        # perform forward pass and calculate training loss
        pred = model(x)
        loss = loss_func(pred, y)

        # calculate the gradient
        loss.backward()

        # check if we are updating model parameters if so update them and zero out previously accumulated gradient
        if (i + 2) % 2 == 0:
            opt.step()
            opt.zero_grad()

        # add the loss to total training loss so far and calculate total number of correct predictions
        total_train_loss +=loss
        train_correct +=(pred.argmax(1) == y).type(torch.float).sum().item()