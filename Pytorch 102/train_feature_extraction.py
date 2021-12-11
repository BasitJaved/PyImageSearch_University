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

