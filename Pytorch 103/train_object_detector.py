from Preprocessing.bbox_regressor import object_detector
from Preprocessing.custom_tensor_dataset import custom_tensor_dataset
from Preprocessing import config
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchvision.models import resnet50
from sklearn.model_selection import train_test_split
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pickle
import torch
import time
import os

# initialize the list of data (images), class labels, target bounding box coordinates, and image paths
print('[INFO] loading dataset...')
data = []
labels = []
bboxes = []
image_paths= []

# loop over all csv files in annotations directory
for csv_path in paths.list_files(config.annots_path, validExts=('.csv')):
    # load the contents of current csv annotation file
    rows = open(csv_path).read().strip().split('\n')

    # loop over the rows
    for row in rows:
        # break the row into filename, bounding box coordinates and class label
        row = row.split(',')
        (file_name, startX, startY, endX, endY, label) = row

        # create the path for the image and load image
        image_path = os.path.sep.join([config.images_path, label, file_name])
        image = cv.imread(image_path)
        (h, w) = image.shape[:2]

        # scale the bounding box coordinates relative to the spatial dimensions of input image
        startX = float(startX)/w
        startY = float(startY)/h
        endX = float(endX) / w
        endY = float(endY) / h

        # preprocessthe image
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (224, 224))

        # update our list of  data, class labels, bounding boxes and image paths
        data.append(image)
        labels.append(label)
        bboxes.append((startX, startY, endX, endY))
        image_paths.append(image_path)

# convert the data, class labels, bounding boxes and image paths to numpy arrays
data = np.array(data, dtype='float32')
labels = np.array(labels)
bboxes = np.array(bboxes, dtype='float32')
image_paths = np.array(image_paths)

# perform label encoding on labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# partition the data into training and testing splis using 80% of data for training and remaining for testing
split = train_test_split(data, labels, bboxes, image_paths, test_size=0.20, random_state=42)

# unpack the data split
(train_images, test_images) = split[:2]
(train_labels, test_labels) = split[2:4]
(train_bboxes, test_bboxes) = split[4:6]
(train_paths, test_paths) = split[6:]

# convert numpy arrays into pytorch tensors
(train_images, test_images) = torch.tensor(train_images), torch.tensor(test_images)
(train_labels, test_labels) = torch.tensor(train_labels), torch.tensor(test_labels)
(train_bboxes, test_bboxes) = torch.tensor(train_bboxes), torch.tensor(test_bboxes)

# define normalize transform
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.mean, std=config.std)
])

# create pytorch datasets
train_ds = custom_tensor_dataset((train_images, train_labels, train_bboxes), transforms=transforms)
test_ds = custom_tensor_dataset((test_images, test_labels, test_bboxes), transforms=transforms)
print(f'[INFO] Total training Samples: {len(train_ds)}')
print(f'[INFO] Total testing Samples: {len(test_ds)}')

# calculate steps per epoch for training and validation set
train_steps = len(train_ds)//config.batch_size
val_steps = len(test_ds)//config.batch_size

# create data loader
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=os.cpu_count(),
                          pin_memory=config.pin_memory)
test_loader = DataLoader(test_ds, batch_size=config.batch_size, num_workers=os.cpu_count(),
                          pin_memory=config.pin_memory)
# write the testing image paths to disk so that we can use them when evaluating/testing our object detector
print('[INFO] Saving testing image paths...')
f = open(config.test_path, 'w')
f.write('\n'.join(test_paths))
f.close()

# load resnet 50 network
resnet = resnet50(pretrained=True)

# freeze all layers so that they will not be updated during training process
for param in resnet.parameters():
    param.requires_grad=False

# create our custom object detector model and flash it to device
object_detect = object_detector(resnet, len(le.classes_))
object_detect = object_detect.to(config.device)

# define loss function
class_loss_function = CrossEntropyLoss()
bbox_loss_function = MSELoss()

# initialize the optimizer, compile the modle and show model summary
opt = Adam(object_detect.parameters(), lr=config.init_lr)
print(object_detect)

# initialize a dictionary to store training history
H = {'total_training_loss': [],
     'total_val_loss': [],
     'train_class_acc': [],
     'val_class_acc': []
     }

