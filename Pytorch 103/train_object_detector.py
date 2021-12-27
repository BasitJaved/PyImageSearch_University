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