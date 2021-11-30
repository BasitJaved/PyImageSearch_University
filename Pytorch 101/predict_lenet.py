import numpy as np
np.random.seed(42)

from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision.transforms import ToTensor
from torchvision.datasets import KMNIST
import argparse
import imutils
import torch
import cv2 as cv

# costruct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, required=True, help='path to trained Pytorch model')
args = vars(ap.parse_args())

# set the device we will be using to train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load KMNIST dataset and randomly grab 10 data points
print('[INFO] loading KMNIST dataset...')
test_data = KMNIST(root='data', train=False, download=True, transform=ToTensor())
idxs = np.random.choice(range(0, len(test_data)), size=(10,))
test_data = Subset(test_data, idxs)

# initialize the test data loader
test_data_loader = DataLoader(test_data, batch_size=1)

# load model and set it to evaluation mode
model = torch.load(args['model']).to(device)
model.eval()

# switch off auto grad
with torch.no_grad():
    # loop over the test set
    for (image, label) in test_data_loader:
        #grab original image and ground truth label
        original_image = image.numpy().squeeze(axis=(0,1))
        get_label = test_data.dataset.classes[label.numpy()[0]]

        # send the input to device and make predictions on it
        image = image.to(device)
        pred = model(image)

        # find the class label index with largest corresponding probabilities
        idx = pred.argmax(axis=1).cpu().numpy()[0]
        pred_label = test_data.dataset.classes[idx]

        # convert the image from grayscale to RGB (to draw on it) and resize it
        original_image = np.dstack([original_image]*3)
        original_image = imutils.resize(original_image, width=128)

        # draw predictions and class label on it
        color = (0, 255, 0) if get_label==pred_label else (0, 0, 255)
        cv.putText(original_image, get_label, (2, 25), cv.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

        # display the results in terminal and show input image
        print(f'[INFO] ground truth label: {get_label}, predicted label: {pred_label}')
        cv.imshow('image', original_image)
        cv.waitKey(0)


