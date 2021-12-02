from Preprocessing import config
from torchvision import models
import numpy as np
import argparse
import torch
import cv2 as cv

def preprocess_image(image):
    # swap color channels from BGR to RGB, resize it and scale pixel values to [0, 1] range
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (config.image_size, config.image_size))
    image = image.astype('float32')/255.0

    # subtract the imagenet mean, divide by imagenet standard deviation, set channel_first ordering and add
    # a batch dimension
    image -= config.mean
    image /= config.std
    image = np.transpose(image, (2, 0, 1)) # pytorch expects channels first ordering
    image = np.expand_dims(image, 0)

    return image

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
ap.add_argument('-m', '--model', type=str,
                default='vgg16',choices=['vgg16', 'vgg19', 'inception', 'densenet', 'resnet'],
                help='Name of pre-trained network to use')
args = vars(ap.parse_args())

# define a dictionary that maps model names to their classes inside torchvision
Models = {
    'vgg16': models.vgg16(pretrained=True),
    'vgg19': models.vgg19(pretrained=True),
    'inception': models.inception_v3(pretrained=True),
    'densenet': models.densenet121(pretrained=True),
    'resnet': models.resnet50(pretrained=True),
}

# load network weights from disk, flash it to the current device and set it to evaluation mode
print(f'[INFO] loading {args["model"]} ...')
model = Models[args['model']].to(config.device)
model.eval()

# load the image from disk, clone and preprocess it
print('[INFO] loading image...')
image = cv.imread(args['image'])
orig = image.copy()
image = preprocess_image(image)

# convert the preprocess image to torch tensor and flash it to the device
image = torch.from_numpy(image)
image = image.to(config.device)

# load the preprocess imagenet labels
print('[INFO] loading Imagenet labels...')
imagenet_labels = dict(enumerate(open(config.in_labels)))

# classify the images and extract the labels
print(f'[INFO] classifying image with {args["model"]} ...')
logits = model(image)
probabilities = torch.nn.Softmax(dim=-1)(logits)
sorted_probabilities = torch.argsort(probabilities, dim=-1, descending=True)

# loop over the predictions and display the rank-5 predictions and corresponding probabilities to terminal
for (i, idx) in enumerate(sorted_probabilities[0, :5]):
    print(f"{i} . {imagenet_labels[idx.item()].strip()}: {probabilities[0, idx.item()]*100}%")

# draw predictions on image and show image
(label, prob) = (imagenet_labels[probabilities.argmax().item()], probabilities.max().item())
cv.putText(orig, f'Label: {label.strip()}, {prob*100}%', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv.imshow('Classification', orig)
cv.waitKey(0)