from Preprocessing import config
from torchvision import transforms
import mimetypes
import argparse
import imutils
import pickle
import torch
import cv2 as cv

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', 'input', required=True, help='Path to input image/text file of image paths')
args = vars(ap.parse_args())

# determine the input file type but assume that we are working with single input image
file_type = mimetypes.guess_type(args['input'])[0]
image_paths = [args['input']]

# if file type is text file then we need to process multiple images
if 'test/plain' == file_type:
    # load the image paths in out testing file
    image_paths = open(args['input']).read().strip().split('\n')

# load our label encoder,  object detector from disk and set it in evaluation mode
print('[INFO] loading object detector...')
model = torch.load(config.model_path).to(config.device)
model.eval()
le = pickle.loads(open(config.le_path, 'rb').read())

# define normalization transforms
transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.mean, std=config.std)
])

# loop over the images that we will be testing using our bounding box regression model
for image_path in image_paths:
    # load the image, copy it, swap its color channels, resize it and bring its channel dimension forward
    image = cv.imread(image_path)
    orig = image.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224))
    image = image.transpose((2, 0, 1))

    # convert image to pytorch tensor, normalize it, flash it to the current device and add a batch dimension
    image = torch.from_numpy(image)
    image = transforms(image).to(config.device)
    image = image.unsqueeze(0)

    # predict the bounding box of object along with the class label
    (box_preds, label_preds) = model(image)
    (startX, startY, endX, endY) = box_preds

    # determine class label with largest predicted probability
    label_preds = torch.nn.Softmax(dim=-1)(label_preds)
    i = label_preds.argmax(dim=-1).cpu()
    label = le.inverse_transform(i)[0]

    # resize the original image such that it fits on screen and grab its dimensions
    orig = imutils.resize(orig, width=600)
    (h, w) = orig.shape[:2]

    # scale the predicted bounding box coordinates based on image dimensions
    startX = int(startX*w)
    startY = int(startY*h)
    endX = int(endX*w)
    endY = int(endY*h)

    # draw predicted bounding box and class labels on image
    y = startY-10 if startY-10>10 else startY+10
    cv.putText(orig, label, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv.rectangle(orig, (startX, startY), (endX, endY), (0,255,0), 2)

    # show output image
    cv.imshow('output', orig)
    cv.waitKey(0)
