from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2 as cv

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, required=True, help='Path to the input image')
ap.add_argument('-m', '--model', type=str, default='frcnn-mobilenet',
                choices=['frcnn-resnet', 'frcnn-mobilenet', 'retinanet'], help='Name of Object detection Model')
ap.add_argument('-l', '--labels', type=str, default='coco_classes.pickle',
                help='Path to file containing list of categories in COCO dataset')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='Minimum probability to filter weak detections')
args = vars(ap.parse_args())

# select the device we will be using to run the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load a list of categories in coco dataset and then generate a set of bounding box colors for each class
classes = pickle.loads(open(args['labels'], 'rb').read())
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# initialize a dictionary containing model name and its corresponding torchvision function call
models = {
    'frcnn-resnet': detection.fasterrcnn_resnet50_fpn,
    'frcnn-mobilenet': detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    'retinanet': detection.retinanet_resnet50_fpn
}

# load model and set it to evaluation mode
model = models[args['model']](pretrained=True, progress=True, num_classes=len(classes),
                              pretrained_backbone=True).to(device)
model.eval()

# load image from disk
image = cv.imread(args['image'])
orig = image.copy()

# convert the image from BGR to RGB channel ordering and change image from channel last to channel first ordering
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))

# add the batch dimension. scale raw pixel intensities to the range [0, 1], and convert image to a floating point tensor
image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)

# send the input to device and pass it through the network to get the detections and predictions
image = image.to(device)
detections = model(image)[0]
print(detections)

#  loop over the detections
for i in range(0, len(detections['boxes'])):
    # extract the confidence associated with predictions
    confidence = detections['scores'][i]

    # filter weak detections by ensuring the confidence is greater then minimum confidence
    if confidence > args['confidence']:
        # extract the index of class label from detections then compute (x, y) coordinates of bounding box for object
        idx = int(detections['labels'][i])
        box = detections['boxes'][i].detach().cpu().numpy()
        (startX, startY, endX, endY) = box.astype('int')

        # display predictions to our terminal
        label = f'{classes[idx]}: {confidence*100:.2f}%'
        print(f'[INFO] {label}')

        # draw bounding box and label on image
        cv.rectangle(orig, (startX, startY), (endX, endY), colors[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv.putText(orig, label, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

cv.imshow('output', orig)
cv.waitKey(0)
