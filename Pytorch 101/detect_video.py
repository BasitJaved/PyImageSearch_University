from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2 as cv

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', type=str, default='frcnn-mobilenet',
                choices=['frcnn-resnet', 'frcnn-mobilenet', 'retinanet'], help='Name of Object detection Model')
ap.add_argument('-l', '--labels', type=str, default='coco_classes.pickle',
                help='Path to file containing list of categories in COCO dataset')
ap.add_argument('-c', '--confidence', type=float, default=0.5, help='Minimum probability to filter weak detections')
args = vars(ap.parse_args())

# set device we will  use to run the model
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

# initialize the vide stream, allow the camera sensor to warm up and initialize FPS counter
print('[INFO] starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from video stream
while True:
    # grab the frame from threaded video stream and resize it to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    orig = frame.copy()

    # conver the frame from BGR to RGB channels ordering and change from from channels last to channels first ordering
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))

    # add a batch dimension, scale the raw pixel intensities to the range [0, 1] and convert frame to a
    # floating point tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame/255.0
    frame = torch.FloatTensor(frame)

    # send the input to device and pass it through the model to get detections and predictions
    frame = frame.to(device)
    detections = model(frame)[0]

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
            label = f'{classes[idx]}: {confidence * 100:.2f}%'

            # draw bounding box and label on image
            cv.rectangle(orig, (startX, startY), (endX, endY), colors[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv.putText(orig, label, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv.imshow('Frame', orig)
    key = cv.waitKey(1)

    # if q is pressed breat the loop
    if key == ord('q'):
        break

    # update FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print(f'[INFO] elapsed time: {fps.elapsed():.2f}')
print(f'[INFO] Approx. FPS: {fps.fps():.2f}')

cv.destroyAllWindows()
vs.stop()