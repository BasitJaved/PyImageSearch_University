import numpy as np
from skimage.exposure import is_low_contrast
import argparse
import imutils
import cv2 as cv

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, default='', help='Path to input video')
ap.add_argument('-t', '--threshold', type=float, default=0.35, help='Threshold for low contrast')
args = vars(ap.parse_args())

# grab the path to the input video
print('[INFO] accessing video stream')
vs = cv.VideoCapture(args['input'] if args['input'] else 0)

# loop over the frames
while True:
    # read frame
    (grabbed, frame) = vs.read()

    # if frame was not read then we have reached end of video
    if not grabbed:
        print('[INFO] No frame from video stream-exiting')
        break

    # Grab the frame resize it and convert to grayscale
    frame = imutils.resize(frame, width=450)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # blur the frame slightly and perform edge detection
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blur, 30, 150)

    # initialize text and color to indicate image is not Low contrast
    text = 'Low Contrast: No'
    color = (0, 255, 0)

    # Check to see if image is low contrast
    if is_low_contrast(gray, fraction_threshold=args['threshold']):
        # Need to fine tune this threshold value fo different scenarios
        # update text and color
        text = 'Low Contrast: Yes'
        color = (0, 0, 255)

    # if not low contrast we continue processing it
    else:
        # find the contours in edge map and find the largest one which we will assume is outline of our card
        cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        # draw largest contour on image
        cv.drawContours(frame, [c], -1, (0, 255, 0), 2)

    # draw text on image
    cv.putText(frame, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Stack the output frame and edge map next to each other
    output = np.dstack([edged]*3) # since edge is single channel so here we are converting it into 3 channel like frame
    output = np.hstack([frame, output])

    # show the output
    cv.imshow('Output', output)
    key = cv.waitKey(1) & 0xFF

    # if q key was pressed break the loop
    if key == ord('q'):
        break
vs.release()
cv.destroyAllWindows()
