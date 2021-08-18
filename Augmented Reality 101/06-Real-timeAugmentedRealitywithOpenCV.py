from augmented_reality import find_and_wrap
from imutils.video import VideoStream
from collections import deque
import argparse
import imutils
import cv2 as cv
import time

# construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', type=str, required=True, help='Path to input video file fro augmented reality')
ap.add_argument('-c', '--cache', type=int, default=-1, help='Whether or not to use cache reference points')
args = vars(ap.parse_args())

# load the aruco dictionary and grab the aruco parameters
arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)
arucoParams = cv.aruco.DetectorParameters_create()

#  initialize the video file stream
print('[INFO] accessing video stream')
vf = cv.VideoCapture(args['input'])

# initialize the queue to maintain next frame from video stream
Q = deque(maxlen=128)

# We need to have a frame in our queue to start our augmented reality pipeline, so read next frame from video
# file source and add it to queue
(grabbed, source) = vf.read()
Q.appendleft(source)

# initialize the video stream
print('[INFO] Starting Video Stream...')
vs = VideoStream(src='vvid.mp4').start()
time.sleep(2.0)

# loop over the frames from video stream
while len(Q) > 0:
    # grab frame from video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # attempt to find the aruco markers in frame and provided they are found, take the current source image
    # and wrap it onto the input frame using augmented reality technique
    warped = find_and_wrap(frame, source, cornerIDs=(923, 1001, 241, 1007), arucoDict=arucoDict,
                           arucoParams=arucoParams, useCache=args['cache'] > 0)
    # if the wrapped image is not None then we know
    # 1) we found the 4 ArUCo markers
    # 2) perspective warp was successfully applied
    if warped is not None:
        # set the frame to the output augmented reality frame and grab the next video file frame from queue
        frame = warped
        source = Q.popleft()

    # for speed/efficiency, we can use a queue to keep the next video frame queue ready for us -- trick is
    # to ensure queue is always (or nearly full)
    if len(Q) != Q.maxlen:
        # read the next frame from video file stream
        (grabbed, nextFrame) = vf.read()

        # if frame was read meaning we are not at the end of our video stream, add frame to queue
        if grabbed:
            Q.append(nextFrame)

    # show the output frame
    cv.imshow('Frame', frame)
    key = cv.waitKey(1) & 0xFF

    # if Q key was pressed, break from loop
    if key == ord('q'):
        break

cv.destroyAllWindows()
vs.stop()
