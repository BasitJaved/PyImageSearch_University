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
cf = cv.VideoCapture(args['input'])
