# These tags help inCamera Calibration, 3D Reconstruction, object detection, object tracking, navigation etc
import apriltag
import cv2 as cv
import argparse

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='Path to image containing AprilTag')
args = vars(ap.parse_args())

# Load the image and convert it to grayscale
image = cv.imread(args['image'])
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Define the April Tag detection options and detect the tags in input image
print('[INFO] Detecting tags...')
options = apriltag.DetectorOptions(families='tag36h11')
detector = apriltag.Detector(options)
results = detector.detect(gray)
print(f'[INFO] Total {results} AprilTags Detected')
