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
options = apriltag.DetectorOptions(families='tag36h11')  # there are a total of 6 families of AprilTags
detector = apriltag.Detector(options)
results = detector.detect(gray)
print(f'[INFO] Total {len(results)} AprilTags Detected')

# Loop over AprilTag detection results
for r in results:
    # extract the bounding box (x, y) coordinates for apriltag and convert each pairs into integers
    (ptA, ptB, ptC, ptD) = r.corners
    ptA = (int(ptA[0]), int(ptA[1]))
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))

    # draw bounding box of AprilTag detection
    cv.line(image, ptA, ptB, (0, 255, 2), 2)
    cv.line(image, ptB, ptC, (0, 255, 2), 2)
    cv.line(image, ptC, ptD, (0, 255, 2), 2)
    cv.line(image, ptD, ptA, (0, 255, 2), 2)

    # draw the center of coordinates of AprilTag
    (cX, cY) = (int(r.cenetr[0]), int(r.cenetr[1]))
    cv.circle(image, (cX, cY), 5, (0, 0, 255), -1)

    # draw Family tag on image
    tagFamily = r.tag_family.decode('utf-8')
    cv.putText(image, tagFamily, (ptA[0], ptA[1]-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(f'[INFO] tag Family: {tagFamily}')

# show image
cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()
