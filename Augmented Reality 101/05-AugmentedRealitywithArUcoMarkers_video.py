import numpy as np
import cv2 as cv
import imutils
import sys

# Load the input image from disk, resize it anf grab it's spatial dimensions
print('[INFO] Loading the input and source image...')
image = cv.imread('pan4.jpg')
image = imutils.resize(image, width=800)
(imgH, imgW) = image.shape[:2]

# Load the aruco dictionary, grab the aruco parameters and detect the markers
arucoDict = cv.aruco.Dictionary_get(cv.aruco.DICT_ARUCO_ORIGINAL)
arucoParm = cv.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv.aruco.detectMarkers(image, arucoDict, parameters=arucoParm)

# if we have not detected 4 markers in image we cannot apply augmented reality technique so will exit out
if len(corners) != 4:
    print('[INFO] Could not find 4 corners... Exiting!!!')
    sys.exit(0)

# once we found the four Markers we can proceed by flattening the ArUCo Ids list and initializing
# out list of reference points
print('[INFO] constructing the Augmented reality visualization...')
ids = ids.flatten()
refPts = []

# Loop over the ids of ArUCo markers in top-left, top-right, bottom-right and bottom-left order
for i in (923, 1001, 241, 1007):
    # Grab the index of the corner with current ID and append the corner coordinates to our list of reference points
    j = np.squeeze(np.where(ids == i))
    corner = np.squeeze(corners[j])
    refPts.append(corner)    # refPts will contain coordinates of bounding boxes, a single BB will have 4 coordinates

# unpack ArUCo reference points and use reference points to define the destination transform Matrix,
# making sure the points are specified in top-left, top-right, bottom-right and bottom-left order
(refPtTL, refPtTR, refPtBR, refPtBL) = refPts
dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
# from first BB we need its (0,0) point, from second we need its (0, 1), from 3rd (1, 0), from 4th (1, 1) same
# as top-left, top-right, bottom-right and bottom-left from each BB
dstMat = np.array(dstMat)

# Construct a mask for the source image now that the perspective warp has taken place (we'll need this mask
# to copy the source image into destination image
mask = np.zeros((imgH, imgW), dtype='uint8')
cv.fillConvexPoly(mask, dstMat.astype('int32'), (255, 255, 255, cv.LINE_AA))

# Give source image a black border surrounding it when applied to source image by applying a dilation operation
rect = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
mask = cv.dilate(mask, rect, iterations=2)

# Create a 3-channel version of mask by stacking it depth wise, such that we can apply warped source
# image into input image
maskScaled = mask.copy()/255.0  # normalizing pixel values
maskScaled = np.dstack([maskScaled]*3)

imageMultiplied = cv.multiply(image.astype(float), 1.0 - maskScaled)

# Load the source Video
cap = cv.VideoCapture('toyota.mp4')

while True:

    ret, source = cap.read()
    # Grab the spatial dimensions of the source image and define the transform matrix for source image
    # in top-left, top-right, bottom-right and bottom-left order
    (srcH, srcW) = source.shape[:2]
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

    # Compute Homography matrix then wrap source image to the destination based on homography
    (H, _) = cv.findHomography(srcMat, dstMat)
    warped = cv.warpPerspective(source, H, (imgW, imgH))

    # Copy warped source image into destination image by
    # 1) multiplying the warped image and mask together
    # 2) multiplying original input image with mask (giving more weight to input image where there are not
    # masked pixels)
    # 3) Adding the resulting multiplications together
    warpedMultiplied = cv.multiply(warped.astype('float'), maskScaled)
    output = cv.add(warpedMultiplied, imageMultiplied)
    output = output.astype('uint8')

    # show image, source image and output of Augmented reality
    cv.imshow('VR', output)
    key = cv.waitKey(5) & 0xFF

    # if q key was pressed break from loop
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
