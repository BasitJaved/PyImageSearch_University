import numpy as np
import cv2 as cv

# Initialize Cached reference points
cached_ref_points = None


def find_and_wrap(frame, source, cornerIDs, arucoDict, arucoParams, useCache=False):
    # grab a reference to our cache referenced points
    global cached_ref_points

    # grab width and height of our frame and source image
    (imgH, imgW) = frame.shape[:2]
    (srcH, srcW) = source.shape[:2]

    # detect aruco markers in frame
    (corners, ids, rejected) = cv.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)

    # if we did not find 4 aruco markers, initialize an empty IDs list, else flatten IDs list
    ids = np.array([]) if len(corners) != 4 else ids.flatten()

    # initialize list of reference points
    refPts = []

    # Loop over the IDs of aruco markers in top-left, top-right, bottom-left, bottom-right order
    for i in cornerIDs:
        # grab the index of corner ID with current ID
        j = np.squeeze(np.where(ids == i))

        # if we recceive an empty list instead of an integer index then we could not find marker with current ID
        if j.size == 0:
            continue

        # otherwise append the corner coordinates to our list of reference points
        corner = np.squeeze(corners[j])
        refPts.append(corner)

    # check to see if we failed to find 4 aruco markers
    if len(refPts) != 4:
        # if we are allowed to use cached reference points fall back to them
        if useCache and cached_ref_points is not None:
            refPts = cached_ref_points

        # other wise return
        else:
            return None

    # if we are allowed to use cached reference points then update the cache with current points
    if useCache:
        cached_ref_points = refPts

    # unpack our aruco reference points and use reference points to define destination transformation matrix,
    # making sure points are specified in top-left, top-right, bottom-left, bottom-right order
    (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
    dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
    dstMat = np.array(dstMat)

    # Grab the spatial dimensions of the source image and define the transform matrix for source image
    # in top-left, top-right, bottom-right and bottom-left order
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])

    # Compute Homography matrix then wrap source image to the destination based on homography
    (H, _) = cv.findHomography(srcMat, dstMat)
    warped = cv.warpPerspective(source, H, (imgW, imgH))

    # Construct a mask for the source image now that the perspective warp has taken place (we'll need this mask
    # to copy the source image into destination image
    mask = np.zeros((imgH, imgW), dtype='uint8')
    cv.fillConvexPoly(mask, dstMat.astype('int32'), (255, 255, 255, cv.LINE_AA))

    # Give source image a black border surrounding it when applied to source image by applying a dilation operation
    rect = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.dilate(mask, rect, iterations=2)

    # Create a 3-channel version of mask by stacking it depth wise, such that we can apply warped source
    # image into input image
    maskScaled = mask.copy() / 255.0  # normalizing pixel values
    maskScaled = np.dstack([maskScaled] * 3)

    # Copy warped source image into destination image by
    # 1) multiplying the warped image and mask together
    # 2) multiplying original input image with mask (giving more weight to input image where there are not
    # masked pixels)
    # 3) Adding the resulting multiplications together
    warpedMultiplied = cv.multiply(warped.astype('float'), maskScaled)
    imageMultiplied = cv.multiply(frame.astype(float), 1.0 - maskScaled)
    output = cv.add(warpedMultiplied, imageMultiplied)
    output = output.astype('uint8')

    # Retuen output
    return output
