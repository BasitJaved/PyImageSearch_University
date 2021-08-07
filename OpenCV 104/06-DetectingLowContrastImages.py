from skimage.exposure import is_low_contrast
from imutils.paths import list_images
import argparse
import imutils
import cv2 as cv

# Construct the argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='Path to input images directory')
ap.add_argument('-t', '--threshold', type=float, default=0.35, help='Threshold for low contrast')
args = vars(ap.parse_args())

# grab the path to the input images
imagePaths = sorted(list(list_images(args['input'])))

# loop over the image paths
for (i, path) in enumerate(imagePaths):
    # load the image, resize it and convert to grayscale
    print(f'[INFO] processing image {i+1}/{len(imagePaths)}')
    image = cv.imread(path)
    image = imutils.resize(image, width=450)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # blur the image slightly and perform edge detection
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blur, 30, 150)

    # initialize text and color to indicate image is not Low contrast
    text = 'Low Contrast: No'
    color = (0, 255, 0)

    # Check to see if image is low contrast
    if is_low_contrast(gray, fraction_threshold=args['threshold']):
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
        cv.drawContours(image, [c], -1, (0, 255, 0), 2)

    # draw text on image
    cv.putText(image, text, (5,25), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show output image and edge map
    cv.imshow('Image', image)
    cv.imshow("Edge", edged)
    cv.waitKey(0)

cv.destroyAllWindows()
