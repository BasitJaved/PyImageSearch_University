import cv2 as cv
import argparse

# Construct the argument parse and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, default='coin5.jpg', help='Path to input image')
args = vars(ap.parse_args())

# Read and display image
image = cv.imread(args['image'])
cv.imshow('Original Image', image)
cv. waitKey(0)

# Convert the image to Gray and display
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)
cv. waitKey(0)

# Blur the image
blur = cv.GaussianBlur(gray, (5, 5), 0)
cv.imshow('Blur Image', blur)
cv. waitKey(0)

# Compute a wide, mid-Range and tight threshold for edges using canny edge detector and display results
wide = cv.Canny(blur, 10, 200)
cv.imshow('Wide', wide)
cv. waitKey(0)

mid = cv.Canny(blur, 30, 150)
cv.imshow('Mid', mid)
cv. waitKey(0)

tight = cv.Canny(blur, 250, 255)
cv.imshow('Tight', tight)
cv. waitKey(0)

# Converting the image to individual channels
(B, G, R) = cv.split(image)
B = cv.GaussianBlur(B, (5, 5), 0)  # Blurring
G = cv.GaussianBlur(G, (5, 5), 0)  # Blurring
R = cv.GaussianBlur(R, (5, 5), 0)  # Blurring
cv.imshow('Blue', B)
cv. waitKey(0)
cv.imshow('Green', G)
cv. waitKey(0)
cv.imshow('Red', R)
cv. waitKey(0)

# Compute a wide, mid-Range and tight threshold for edges  for individual channels
# Blue
wideB = cv.Canny(B, 10, 200)
cv.imshow('Blue Wide', wideB)
cv. waitKey(0)

midB = cv.Canny(B, 30, 150)
cv.imshow('Blue Mid', midB)
cv. waitKey(0)

tightB = cv.Canny(B, 250, 255)
cv.imshow('Blue Tight', tightB)
cv. waitKey(0)

# Green
wideG = cv.Canny(G, 10, 200)
cv.imshow('Green Wide', wideG)
cv. waitKey(0)

midG = cv.Canny(G, 30, 150)
cv.imshow('Green Mid', midG)
cv. waitKey(0)

tightG = cv.Canny(G, 250, 255)
cv.imshow('Green Tight', tightG)
cv. waitKey(0)

# Red
wideR = cv.Canny(R, 10, 200)
cv.imshow('Red Wide', wideR)
cv. waitKey(0)

midR = cv.Canny(R, 30, 150)
cv.imshow('Red Mid', midR)
cv. waitKey(0)

tightR = cv.Canny(R, 250, 255)
cv.imshow('Red Tight', tightR)
cv. waitKey(0)

# Merging all
finalCanny = cv.merge((tightB, tightG, tightR))
cv.imshow('Final Canny', finalCanny)
cv. waitKey(0)

cv.destroyAllWindows()
