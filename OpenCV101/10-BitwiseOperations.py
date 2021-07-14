import cv2 as cv
import numpy as np

# draw a rectangle
rectangle = np.zeros((300, 300), dtype= 'uint8')
cv.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv.imshow('Rectangle', rectangle)
cv.waitKey(0)

# draw a circle
circle = np.zeros((300, 300), dtype= 'uint8')
cv.circle(circle, (150, 150), 150, 255, -1)
cv.imshow('Circle', circle)
cv.waitKey(0)

# Performing bitwise AND
bitwiseAnd = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise And', bitwiseAnd)
cv.waitKey(0)

# Performing bitwise OR
bitwiseOr = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwiseOr)
cv.waitKey(0)

# Performing bitwise XOR
bitwiseXor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise XOR', bitwiseXor)
cv.waitKey(0)

# Performing bitwise Not
bitwiseNot = cv.bitwise_not(circle)
cv.imshow('Bitwise Not', bitwiseNot)
cv.waitKey(0)

# applying Not operation on an image
image = cv.imread('00.jpg')
bitwiseNotIm = cv.bitwise_not(image)
cv.imshow('Bitwise Not Image', bitwiseNotIm)
cv.waitKey(0)

cv.destroyAllWindows()