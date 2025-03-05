import cv2 as cv

img = cv.imread('./templates/leftBoundary.jpg')
print(f"dimesions are {img.shape[0]} and {img.shape[1]}")