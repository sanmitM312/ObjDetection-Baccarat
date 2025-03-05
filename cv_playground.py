import cv2 as cv
import numpy as np
import os

# Load haystack image
haystack_img = cv.imread('casino2.jpg')

# Check if haystack image loaded successfully
if haystack_img is None:
    print("Error: Could not load haystack image")
    exit()

(h, w) = haystack_img.shape[:2]
print(f"Original Size: Width={w}, Height={h}")

# Convert haystack image to grayscale
haystack_gray = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)

# Directory containing template images
templates_dir = 'templates'

# Check if templates directory exists
if not os.path.exists(templates_dir):
    print(f"Error: Directory '{templates_dir}' does not exist")
    exit()

# Iterate over all template images in the directory
for template_name in os.listdir(templates_dir):
    template_path = os.path.join(templates_dir, template_name)
    needle_img = cv.imread(template_path)

    # Check if template image loaded successfully
    if needle_img is None:
        print(f"Error: Could not load template image '{template_name}'")
        continue

    # Convert template image to grayscale
    needle_gray = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)

    # Perform template matching
    result = cv.matchTemplate(haystack_gray, needle_gray, cv.TM_CCOEFF_NORMED)

    # Get the best match position
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

    print(f"Template '{template_name}' best match top left position: {max_loc}")
    print(f"Template '{template_name}' best match confidence: {max_val}")

    threshold = 0.90
    if max_val >= threshold:
        print(f"Found '{template_name}'")

        needle_w = needle_img.shape[1]
        needle_h = needle_img.shape[0]
        top_left = max_loc
        bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

        # Draw rectangle on the original color image
        cv.rectangle(haystack_img, top_left, bottom_right, 
                    color=(0, 255, 0), thickness=2, lineType=cv.LINE_4)
    else:
        print(f"'{template_name}' not found")

# Make the window resizable
cv.namedWindow('Result', cv.WINDOW_NORMAL)  # Allows resizing
cv.resizeWindow('Result', 1000, 700)  # Set initial window size

# cv.imshow('Result', haystack_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

cv.imwrite("output.jpg", haystack_img)
