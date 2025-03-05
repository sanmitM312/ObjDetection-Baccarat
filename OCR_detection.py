import cv2
import numpy as np
import pytesseract

def detect_timer(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # We'll focus on the top-left corner of the image
    # The green timer box appears to be in this region
    height, width = image.shape[:2]
    top_left_region = image[0:int(height/4), 0:int(width/4)]
    
    cv2.imshow(f"Top left region", top_left_region)
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Convert to HSV color space to isolate the green box
    hsv = cv2.cvtColor(top_left_region, cv2.COLOR_BGR2HSV)
    cv2.imshow(f"Top left region", hsv)
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Define range for green color
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    
    # Create a mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    timer_info = {"detected": False, "number": None, "status": None}
    
    if contours:
        # Find the largest green contour (should be the timer box)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        

        # Extract the timer box region
        timer_box = top_left_region[y+h:y+3*h, x:x+w]
        
        # Get the number inside the timer box
        # Convert to grayscale for better text detection
        gray_timer = cv2.cvtColor(timer_box, cv2.COLOR_BGR2GRAY)

        cv2.imshow(f"Timer", gray_timer)
        cv2.waitKey(0);
        cv2.destroyAllWindows();

        _, binary_timer = cv2.threshold(gray_timer, 200, 255, cv2.THRESH_BINARY)
        cv2.imshow(f"Timer", binary_timer)
        cv2.waitKey(0);
        cv2.destroyAllWindows();

        # Use pytesseract to extract text (the number)
        timer_number = pytesseract.image_to_string(gray_timer, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
        timer_number = timer_number.strip()
        
        # Look for the status text above the timer
        status_region = top_left_region[y:y+h, x:x+w]
        cv2.imshow(f"Station region", status_region)
        cv2.waitKey(0);
        cv2.destroyAllWindows();

        gray_status = cv2.cvtColor(status_region, cv2.COLOR_BGR2GRAY)
        _, binary_status = cv2.threshold(gray_status, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Use pytesseract to extract the status text
        status_text = pytesseract.image_to_string(binary_status).strip()
        
        # If we found both a number and status, mark as detected
        if timer_number and status_text:
            timer_info = {
                "detected": True,
                "number": timer_number,
                "status": status_text
            }
        
        # For debugging: draw rectangle around detected timer
        cv2.rectangle(top_left_region, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imwrite('detected_timer.jpg', image)
    
    return timer_info

def main():
    image_path = 'casino5.png'  # Replace with your image path
    timer_info = detect_timer(image_path)
    
    if timer_info["detected"]:
        print(f"Timer detected!")
        print(f"Status: {timer_info['status']}")
        print(f"Number: {timer_info['number']}")
    else:
        print("Timer not detected in the image.")

if __name__ == "__main__":
    main()