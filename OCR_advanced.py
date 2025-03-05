import cv2
import numpy as np
import pytesseract

def detect_timer_status_and_number(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Could not load image"}
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Focus on the top-left region of the image
    top_left_region = image[0:int(height/4), 0:int(width/4)]
    cv2.imwrite('top_left_region.jpg', top_left_region)

    
    # Convert to HSV color space
    hsv = cv2.cvtColor(top_left_region, cv2.COLOR_BGR2HSV)
    cv2.imwrite('top_left_region.jpg', hsv)
    
    # Define color ranges for detection
    # Red (for "Stop Bet")
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Green (for "Please Bet")
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    
    # Create masks for each color
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Combine masks to detect either red or green
    combined_mask = cv2.bitwise_or(red_mask, green_mask)
    cv2.imwrite('combined_mask.jpg', combined_mask)
    
    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = {"status": None, "number": None}
    
    if contours:
        # Find the largest contour (should be the timer/status box)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Extract the timer/status box
        status_box = top_left_region[y:y+h, x:x+w]
        cv2.imwrite('status_box.jpg', status_box)
        
        # Use OCR to get the status text
        status_text = pytesseract.image_to_string(status_box).strip()
        result["status"] = status_text
        
        # Now find the number below the status box
        # Define a region directly below the status box
        below_y = y + h
        below_h = int(h * 1.5)  # Adjust this height as needed
        below_region = top_left_region[below_y:below_y+below_h, x:x+w]
        
        # Check if the region is valid
        if below_region.size > 0:
            cv2.imwrite('below_region.jpg', below_region)
            
            # Convert to grayscale for better text detection
            gray_below = cv2.cvtColor(below_region, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to isolate text
            _, binary_below = cv2.threshold(gray_below, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite('binary_below.jpg', binary_below)
            
            # Use OCR with configuration for single digits
            config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'
            number_text = pytesseract.image_to_string(binary_below, config=config).strip()
            
            # If no number is found, try a different approach with adaptive threshold
            if not number_text:
                adaptive_thresh = cv2.adaptiveThreshold(gray_below, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                        cv2.THRESH_BINARY_INV, 11, 2)
                cv2.imwrite('adaptive_thresh.jpg', adaptive_thresh)
                number_text = pytesseract.image_to_string(adaptive_thresh, config=config).strip()
            
            result["number"] = number_text if number_text else None
    
    # If the above approach failed, try a direct template matching approach for common digits
    if result["number"] is None:
        # This approach would be good if the numbers are always in the same font/style
        # We would create templates for digits 0-9 and match them against the image
        # For simplicity, we'll just use a more aggressive text detection approach here
        
        # Create a wider region to scan for the number
        status_y = y + h if contours else 0
        number_region = top_left_region[status_y:status_y+50, 0:int(width/4)]
        
        if number_region.size > 0:
            cv2.imwrite('number_region_wide.jpg', number_region)
            
            # Convert to grayscale
            gray_number = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, binary_number = cv2.threshold(gray_number, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Use OCR with digit whitelist
            number_text = pytesseract.image_to_string(binary_number, config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789').strip()
            
            # Extract just the first digit if multiple were found
            if number_text:
                result["number"] = number_text[0] if len(number_text) > 0 else None
    
    return result

def main():
    image_path = 'casino5.jpg'  # Replace with your image path
    result = detect_timer_status_and_number(image_path)
    
    print("Detection results:")
    print(f"Status: {result['status']}")
    print(f"Number below status: {result['number']}")
    
    if "error" in result:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    main()