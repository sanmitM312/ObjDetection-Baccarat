import cv2 as cv
import numpy as np
import os
import re
import time
import pytesseract
import fnmatch
def convert_to_png(image_path):
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    png_path = os.path.splitext(image_path)[0] + '.png'
    cv.imwrite(png_path, img)
    return png_path

def resize(haystack_img):
    # Resize result_img only if required and maintain aspect ratio
    max_width, max_height = 1280, 720
    height, width = haystack_img.shape[:2]
    if width > max_width or height > max_height:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        haystack_img = cv.resize(haystack_img, (new_width, new_height), interpolation=cv.INTER_AREA)

    return haystack_img

def scale_up_coordinates(original_img, resized_img, coordinates, temp):
    """
    Scale up the coordinates from the resized image to the original image dimensions.
    
    Args:
        original_img: Original image before resizing
        resized_img: Resized image
        coordinates: Tuple of (x, y) coordinates in the resized image
        
    Returns:
        Tuple of (x, y) coordinates scaled up to the original image dimensions
    """
    original_height, original_width = original_img.shape[:2]
    resized_height, resized_width = resized_img.shape[:2]
    
    scale_x = original_width / resized_width
    scale_y = original_height / resized_height
    
    # print("Scaled down coords :", coordinates)
    original_x = int(coordinates[0] * scale_x)
    original_y = int(coordinates[1] * scale_y)
    # print("Original coords :", coordinates)
    width = int(temp[0] * scale_x)
    height = int(temp[1] * scale_y)
    
    return (original_x, original_y),(width,height)

def multi_scale_template_matching(haystack_path, needle_folder, threshold=0.6, scale_range=(0.5, 1.5), scale_steps=5):
    """
    Match multiple template images against a single haystack image at multiple scales
    
    Args:
        haystack_path: Path to the main image to search in
        needle_folder: Folder containing multiple template images to search for
        threshold: Minimum confidence threshold for matches
        scale_range: Tuple of (min_scale, max_scale) to try
        scale_steps: Number of different scales to try
        
    Returns:
        Annotated haystack image and list of match information
    """
    # Load haystack image
    haystack_img = cv.imread(haystack_path)

    if haystack_img is None:
        print(f"Error: Could not load haystack image: {haystack_path}")
        return None, []
    
    print(f"{haystack_img.shape} aspect ratio {haystack_img.shape[0]/haystack_img.shape[1]}")
    
    # haystack_img = resize(haystack_img)
    
    
    colored_needle_names = ['chip_1000','chip_10_2','chip_10','confirm','cancel','leftBoundary','stop_bet_open', 'stop_bet_settle', 'please_bet']

    all_matches = []
    ring_bounds = []
    banker_bounds = []
    player_bounds = []
    status = []
    # Get all needle template files from the folder
    needle_files = [os.path.join(needle_folder, f) for f in os.listdir(needle_folder)]

    if not needle_files:
        print(f"Error: No template images found in folder: {needle_folder}")
        return None, []
    
    result_img = haystack_img.copy()
    # preprocess
    haystack_img = resize(haystack_img)
    haystack_gray = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)

    # Generate scale factors
    min_scale, max_scale = scale_range
    scale_factors = np.linspace(min_scale, max_scale, scale_steps)
    
    # Load the bottom_bounds needle template
    bottom_bounds_path = os.path.join(needle_folder, 'bottom_bounds.png')
    bottom_bounds_template = cv.imread('./templates/bottom_bounds.png', cv.IMREAD_GRAYSCALE)
    
    if bottom_bounds_template is None:
        print(f"Error: Could not load bottom_bounds template: {bottom_bounds_path}")
        return None, []

    best_bottom_match = None
    for scale in scale_factors:
        # Calculate new dimensions
        new_width = int(bottom_bounds_template.shape[1] * scale)
        new_height = int(bottom_bounds_template.shape[0] * scale)
        
        # Skip if scaled dimensions are too small
        if new_width < 10 or new_height < 10:
            continue
        
        # Resize the template
        scaled_bottom_bounds = cv.resize(bottom_bounds_template, (new_width, new_height), interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
        
        # Template matching
        result = cv.matchTemplate(haystack_gray, scaled_bottom_bounds, cv.TM_CCOEFF_NORMED)

        # Get the best match for this scale
        _, max_val, _, max_loc = cv.minMaxLoc(result)
        
        # Update best match if this is better
        if best_bottom_match is None or max_val > best_bottom_match['confidence']:
            best_bottom_match = {
                'position': max_loc,
                'size': (new_width, new_height),
                'confidence': max_val,
                'scale': scale
            }
            best_bottom_match["position"], best_bottom_match["size"] = scale_up_coordinates(result_img, haystack_img, best_bottom_match["position"], best_bottom_match["size"])
    
    color = (0,0,255)
    cv.rectangle(result_img, best_bottom_match["position"], (best_bottom_match["position"][0]+best_bottom_match["size"][0],best_bottom_match["position"][1]+best_bottom_match["size"][1]), color, 2)

    # Process each needle template
    for needle_path in needle_files:
        needle_name = os.path.basename(needle_path)
        
        # Load the needle image
        needle_img = cv.imread(needle_path)
        if needle_img is None:
            print(f"  Warning: Could not load template: {needle_path}")
            continue
    
        use_colored_matching = any(re.match(f'^{name}', needle_name) for name in colored_needle_names)

        if use_colored_matching:
            needle_template = needle_img
        else:
            # Convert needle to grayscale for template matching
            needle_template = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)

        # Convert needle to grayscale for template matching
        # needle_gray = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)
        # cv.imshow(f"Needle: {needle_name}", needle_gray)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        best_match = None
        best_scale = 1.0
        # scale_factors = (scale_factors[0],scale_up)
       # Try different scales
        for scale in scale_factors:
            # Calculate new dimensions
            new_width = int(needle_template.shape[1] * scale)
            new_height = int(needle_template.shape[0] * scale)
            
            # Skip if scaled dimensions are too small
            if new_width < 10 or new_height < 10:
                continue
            
            # Resize the template
            scaled_needle = cv.resize(needle_template, (new_width, new_height), interpolation=cv.INTER_AREA if scale < 1 else cv.INTER_LINEAR)
            
            # needle_template = needle_gray[:,:,0:2]
            # alpha = needle_template[:,:,3]
            # needle_alpha_merged = cv.merge([alpha,alpha,alpha])
            # Template matching
            if use_colored_matching:
                result = cv.matchTemplate(haystack_img, scaled_needle, cv.TM_CCOEFF_NORMED)
            else:
                result = cv.matchTemplate(haystack_gray, scaled_needle, cv.TM_CCOEFF_NORMED)

            # Get the best match for this scale
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            

            # Update best match if this is better
            if best_match is None or max_val > best_match['confidence']:
                best_match = {
                    'template': needle_name,
                    'position': max_loc,
                    'size': (new_width, new_height),
                    'confidence': max_val,
                    'scale': scale,
                    'mid' : (new_width//2, new_height//2)
                }
                # print("The scale is ", scale)
                best_match["position"],best_match["size"] = scale_up_coordinates(result_img, haystack_img,best_match["position"],best_match["size"])

        print(f"Needle image {needle_name} shape {needle_img.shape} {needle_img.shape[0]/needle_img.shape[1]} {best_match['confidence']}")
        # If we found a match above threshold
        

        # dynamically check for the threshold as the max of the stop bet block
        if best_match and best_match['confidence'] >= threshold:
            roi_x, roi_y = best_match['position']
            roi_w, roi_h = best_match['size']


            
            # print(f"Needle image {needle_name}")

            # if not any(re.match(f'^{name}', needle_name) for name in allowed_needle_names):
            #     continue
           


            if needle_name.startswith('stop_bet_open'):
                status.append({
                    "status" : "Stop Bet",
                    "isNumber": False,
                    "content": "Open Cards"
                })
            
            if needle_name.startswith('stop_bet_settle'):

                new_height = 3*roi_h # thora hacky   # Adjust height based on the region of interest
                roi_y -=2
                # cv.rectangle(result_img, (roi_x//2, roi_y+roi_h), (roi_x + roi_w, roi_y + new_height), (255,0,0), 2)
                # Ensure the slicing is within bounds
                if roi_y + roi_h < result_img.shape[0] and roi_y + new_height < result_img.shape[0] and (roi_x + roi_w//2) < result_img.shape[1] and roi_x + roi_w < result_img.shape[1]: 
                    # cv.rectangle(result_img, (roi_x//2, roi_y+roi_h), (roi_x + roi_w, roi_y + new_height), (255,0,0), 2)
                    s_img = result_img[roi_y:roi_y+new_height, roi_x + roi_w//2: roi_x+roi_w]
                    # cv.imshow("Result", s_img)
                    # cv.waitKey(0)
                    # cv.destroyAllWindows()
                    gs_img = cv.cvtColor(s_img, cv.COLOR_BGR2GRAY)

                    dim_top = (roi_x + roi_w//2,roi_y)
                    dim_bottom = (roi_x+roi_w,roi_y+new_height)

                    color = (0,0,255)
                    cv.rectangle(result_img, dim_top,dim_bottom, color, 2)            

                # cv.imshow("Result", gs_img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

                status.append({
                    "status" : "Stop Bet",
                    "isNumber": False,
                    "content": "Settle"
                })

            if needle_name.startswith('please_bet'):
                # add ocr logic
                # get the result  
                # print("Inside please bet")
                 
                # Define the region of interest (ROI) for OCR
                new_height = int(2 * roi_h)  # Adjust height based on the region of interest
                roi_y -= 2  # Adjust y-coordinate if needed

                # Ensure the slicing is within bounds
                if roi_y + new_height < haystack_img.shape[0] and (roi_x + roi_w) < haystack_img.shape[1]:
                    sub_img = haystack_img[roi_y:roi_y + new_height, roi_x:roi_x + roi_w]
                    gray_sub_img = cv.cvtColor(sub_img, cv.COLOR_BGR2GRAY)

                    if gray_sub_img is not None:
                        # Perform OCR on the grayscale sub-image
                        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                        text = pytesseract.image_to_string(gray_sub_img, config=custom_config)
                        # print(f"OCR Result: {text.strip()}")

                        # Draw a rectangle around the ROI
                        cv.rectangle(result_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + new_height), (0, 0, 255), 2)

                        status.append({
                            "status": "Please Bet",
                            "isNumber": True,
                            "content": text.strip()
                        })
                
                # start = time.time()
                # custom_config = r'--psm 6'
                # text = pytesseract.image_to_string(gray_sub_img, config=custom_config)
                # print(time.time() - start)
                # print(f"OCR Result: {text.strip()}")

                # cv.imshow("Result", gray_sub_img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

                # dim_top = (roi_x + roi_w//2,roi_y)
                # dim_bottom = (roi_x+roi_w,roi_y+new_height)

                # color = (0,0,255)
                # cv.rectangle(result_img, dim_top,dim_bottom, color, 2) 


                # status.append({
                #     "status" : "Please Bet",
                #     "isNumber": True,
                #     "content": text
                # })
        
                
            # if(re.match(r'one.*',needle_name)):
            #     banker_bounds.append((roi_x+roi_w,roi_y-int(1.5*roi_h)))
            #     player_bounds.append((roi_x+roi_w,roi_y))

            # # if(re.match(r'oneLeft.*',needle_name) and best_match["oneLeft"]['confidence'] > best_match['one']['confidence']):
            # #     banker_bounds.append((roi_x+roi_w,roi_y-int(1.5*roi_h)))
            # #     player_bounds.append((roi_x+roi_w,roi_y))

            # if(re.match(r'eight.*',needle_name)):
            #     banker_bounds.append((roi_x, roi_y))
            #     player_bounds.append((roi_x,roi_y+roi_h))
            
            if re.match(r'leftBoundary.*', needle_name):
                top_left_bound = roi_x + roi_w
                top_left_bound -= 2
                ring_bounds.append((top_left_bound, roi_y + roi_h))
                # print(f"ringgggggggggggggggggggggggggggg1 : {(top_left_bound, roi_y + roi_h)}")
            
            if re.match(r'rightBoundary.*', needle_name):
                # if len(banker_bounds)==2:
                #     ring_bounds.pop()
                # print(f"ringgggggggggggggggggggggggggggg : {(roi_x, roi_y + roi_h)}")
                ring_bounds.append((roi_x, roi_y + roi_h))
        
            # if len(banker_bounds) == 2:
            #     all_matches.append({
            #         'template' : 'banker',
            #         'position' : banker_bounds[0],
            #         'size' :    (banker_bounds[1][0]-banker_bounds[0][0],banker_bounds[1][1]-banker_bounds[0][1]),
            #         'confidence': 1.0,
            #         'scale' : 1.0

            #     })
            # if len(player_bounds) == 2:
            #     all_matches.append({
            #         'template' : 'player',
            #         'position' : player_bounds[0],
            #         'size' :    (player_bounds[1][0]-player_bounds[0][0],player_bounds[1][1]-player_bounds[0][1]),
            #         'confidence': 1.0,
            #         'scale' : 1.0
            #     })
            if len(ring_bounds) == 2:
                # print("hello worldddddddd ",ring_bounds[1][0]-ring_bounds[0][0],ring_bounds[1][1]-ring_bounds[0][1])
                all_matches.append({
                    'template' : 'ring',
                    'position' : ring_bounds[0],
                    'size' :    (ring_bounds[1][0]-ring_bounds[0][0],ring_bounds[1][1]-ring_bounds[0][1]),
                    'confidence': 1.0,
                    'scale' : 1.0
                })

            all_matches.append(best_match)
            # if not any(re.match(f'^{name}', needle_name) for name in allowed_needle_names):
            #     continue



            # color = (0,0,255)
            # cv.rectangle(result_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, 2)            

            # label = f"{needle_name}: {best_match['confidence']:.2f} (scale: {best_match['scale']:.2f})"
            # cv.putText(result_img, label, (roi_x, roi_y - 5),
            #             cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Sort matches by confidence
    # allowed_needle_names = ['rightBoundary','leftBoundary','stop_bet_open', 'stop_bet_settle', 'please_bet','confirm','cancel','chip_10','chip_50','chip_100','chip_500','chip_1000','banker','player','ring']
    allowed_needle_names = ['banker_1','player_1','stop_bet_open', 'stop_bet_settle', 'please_bet','confirm','cancel','chip_10','chip_50','chip_100','chip_500','chip_1000','banker','player','ring']

    all_matches = [match for match in all_matches if any(re.match(f'^{name}', match['template']) for name in allowed_needle_names)]
    # Keep only the highest confidence match for each template
    unique_matches = {}
    for match in sorted(all_matches, key=lambda x: x['confidence'], reverse=True):
        if match['template'] not in unique_matches:
            unique_matches[match['template']] = match
    
    all_matches = list(unique_matches.values())

    return result_img, all_matches, ring_bounds,banker_bounds,player_bounds

def detect_chips_simple(haystack_img, result_img, all_matches):
    """Simple detection of chip area without doing complex processing"""
    # Just focus on the area where chips are typically found
    height, width = haystack_img.shape[:2]
    
    # Chip area is usually in the bottom right
    chip_area_x = int(width * 0.8)
    chip_area_y = int(height * 0.8)
    chip_area_w = int(width * 0.2)
    chip_area_h = int(height * 0.2)
    
    # Draw a rectangle around the likely chip area
    print(f"Drawing {needle_name}")
    cv.rectangle(result_img, 
                (chip_area_x, chip_area_y), 
                (chip_area_x + chip_area_w, chip_area_y + chip_area_h), 
                (0, 255, 255), 2)
    
    cv.putText(result_img, "Chip Area", (chip_area_x, chip_area_y - 5),
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Add this area to matches
    all_matches.append({
        'template': "chip_area",
        'position': (chip_area_x, chip_area_y),
        'size': (chip_area_w, chip_area_h),
        'confidence': 0.5,
        'scale': 1.0
    })

def main():
    haystack_folder = 'haystacks'
    needle_folder = 'templates'
    output_folder = 'output'

    start = time.time()
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all haystack images from the folder
    haystack_files = [os.path.join(haystack_folder, f) for f in os.listdir(haystack_folder) if fnmatch.fnmatch(f, '*.png')]

    if not haystack_files:
        print(f"Error: No haystack images found in folder: {haystack_folder}")
        return

    for haystack_path in haystack_files:
        # Convert image to PNG if needed
        haystack_path = convert_to_png(haystack_path)

        # Define scale range and steps
        scale_range = (0.5, 1.5)  # Try scales from 50% to 150%
        scale_steps = 7  # Try 7 different scales within this range
        
        # Perform multi-scale template matching
        result_img, matches, ring_bounds, banker_bounds, player_bounds = multi_scale_template_matching(
            haystack_path, 
            needle_folder,
            threshold=0.65,
            scale_range=scale_range,
            scale_steps=scale_steps
        )
        
        # Process turn ROI if boundaries were found
        ans = {}
        for match in matches:
            if match['template'] == 'ring':
                ans = match
            elif match["confidence"] > 0.65:
                color = (0,0,255)
                cv.rectangle(result_img, match["position"], (match["position"][0]+match["size"][0],match["position"][1]+match["size"][1]), color, 2)
                label = f"{match['template']}"
                cv.putText(result_img, label, (match["position"][0],match["position"][1]-10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        image = cv.imread(haystack_path)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Define lower and upper bounds for grey color
        lower_gray = 180  # Adjust based on image intensity
        upper_gray = 220  # Adjust based on image intensity
        
        x_start, y_start = ans['position']
        x_end, y_end = (ans['size'][0] + ans['position'][0], ans['size'][1] + ans['position'][1])
        gray = gray[y_start:y_end, x_start:x_end]
        # Create a mask to extract grey lines
        mask = cv.inRange(gray, lower_gray, upper_gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv.GaussianBlur(mask, (5, 5), 0)
        
        # Use edge detection
        edges = cv.Canny(blurred, 50, 150, apertureSize=3)

        # Use Hough Line Transform to detect straight vertical lines
        lines = cv.HoughLinesP(edges, 1, np.pi / 90, 30, minLineLength=10, maxLineGap=5)
        
        # Draw only vertical lines on the original image
        result = image.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) > 85:  # Ensure the line is close to vertical
                    ans_top_left = (x2 + x_start, y2 + y_start)
                    cv.line(result, (x1 + x_start, y1 + y_start), (x2 + x_start, y2 + y_start), (0, 255, 0), 2)
                    break

        color = (0,0,255)
        cv.rectangle(result_img, ans_top_left, (x_end, y_end), color, 2)
              
        result_path = os.path.join(output_folder, f"detection_result_{os.path.basename(haystack_path)}")
        cv.imwrite(result_path, result_img)
        print(f"Result saved in {result_path}")
    
    print(f"Total processing time: {time.time() - start} seconds")

if __name__ == "__main__":
    main()
