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
    max_width, max_height = 1333, 755
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

def calculate_scale_factor(image, target_width=2500, target_height=1600):
    """
    Calculate the scale factor for an image to match the target dimensions.
    
    Args:
        image: Input image
        target_width: Target width to compare against
        target_height: Target height to compare against
        
    Returns:
        Scale factor to resize the image to match the target dimensions
    """
    height, width = image.shape[:2]
    width_scale = target_width / width
    height_scale = target_height / height
    return min(width_scale, height_scale)

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
    
    
    haystack_img = resize(haystack_img)
    scale_factor = calculate_scale_factor(haystack_img)
    print(f"scale factor is {scale_factor}")
    print(f"{haystack_img.shape} aspect ratio {haystack_img.shape[0]/haystack_img.shape[1]}")


    haystack_gray = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
    

    result_img = haystack_img.copy()
    
    allowed_needle_names = ['chip', 'stop_bet_open', 'stop_bet_settle', 'please_bet','confirm','cancel']
    colored_needle_names = ['chip_10','chip_50','chip_100','chip_500','chip_1000','stop_bet_open', 'stop_bet_settle', 'please_bet','confirm','cancel']

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
    
    # Generate scale factors
    min_scale, max_scale = scale_range
    scale_factors = np.linspace(min_scale, max_scale, scale_steps)
    
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
            # needle_alpha_merged = cv2.merge([alpha,alpha,alpha])
            # Template matching
            if use_colored_matching:
                result = cv.matchTemplate(haystack_img, scaled_needle, cv.TM_CCOEFF_NORMED)
            else:
                result = cv.matchTemplate(haystack_gray, scaled_needle, cv.TM_CCOEFF_NORMED)

            # Get the best match for this scale
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            
            # if fnmatch.fnmatch(needle_name, 'chip_1000*'):
            #     roi_x, roi_y =  max_loc
            #     roi_w, roi_h = (new_width, new_height)

            #     cv.rectangle(result_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (225,0,255), 2)            
            #     cv.imshow("chip1000",result_img)
            #     cv.waitKey(0)
            #     cv.destroyAllWindows()

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
                if roi_y + roi_h < haystack_img.shape[0] and roi_y + new_height < haystack_img.shape[0] and (roi_x + roi_w//2) < haystack_img.shape[1] and roi_x + roi_w < haystack_img.shape[1]: 
                    # cv.rectangle(result_img, (roi_x//2, roi_y+roi_h), (roi_x + roi_w, roi_y + new_height), (255,0,0), 2)
                    s_img = haystack_img[roi_y:roi_y+new_height, roi_x + roi_w//2: roi_x+roi_w]
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
                 
                new_height = roi_w - 10*roi_h # thora hacky 
                roi_y-=2
                cv.rectangle(result_img, (roi_x//2, roi_y+roi_h), (roi_x + roi_w, roi_y + new_height), (255,0,0), 2)
                sub_img = haystack_img[roi_y+roi_h:roi_y+new_height, (roi_x + roi_w//2): roi_x+roi_w]
                gray_sub_img = cv.cvtColor(sub_img, cv.COLOR_BGR2GRAY)
                
                start = time.time()
                text = pytesseract.image_to_string(gray_sub_img, config='--psm 6')
                print(time.time() - start)
                print(f"OCR Result: {text.strip()}")

                # cv.imshow("Result", sub_img)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

                dim_top = (roi_x + roi_w//2,roi_y)
                dim_bottom = (roi_x+roi_w,roi_y+new_height)

                color = (0,0,255)
                cv.rectangle(result_img, dim_top,dim_bottom, color, 2) 

                

                status.append({
                    "status" : "Please Bet",
                    "isNumber": True,
                    "content": "1"
                })
                
            if needle_name.startswith('p_pair'):
                # print("Inside p_pair ")
                banker_tl = roi_x+roi_w
                banker_tl -= 2
                banker_bounds.append((banker_tl,roi_y+roi_h))
            if(re.match(r'one.*',needle_name)):
                player_bounds.append((roi_x+roi_w,roi_y))

            if(re.match(r'eight.*',needle_name)):
                banker_bounds.append((roi_x, roi_y))
                player_bounds.append((roi_x,roi_y+roi_h))

            if re.match(r'leftBoundary.*', needle_name):
                top_left_bound = roi_x + roi_w
                top_left_bound -= 2
                ring_bounds.append((top_left_bound, roi_y + roi_h))
            
            if re.match(r'rightBoundary.*', needle_name):
                ring_bounds.append((roi_x, roi_y + roi_h))

            all_matches.append(best_match)
            if not any(re.match(f'^{name}', needle_name) for name in allowed_needle_names):
                continue

            color = (0,0,255)
            cv.rectangle(result_img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), color, 2)            
            label = f"{needle_name}: {best_match['confidence']:.2f} (scale: {best_match['scale']:.2f})"
            cv.putText(result_img, label, (roi_x, roi_y - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
    # Sort matches by confidence
    all_matches.sort(key=lambda x: x['confidence'], reverse=True)
    
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
    
    # Define scale range and steps
    scale_range = (0.5, 1.5)  # Try scales from 50% to 150%
    scale_steps = 5  # Try 5 different scales within this range
    
    # Process each haystack image in the folder
    for haystack_name in os.listdir(haystack_folder):
        haystack_path = os.path.join(haystack_folder, haystack_name)
        
        # Convert image to PNG if needed
        haystack_path = convert_to_png(haystack_path)
        
        # Perform multi-scale template matching
        result_img, matches, ring_bounds, banker_bounds, player_bounds = multi_scale_template_matching(
            haystack_path, 
            needle_folder,
            threshold=0.7,
            scale_range=scale_range,
            scale_steps=scale_steps
        )
        
        # Process turn ROI if boundaries were found
        if len(ring_bounds) == 2:
            color = (0,0,255)
            cv.rectangle(result_img, ring_bounds[0], ring_bounds[1], color, 2)
            label = f"TURN ROI"
            cv.putText(result_img, label, (ring_bounds[0][0], ring_bounds[0][1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if len(banker_bounds) == 2:
            color = (0,0,255)
            cv.rectangle(result_img, banker_bounds[0], banker_bounds[1], color, 2)
            label = f"BANKER ROI"
            cv.putText(result_img, label, (banker_bounds[0][0], banker_bounds[0][1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if len(player_bounds) == 2:
            color = (0,0,255)
            cv.rectangle(result_img, player_bounds[0], player_bounds[1], color, 2)
            label = f"PLAYER ROI"
            cv.putText(result_img, label, (player_bounds[0][0], player_bounds[0][1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save result
        result_name = f"multi_template_{os.path.splitext(haystack_name)[0]}_result.png"
        result_path = os.path.join(output_folder, result_name)
        cv.imwrite(result_path, result_img)
        print(f"Result saved in {result_path}")
        
        # Print detection summary
        print(f"Total elements detected in {haystack_name}: {len(matches)}")
        for i, match in enumerate(matches):
            print(f"{i+1}. {match['template']} - Confidence: {match['confidence']:.2f} - Scale: {match['scale']:.2f}")
    
    print(f"Total time taken: {time.time() - start} seconds")

if __name__ == "__main__":
    main()