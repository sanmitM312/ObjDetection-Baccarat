import cv2 as cv
import numpy as np
import os

def multi_template_matching(haystack_path, needle_folder, threshold=0.7, color_weight=0.5):
    """
    Match multiple template images against a single haystack image
    
    Args:
        haystack_path: Path to the main image to search in
        needle_folder: Folder containing multiple template images to search for
        threshold: Minimum confidence threshold for matches
        color_weight: Weight to balance color vs shape matching
        
    Returns:
        Annotated haystack image and list of match information
    """
    # Load haystack image
    haystack_img = cv.imread(haystack_path)
    if haystack_img is None:
        print(f"Error: Could not load haystack image: {haystack_path}")
        return None, []
    
    print(f"Haystack image size: {haystack_img.shape}")
    
    haystack_gray = cv.cvtColor(haystack_img, cv.COLOR_BGR2GRAY)
    haystack_hsv = cv.cvtColor(haystack_img, cv.COLOR_BGR2HSV)
    
    result_img = haystack_img.copy()
    all_matches = []
    
    # Get all needle template files from the folder
    needle_files = [os.path.join(needle_folder, f) for f in os.listdir(needle_folder)]
    
    if not needle_files:
        print(f"Error: No template images found in folder: {needle_folder}")
        return None, []
    
    # Process each needle template
    for needle_index, needle_path in enumerate(needle_files):
        needle_name = os.path.basename(needle_path)
        print(f"Processing template {needle_index+1}/{len(needle_files)}: {needle_name}")
        
        # Load the needle image
        needle_img = cv.imread(needle_path)
        if needle_img is None:
            print(f"  Warning: Could not load template: {needle_path}")
            continue
            
        # Get the color profile of this needle
        needle_hsv = cv.cvtColor(needle_img, cv.COLOR_BGR2HSV)
        needle_hist = cv.calcHist([needle_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv.normalize(needle_hist, needle_hist, 0, 1, cv.NORM_MINMAX)
        
        # Convert needle to grayscale for template matching
        needle_gray = cv.cvtColor(needle_img, cv.COLOR_BGR2GRAY)
        
        # Try multiple scales for this template
        best_match = None
        best_score = -1
        best_scale = 1.0
        
        scale_factors = [0.9, 0.95, 1.0, 1.05, 1.1]
        for scale in scale_factors:
            width = int(needle_img.shape[1] * scale)
            height = int(needle_img.shape[0] * scale)
            resized_needle = cv.resize(needle_gray, (width, height), interpolation=cv.INTER_AREA)
            
            # Template matching
            result = cv.matchTemplate(haystack_gray, resized_needle, cv.TM_CCOEFF_NORMED)
            
            # Get the best match location
            _, max_val, _, max_loc = cv.minMaxLoc(result)
            
            if max_val >= threshold:
                roi_x, roi_y = max_loc
                roi_w, roi_h = width, height
                
                # Check if within boundaries
                if roi_x + roi_w <= haystack_img.shape[1] and roi_y + roi_h <= haystack_img.shape[0]:
                    roi = haystack_hsv[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    
                    # Calculate color histogram of ROI
                    roi_hist = cv.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
                    cv.normalize(roi_hist, roi_hist, 0, 1, cv.NORM_MINMAX)
                    
                    # Compare histograms
                    hist_match = cv.compareHist(needle_hist, roi_hist, cv.HISTCMP_CORREL)
                    
                    # Combined score
                    combined_score = (1 - color_weight) * max_val + color_weight * hist_match
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_match = max_loc
                        best_scale = scale
        
        # If we found a match for this template
        if best_match is not None:
            # Calculate dimensions based on best scale
            best_width = int(needle_img.shape[1] * best_scale)
            best_height = int(needle_img.shape[0] * best_scale)
            
            # Color analysis for better boundary detection
            roi_x, roi_y = best_match
            roi = haystack_img[roi_y:roi_y+best_height, roi_x:roi_x+best_width]
            roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
            
            # Get color range from this needle
            h, s, v = cv.split(needle_hsv)
            h_mean, h_std = np.mean(h), np.std(h)
            s_mean, s_std = np.mean(s), np.std(s)
            v_mean, v_std = np.mean(v), np.std(v)
            
            # Create mask based on dominant colors
            lower_bound = np.array([max(0, h_mean - h_std * 2), 
                                   max(0, s_mean - s_std * 2), 
                                   max(0, v_mean - v_std * 2)])
            upper_bound = np.array([min(180, h_mean + h_std * 2), 
                                   min(255, s_mean + s_std * 2), 
                                   min(255, v_mean + v_std * 2)])
            
            mask = cv.inRange(roi_hsv, lower_bound, upper_bound)
            
            # Find contours
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            if contours and len(contours) > 0:
                # Find largest contour
                largest_contour = max(contours, key=cv.contourArea)
                
                if cv.contourArea(largest_contour) > 50:
                    # Draw contour with random color for this template
                    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    cv.drawContours(result_img, [largest_contour], 0, color, 2, offset=(roi_x, roi_y))
                    
                    # Get bounding box
                    x, y, w, h = cv.boundingRect(largest_contour)
                    x += roi_x
                    y += roi_y
                    
                    # Draw rectangle
                    cv.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                    
                    # Add template name and confidence
                    label = f"{needle_name}: {best_score:.2f}"
                    cv.putText(result_img, label, (x, y - 10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    all_matches.append({
                        'template': needle_name,
                        'position': (x, y),
                        'size': (w, h),
                        'confidence': best_score,
                        'color': color
                    })
            else:
                # Just use the rectangle if no good contour found
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                cv.rectangle(result_img, best_match, 
                           (best_match[0] + best_width, best_match[1] + best_height), color, 2)
                
                label = f"{needle_name}: {best_score:.2f}"
                cv.putText(result_img, label, (best_match[0], best_match[1] - 10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                all_matches.append({
                    'template': needle_name,
                    'position': best_match,
                    'size': (best_width, best_height),
                    'confidence': best_score,
                    'color': color
                })
    
    # Sort matches by confidence
    all_matches.sort(key=lambda x: x['confidence'], reverse=True)
    
    return result_img, all_matches

# Example usage
def main():
    haystack_folder = 'haystacks'
    needle_folder = 'templates'
    output_folder = 'output'

    # Create necessary folders if they don't exist
    if not os.path.exists(needle_folder):
        os.makedirs(needle_folder)
        print(f"Created templates folder: {needle_folder}")
        print("Please add your template images to this folder and run again.")
        return

    if not os.path.exists(haystack_folder):
        os.makedirs(haystack_folder)
        print(f"Created haystacks folder: {haystack_folder}")
        print("Please add your haystack images to this folder and run again.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Process each haystack image in the folder
    haystack_files = [os.path.join(haystack_folder, f) for f in os.listdir(haystack_folder)]
    
    for haystack_path in haystack_files:
        result_img, matches = multi_template_matching(
            haystack_path, 
            needle_folder,
            threshold=0.65,
            color_weight=0.6
        )
        
        if result_img is not None and matches:
            haystack_name = os.path.basename(haystack_path)
            output_path = os.path.join(output_folder, f'multi_template_{haystack_name}_result.png')
            cv.imwrite(output_path, result_img)
            print(f"Result saved as '{output_path}'")
        else:
            print(f"No matches found for '{haystack_path}'")

if __name__ == "__main__":
    main()