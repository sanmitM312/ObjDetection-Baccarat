import cv2
import numpy as np
import matplotlib.pyplot as plt

def orb_template_matching(template_path, image_path, min_match_count=10, distance_threshold=0.75):
    """
    Performs scale-invariant template matching using ORB features.
    
    Args:
        template_path (str): Path to the template image
        image_path (str): Path to the haystack image
        min_match_count (int): Minimum number of good matches to consider detection valid
        distance_threshold (float): Threshold for good matches (0.0-1.0, lower is stricter)
        
    Returns:
        list: List of dictionaries containing match information (coordinates, scale, etc.)
    """
    # Read images
    template = cv2.imread(template_path)
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(template_gray.shape)
    print(image.shape)
    cv2.imshow("sjfskf",template_gray)
    cv2.waitKey(0)
    cv2.imshow("hskjf",image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Initialize ORB detector (increase nfeatures for better matching)
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    
    # Find keypoints and descriptors
    template_keypoints, template_descriptors = orb.detectAndCompute(template_gray, None)
    image_keypoints, image_descriptors = orb.detectAndCompute(image_gray, None)
    
    # Check if keypoints were found
    if template_keypoints is None or len(template_keypoints) == 0:
        print("No keypoints found in template image.")
        return []
    
    if image_keypoints is None or len(image_keypoints) == 0:
        print("No keypoints found in haystack image.")
        return []
    
    # Create BFMatcher object with Hamming distance (suitable for binary descriptors like ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Match descriptors using knnMatch
    matches = bf.knnMatch(template_descriptors, image_descriptors, k=2)
    
    # Apply ratio test to find good matches
    good_matches = []
    for pair in matches:
        if len(pair) == 2:  # Sometimes knnMatch may return only 1 match
            m, n = pair
            if m.distance < distance_threshold * n.distance:
                good_matches.append(m)
    
    results = []
    print(good_matches)
    if len(good_matches) >= min_match_count:
        # Extract coordinates of matched keypoints
        src_pts = np.float32([template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([image_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Check if homography was successfully found
        if H is None:
            print("Could not find a valid homography.")
            return []
        
        # Get template dimensions
        h, w = template_gray.shape
        
        # Define template corners
        template_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        # Transform corners to image coordinates
        transformed_corners = cv2.perspectiveTransform(template_corners, H)
        
        # Calculate center point
        center_x = np.mean([p[0][0] for p in transformed_corners])
        center_y = np.mean([p[0][1] for p in transformed_corners])
        
        # Calculate scale (approximate by averaging x and y scale factors)
        original_width = np.sqrt((w-1)**2)
        original_height = np.sqrt((h-1)**2)
        
        transformed_width = np.sqrt((transformed_corners[3][0][0] - transformed_corners[0][0][0])**2 + 
                                   (transformed_corners[3][0][1] - transformed_corners[0][0][1])**2)
        transformed_height = np.sqrt((transformed_corners[1][0][0] - transformed_corners[0][0][0])**2 + 
                                    (transformed_corners[1][0][1] - transformed_corners[0][0][1])**2)
        
        scale_x = transformed_width / original_width
        scale_y = transformed_height / original_height
        scale = (scale_x + scale_y) / 2
        
        # Get inliers count (matches that fit the homography)
        inliers_count = np.sum(mask)
        
        # Store match information
        match_info = {
            'center': (center_x, center_y),
            'corners': transformed_corners.tolist(),
            'scale': scale,
            'confidence': inliers_count / len(good_matches),
            'homography': H.tolist(),
            'num_matches': len(good_matches),
            'inliers': inliers_count
        }
        
        results.append(match_info)
    
    return results

def visualize_matches(template_path, image_path, matches):
    """
    Visualizes the matches found in the haystack image.
    
    Args:
        template_path (str): Path to the template image
        image_path (str): Path to the haystack image
        matches (list): List of match dictionaries returned by orb_template_matching
    """
    # Read images
    template = cv2.imread(template_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create a copy for drawing
    result_img = image_rgb.copy()
    
    plt.figure(figsize=(12, 8))
    
    for i, match in enumerate(matches):
        # Draw bounding box
        corners = np.array(match['corners']).reshape(-1, 2).astype(int)
        for j in range(4):
            pt1 = tuple(corners[j])
            pt2 = tuple(corners[(j+1) % 4])
            cv2.line(result_img, pt1, pt2, (0, 255, 0), 3)
        
        # Draw center point
        center = tuple(map(int, match['center']))
        cv2.circle(result_img, center, 5, (255, 0, 0), -1)
        
        # Add scale and confidence text
        text = f"Scale: {match['scale']:.2f}, Conf: {match['confidence']:.2f}"
        cv2.putText(result_img, text, (corners[0][0], corners[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    plt.imshow(result_img)
    plt.title(f"Found {len(matches)} matches")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_keypoints_matches(template_path, image_path, max_matches=30):
    """
    Visualizes keypoints and matches between template and image.
    Useful for debugging and understanding the matching process.
    
    Args:
        template_path (str): Path to the template image
        image_path (str): Path to the haystack image
        max_matches (int): Maximum number of matches to display
    """
    # Read images
    template = cv2.imread(template_path)
    image = cv2.imread(image_path)
    
    # Convert to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    
    # Find keypoints and descriptors
    template_keypoints, template_descriptors = orb.detectAndCompute(template_gray, None)
    image_keypoints, image_descriptors = orb.detectAndCompute(image_gray, None)
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(template_descriptors, image_descriptors)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw top matches
    result = cv2.drawMatches(template, template_keypoints, image, image_keypoints, 
                             matches[:max_matches], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Convert to RGB for matplotlib
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(result_rgb)
    plt.title(f'Top {min(max_matches, len(matches))} ORB matches')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    template_path = "image.png"
    image_path = "casino8_mac.jpg"
    
    # Uncomment to visualize keypoints and matches (helpful for debugging)
    visualize_keypoints_matches(template_path, image_path)
    
    matches = orb_template_matching(template_path, image_path, min_match_count=10)
    
    if matches:
        print(f"Found {len(matches)} matches:")
        for i, match in enumerate(matches):
            print(f"Match {i+1}:")
            print(f"  Center: {match['center']}")
            print(f"  Scale: {match['scale']:.2f}")
            print(f"  Confidence: {match['confidence']:.2f}")
            print(f"  Inliers/Matches: {match['inliers']}/{match['num_matches']}")
        
        visualize_matches(template_path, image_path, matches)
    else:
        print("No matches found.")