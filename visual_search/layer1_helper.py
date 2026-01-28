import cv2
import numpy as np

def layer_1_structural_salience(image_path):
    img = cv2.imread(image_path)
    if img is None: return {"error": "Image not found"}
    
    h_img, w_img = img.shape[:2]
    total_area = h_img * w_img
    
    # 1. Structural Truth Map (Canny Edges)
    # Use median pixel intensity to set Canny thresholds automatically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    edges = cv2.Canny(gray, lower, upper)
    
    # DILATION: Create a "Landing Zone" for valid contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_landing_zone = cv2.dilate(edges, kernel)

    # 2. Candidate Generation (Smart Otsu)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Auto-Invert Logic (Check corners for light background)
    corners = [gray[0,0], gray[0, w_img-1], gray[h_img-1, 0], gray[h_img-1, w_img-1]]
    if np.mean(corners) > 140: 
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        
    _, binary = cv2.threshold(blurred, 0, 255, thresh_type)
    
    # Clean up candidates
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel) 
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    # Contrast Map for scoring
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    for c in contours:
        area = cv2.contourArea(c)
        
        # --- A. STRICT REJECTION LOGIC ---
        
        # 1. Ignore tiny noise
        if area < 500: continue
        
        # 2. Strict Canvas Limit:
        # If the shape is more than 90% of the image, it's the background/frame.
        if area > (0.90 * total_area): 
            continue 
            
        # 3. Border Touch Check:
        # If it touches the edge AND is reasonably large (>20%), it's likely a crop artifact.
        x, y, w, h = cv2.boundingRect(c)
        touches_border = (x <= 5) or (y <= 5) or (x+w >= w_img-5) or (y+h >= h_img-5)
        
        if touches_border and area > (0.20 * total_area):
            continue

        # --- B. EDGE SUPPORT CALCULATION (Restored) ---
        
        # 1. Create a mask of JUST the contour perimeter
        mask_perimeter = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(mask_perimeter, [c], -1, 255, 1) # Thickness = 1
        
        # 2. Measure overlap with the "Truth Map" (Canny)
        overlap = cv2.bitwise_and(mask_perimeter, edge_landing_zone)
        
        # 3. Calculate Ratio
        perimeter_pixels = cv2.countNonZero(mask_perimeter)
        overlap_pixels = cv2.countNonZero(overlap)
        
        if perimeter_pixels > 0:
            edge_support = overlap_pixels / perimeter_pixels
        else:
            edge_support = 0
            
        # 4. Filter: If it's a ghost (no edges), kill it.
        if edge_support < 0.25: 
            continue

        # --- C. SCORING ---
        
        # Calculate mean contrast inside the shape
        mask_filled = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask_filled, [c], -1, 255, -1)
        mean_contrast = cv2.mean(magnitude, mask=mask_filled)[0]
        
        # Score: (Area^1.5) * Contrast * EdgeSupport
        score = (area ** 1.5) * mean_contrast * edge_support
        
        candidates.append({
            "score": score,
            "contour": c,
            "metrics": {
                "area": area,
                "edge_support": round(edge_support, 2),
                "mean_contrast": round(mean_contrast, 2)
            }
        })

    if not candidates: 
        return {"layer": 1, "status": "no_structurally_valid_object"}

    # 3. Select Winner
    winner = max(candidates, key=lambda x: x["score"])
    c = winner["contour"]
    
    # 4. Final Geometric Metrics
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0: perimeter = 1
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0
    
    epsilon = 0.03 * perimeter 
    approx = cv2.approxPolyDP(c, epsilon, True)
    vertices = len(approx)

    # 5. Classification
    if circularity > 0.82:
        label = "Circle"
    elif solidity < 0.88:
        label = "Complex" 
    else:
        label = "Polygon"

    return {
        "layer": 1,
        "classification": {
            "label": label,
            "confidence": round(winner["metrics"]["edge_support"], 2)
        },
        "geometry": {
            "vertices": vertices,
            "circularity": round(circularity, 3),
            "solidity": round(solidity, 3),
            "area": area
        },
        "debug_data": {
            "edge_support_ratio": winner["metrics"]["edge_support"],
            "contrast_intensity": winner["metrics"]["mean_contrast"]
        }
    }