import cv2
import numpy as np

def layer_2_context_probe(image_path, layer_1_result):
    # 1. Input Validation
    if "geometry" not in layer_1_result:
        return {"status": "skipped_no_layer1_target"}
        
    img = cv2.imread(image_path)
    if img is None: return {"status": "error_loading_image"}
    
    h, w = img.shape[:2]
    area_ratio = layer_1_result['geometry']['area'] / (h * w)
    l1_label = layer_1_result['classification']['label']
    
    # Retrieve "Content" metrics from Layer 1 debug data
    # (We want to know: "Is there something on it?")
    internal_contrast = 0
    if "debug_data" in layer_1_result:
        internal_contrast = layer_1_result["debug_data"].get("contrast_intensity", 0)

    # --- SHORTCUT: THE "SEE WHAT IS THERE" CHECK ---
    # Logic: If it's big, distinct (solid), and has surface details, it's an object.
    # We don't care if the edge is perfectly circular (stitching issues).
    
    is_big_enough = area_ratio > 0.15
    is_distinct = layer_1_result['geometry']['solidity'] > 0.85  # Convex-ish
    has_content = internal_contrast > 20  # Arbitrary unit from Sobel magnitude
    
    if is_big_enough and is_distinct:
        # It's definitely a "Whole Object" (not a fragment)
        # Now, what shape is it?
        circ = layer_1_result['geometry']['circularity']
        
        container_type = "Unknown_Artifact"
        if circ > 0.60:
            container_type = "Round_Artifact" # "This is round"
        else:
            container_type = "Irregular_Artifact"
            
        return {
            "layer": 2,
            "context_classification": {
                "label": "Self-Contained",
                "inferred_container": container_type, 
                "confidence": 1.0
            },
            "metrics": { "fit_error": 0.0, "content_score": internal_contrast }
        }

    # 2. Setup Probe Center (For smaller fragments/specks)
    if "contour" in layer_1_result and layer_1_result["contour"] is not None:
        c = layer_1_result["contour"]
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            x, y, w_box, h_box = cv2.boundingRect(c)
            cx, cy = x + w_box//2, y + h_box//2
    else:
        cx, cy = int(w/2), int(h/2)

    # 3. Wall Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Smart Background Check
    corners = [gray[0,0], gray[0, w-1], gray[h-1, 0], gray[h-1, w-1]]
    avg_bg = np.mean(corners)
    
    if avg_bg > 200:
        enhanced = gray 
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
    edges = cv2.Canny(enhanced, 30, 100)
    
    # Masking: Only mask if it's a small "Speck" inside the object
    if area_ratio < 0.10 and "contour" in layer_1_result:
        cv2.drawContours(edges, [layer_1_result["contour"]], -1, 0, -1)
    
    # 4. Fire Rays
    wall_points = []
    for angle in range(0, 360, 5):
        rad = np.deg2rad(angle)
        dx, dy = np.cos(rad), np.sin(rad)
        
        start_dist = np.sqrt(layer_1_result['geometry']['area']) / 2 + 5
        max_dist = min(w, h) / 1.1
        
        for d in range(int(start_dist), int(max_dist), 4): 
            px, py = int(cx + dx * d), int(cy + dy * d)
            if px < 0 or px >= w or py < 0 or py >= h: break
            
            if edges[py, px] > 0:
                wall_points.append([px, py])
                break
                
    # 5. Geometric Fitting (The "Container" Check)
    enclosure_confidence = len(wall_points) / 72.0
    
    context_label = "Open/Fragment"
    container_shape = "None"
    fit_error = 1.0
    
    if enclosure_confidence > 0.35 and len(wall_points) > 10:
        points_array = np.array(wall_points, dtype=np.int32)
        (fit_cx, fit_cy), radius = cv2.minEnclosingCircle(points_array)
        
        errors = []
        for pt in wall_points:
            dist = np.sqrt((pt[0] - fit_cx)**2 + (pt[1] - fit_cy)**2)
            errors.append(abs(dist - radius))
            
        median_error = np.median(errors)
        fit_error = median_error / radius 
        
        # If the inferred container is round-ish, we assume it's the parent object
        if fit_error < 0.20:
            context_label = "Enclosed (Clean)"
            container_shape = "Round_Artifact"
        elif fit_error < 0.50: # Very generous for hammered/stitched items
            context_label = "Enclosed (Rough)" 
            container_shape = "Round_Artifact"
        else:
            context_label = "Enclosed (Irregular)"
            container_shape = "Fragment"
            
    return {
        "layer": 2,
        "context_classification": {
            "label": context_label,
            "inferred_container": container_shape,
            "confidence": round(enclosure_confidence, 2)
        },
        "metrics": {
            "wall_hits": len(wall_points),
            "fit_error": round(fit_error, 3)
        }
    }