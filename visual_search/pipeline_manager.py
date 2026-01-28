import cv2
import numpy as np
import os
from src.layer1_geometry import layer_1_structural_salience
from src.layer2_context import layer_2_context_probe

MAX_DIMENSION = 3200 

def _load_and_resize(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, 1.0
    if image_path.lower().endswith(".png"):
        img_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img_alpha is not None and img_alpha.shape[2] == 4:
            alpha = img_alpha[:,:,3]
            img[alpha == 0] = [255, 255, 255]
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale

def analyze_image(image_path):
    img, scale = _load_and_resize(image_path)
    if img is None: return {"status": "error", "error": "Load failed"}
    working_path = f"temp_work_{os.path.basename(image_path)}.jpg"
    cv2.imwrite(working_path, img)
    
    try:
        # 1. Run Layer 1
        l1_result = layer_1_structural_salience(working_path, sensitivity="standard")
        
        if "objects" not in l1_result:
             l1_result = layer_1_structural_salience(working_path, sensitivity="high")
             
        if "objects" not in l1_result:
             return {"status": "failed", "last_error": "No objects found"}

        # 2. Run Layer 2 on EACH object
        detected_objects = l1_result["objects"]
        final_results = []
        
        for i, obj_data in enumerate(detected_objects):
            single_l1 = {
                "layer": 1,
                "classification": obj_data["classification"],
                "geometry": obj_data["geometry"],
                "contour": obj_data["contour"],
                "debug_data": obj_data["debug_data"]
            }
            
            l2_result = layer_2_context_probe(working_path, single_l1)
            
            container = l2_result['context_classification']['inferred_container']
            
            # === SEMANTIC SALVAGE RULE (Commit 3) ===
            # High coin_likelihood overrides irregular geometry
            # Handles: incuse reverses, thick flans, shadow-heavy photos
            coin_likelihood = single_l1['geometry'].get('coin_likelihood', 0)
            area = single_l1['geometry']['area']
            circularity = single_l1['geometry']['circularity']
            
            # Strong coin evidence - trust it even if geometry irregular
            if coin_likelihood > 0.65 and area > 10000:
                if circularity > 0.60:
                    final_decision = "Round Object (Coin/Medallion)"
                else:
                    final_decision = "Round Object (Coin/Medallion)"  # Still a coin!
            
            # Standard mapping for lower confidence
            elif container == "Round_Artifact":
                final_decision = "Round Object (Coin/Medallion)"
            elif container == "Coin_Planchet":
                final_decision = "Round Object (Coin)"
            elif container == "Fragment":
                final_decision = "Fragment"
            elif obj_data['classification']['label'] in ["Circle", "Polygon"]:
                final_decision = "Geometric Form"
            else:
                final_decision = "Artifact (Isolated)"
            
            final_results.append({
                "id": i+1,
                "final_classification": final_decision,
                "layer_1": single_l1,
                "layer_2": l2_result
            })

        return {
            "status": "success", 
            "detections": final_results 
        }

    finally:
        if os.path.exists(working_path): os.remove(working_path)
