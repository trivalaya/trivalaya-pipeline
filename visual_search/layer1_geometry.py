"""
Layer 1: Structural Salience Detection

Detects and classifies objects in coin images based on geometric properties.
Now includes two-coin resolution for auction-style obv/rev images.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from src.config import Layer1Config, CoinConfig
from src.math_utils import (
    detect_background_histogram,
    is_contour_valid,
    compute_circularity_safe,
    non_maximum_suppression,
    validate_rim_recovery
)
from src.rim_logic import recover_rim

# Import two-coin resolver (graceful fallback if not available)
try:
    from src.two_coin_resolver import TwoCoinResolver, CoinPairConfig
    TWO_COIN_RESOLVER_AVAILABLE = True
except ImportError:
    TWO_COIN_RESOLVER_AVAILABLE = False


def layer_1_structural_salience(image_path: str, sensitivity: str = "standard") -> Dict:
    """
    Main entry point for Layer 1 structural analysis.
    
    Now includes two-coin detection: if a single merged blob is detected
    with low circularity and wide aspect ratio, attempts to split it into
    separate obverse/reverse detections.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"layer": 1, "status": "error", "error": "image_load_failed"}

    h, w = img.shape[:2]
    total_area = h * w

    # --- Grayscale + background handling ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_bg, _ = detect_background_histogram(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # --- Threshold polarity ---
    thresh_type = (
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        if avg_bg > Layer1Config.BRIGHT_BACKGROUND_THRESHOLD
        else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # --- Edge detection ---
    v = np.median(gray_enhanced)
    sigma = Layer1Config.Standard.CANNY_SIGMA
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray_enhanced, lower, upper)

    edge_zone = cv2.dilate(
        edges,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )

    # --- Binary segmentation ---
    blurred = cv2.GaussianBlur(gray_enhanced, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, thresh_type)

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2
    )

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- Initial candidate extraction ---
    candidates: List[Dict] = []

    for c in contours:
        valid, _ = is_contour_valid(c, min_area=Layer1Config.Standard.MIN_AREA_PX)
        if not valid:
            continue

        area = cv2.contourArea(c)
        if area > Layer1Config.MAX_AREA_RATIO * total_area:
            continue

        # --- Edge support ---
        perimeter_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(perimeter_mask, [c], -1, 255, 1)
        perimeter_px = cv2.countNonZero(perimeter_mask)
        if perimeter_px == 0:
            continue

        overlap = cv2.bitwise_and(perimeter_mask, edge_zone)
        edge_support = cv2.countNonZero(overlap) / perimeter_px

        # Allow circular shapes through even with weaker edges
        circularity = compute_circularity_safe(c)
        if edge_support < Layer1Config.Standard.EDGE_SUPPORT_MIN and circularity < 0.70:
            continue

        # --- Optional rim recovery ---
        final_c = c
        recovered = False

        if CoinConfig.ENABLE_RIM_RECOVERY and circularity < Layer1Config.CIRCULARITY_RELAXED:
            new_c, conf = recover_rim(img, c)
            if new_c is not None and validate_rim_recovery(new_c, c, (h, w)):
                final_c = new_c
                recovered = True
                area = cv2.contourArea(final_c)
                circularity = compute_circularity_safe(final_c)

        if final_c is None or len(final_c) < 3:
            continue

        hull = cv2.convexHull(final_c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        bbox = cv2.boundingRect(final_c)

        coin_likelihood = (
            0.45 * circularity +
            0.35 * edge_support +
            0.20 * solidity
        )

        label = "Circle" if circularity > Layer1Config.CIRCULARITY_THRESHOLD else "Geometric"

        candidates.append({
            "score": area * edge_support,
            "classification": {
                "label": label,
                "confidence": round(edge_support, 2)
            },
            "geometry": {
                "area": int(area),
                "circularity": round(circularity, 3),
                "solidity": round(solidity, 3),
                "coin_likelihood": round(coin_likelihood, 3)
            },
            "contour": final_c,
            "bbox": bbox,
            "debug_data": {
                "rim_recovered": recovered
            }
        })

    if not candidates:
        return {"layer": 1, "status": "no_structurally_valid_object"}

    # --- NMS ---
    candidates = non_maximum_suppression(
        candidates,
        iou_threshold=Layer1Config.NMS_IOU_THRESHOLD,
        containment_threshold=0.40
    )

    # === TWO-COIN RESOLUTION CHECK ===
    # Trigger conditions:
    # 1. Exactly one candidate
    # 2. Low circularity (merged blob signature)  
    # 3. Wide aspect ratio (suggests side-by-side coins)
    # 4. Large area (not a fragment)
    
    if TWO_COIN_RESOLVER_AVAILABLE and len(candidates) == 1:
        two_coin_result = _try_two_coin_resolution(
            img, binary, gray_enhanced, candidates[0], (h, w)
        )
        
        if two_coin_result is not None:
            return two_coin_result

    return {
        "layer": 1,
        "status": "success",
        "objects": candidates[:Layer1Config.MAX_DETECTIONS]
    }


def _try_two_coin_resolution(img: np.ndarray,
                              binary: np.ndarray, 
                              gray: np.ndarray,
                              candidate: Dict,
                              shape: tuple) -> Optional[Dict]:
    """
    Check if single candidate is a merged two-coin blob and attempt to split.
    
    Returns:
        Dict with split results if successful, None if not triggered or failed
    """
    h, w = shape
    
    # Use resolver's should_trigger for consistent threshold behavior
    resolver = TwoCoinResolver()
    should_trigger, _ = resolver.should_trigger([candidate], (h, w))
    
    if not should_trigger:
        return None
    
    # Attempt resolution (reuse the same resolver instance)
    # --- CRITICAL FIX: Pass 'bbox' to the optimized resolver ---
    try:
        result = resolver.resolve(img, binary, gray, candidate['bbox'])
    except TypeError:
        # Fallback if old version of resolver is somehow loaded
        result = resolver.resolve(img, binary, gray)

    if result['status'] != 'split':
        # Failed - return None to fall back to single-blob result
        # But flag it for review
        return {
            "layer": 1,
            "status": "success",
            "objects": [{
                **candidate,
                "debug_data": {
                    **candidate.get("debug_data", {}),
                    "two_coin_attempted": True,
                    "two_coin_failed": result.get('debug', {})
                }
            }],
            "two_coin_resolution": {
                "triggered": True,
                "status": "failed",
                "method": result.get('method'),
                "debug": result.get('debug')
            }
        }
    
    # Success! Convert split coins to candidate format
    split_candidates = []
    
    for coin in result['coins']:
        # The crop is a sub-image, we need to record its position in the original
        crop = coin['crop']
        
        # Create a synthetic contour from the circular detection
        cx, cy = coin['center']
        r = coin['radius']
        
        # Generate circle contour points, clamped to image bounds
        theta = np.linspace(0, 2 * np.pi, 64)
        contour_points = []
        for t in theta:
            px = int(np.clip(cx + r * np.cos(t), 0, w - 1))
            py = int(np.clip(cy + r * np.sin(t), 0, h - 1))
            contour_points.append([px, py])
        contour_points = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
        
        area = np.pi * r * r
        
        split_candidates.append({
            "score": area,  # Simplified scoring for split coins
            "classification": {
                "label": "Circle",
                "confidence": 0.85  # High confidence since Hough validated
            },
            "geometry": {
                "area": int(area),
                "circularity": 0.95,  # Circles by definition
                "solidity": 0.95,
                "coin_likelihood": 0.90
            },
            "contour": contour_points,
            "bbox": coin['bbox'],
            "center": coin['center'],
            "radius": coin['radius'],
            "side": coin['side'],
            "debug_data": {
                "from_two_coin_split": True,
                "split_method": result['method']
            }
        })
    
    return {
        "layer": 1,
        "status": "success",
        "objects": split_candidates,
        "two_coin_resolution": {
            "triggered": True,
            "status": "split",
            "method": result['method'],
            "pair_score": result['debug'].get('score', 0)
        }
    }