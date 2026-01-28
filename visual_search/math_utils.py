import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

# Import the config we just created
try:
    from src.config import Layer1Config, RimRecoveryConfig
except ImportError:
    # Fallback for standalone testing
    import logging
    logging.warning("Could not import config - using defaults")
    class Layer1Config:
        NMS_IOU_THRESHOLD = 0.50
        NMS_CONTAINMENT_THRESHOLD = 0.50

logger = logging.getLogger(__name__)

def compute_iou(bbox1: Tuple[int, int, int, int], 
                bbox2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Returns a value between 0.0 and 1.0.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection rectangle
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def non_maximum_suppression(candidates: List[Dict], 
                            iou_threshold: float = 0.50,
                            containment_threshold: float = 0.50) -> List[Dict]:
    """
    Hybrid NMS handling both overlapping duplicates AND containment.
    
    FIXED: Now properly uses IoU for overlap detection instead of one-sided intersection.
    
    Strategy:
    1. Sorts by Area (Largest first) - ensures parent objects are processed before children
    2. IoU check - removes candidates with high overlap relative to UNION (same-size duplicates)
    3. Containment check - removes small objects mostly inside larger ones (fragment suppression)
    4. Center proximity - catches near-identical detections with slightly different shapes
    
    Args:
        candidates: List of detection dicts with 'bbox' and 'geometry' keys
        iou_threshold: Overlap threshold for IoU-based suppression (default 0.50)
        containment_threshold: Overlap threshold for containment suppression (default 0.50)
    
    Returns:
        Filtered list of unique detections
    """
    if not candidates: 
        return []
    
    # Sort by area (largest first) - CRITICAL for parent-child logic
    candidates.sort(key=lambda x: x["geometry"]["area"], reverse=True)
    
    kept = []
    
    for cand in candidates:
        is_duplicate = False
        x1, y1, w1, h1 = cand['bbox']
        cand_area = w1 * h1
        
        for accepted in kept:
            x2, y2, w2, h2 = accepted['bbox']
            
            # Calculate intersection rectangle
            ix = max(x1, x2)
            iy = max(y1, y2)
            iw = min(x1+w1, x2+w2) - ix
            ih = min(y1+h1, y2+h2) - iy
            
            if iw > 0 and ih > 0:
                intersection = iw * ih
                accepted_area = w2 * h2
                
                # === CHECK 1: IoU (Intersection over Union) ===
                # FIXED: Calculate overlap relative to BOTH objects (union)
                # This catches same-size duplicates or significant overlaps
                union = cand_area + accepted_area - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > iou_threshold:
                    is_duplicate = True
                    logger.debug(f"Suppressing duplicate: IoU={iou:.3f} > {iou_threshold}")
                    break
                
                # === CHECK 2: Containment (Parent-Child Suppression) ===
                # Calculate overlap relative to the SMALLER candidate
                # Since sorted by area, 'accepted' is the parent, 'cand' is potential child
                overlap_ratio = intersection / cand_area
                
                if overlap_ratio > containment_threshold:
                    is_duplicate = True
                    logger.debug(f"Suppressing contained object: {overlap_ratio:.3f} of candidate inside accepted")
                    break
                
                # === CHECK 3: Center Proximity (Bullseye Rule) ===
                # If centers are very close, they're the same object
                # even if bbox shapes differ slightly
                cx1, cy1 = x1 + w1//2, y1 + h1//2
                cx2, cy2 = x2 + w2//2, y2 + h2//2
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                
                # Distance threshold: 10% of the larger object's size
                if dist < (max(w2, h2) * 0.10):
                    is_duplicate = True
                    logger.debug(f"Suppressing near-identical center: dist={dist:.1f}px")
                    break
        
        if not is_duplicate:
            kept.append(cand)
            
    logger.info(f"NMS: {len(candidates)} candidates → {len(kept)} kept")
    return kept

def is_contour_valid(contour: np.ndarray, 
                     min_area: float = 10.0,
                     min_perimeter: float = 10.0) -> Tuple[bool, str]:
    """
    Validate contour for degenerate cases BEFORE using it.
    Rejects lines, single points, and impossible geometries.
    """
    if contour is None or len(contour) < 3:
        return False, "insufficient_points"
    
    area = cv2.contourArea(contour)
    if area < min_area:
        return False, "area_too_small"
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter < min_perimeter:
        return False, "perimeter_degenerate"
        
    # Check for numerical stability (div by zero prevention)
    if perimeter == 0:
        return False, "zero_perimeter"
    
    # Geometric sanity check: perimeter must be at least circle-like
    # Perimeter of circle = 2*sqrt(pi*area) ≈ 3.54 * sqrt(area)
    # A straight line back-and-forth has perimeter ~ 2*length, area ~ 0.
    # We use a loose bound: perimeter < 2 * sqrt(area) implies impossible compactness 
    # (actually circle is ~3.54, square is 4.0). 
    # If perimeter is smaller than 2*sqrt(area), mathematical error occurred.
    if perimeter < np.sqrt(area) * 2:
        return False, "impossible_geometry"
    
    return True, "valid"

# ... (Keep imports and compute_iou/nms/is_contour_valid as is) ...

def detect_background_histogram(gray_image: np.ndarray) -> Tuple[float, str]:
    """
    Robust background detection prioritizing corner sampling (standard for coins),
    falling back to histogram only if corners are noisy.
    """
    h, w = gray_image.shape
    
    # 1. Corner Sampling (Robust Median)
    # Sample 5x5 patches from corners to avoid single-pixel noise
    corners = []
    margin = 5
    if h > 20 and w > 20:
        corners.extend(gray_image[0:margin, 0:margin].flatten())       # Top-Left
        corners.extend(gray_image[0:margin, w-margin:w].flatten())     # Top-Right
        corners.extend(gray_image[h-margin:h, 0:margin].flatten())     # Bottom-Left
        corners.extend(gray_image[h-margin:h, w-margin:w].flatten())   # Bottom-Right
        
        corner_median = np.median(corners)
        corner_std = np.std(corners)
        
        # If corners are consistent (std < 15), trust them absolutely.
        # This fixes "Coin fills the frame" issues.
        if corner_std < 15:
            bg_type = "light" if corner_median > 127 else "dark"
            return float(corner_median), bg_type

    # 2. Histogram Fallback (Complex scenes only)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    dark_peak = np.sum(hist[0:50])
    light_peak = np.sum(hist[205:256])
    
    if light_peak > dark_peak * 2:
        bright_region = gray_image[gray_image > 200]
        bg_value = np.mean(bright_region) if len(bright_region) > 0 else 240
        return float(bg_value), "light"
    elif dark_peak > light_peak * 2:
        dark_region = gray_image[gray_image < 50]
        bg_value = np.mean(dark_region) if len(dark_region) > 0 else 20
        return float(bg_value), "dark"
    
    return float(np.argmax(hist)), "mixed"

# ... (Keep remaining functions) ...

def compute_circularity_safe(contour: np.ndarray) -> float:
    """Safe circularity calculation handling invalid contours."""
    is_valid, _ = is_contour_valid(contour)
    if not is_valid: return 0.0
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # 4*pi*area / perim^2
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return min(circularity, 1.0)

def validate_rim_recovery(recovered_contour: np.ndarray,
                          seed_contour: np.ndarray,
                          image_shape: Tuple[int, int]) -> bool:
    """
    Sanity checks for rim recovery output.
    Ensures the 'recovered' rim actually relates to the 'seed' fragment.
    """
    h, w = image_shape
    
    # 1. Basic Validity
    is_valid, _ = is_contour_valid(recovered_contour, min_area=100)
    if not is_valid: return False
    
    # 2. Image Bounds
    x, y, w_box, h_box = cv2.boundingRect(recovered_contour)
    if w_box > w * 1.1 or h_box > h * 1.1: return False
    
    # 3. Center Alignment
    seed_moments = cv2.moments(seed_contour)
    rec_moments = cv2.moments(recovered_contour)
    
    if seed_moments["m00"] == 0 or rec_moments["m00"] == 0: return False
    
    seed_cx = seed_moments["m10"] / seed_moments["m00"]
    seed_cy = seed_moments["m01"] / seed_moments["m00"]
    
    rec_cx = rec_moments["m10"] / rec_moments["m00"]
    rec_cy = rec_moments["m01"] / rec_moments["m00"]
    
    # Distance between centers
    dist = np.sqrt((seed_cx - rec_cx)**2 + (seed_cy - rec_cy)**2)
    
    # Allow deviation up to 30% of the recovered size
    if dist > max(w_box, h_box) * 0.3: return False
    
    # 4. Size Logic (Parent must be > Child)
    # Recovered rim should be at least 90% size of seed (allows for minor discrepancies)
    if cv2.contourArea(recovered_contour) < cv2.contourArea(seed_contour) * 0.9: 
        return False
        
    return True
def fit_circle_to_points(points: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[float], float]:
    """
    Kasa's method: Least squares circle fit
    
    Given scattered points (e.g., from fragment contours), find the best-fit 
    circle center and radius. This FORCES a circle fit even on fragments,
    unlike Hough which rejects incomplete arcs.
    
    Args:
        points: Nx2 array of (x, y) coordinates
    
    Returns:
        (center, radius, confidence) tuple:
        - center: (cx, cy) or None if fit fails
        - radius: float or None if fit fails
        - confidence: 0.0 to 1.0 based on fit quality (RMS error)
    """
    # Convert to numpy array
    if isinstance(points, list):
        points = np.array(points, dtype=np.float64)
    else:
        points = points.astype(np.float64)
    
    # Need at least 3 points for a circle
    if len(points) < 3:
        logger.debug("Cannot fit circle: fewer than 3 points")
        return None, None, 0.0
    
    # Reshape if needed
    if len(points.shape) == 3:  # Contour format (N, 1, 2)
        points = points.reshape(-1, 2)
    
    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Build design matrix for least squares
    # We solve: [x, y, 1] * [a, b, c]^T = x^2 + y^2
    A = np.column_stack([x, y, np.ones(len(x))])
    b = x**2 + y**2
    
    # Solve least squares
    try:
        params, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        logger.debug("Circle fit failed: singular matrix")
        return None, None, 0.0
    
    a, b_param, c = params
    
    # Convert to center and radius
    # Circle equation: (x - cx)^2 + (y - cy)^2 = r^2
    # Expanded: x^2 + y^2 - 2*cx*x - 2*cy*y + (cx^2 + cy^2 - r^2) = 0
    # Comparing: -2*cx = a, -2*cy = b, cx^2 + cy^2 - r^2 = c
    cx = a / 2.0
    cy = b_param / 2.0
    radius_squared = c + cx**2 + cy**2
    
    if radius_squared < 0:
        logger.debug("Circle fit failed: negative radius")
        return None, None, 0.0
    
    radius = np.sqrt(radius_squared)
    
    # Calculate fit quality (RMS error normalized by radius)
    distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    errors = np.abs(distances - radius)
    rms_error = np.sqrt(np.mean(errors**2))

    # Geometric fit confidence (how well points lie on a circle)
    normalized_error = rms_error / radius if radius > 0 else 1.0
    geometric_conf = 1.0 / (1.0 + normalized_error)

    # Arc coverage factor (how much of the circle is represented)
    # For scattered fragments, we need to know: is this 10% or 90% of a circle?
    point_angles = np.arctan2(y - cy, x - cx)
    point_angles = np.sort(point_angles)

    # Calculate angular gaps
    gaps = np.diff(point_angles)
    gaps = np.append(gaps, (2*np.pi - point_angles[-1] + point_angles[0]))

    # Coverage = 1.0 - (largest_gap / 2π)
    # Full circle: all gaps small → coverage ≈ 1.0
    # Partial arc: one huge gap → coverage < 1.0
    largest_gap = np.max(gaps)
    arc_coverage = 1.0 - (largest_gap / (2 * np.pi))

    # Combined confidence: geometry × coverage
    # Full circle with good fit: both ≈ 1.0 → conf ≈ 1.0
    # Fragment arc with good fit: coverage ≈ 0.2 → conf ≈ 0.5
    confidence = geometric_conf * np.sqrt(arc_coverage)

    logger.debug(f"Circle fit: center=({cx:.1f}, {cy:.1f}), r={radius:.1f}, "
                f"geom_conf={geometric_conf:.3f}, coverage={arc_coverage:.3f}, "
                f"final_conf={confidence:.3f}")
    
    return (cx, cy), radius, confidence


def validate_with_annulus_support(image: np.ndarray, 
                                  center: Tuple[float, float], 
                                  radius: float,
                                  band_width: int = 5) -> float:
    """
    Check edge support in a RING (annulus), not a single-pixel circle.
    
    This accounts for ancient coins with wobbly, irregular rims where edges
    may not lie on a perfect mathematical circle.
    
    Args:
        image: BGR or grayscale image
        center: (cx, cy) circle center
        radius: Circle radius
        band_width: Width of annulus in pixels (default: 5)
                   Edges within radius ± band_width count as support
    
    Returns:
        Support ratio: 0.0 to 1.0 (fraction of annulus with edge pixels)
    """
    h, w = image.shape[:2]
    cx, cy = int(center[0]), int(center[1])
    r = int(radius)
    
    # Validate bounds
    if r < band_width:
        logger.debug(f"Radius {r} too small for band_width {band_width}")
        return 0.0
    
    # Create annulus mask
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    
    cv2.circle(outer_mask, (cx, cy), r + band_width, 255, -1)
    cv2.circle(inner_mask, (cx, cy), max(1, r - band_width), 255, -1)
    
    # Annulus = outer - inner
    annulus = cv2.subtract(outer_mask, inner_mask)
    
    # Detect edges
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    edges = cv2.Canny(gray, 50, 150)
    
    # Count overlap
    overlap = cv2.bitwise_and(annulus, edges)
    
    annulus_pixels = cv2.countNonZero(annulus)
    overlap_pixels = cv2.countNonZero(overlap)
    
    if annulus_pixels == 0:
        return 0.0
    
    support = overlap_pixels / annulus_pixels
    
    logger.debug(f"Annulus support: {overlap_pixels}/{annulus_pixels} = {support:.3f} "
                f"(band={band_width}px)")
    
    return support


def merge_by_center_consensus(hypotheses: list, 
                              tolerance_px: float = 50, 
                              radius_tolerance: float = 0.15) -> list:
    """
    Merge circle hypotheses that agree on center and radius.
    
    Key insight: Even if individual fragment fits have low confidence,
    if multiple fragments agree on the same circle geometry, that's
    strong evidence they belong to the same coin.
    
    This prevents "fragment explosion" where one coin becomes 4+ outputs.
    
    Args:
        hypotheses: List of dicts with keys: 'center', 'radius', 'confidence'
        tolerance_px: Max distance between centers to consider "same" (pixels)
        radius_tolerance: Max relative difference in radii (fraction)
    
    Returns:
        List of merged hypotheses (each representing one unified coin)
    """
    if not hypotheses:
        return []
    
    merged_groups = []
    used = set()
    
    for i, hyp1 in enumerate(hypotheses):
        if i in used:
            continue
        
        # Start new group
        group = [hyp1]
        used.add(i)
        
        cx1, cy1 = hyp1['center']
        r1 = hyp1['radius']
        
        # Find agreeing hypotheses
        for j in range(i + 1, len(hypotheses)):
            if j in used:
                continue
            
            hyp2 = hypotheses[j]
            cx2, cy2 = hyp2['center']
            r2 = hyp2['radius']
            
            # Check center proximity
            center_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            
            # Check radius agreement
            radius_diff = abs(r1 - r2) / max(r1, r2) if max(r1, r2) > 0 else 1.0
            
            if center_dist < tolerance_px and radius_diff < radius_tolerance:
                group.append(hyp2)
                used.add(j)
                logger.debug(f"Merged hypothesis {j} with {i}: "
                           f"center_dist={center_dist:.1f}px, radius_diff={radius_diff:.2%}")
        
        merged_groups.append(group)
    
    # Create unified candidates
    results = []
    for group in merged_groups:
        # Average center and radius from all agreeing hypotheses
        avg_cx = np.mean([h['center'][0] for h in group])
        avg_cy = np.mean([h['center'][1] for h in group])
        avg_radius = np.mean([h['radius'] for h in group])
        
        # Confidence boost from consensus
        # More agreeing fragments = higher confidence
        consensus_boost = min(1.0, len(group) / 4.0)  # Cap at 4 fragments
        base_conf = np.mean([h['confidence'] for h in group])
        final_conf = base_conf * 0.5 + consensus_boost * 0.5
        
        needs_review = final_conf < 0.50 or len(group) == 1
        
        if len(group) > 1:
            logger.info(f"✓ CONSENSUS MERGE: {len(group)} fragments → 1 coin "
                       f"(center=({avg_cx:.0f}, {avg_cy:.0f}), r={avg_radius:.0f}, "
                       f"conf={final_conf:.3f})")
        
        results.append({
            'center': (avg_cx, avg_cy),
            'radius': avg_radius,
            'confidence': final_conf,
            'merged_from': len(group),
            'needs_review': needs_review,
            'source_hypotheses': group
        })
    
    return results


# ============================================================================
# END OF COMMIT 1 ADDITIONS
# ============================================================================