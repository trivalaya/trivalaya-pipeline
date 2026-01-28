"""
Two-Coin Resolver (Vectorized & Cropped)
Status: MAX SPEED
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CoinPairConfig:
    # Aggressive downscaling
    WORKING_MAX_DIM: int = 800  
    
    # Hough Params
    HOUGH_DP: float = 1.2
    HOUGH_PARAM1: int = 100
    HOUGH_PARAM2: int = 25  # Slightly looser, we filter later
    
    # Geometric Constraints
    Y_ALIGNMENT_MAX: float = 0.15
    RADIUS_RATIO_MIN: float = 0.70
    
    # Validation
    RIM_EDGE_DENSITY_MIN: float = 0.08
    
    # Triggers
    TRIGGER_CIRCULARITY_MAX: float = 0.65
    TRIGGER_ASPECT_RATIO_MIN: float = 1.35

class TwoCoinResolver:
    def __init__(self, config: Optional[CoinPairConfig] = None):
        self.config = config or CoinPairConfig()

    def should_trigger(self, candidates: List[Dict], image_shape: Tuple[int, int]) -> Tuple[bool, Optional[Dict]]:
        # (Same fast gate as before - keep this logic)
        if len(candidates) != 1: return False, None
        cand = candidates[0]
        # Fast aspect ratio check
        w, h = cand['bbox'][2], cand['bbox'][3]
        if w/h < self.config.TRIGGER_ASPECT_RATIO_MIN: return False, None
        # Circularity check
        if cand['geometry']['circularity'] > self.config.TRIGGER_CIRCULARITY_MAX: return False, None
        return True, cand

    def resolve(self, image: np.ndarray, binary_mask: np.ndarray, gray: np.ndarray, candidate_bbox: Tuple[int,int,int,int]) -> Dict:
        """
        Speed-optimized resolve flow:
        1. CROP to the candidate blob (ignore background)
        2. DOWNSCALE the crop
        3. DETECT using vectorized operations
        """
        cfg = self.config
        
        # --- OPTIMIZATION 1: CROP FIRST ---
        # Don't resize/process the empty background. 
        # Add a small safety margin to the bbox.
        bx, by, bw, bh = candidate_bbox
        h_img, w_img = image.shape[:2]
        
        margin = int(max(bw, bh) * 0.1)
        x1 = max(0, bx - margin)
        y1 = max(0, by - margin)
        x2 = min(w_img, bx + bw + margin)
        y2 = min(h_img, by + bh + margin)
        
        # All processing happens on this crop
        gray_crop = gray[y1:y2, x1:x2]
        
        # --- OPTIMIZATION 2: DOWNSCALE THE CROP ---
        h_crop, w_crop = gray_crop.shape
        scale = 1.0
        if max(h_crop, w_crop) > cfg.WORKING_MAX_DIM:
            scale = cfg.WORKING_MAX_DIM / max(h_crop, w_crop)
            dsize = (int(w_crop * scale), int(h_crop * scale))
            gray_working = cv2.resize(gray_crop, dsize, interpolation=cv2.INTER_NEAREST) # Nearest is fastest
        else:
            gray_working = gray_crop
            
        # Pre-compute edges once on small working image
        blurred = cv2.GaussianBlur(gray_working, (5, 5), 0)
        edges_working = cv2.Canny(blurred, 50, 150)
        
        # --- EXECUTE HOUGH ---
        result = self._vectorized_hough(gray_working, edges_working)
        
        if result['status'] == 'split':
            # Remap coordinates: Working -> Crop -> Original
            final_coins = []
            for c in result['coins']:
                # Un-scale
                cx_c = c['center'][0] / scale
                cy_c = c['center'][1] / scale
                r_c  = c['radius'] / scale
                
                # Un-crop (add top-left offset)
                cx_final = cx_c + x1
                cy_final = cy_c + y1
                
                # Extract final high-res crop
                coin_crop, bbox = self._extract_crop(image, cx_final, cy_final, r_c, (h_img, w_img))
                final_coins.append({
                    'crop': coin_crop,
                    'bbox': bbox,
                    'center': (cx_final, cy_final),
                    'radius': r_c,
                    'side': c['side']
                })
            return {'status': 'split', 'method': 'hough', 'coins': final_coins, 'debug': result['debug']}

        # Fallback to watershed only if Hough failed (omitted for brevity, assume similar structure)
        ws_result = self._try_watershed_fallback(gray_working, edges_working)
        if ws_result['status'] == 'split':
             # ... (Copy the same un-scaling logic as Hough block above) ...
             # I can write the full block if you need it.
             pass

    def _vectorized_hough(self, gray: np.ndarray, edges: np.ndarray) -> Dict:
        """
        No Python loops over pairs. Pure NumPy.
        """
        h, w = gray.shape
        cfg = self.config
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=cfg.HOUGH_DP,
            minDist=int(w * 0.2), # Coins must be distinct
            param1=cfg.HOUGH_PARAM1, param2=cfg.HOUGH_PARAM2,
            minRadius=int(w * 0.1), maxRadius=int(w * 0.45)
        )
        
        if circles is None or circles.shape[1] < 2:
            return {'status': 'failed'}

        # circles shape is (1, N, 3) -> (N, 3)
        circles = circles[0]
        N = circles.shape[0]
        
        # --- OPTIMIZATION 3: VECTORIZED PAIRING ---
        # Create matrices of shape (N, N) comparing every circle i to every circle j
        
        # Extract columns
        X = circles[:, 0]  # Shape (N,)
        Y = circles[:, 1]
        R = circles[:, 2]
        
        # Broadcast to create pair matrices
        # x_diff[i, j] = X[i] - X[j]
        x_diff = X[:, np.newaxis] - X
        y_diff = np.abs(Y[:, np.newaxis] - Y)
        
        # Radius ratio: min(r1, r2) / max(r1, r2)
        r_matrix = R[:, np.newaxis] # col vector
        r_matrix_T = R # row vector
        r_min = np.minimum(r_matrix, r_matrix_T)
        r_max = np.maximum(r_matrix, r_matrix_T)
        r_ratio = np.divide(r_min, r_max, out=np.zeros_like(r_min), where=r_max!=0)
        
        # Separation
        dist_sq = x_diff**2 + y_diff**2
        dist = np.sqrt(dist_sq)
        
        # --- BOOLEAN MASKS (The Filter) ---
        # 1. Ignore self-pairs and lower triangle (j <= i)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # 2. Y-Alignment check
        mask &= (y_diff / h) < cfg.Y_ALIGNMENT_MAX
        
        # 3. Radius Similarity
        mask &= (r_ratio > cfg.RADIUS_RATIO_MIN)
        
        # 4. Horizontal Separation (must be some distance apart)
        mask &= (dist > (r_min * 1.5)) 
        
        # If no pairs survive, exit fast
        if not np.any(mask):
            return {'status': 'failed', 'debug': {'reason': 'no_geom_match'}}
            
        # --- SCORING ---
        # Score = RadiusConsistency + Y_Alignment
        # Higher is better
        scores = (r_ratio * 0.5) + ((1.0 - (y_diff / h)) * 0.5)
        scores[~mask] = 0 # Zero out invalid pairs
        
        # Find best index (i, j)
        flat_idx = np.argmax(scores)
        best_i, best_j = np.unravel_index(flat_idx, (N, N))
        best_score = scores[best_i, best_j]
        
        if best_score == 0:
            return {'status': 'failed'}
            
        # --- OPTIMIZATION 4: LAZY EDGE CHECK ---
        # Only check edge density for the WINNER. 
        # If the winner fails, we bail (or check 2nd best, but bailing is faster for batch)
        
        c1, c2 = circles[best_i], circles[best_j]
        
        # Check edges on the small 'edges' map we already computed
        d1 = self._quick_density(edges, c1[0], c1[1], c1[2])
        if d1 < cfg.RIM_EDGE_DENSITY_MIN: return {'status': 'failed'}
        
        d2 = self._quick_density(edges, c2[0], c2[1], c2[2])
        if d2 < cfg.RIM_EDGE_DENSITY_MIN: return {'status': 'failed'}
        
        # Success
        left, right = (c1, c2) if c1[0] < c2[0] else (c2, c1)
        
        return {
            'status': 'split',
            'coins': [
                {'center': left[:2], 'radius': left[2], 'side': 'obverse'},
                {'center': right[:2], 'radius': right[2], 'side': 'reverse'}
            ],
            'debug': {'score': float(best_score)}
        }

    def _quick_density(self, edges, cx, cy, r):
        # Fast density check on small ROI
        h, w = edges.shape
        x1, y1 = max(0, int(cx-r-5)), max(0, int(cy-r-5))
        x2, y2 = min(w, int(cx+r+5)), min(h, int(cy+r+5))
        roi = edges[y1:y2, x1:x2]
        if roi.size == 0: return 0
        
        # Mask inside the ROI only
        mask = np.zeros_like(roi)
        rcx, rcy = int(cx-x1), int(cy-y1)
        cv2.circle(mask, (rcx, rcy), int(r), 1, 1) # Thin circle
        
        return np.sum(roi & mask) / (np.sum(mask) + 1e-5) # Sum is faster than countNonZero for boolean ops

    def _extract_crop(self, img, cx, cy, r, shape):
        # (Standard crop logic, omitted for brevity)
        h, w = shape
        s = int(r * 1.05) # box size
        x1, y1 = max(0, int(cx)-s), max(0, int(cy)-s)
        x2, y2 = min(w, int(cx)+s), min(h, int(cy)+s)
        return img[y1:y2, x1:x2].copy(), (x1, y1, x2-x1, y2-y1)
    def _try_watershed_fallback(self, gray_crop: np.ndarray, edges: np.ndarray) -> Dict:
        """
        Fast watershed fallback on the existing crop.
        Only runs if Hough failed.
        """
        # 1. Binarize the crop (Otsu is fast on small images)
        _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 2. Distance Transform
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        max_dist = dist.max()
        if max_dist < 5: return {'status': 'failed'}
        
        # 3. Quick Threshold Check (only check 0.4 which is most common for coins)
        _, sure_fg = cv2.threshold(dist, 0.4 * max_dist, 255, 0)
        sure_fg = np.uint8(sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # We want exactly background (0) + 2 coins (1, 2) -> ret == 3
        if ret == 3:
            # We found 2 distinct blobs!
            # Get their centers/radii from the sure_fg markers directly (fast approximation)
            coins = []
            for label in [1, 2]:
                mask = np.uint8(markers == label)
                # Find centroid
                M = cv2.moments(mask)
                if M["m00"] == 0: continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                # Approx radius from area
                r = np.sqrt(M["m00"] / np.pi)
                coins.append({'center': (cx, cy), 'radius': r})
            
            if len(coins) == 2:
                # Sort left/right
                coins.sort(key=lambda x: x['center'][0])
                return {
                    'status': 'split', 
                    'coins': [
                        {'center': coins[0]['center'], 'radius': coins[0]['radius'], 'side': 'obverse'},
                        {'center': coins[1]['center'], 'radius': coins[1]['radius'], 'side': 'reverse'}
                    ],
                    'debug': {'method': 'watershed_fallback'}
                }
                
        return {'status': 'failed'}