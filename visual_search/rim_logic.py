import cv2
import numpy as np
import logging
from typing import Optional, Tuple

try:
    from src.config import RimRecoveryConfig
    from src.math_utils import (
        validate_rim_recovery,
        fit_circle_to_points,
        validate_with_annulus_support
    )
except ImportError:
    logging.warning("Config/Utils import failed. Using defaults.")
    class RimRecoveryConfig:
        EDGE_SUPPORT_MIN = 0.15
        EDGE_SUPPORT_FALLBACK = 0.12
    
    def validate_rim_recovery(rec, seed, shape): 
        return True
    def fit_circle_to_points(pts):
        return None, None, 0.0
    def validate_with_annulus_support(img, center, radius, band=5):
        return 0.0

logger = logging.getLogger(__name__)


def geometric_fit_recovery(image_bgr, seed_contour):
    """
    PRIMARY: Geometric circle fitting (works on fragments).
    Fits circle to existing points instead of detecting.
    """
    if seed_contour is None or len(seed_contour) < 10:
        return None, 0
    
    h, w = image_bgr.shape[:2]
    points = seed_contour.reshape(-1, 2)
    
    center, radius, fit_conf = fit_circle_to_points(points)
    
    if center is None:
        logger.debug("Geometric fit failed")
        return None, 0
    
    cx, cy = center
    
    if radius < 10 or radius > min(w, h) / 2:
        logger.debug(f"Invalid radius {radius:.0f}")
        return None, 0
    
    if not (0 <= cx < w and 0 <= cy < h):
        logger.debug(f"Center outside image")
        return None, 0
    
    band_width = max(5, int(0.02 * radius))
    edge_support = validate_with_annulus_support(image_bgr, center, radius, band_width)
    
    combined_conf = fit_conf * 0.5 + edge_support * 0.5
    
    logger.info(f"Geometric fit: r={radius:.0f}px, fit={fit_conf:.3f}, edge={edge_support:.3f}, combined={combined_conf:.3f}")
    
    if combined_conf > 0.20:
        theta = np.linspace(0, 2*np.pi, 360)
        x_pts = cx + radius * np.cos(theta)
        y_pts = cy + radius * np.sin(theta)
        contour = np.column_stack((x_pts, y_pts)).astype(np.int32).reshape((-1, 1, 2))
        logger.info(f"SUCCESS: Geometric fit conf={combined_conf:.3f}")
        return contour, combined_conf
    
    return None, 0


def recover_rim(image_bgr, seed_contour):
    """Use geometric fitting (replaces old Hough method)"""
    return geometric_fit_recovery(image_bgr, seed_contour)
