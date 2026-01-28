"""
Configuration Module for Coin Detection Pipeline
All magic numbers and thresholds consolidated here for tuning and maintainability.
"""

# === IMAGE PREPROCESSING ===
MAX_DIMENSION = 3200  # Resize limit to prevent memory issues

# === LAYER 1: GEOMETRY DETECTION ===
class Layer1Config:
    """Structural salience detection parameters"""
    
    # Sensitivity Presets
    class Standard:
        EDGE_SUPPORT_MIN = 0.05  # Minimum edge overlap to consider contour valid
        CANNY_SIGMA = 0.33       # Canny threshold calculation multiplier
        MIN_AREA_PX = 300        # Minimum contour area in pixels
        
    class High:
        EDGE_SUPPORT_MIN = 0.02
        CANNY_SIGMA = 0.60
        MIN_AREA_PX = 200
    
    # Morphological Operations
    CLOSE_KERNEL_SIZE_STANDARD = 7  # Kernel for closing gaps (standard mode)
    CLOSE_KERNEL_SIZE_HIGH = 9      # Kernel for closing gaps (high sensitivity)
    CLOSE_ITERATIONS = 2            # Number of closing operations
    
    # Background Detection
    BRIGHT_BACKGROUND_THRESHOLD = 110   # Gray value to classify as light background
    VERY_BRIGHT_THRESHOLD = 200         # For aggressive foreground filtering
    
    # Contour Filtering
    MAX_AREA_RATIO = 0.98       # Maximum contour area as fraction of image
    BORDER_TOLERANCE_PX = 2     # Pixels from edge to consider "touching border"
    BORDER_TOUCH_MAX_AREA = 0.30  # Max area ratio for border-touching contours
    
    # Geometry Classification
    CIRCULARITY_THRESHOLD = 0.82  # Minimum circularity to label as "Circle"
    SOLIDITY_THRESHOLD = 0.88     # Maximum solidity for "Complex" shapes
    POLYGON_APPROXIMATION = 0.03  # Epsilon for polygon vertex reduction
    
    # === RIM RECOVERY TRIGGERS (Layer 1.5) ===
    CIRCULARITY_STRICT = 0.90     # OLD threshold - too strict for ancient coins
    CIRCULARITY_RELAXED = 0.75    # NEW threshold - triggers recovery earlier
    MIN_AREA_RATIO = 0.20         # Trigger if object is small (was 0.40)
    RIM_GAP_THRESHOLD = 0.85      # Expected area completeness ratio
    MIN_SOLIDITY_FOR_RECOVERY = 0.85
    
    # NMS (Non-Maximum Suppression)
    NMS_IOU_THRESHOLD = 0.50      # IoU overlap threshold for duplicate detection
    NMS_CENTER_DISTANCE_RATIO = 0.10  # Max center distance as fraction of bbox
    MAX_DETECTIONS = 5            # Maximum objects to return per image


# === LAYER 1.5: RIM RECOVERY ===
class RimRecoveryConfig:
    """Circle hypothesis and rim reconstruction parameters"""
    
    SEARCH_PAD_RATIO = 1.0        # Search area around seed (multiplier of bbox size)
    SEED_RADIUS_TOLERANCE = 0.9   # Min ratio of recovered rim to seed size
    
    # Hough Circle Parameters
    HOUGH_DP = 1.5                # Inverse accumulator resolution
    HOUGH_MIN_DIST = 1000         # Minimum distance between circle centers
    HOUGH_PARAM1 = 100            # Canny edge threshold for Hough
    HOUGH_PARAM2 = 30             # Accumulator threshold (lower = more circles)
    
    # Validation
    CENTER_ALIGNMENT_TOLERANCE = 0.8  # Max distance from seed center (as ratio of radius)
    EDGE_SUPPORT_MIN = 0.15       # Minimum edge confirmation (was 0.08 - too low!)
    EDGE_SUPPORT_FALLBACK = 0.12  # Lower threshold for poor quality images
    EDGE_RING_THICKNESS = 3       # Pixels for edge sampling ring
    
    # Fit Quality
    FIT_ERROR_EXCELLENT = 0.20    # Median error/radius for "Clean" classification
    FIT_ERROR_ACCEPTABLE = 0.50   # Maximum error for "Rough" classification


# === LAYER 2: CONTEXT ANALYSIS ===
class Layer2Config:
    """Scene understanding and container detection parameters"""
    
    # Self-Contained Object Detection (Shortcut)
    AREA_RATIO_LARGE = 0.15       # Objects >15% of image are "big"
    SOLIDITY_DISTINCT = 0.85      # Solidity threshold for "distinct" objects
    MIN_INTERNAL_CONTRAST = 20    # Sobel magnitude for "has content"
    
    # Ray Casting
    RAY_ANGLE_STEP_DEG = 5        # Angular resolution (72 rays at 5°)
    RAY_STEP_SIZE_PX = 4          # Distance between samples along ray
    RAY_DISTANCE_RATIO = 1.1      # Max ray length as fraction of image size
    
    # Edge Detection
    CANNY_LOW = 30
    CANNY_HIGH = 100
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_SIZE = (8, 8)
    
    # Container Detection
    ENCLOSURE_CONFIDENCE_MIN = 0.35  # Minimum ray hit ratio to infer container
    MIN_WALL_POINTS = 10            # Minimum edge points for fitting
    MASK_SMALL_OBJECTS_RATIO = 0.10  # Only mask objects <10% of image
    
    # Classification Thresholds
    CIRCULARITY_ROUND = 0.60      # Minimum circularity for "Round_Artifact"


# === EXTRACTION PIPELINE ===
class ExtractionConfig:
    """File output and naming conventions"""
    
    CROP_MARGIN_RATIO = 0.05      # Margin around bounding box (5% of max dimension)
    
    # Domain-agnostic classification mapping
    GENERIC_CLASS_MAP = {
        "Circle": "circular_objects",
        "Polygon": "polygonal_objects",
        "Complex": "irregular_objects",
        "Fragment": "fragments",
        "Unknown": "review"
    }


# === COIN-SPECIFIC CONFIGURATION ===
# NOTE: These settings are for numismatic analysis
# When adding other domains (pottery, PCB, etc.), create separate config classes
class CoinConfig:
    """Numismatic-specific parameters and behaviors"""
    
    # Rim Recovery (should this be attempted?)
    ENABLE_RIM_RECOVERY = True
    
    # Scene Understanding
    ASSUME_TWO_OBJECTS_ARE_PAIR = True  # Infer obv/rev from 2-object scenes
    
    # Classification Mapping (user-friendly terms)
    CLASS_MAP = {
        "Round Object (Coin/Medallion)": "coins",
        "Round Object (Coin)": "coins",
        "Geometric Form": "fragments",
        "Fragment": "fragments",
        "Artifact (Isolated)": "artifacts",
        "Unknown": "review"
    }
    
    # Quality thresholds for "coin-like" objects
    MIN_CIRCULARITY = 0.65  # Minimum to consider as potential coin
    MIN_EDGE_QUALITY = 0.12  # Minimum edge support for coin detection


# === DIAGNOSTIC / DEBUG ===
class DiagnosticConfig:
    """Parameters for visualization and debugging tools"""
    
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_SIZE = (8, 8)
    CANNY_SIGMA_MULTIPLIER = 0.33
    
    CORNER_SAMPLES = 4  # Number of corner pixels to sample for BG detection
    LIGHT_BG_THRESHOLD = 127  # Gray value threshold for inverting


# === VALIDATION ===
def validate_config():
    """Sanity checks for configuration consistency"""
    assert Layer1Config.CIRCULARITY_RELAXED < Layer1Config.CIRCULARITY_STRICT, \
        "Relaxed circularity must be lower than strict"
    assert RimRecoveryConfig.EDGE_SUPPORT_MIN > RimRecoveryConfig.EDGE_SUPPORT_FALLBACK, \
        "Primary edge support must exceed fallback"
    assert Layer1Config.NMS_IOU_THRESHOLD <= 1.0, "IoU must be <= 1.0"
    print("✓ Configuration validated")

if __name__ == "__main__":
    validate_config()