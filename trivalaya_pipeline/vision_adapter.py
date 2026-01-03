"""
Vision Adapter: Interface to trivalaya-vision module.

Imports the vision pipeline from your existing repo and provides
a clean interface for the orchestration pipeline.

This adapter pattern allows trivalaya-vision to evolve independently
and be reused for other artifact types (pottery, seals, etc.)
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .config import PathConfig, VisionConfig
from .catalog import CoinDetection


@dataclass
class VisionResult:
    """Standardized result from vision processing."""
    status: str  # "success", "no_detections", "error"
    detections: List[Dict]
    error: Optional[str] = None
    
    @property
    def coin_count(self) -> int:
        return len(self.detections)


class VisionAdapter:
    """
    Adapter for trivalaya-vision module.
    
    Handles:
    - Dynamic import of vision module
    - Running the detection pipeline
    - Extracting and saving coin images
    - Converting results to catalog format
    """
    
    def __init__(
        self,
        paths: PathConfig = None,
        vision_config: VisionConfig = None
    ):
        self.paths = paths or PathConfig()
        self.config = vision_config or VisionConfig()
        
        self._vision_loaded = False
        self._analyze_image = None
        self._layer1 = None
        self._layer2 = None
        
        self._load_vision_module()
    
    def _load_vision_module(self):
        """Dynamically import trivalaya-vision."""
        vision_path = self.paths.vision_module
        
        if not vision_path.exists():
            print(f"⚠ Vision module not found at: {vision_path}")
            return
        
        # Add to Python path
        src_path = vision_path / "src" if (vision_path / "src").exists() else vision_path
        if str(vision_path) not in sys.path:
            sys.path.insert(0, str(vision_path))
        
        try:
            from src.pipeline_manager import analyze_image
            from src.layer1_geometry import layer_1_structural_salience
            from src.layer2_context import layer_2_context_probe
            
            self._analyze_image = analyze_image
            self._layer1 = layer_1_structural_salience
            self._layer2 = layer_2_context_probe
            self._vision_loaded = True
            
            print(f"✓ Vision module loaded from: {vision_path}")
            
        except ImportError as e:
            print(f"⚠ Failed to import vision module: {e}")
            self._vision_loaded = False
    
    @property
    def is_available(self) -> bool:
        """Check if vision module is loaded."""
        return self._vision_loaded
    
    def process_image(self, image_path: Path) -> VisionResult:
        """
        Run vision pipeline on an image.
        
        Args:
            image_path: Path to the source image
            
        Returns:
            VisionResult with detected coins
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            return VisionResult(
                status="error",
                detections=[],
                error=f"Image not found: {image_path}"
            )
        
        if self._vision_loaded:
            return self._run_vision_pipeline(image_path)
        else:
            return self._run_fallback_detection(image_path)
    
    def _run_vision_pipeline(self, image_path: Path) -> VisionResult:
        """Run the full trivalaya-vision pipeline."""
        try:
            result = self._analyze_image(str(image_path))
            
            if result.get('status') == 'success':
                return VisionResult(
                    status="success",
                    detections=result.get('detections', [])
                )
            else:
                return VisionResult(
                    status="no_detections",
                    detections=[],
                    error=result.get('last_error')
                )
                
        except Exception as e:
            return VisionResult(
                status="error",
                detections=[],
                error=str(e)
            )
    
    def _run_fallback_detection(self, image_path: Path) -> VisionResult:
        """
        Basic fallback when vision module unavailable.
        Simple contour-based detection.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return VisionResult(status="error", detections=[], error="Failed to load")
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Background detection
        corners = [gray[0, 0], gray[0, w-1], gray[h-1, 0], gray[h-1, w-1]]
        is_bright_bg = np.mean(corners) > 127
        
        # Threshold
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh_type = cv2.THRESH_BINARY_INV if is_bright_bg else cv2.THRESH_BINARY
        _, binary = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)
        
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < 1000 or area > 0.95 * h * w:
                continue
            
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            coin_likelihood = 0.5 * circularity + 0.3 * solidity + 0.2
            
            if circularity > 0.75:
                final_class = "Round Object (Coin/Medallion)"
            elif circularity > 0.5:
                final_class = "Geometric Form"
            else:
                final_class = "Artifact (Isolated)"
            
            bbox = cv2.boundingRect(c)
            
            detections.append({
                'id': i + 1,
                'final_classification': final_class,
                'layer_1': {
                    'classification': {'label': 'Circle' if circularity > 0.8 else 'Polygon'},
                    'geometry': {
                        'area': int(area),
                        'circularity': round(circularity, 3),
                        'solidity': round(solidity, 3),
                        'coin_likelihood': round(coin_likelihood, 3),
                    },
                    'contour': c,
                    'bbox': bbox,
                },
                'layer_2': {
                    'context_classification': {
                        'inferred_container': 'Round_Artifact' if circularity > 0.6 else 'Unknown',
                        'confidence': 0.7,
                    }
                }
            })
        
        # Sort by X position (left = obverse, right = reverse)
        detections.sort(key=lambda d: d['layer_1']['bbox'][0])
        
        if detections:
            return VisionResult(status="success", detections=detections)
        else:
            return VisionResult(status="no_detections", detections=[])
    
    def extract_coins(
        self,
        image_path: Path,
        detections: List[Dict],
        output_dir: Path,
        base_name: str
    ) -> List[Dict]:
        """
        Extract individual coin images from detections.
        
        Args:
            image_path: Source image path
            detections: List of detection dicts from vision pipeline
            output_dir: Where to save extracted images
            base_name: Base filename for outputs
            
        Returns:
            List of extraction results with paths and metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        img = cv2.imread(str(image_path))
        if img is None:
            return []
        
        h, w = img.shape[:2]
        results = []
        num_detections = len(detections)
        
        for i, det in enumerate(detections):
            # Determine obverse/reverse for two-coin scenes
            if self.config.assume_obv_rev_pair and num_detections == 2:
                suffix = "_obv" if i == 0 else "_rev"
                side = "obverse" if i == 0 else "reverse"
            else:
                suffix = f"_{i+1:02d}" if num_detections > 1 else ""
                side = "unknown"
            
            # Get contour
            layer1 = det.get('layer_1', {})
            contour = layer1.get('contour')
            
            if contour is None:
                continue
            
            if isinstance(contour, list):
                contour = np.array(contour, dtype=np.int32)
            
            # Bounding box with margin
            x, y, bw, bh = cv2.boundingRect(contour)
            margin = int(max(bw, bh) * self.config.crop_margin_ratio)
            
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w, x + bw + margin)
            y2 = min(h, y + bh + margin)
            
            # === Crop ===
            crop = img[y1:y2, x1:x2].copy()
            crop_path = output_dir / f"{base_name}{suffix}_crop.jpg"
            cv2.imwrite(str(crop_path), crop)
            
            # === Transparent PNG ===
            transparent_path = None
            if self.config.save_transparent:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mask_crop = mask[y1:y2, x1:x2]
                
                b, g, r = cv2.split(crop)
                transparent = cv2.merge([b, g, r, mask_crop])
                transparent_path = output_dir / f"{base_name}{suffix}_transparent.png"
                cv2.imwrite(str(transparent_path), transparent)
            
            # === Normalized (224x224 for ML) ===
            normalized_path = None
            highres_path = None
            
            if self.config.save_normalized:
                crop_h, crop_w = crop.shape[:2]
                target_size = max(crop_h, crop_w)
                
                # Center on white square
                square = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
                y_off = (target_size - crop_h) // 2
                x_off = (target_size - crop_w) // 2
                square[y_off:y_off+crop_h, x_off:x_off+crop_w] = crop
                
                # 224x224
                normalized = cv2.resize(square, (224, 224), interpolation=cv2.INTER_AREA)
                normalized_path = output_dir / f"{base_name}{suffix}_224.jpg"
                cv2.imwrite(str(normalized_path), normalized)
                
                # 512x512 high-res
                highres = cv2.resize(square, (512, 512), interpolation=cv2.INTER_AREA)
                highres_path = output_dir / f"{base_name}{suffix}_512.jpg"
                cv2.imwrite(str(highres_path), highres)
            
            # Build result
            geom = layer1.get('geometry', {})
            results.append({
                'detection_index': i,
                'inferred_side': side,
                'crop_path': str(crop_path),
                'transparent_path': str(transparent_path) if transparent_path else '',
                'normalized_path': str(normalized_path) if normalized_path else '',
                'highres_path': str(highres_path) if highres_path else '',
                'bbox': (x1, y1, x2 - x1, y2 - y1),
                'circularity': geom.get('circularity', 0),
                'solidity': geom.get('solidity', 0),
                'coin_likelihood': geom.get('coin_likelihood', 0),
                'edge_support': layer1.get('classification', {}).get('confidence', 0),
                'final_classification': det.get('final_classification', ''),
                'layer2_container': det.get('layer_2', {}).get('context_classification', {}).get('inferred_container', ''),
                'vision_metadata': det,
            })
        
        return results
    
    def result_to_detections(
        self,
        auction_record_id: int,
        extractions: List[Dict]
    ) -> List[CoinDetection]:
        """
        Convert extraction results to CoinDetection objects for database.
        """
        detections = []
        
        for ext in extractions:
            detection = CoinDetection(
                auction_record_id=auction_record_id,
                detection_index=ext['detection_index'],
                inferred_side=ext['inferred_side'],
                crop_path=ext['crop_path'],
                transparent_path=ext['transparent_path'],
                normalized_path=ext['normalized_path'],
                highres_path=ext['highres_path'],
                circularity=ext['circularity'],
                solidity=ext['solidity'],
                coin_likelihood=ext['coin_likelihood'],
                edge_support=ext['edge_support'],
                bbox_x=ext['bbox'][0],
                bbox_y=ext['bbox'][1],
                bbox_w=ext['bbox'][2],
                bbox_h=ext['bbox'][3],
                final_classification=ext['final_classification'],
                layer2_container=ext['layer2_container'],
                vision_metadata=json.dumps(
                    ext['vision_metadata'], 
                    default=self._json_serializer
                ),
            )
            detections.append(detection)
        
        return detections
    
    @staticmethod
    def _json_serializer(obj):
        """Handle numpy types in JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)}")
