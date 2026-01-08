#!/usr/bin/env python3
"""
app_integrated.py - Trivalaya Coin Classifier with Geometric Preprocessing

CRITICAL FIX: The classifier was trained on preprocessed images (isolated coins,
normalized backgrounds) but was receiving raw photos. This version integrates
the Layer-1 geometric extractor into the inference pipeline.

Pipeline:
    Raw Photo → Layer-1 Geometry → Isolated Coin PNG → Classifier → Prediction
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
import time
import json
import tempfile
from io import BytesIO

# --- CONFIGURATION ---
os.environ["NNPACK_MODE"] = "0"

MODEL_PATH = "best_model.pth"
ROMAN_SPECIALIST_PATH = "roman_specialist.pth"
CORRECTIONS_DIR = "trivalaya_data/04_corrections"

CLASSES = [
    'byzantine', 'celtic', 'greek', 'islamic', 'medieval',
    'persian', 'roman_imperial', 'roman_provincial', 'roman_republican'
]

ROMAN_CLASSES = ['roman_imperial', 'roman_provincial', 'roman_republican']


# =============================================================================
# LAYER 1: GEOMETRIC EXTRACTION (Embedded)
# =============================================================================

class Layer1Config:
    """Embedded config for geometric extraction"""
    BRIGHT_BACKGROUND_THRESHOLD = 110
    CANNY_SIGMA = 0.33
    MIN_AREA_PX = 300
    MAX_AREA_RATIO = 0.98
    EDGE_SUPPORT_MIN = 0.05
    CIRCULARITY_THRESHOLD = 0.82
    CIRCULARITY_RELAXED = 0.75


def detect_background_histogram(gray_image: np.ndarray) -> tuple:
    """Detect background type from corner sampling"""
    h, w = gray_image.shape
    margin = 5
    
    if h > 20 and w > 20:
        corners = []
        corners.extend(gray_image[0:margin, 0:margin].flatten())
        corners.extend(gray_image[0:margin, w-margin:w].flatten())
        corners.extend(gray_image[h-margin:h, 0:margin].flatten())
        corners.extend(gray_image[h-margin:h, w-margin:w].flatten())
        
        corner_median = np.median(corners)
        corner_std = np.std(corners)
        
        if corner_std < 15:
            bg_type = "light" if corner_median > 127 else "dark"
            return float(corner_median), bg_type
    
    # Fallback to histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
    return float(np.argmax(hist)), "mixed"


def compute_circularity_safe(contour: np.ndarray) -> float:
    """Safe circularity calculation"""
    if contour is None or len(contour) < 5:
        return 0.0
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0 or area == 0:
        return 0.0
    
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return min(circularity, 1.0)


def extract_coin_from_image(image_bgr: np.ndarray) -> tuple:
    """
    Layer-1 geometric extraction: Find and isolate the coin from a raw photo.
    
    Returns:
        (cropped_coin_bgr, detection_info) or (None, error_info)
    """
    if image_bgr is None:
        return None, {"error": "invalid_image"}
    
    h, w = image_bgr.shape[:2]
    total_area = h * w
    
    # Grayscale + background detection
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    avg_bg, bg_type = detect_background_histogram(gray)
    
    # CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Threshold polarity based on background
    if avg_bg > Layer1Config.BRIGHT_BACKGROUND_THRESHOLD:
        thresh_type = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    else:
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    
    # Edge detection for validation
    v = np.median(gray)
    sigma = Layer1Config.CANNY_SIGMA
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)
    edge_zone = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    
    # Binary segmentation
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, thresh_type)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=2
    )
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, {"error": "no_contours_found"}
    
    # Score candidates
    candidates = []
    
    for c in contours:
        area = cv2.contourArea(c)
        
        # Filter by size
        if area < Layer1Config.MIN_AREA_PX:
            continue
        if area > Layer1Config.MAX_AREA_RATIO * total_area:
            continue
        
        # Edge support calculation
        perimeter_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(perimeter_mask, [c], -1, 255, 1)
        perimeter_px = cv2.countNonZero(perimeter_mask)
        
        if perimeter_px == 0:
            continue
        
        overlap = cv2.bitwise_and(perimeter_mask, edge_zone)
        edge_support = cv2.countNonZero(overlap) / perimeter_px
        
        circularity = compute_circularity_safe(c)
        
        # Allow circular shapes through even with weaker edges
        if edge_support < Layer1Config.EDGE_SUPPORT_MIN and circularity < 0.70:
            continue
        
        # Compute solidity
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Coin likelihood score
        coin_likelihood = 0.45 * circularity + 0.35 * edge_support + 0.20 * solidity
        
        candidates.append({
            "contour": c,
            "area": area,
            "circularity": circularity,
            "edge_support": edge_support,
            "solidity": solidity,
            "coin_likelihood": coin_likelihood,
            "bbox": cv2.boundingRect(c)
        })
    
    if not candidates:
        return None, {"error": "no_valid_candidates"}
    
    # Select best candidate (highest coin likelihood)
    best = max(candidates, key=lambda x: x["coin_likelihood"])
    
    # Crop with margin
    x, y, bw, bh = best["bbox"]
    margin = int(max(bw, bh) * 0.05)
    
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)
    
    cropped = image_bgr[y1:y2, x1:x2].copy()
    
    # === CRITICAL: Match training pipeline normalization ===
    # Training images are centered on white squares, then resized
    # This must match VisionAdapter.extract_coins() behavior
    
    crop_h, crop_w = cropped.shape[:2]
    target_size = max(crop_h, crop_w)
    
    # Create white square canvas
    square = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    
    # Center the crop on the white background
    y_off = (target_size - crop_h) // 2
    x_off = (target_size - crop_w) // 2
    square[y_off:y_off+crop_h, x_off:x_off+crop_w] = cropped
    
    # The normalized output matches training data format
    normalized = square
    
    info = {
        "status": "success",
        "circularity": best["circularity"],
        "edge_support": best["edge_support"],
        "coin_likelihood": best["coin_likelihood"],
        "original_size": (w, h),
        "crop_size": (crop_w, crop_h),
        "normalized_size": (target_size, target_size),
        "background_type": bg_type
    }
    
    return normalized, info


# =============================================================================
# MODEL LOADING
# =============================================================================

@st.cache_resource
def load_general_model():
    """Load the main classifier"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        return None, None


@st.cache_resource
def load_roman_specialist():
    """Load Roman specialist model if available"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(ROMAN_SPECIALIST_PATH):
        return None, device
    
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(ROMAN_CLASSES))
    
    try:
        checkpoint = torch.load(ROMAN_SPECIALIST_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    except:
        return None, device


# =============================================================================
# INFERENCE
# =============================================================================

def get_transform(img_size: int = 128):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def classify_coin(model, image_pil, device, classes):
    """Run classification on a PIL image"""
    transform = get_transform()
    input_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    probs_np = probs.cpu().numpy()[0]
    top3_idx = probs_np.argsort()[-3:][::-1]
    
    return {
        "prediction": classes[top3_idx[0]],
        "confidence": float(probs_np[top3_idx[0]]),
        "top3": [(classes[i], float(probs_np[i])) for i in top3_idx],
        "all_probs": {classes[i]: float(probs_np[i]) for i in range(len(classes))}
    }


def run_pipeline(image_pil, general_model, roman_model, device):
    """
    Full pipeline: Geometric extraction → Classification → (Optional) Roman specialist
    """
    # Convert PIL to OpenCV BGR
    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Step 1: Geometric extraction
    extracted_bgr, extraction_info = extract_coin_from_image(image_bgr)
    
    if extracted_bgr is None:
        return {
            "status": "extraction_failed",
            "error": extraction_info.get("error", "unknown"),
            "extraction_info": extraction_info
        }
    
    # Convert extracted coin back to PIL for classifier
    extracted_rgb = cv2.cvtColor(extracted_bgr, cv2.COLOR_BGR2RGB)
    extracted_pil = Image.fromarray(extracted_rgb)
    
    # Step 2: General classification
    general_result = classify_coin(general_model, extracted_pil, device, CLASSES)
    
    result = {
        "status": "success",
        "extraction_info": extraction_info,
        "extracted_image": extracted_pil,
        "level1_prediction": general_result["prediction"],
        "level1_confidence": general_result["confidence"],
        "level1_top3": general_result["top3"],
        "level1_all_probs": general_result["all_probs"],
        "used_specialist": False
    }
    
    # Step 3: Roman specialist (if applicable)
    is_roman = general_result["prediction"] in ROMAN_CLASSES
    
    if is_roman and roman_model is not None:
        roman_result = classify_coin(roman_model, extracted_pil, device, ROMAN_CLASSES)
        
        result["used_specialist"] = True
        result["final_prediction"] = roman_result["prediction"]
        result["final_confidence"] = roman_result["confidence"]
        result["roman_probs"] = roman_result["all_probs"]
    else:
        result["final_prediction"] = general_result["prediction"]
        result["final_confidence"] = general_result["confidence"]
    
    return result


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Trivalaya Classifier", page_icon="🪙", layout="wide")
st.title("🪙 Trivalaya: Coin Classifier")
st.caption("With integrated geometric preprocessing")

# Load models
general_model, device = load_general_model()
roman_model, _ = load_roman_specialist()

# Sidebar status
with st.sidebar:
    st.subheader("Pipeline Status")
    st.success("✅ Layer-1: Geometric Extractor")
    
    if general_model:
        st.success("✅ Layer-2: General Classifier")
    else:
        st.error("❌ General Classifier Missing")
    
    if roman_model:
        st.success("✅ Layer-3: Roman Specialist")
    else:
        st.info("ℹ️ Roman Specialist Not Loaded")
    
    st.markdown("---")
    st.caption(f"Device: {device}")
    
    # Preprocessing toggle (for debugging)
    skip_preprocessing = st.checkbox("Skip preprocessing (debug)", value=False)

# File upload
uploaded_file = st.file_uploader("Upload a coin image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None and general_model is not None:
    # Load image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Run pipeline
    if skip_preprocessing:
        # Debug mode: skip extraction (old behavior)
        result = {
            "status": "success",
            "extracted_image": image,
            "extraction_info": {"status": "skipped"},
            "used_specialist": False
        }
        general_result = classify_coin(general_model, image, device, CLASSES)
        result["level1_prediction"] = general_result["prediction"]
        result["level1_confidence"] = general_result["confidence"]
        result["level1_top3"] = general_result["top3"]
        result["final_prediction"] = general_result["prediction"]
        result["final_confidence"] = general_result["confidence"]
    else:
        result = run_pipeline(image, general_model, roman_model, device)
    
    # Display results
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Extracted")
        if result["status"] == "success":
            st.image(result["extracted_image"], use_container_width=True)
            
            # Show extraction metrics
            info = result["extraction_info"]
            if "circularity" in info:
                st.caption(f"Circularity: {info['circularity']:.2f} | "
                          f"Edge: {info['edge_support']:.2f} | "
                          f"Likelihood: {info['coin_likelihood']:.2f}")
        else:
            st.error(f"Extraction failed: {result.get('error', 'unknown')}")
            st.info("Try adjusting the image or using 'Skip preprocessing' for debugging")
    
    with col3:
        st.subheader("Prediction")
        
        if result["status"] == "success":
            final_pred = result["final_prediction"]
            final_conf = result["final_confidence"]
            
            # Color-coded confidence
            display_label = final_pred.upper().replace('_', ' ')
            if final_conf > 0.8:
                st.success(f"**{display_label}** ({final_conf*100:.1f}%)")
            elif final_conf > 0.5:
                st.warning(f"**{display_label}** ({final_conf*100:.1f}%)")
            else:
                st.error(f"**{display_label}** ({final_conf*100:.1f}%)")
            
            if result.get("used_specialist"):
                st.caption("🎯 Refined by Roman Specialist")
            
            st.markdown("---")
            
            # Top-3 from general model
            st.markdown("**Top 3:**")
            for label, score in result["level1_top3"]:
                st.progress(score, text=f"{label} ({score*100:.1f}%)")
    
    # Feedback section
    st.markdown("---")
    
    col_confirm, col_correct = st.columns(2)
    
    with col_confirm:
        if st.button(f"✅ Confirm: {result.get('final_prediction', 'N/A')}", 
                     disabled=result["status"] != "success"):
            if result["status"] == "success":
                # Save confirmed image
                save_dir = os.path.join(CORRECTIONS_DIR, result["final_prediction"])
                os.makedirs(save_dir, exist_ok=True)
                
                timestamp = int(time.time())
                filename = f"confirmed_{timestamp}_{uploaded_file.name}"
                save_path = os.path.join(save_dir, filename)
                
                # Save the EXTRACTED image (what the model saw)
                result["extracted_image"].save(save_path)
                
                # Log
                log_entry = {
                    "timestamp": timestamp,
                    "filename": filename,
                    "type": "confirmation",
                    "label": result["final_prediction"],
                    "confidence": result["final_confidence"],
                    "extraction_info": result["extraction_info"]
                }
                
                log_path = os.path.join(CORRECTIONS_DIR, "feedback_log.jsonl")
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                
                st.success(f"✅ Confirmed as {result['final_prediction']}")
    
    with col_correct:
        with st.expander("🔧 Correction needed?"):
            correct_label = st.selectbox("Actual class:", CLASSES, index=None)
            
            if st.button("💾 Save Correction"):
                if correct_label and result["status"] == "success":
                    save_dir = os.path.join(CORRECTIONS_DIR, correct_label)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    timestamp = int(time.time())
                    filename = f"correction_{timestamp}_{uploaded_file.name}"
                    save_path = os.path.join(save_dir, filename)
                    
                    # Save EXTRACTED image
                    result["extracted_image"].save(save_path)
                    
                    # Log with full pipeline info
                    log_entry = {
                        "timestamp": timestamp,
                        "filename": filename,
                        "type": "correction",
                        "original_prediction": result["final_prediction"],
                        "original_confidence": result["final_confidence"],
                        "corrected_label": correct_label,
                        "level1_prediction": result["level1_prediction"],
                        "level1_confidence": result["level1_confidence"],
                        "used_specialist": result.get("used_specialist", False),
                        "extraction_info": result["extraction_info"]
                    }
                    
                    log_path = os.path.join(CORRECTIONS_DIR, "corrections_log.jsonl")
                    with open(log_path, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
                    
                    st.success(f"✅ Saved correction: {correct_label}")
                elif not correct_label:
                    st.warning("⚠️ Select a label first")

# Footer
st.markdown("---")
st.caption("Trivalaya Pipeline: Raw Photo → Geometric Extraction → Classification")