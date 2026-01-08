import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import shutil
import time
import json  # <--- NEW IMPORT

# --- CONFIGURATION ---
os.environ["NNPACK_MODE"] = "0"  # Silence hardware warning
MODEL_PATH = "best_model.pth"
CORRECTIONS_DIR = "trivalaya_data/04_corrections"
CLASSES = [
    'byzantine', 
    'celtic', 
    'greek', 
    'islamic', 
    'medieval', 
    'persian', 
    'roman_imperial', 
    'roman_provincial', 
    'roman_republican'
]

# --- MODEL LOADING (Cached) ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# --- UI LAYOUT ---
st.set_page_config(page_title="Trivalaya Classifier", page_icon="🪙")
st.title("🪙 Trivalaya: Coin Classifier")

model, device = load_model()
if model:
    st.sidebar.success("MobileNetV2 Loaded")

uploaded_file = st.file_uploader("Choose a coin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption='Uploaded Coin', use_container_width=True)

    # Inference
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
    # Results
    top_prob, top_class = torch.topk(probabilities, 3)
    # Convert tensors to python lists for JSON serialization
    top_prob_list = top_prob.cpu().numpy()[0].tolist() 
    top_class_list = top_class.cpu().numpy()[0].tolist()
    
    primary_label = CLASSES[top_class_list[0]]
    primary_score = top_prob_list[0]
    
    with col2:
        st.subheader("Prediction")
        if primary_score > 0.8:
            st.success(f"**{primary_label.upper()}** ({primary_score*100:.1f}%)")
        elif primary_score > 0.5:
            st.warning(f"**{primary_label.upper()}** ({primary_score*100:.1f}%) - Uncertain")
        else:
            st.error(f"**{primary_label.upper()}** ({primary_score*100:.1f}%) - Low Confidence")
            
        st.markdown("---")
        for i in range(3):
            label = CLASSES[top_class_list[i]]
            score = top_prob_list[i]
            st.progress(score, text=f"{label} ({score*100:.1f}%)")

    # --- FEEDBACK LOOP (With JSON Logging) ---
    st.markdown("---")
    st.write("### Feedback")
    
    col_correct, col_wrong = st.columns(2)
    
    # CASE 1: The Model was RIGHT
    with col_correct:
        if st.button(f"✅ Confirm {primary_label}"):
            # Save to the predicted class folder (reinforcement)
            save_dir = os.path.join(CORRECTIONS_DIR, primary_label) # Use predicted label
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = int(time.time())
            filename = f"confirmed_{timestamp}_{uploaded_file.name}"
            save_path = os.path.join(save_dir, filename)
            
            uploaded_file.seek(0)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Log it as a "success"
            log_entry = {
                "timestamp": timestamp,
                "filename": filename,
                "type": "confirmation", # Mark as confirmation
                "label": primary_label,
                "confidence": primary_score
            }
            with open(os.path.join(CORRECTIONS_DIR, "feedback_log.jsonl"), "a") as f:
                f.write(json.dumps(log_entry) + "\n")
                
            st.success(f"Saved as verified {primary_label}!")

    # CASE 2: The Model was WRONG
    with col_wrong:
        with st.expander("Correction needed?"):
            correct_label = st.selectbox("Actual Label:", CLASSES, index=None)
            if st.button("💾 Save Correction"):
                if correct_label:
                # 1. Save Image
                    save_dir = os.path.join(CORRECTIONS_DIR, correct_label)
                    os.makedirs(save_dir, exist_ok=True)
                    timestamp = int(time.time())
                    filename = f"correction_{timestamp}_{uploaded_file.name}"
                    save_path = os.path.join(save_dir, filename)
                
                    uploaded_file.seek(0)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # 2. Log Metadata (The Enhancement)
                log_entry = {
                    "timestamp": timestamp,
                    "filename": filename,
                    "original_prediction": primary_label,
                    "original_confidence": primary_score,
                    "corrected_label": correct_label,
                    "top3": [
                        {"label": CLASSES[idx], "score": score} 
                        for idx, score in zip(top_class_list, top_prob_list)
                    ]
                }
                
                log_path = os.path.join(CORRECTIONS_DIR, "corrections_log.jsonl")
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                st.success(f"✅ Saved image & log for {correct_label}")
            else:
                st.warning("⚠️ Please select a label first.")