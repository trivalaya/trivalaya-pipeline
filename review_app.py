import streamlit as st
import mysql.connector
from PIL import Image
import os

# --- CONFIGURATION ---
DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "auction_user",
    "password": "Veritas@2024",
    "database": "auction_data"
}

CLASSES = [
    'byzantine', 'celtic', 'greek', 'islamic', 'medieval',
    'persian', 'roman_imperial', 'roman_provincial', 'roman_republican'
]

# --- DATABASE FUNCTIONS ---

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

def fetch_stats():
    """Get counts: Flagged reviews vs Total Unverified."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Count flagged specifically
    cursor.execute("SELECT COUNT(*) FROM ml_dataset WHERE needs_review = 1")
    flagged = cursor.fetchone()[0]
    
    # Count total remaining unverified (workload remaining)
    cursor.execute("SELECT COUNT(*) FROM ml_dataset WHERE is_verified = 0")
    total_unverified = cursor.fetchone()[0]
    
    conn.close()
    return flagged, total_unverified

def fetch_next_record(show_all_unverified=False):
    """
    Fetch one record.
    Priority 1: 'needs_review = 1' (The burning fires)
    Priority 2: 'is_verified = 0' (The rest of the pile, if enabled)
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Select subperiod now too
    cols = "id, image_path, period, subperiod, raw_label, label_confidence, needs_review"
    
    if show_all_unverified:
        # Get anything unverified, prioritizing flagged ones first
        query = f"""
            SELECT {cols} FROM ml_dataset 
            WHERE is_verified = 0 
            ORDER BY needs_review DESC, label_confidence ASC 
            LIMIT 1
        """
    else:
        # Strict mode: Only flagged items
        query = f"""
            SELECT {cols} FROM ml_dataset 
            WHERE needs_review = 1 
            LIMIT 1
        """
        
    cursor.execute(query)
    record = cursor.fetchone()
    conn.close()
    return record

def update_record(record_id, period=None, subperiod=None):
    """
    Update the record.
    - Always marks is_verified = 1 (Never see it again).
    - Always clears needs_review = 0.
    - Updates period/subperiod if provided.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build the update query dynamically based on what changed
    updates = ["needs_review = 0", "is_verified = 1"]
    params = []
    
    if period:
        updates.append("period = %s")
        params.append(period)
        
    if subperiod is not None:  # Allow empty string to clear it
        updates.append("subperiod = %s")
        params.append(subperiod)
        
    updates_sql = ", ".join(updates)
    params.append(record_id)
    
    query = f"UPDATE ml_dataset SET {updates_sql} WHERE id = %s"
    
    cursor.execute(query, tuple(params))
    conn.commit()
    conn.close()

def delete_record(record_id):
    """Delete garbage records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ml_dataset WHERE id = %s", (record_id,))
    conn.commit()
    conn.close()
def fetch_next_record(mode):
    if mode == "Flagged Items":
        query = """
            SELECT * FROM ml_dataset 
            WHERE needs_review = 1 
            ORDER BY label_confidence ASC 
            LIMIT 1
        """
    elif mode == "Low Confidence":
        query = """
            SELECT * FROM ml_dataset 
            WHERE is_verified = 0 
            AND label_confidence < 0.7
            ORDER BY label_confidence ASC 
            LIMIT 1
        """
    elif mode == "High Confidence Audit":
        # Randomly sample high-confidence records to catch systematic errors
        query = """
            SELECT * FROM ml_dataset 
            WHERE is_verified = 0 
            AND label_confidence > 0.85
            ORDER BY RAND() 
            LIMIT 1
        """
    elif mode == "Random Sample":
        query = """
            SELECT * FROM ml_dataset 
            WHERE is_verified = 0 
            ORDER BY RAND() 
            LIMIT 1
        """
# --- UI LAYOUT ---

st.set_page_config(page_title="Trivalaya Labeler", layout="wide")

st.title("🛡️ Dataset Verifier")

# 1. Sidebar & Stats
with st.sidebar:
    st.header("Settings")
    # The "Boredom Toggle"
    show_all = st.checkbox("Show all unverified records", 
                          help="If checked, you will keep seeing images even after the 'Needs Review' queue is empty.")
    
    st.divider()
    
    try:
        flagged_count, total_count = fetch_stats()
        st.metric("🚩 Needs Review", flagged_count)
        st.metric("📉 Total Unverified", total_count)
        st.progress(1.0 - (total_count / 6000) if total_count < 6000 else 0, text="Completion Progress")
        review_mode = st.radio("Review Mode",
        ["Flagged Items", "Low Confidence", "High Confidence Audit", "Random Sample"]
        )
    except Exception as e:
        st.error("DB Connection lost")
        st.stop()
        
    st.info("**Hotkeys:**\n\n**C**: Confirm\n**D**: Delete\n**R**: Refresh")

# 2. Fetch Record
try:
    record = fetch_next_record(show_all_unverified=show_all)
except Exception as e:
    st.error(f"⚠️ Database Error: {e}")
    st.stop()

# 3. Main Interface
if not record:
    if not show_all:
        st.success("🎉 'Needs Review' queue is empty!")
        st.info("Check the box in the sidebar to review the rest of the unverified images.")
    else:
        st.balloons()
        st.success("🏆 YOU ARE DONE! The entire dataset is verified.")
        
    if st.button("Refresh Check"):
        st.rerun()
else:
    # Layout: Image on Left, Data/Controls on Right
    col_img, col_data = st.columns([1, 1])

    with col_img:
        # Handle path resolution
        if record['image_path'] and os.path.exists(record['image_path']):
            image = Image.open(record['image_path'])
            st.image(image, use_container_width=True)
        else:
            st.error(f"❌ Image missing on disk: {record.get('image_path', 'None')}")
            # If image is missing, maybe offer quick delete?
            if st.button("Delete (Missing File)"):
                delete_record(record['id'])
                st.rerun()

    with col_data:
        # Display Raw Context
        with st.expander("📄 Raw Auction Label (Context)", expanded=True):
            st.caption(record['raw_label'] or "N/A")

        # Metadata Row
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Period:** {record['period']}")
        
        conf = record.get('label_confidence', 0) or 0
        c2.metric("Confidence", f"{conf:.1%}")
        
        if record['needs_review']:
            c3.error("🚩 Flagged")
        else:
            c3.success("Routine Check")
        # After line 187
        if record['needs_review']:
            c3.error("🚩 Flagged")
            
            # NEW: Show why it was flagged
            if conf < 0.6:
                st.warning("⚠️ Low confidence parsing")
            elif record.get('raw_label'):
                # Check for multiple period signals
                from trivalaya_pipeline.label_parser import LabelParser
                parser = LabelParser()
                result = parser.parse(record['raw_label'], "")
                if result.warnings:
                    for w in result.warnings:
                        st.caption(f"⚠️ {w}")

        st.divider()
        
        # --- FORM AREA ---
        st.subheader("Edit & Save")
        
        with st.form("verify_form"):
            # 1. Period Selector (Defaults to current DB value)
            current_period_index = CLASSES.index(record['period']) if record['period'] in CLASSES else 0
            new_period = st.selectbox("Period", CLASSES, index=current_period_index)
            
            # 2. Subperiod Input (New!)
            current_sub = record.get('subperiod') or ""
            new_subperiod = st.text_input("Subperiod (Optional)", value=current_sub, placeholder="e.g. Flavian, Umayyad, Late Byzantine")
            
            # Submit Buttons
            col_save, col_del = st.columns([3, 1])
            
            with col_save:
                submitted = st.form_submit_button("✅ Confirm / Save Changes", type="primary", use_container_width=True)
            
            if submitted:
                # Update DB
                update_record(
                    record['id'], 
                    period=new_period, 
                    subperiod=new_subperiod
                )
                st.toast("Saved!")
                st.rerun()

        # Delete outside the form to avoid validation weirdness
        with st.expander("Danger Zone"):
            if st.button("🗑️ Delete Record (Not a coin)", type="secondary"):
                delete_record(record['id'])
                st.warning("Deleted")
                st.rerun()