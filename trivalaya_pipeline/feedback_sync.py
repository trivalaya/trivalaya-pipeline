#!/usr/bin/env python3
"""
feedback_sync.py - Merge classifier corrections into database.
Runs before training to pick up JSONL feedback.
"""

import json
from pathlib import Path
import mysql.connector

CORRECTIONS_LOG = Path("trivalaya_data/04_corrections/corrections_log.jsonl")
FEEDBACK_LOG = Path("trivalaya_data/04_corrections/feedback_log.jsonl")

def sync_to_database():
    conn = mysql.connector.connect(...)
    cursor = conn.cursor()
    
    # 1. Process corrections (model was wrong)
    if CORRECTIONS_LOG.exists():
        with open(CORRECTIONS_LOG) as f:
            for line in f:
                entry = json.loads(line)
                # Find the ml_dataset record by image path
                cursor.execute("""
                    INSERT INTO verified_feedback 
                    (ml_dataset_id, model_prediction, model_confidence, 
                     human_label, feedback_type, feedback_source)
                    SELECT id, %s, %s, %s, 'correction', 'classifier_app'
                    FROM ml_dataset 
                    WHERE image_path LIKE %s
                """, (
                    entry['original_prediction'],
                    entry['original_confidence'],
                    entry['corrected_label'],
                    f"%{entry['filename']}"
                ))
                
    # 2. Process confirmations (model was right)
    if FEEDBACK_LOG.exists():
        with open(FEEDBACK_LOG) as f:
            for line in f:
                entry = json.loads(line)
                cursor.execute("""
                    INSERT INTO verified_feedback 
                    (ml_dataset_id, model_prediction, model_confidence,
                     human_label, feedback_type, feedback_source)
                    SELECT id, %s, %s, %s, 'confirm', 'classifier_app'
                    FROM ml_dataset
                    WHERE image_path LIKE %s
                """, (
                    entry['label'],
                    entry['confidence'],
                    entry['label'],
                    f"%{entry['filename']}"
                ))
    
    conn.commit()
    
    # 3. Mark verified in ml_dataset
    cursor.execute("""
        UPDATE ml_dataset m
        JOIN verified_feedback f ON m.id = f.ml_dataset_id
        SET m.is_verified = 1, m.needs_review = 0
        WHERE f.feedback_type IN ('confirm', 'correction')
    """)
    
    conn.commit()

if __name__ == "__main__":
    sync_to_database()