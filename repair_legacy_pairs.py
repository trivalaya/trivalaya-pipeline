"""
Legacy Repair Script (Standalone):
Finds auction records with exactly 2 detections and merges them into
a single 'Coin' identity with Obverse/Reverse assigned spatially.
"""

import sys
import os
import mysql.connector
from typing import List, Dict, Tuple

# --- 1. Helper Functions (Copied from catalog_additions.py) ---

def sort_detections_spatially(detections: List[Dict]) -> Tuple[List[Dict], str]:
    """Sort detections left-to-right (horizontal) or top-to-bottom (vertical)."""
    if len(detections) != 2:
        return detections, 'ambiguous'
    
    d1, d2 = detections
    
    # Calculate centers
    c1_x = d1.get('bbox_x', 0) + d1.get('bbox_w', 0) / 2
    c1_y = d1.get('bbox_y', 0) + d1.get('bbox_h', 0) / 2
    c2_x = d2.get('bbox_x', 0) + d2.get('bbox_w', 0) / 2
    c2_y = d2.get('bbox_y', 0) + d2.get('bbox_h', 0) / 2
    
    x_diff = abs(c1_x - c2_x)
    y_diff = abs(c1_y - c2_y)
    
    avg_width = (d1.get('bbox_w', 100) + d2.get('bbox_w', 100)) / 2
    avg_height = (d1.get('bbox_h', 100) + d2.get('bbox_h', 100)) / 2
    
    x_threshold = avg_width * 0.3
    y_threshold = avg_height * 0.3
    
    if x_diff > y_diff and x_diff > x_threshold:
        # Horizontal: Left is Obverse
        sorted_dets = sorted(detections, key=lambda d: d.get('bbox_x', 0) + d.get('bbox_w', 0) / 2)
        return sorted_dets, 'left_right'
    elif y_diff > x_diff and y_diff > y_threshold:
        # Vertical: Top is Obverse
        sorted_dets = sorted(detections, key=lambda d: d.get('bbox_y', 0) + d.get('bbox_h', 0) / 2)
        return sorted_dets, 'top_bottom'
    else:
        return detections, 'ambiguous'

def assign_sides_from_spatial_sort(detections: List[Dict], assume_obv_rev: bool = True):
    if len(detections) == 2 and assume_obv_rev:
        sorted_dets, layout = sort_detections_spatially(detections)
        if layout == 'ambiguous':
            return [(sorted_dets[0], 'unknown', 0.0), (sorted_dets[1], 'unknown', 0.0)]
        
        confidence = 0.8 if layout in ('left_right', 'top_bottom') else 0.5
        return [
            (sorted_dets[0], 'obverse', confidence),
            (sorted_dets[1], 'reverse', confidence)
        ]
    return [(d, 'unknown', 0.0) for d in detections]


# --- 2. Database Handler Class ---

class DatabaseHandler:
    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.ssl_disabled = True
        self.conn = None
        self.setup_database()

    def setup_database(self):
        # Initial connection to check/create DB if needed
        try:
            conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                ssl_disabled=self.ssl_disabled
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            conn.commit()
            cursor.close()
            conn.close()

            # Establish persistent connection to specific DB
            self.conn = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                ssl_disabled=self.ssl_disabled
            )
            print("Database connection established.")
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
            sys.exit(1)

    def run_repair(self):
        print("Starting Legacy Pair Repair...")
        cursor = self.conn.cursor(dictionary=True)

        try:
            # 1. Find auction records that have exactly 2 detections
            print("Scanning for candidate pairs...")
            cursor.execute("""
                SELECT auction_record_id, COUNT(*) as cnt 
                FROM coin_detections 
                GROUP BY auction_record_id 
                HAVING cnt = 2
            """)
            candidates = cursor.fetchall()
            print(f"Found {len(candidates)} candidate lots to check.")

            repaired_count = 0
            
            for row in candidates:
                rec_id = row['auction_record_id']
                
                # Fetch detections
                cursor.execute("SELECT * FROM coin_detections WHERE auction_record_id = %s", (rec_id,))
                detections = cursor.fetchall()

                # 2. Apply Spatial Sorting
                assignments = assign_sides_from_spatial_sort(detections, assume_obv_rev=True)
                
                sides = {a[1] for a in assignments}
                if 'obverse' in sides and 'reverse' in sides:
                    
                    # 3. Create NEW joined Coin identity
                    cursor.execute("""
                        INSERT INTO coins (
                            auction_record_id, group_index, pairing_method, pairing_confidence
                        ) VALUES (%s, 0, 'spatial', 0.9)
                    """, (rec_id,))
                    new_coin_id = cursor.lastrowid
                    
                    # 4. Update detections
                    for det, side, conf in assignments:
                        cursor.execute("""
                            UPDATE coin_detections 
                            SET coin_id = %s, side = %s, side_confidence = %s 
                            WHERE id = %s
                        """, (new_coin_id, side, conf, det['id']))
                    
                    repaired_count += 1
                    
                    if repaired_count % 100 == 0:
                        print(f"Repaired {repaired_count} pairs...", end='\r')
                        self.conn.commit()

            self.conn.commit()
            print(f"\nSuccess! Repaired {repaired_count} coins into Obverse/Reverse pairs.")
            
            # 5. Cleanup orphans
            print("Cleaning up orphaned single-image coins...")
            cursor.execute("""
                DELETE FROM coins 
                WHERE id NOT IN (SELECT DISTINCT coin_id FROM coin_detections WHERE coin_id IS NOT NULL)
            """)
            self.conn.commit()
            print("Cleanup done.")

        except mysql.connector.Error as err:
            print(f"Error during repair: {err}")
            self.conn.rollback()
        finally:
            cursor.close()
            self.conn.close()


# --- 3. Main Execution ---

if __name__ == "__main__":
    # You can pass env vars or hardcode here
    db_handler = DatabaseHandler(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'auction_user'),
        password=os.getenv('DB_PASSWORD'),  # Will be read from environment
        database=os.getenv('DB_NAME', 'auction_data')
    )
    
    db_handler.run_repair()