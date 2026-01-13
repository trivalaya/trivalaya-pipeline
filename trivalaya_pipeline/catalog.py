"""
Catalog Database: MySQL-based storage extending your existing auction_data schema.

Adds tables for:
- coin_detections: Vision pipeline outputs
- ml_dataset: ML-ready entries with parsed labels

Reuses your existing auction_data table from trivalaya-data.
"""

import json
import hashlib
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

import mysql.connector
from mysql.connector import Error as MySQLError

from .config import MySQLConfig


@dataclass
class AuctionRecord:
    """
    Mirrors your existing auction_data table.
    Used for type hints and data transfer.
    """
    id: Optional[int] = None
    lot_number: int = 0
    title: str = ""
    description: str = ""
    current_bid: str = ""
    image_url: str = ""
    image_path: str = ""
    closing_date: str = ""
    timestamp: Optional[str] = None
    
    # Extended fields (added by pipeline)
    site: str = ""
    auction_id: str = ""
    vision_processed: bool = False


@dataclass
class CoinDetection:
    """Vision pipeline output for a single detected coin."""
    
    id: Optional[int] = None
    auction_record_id: int = 0
    
    # Detection metadata
    detection_index: int = 0
    inferred_side: str = ""  # "obverse", "reverse", "unknown"
    
    # Extracted image paths
    crop_path: str = ""
    transparent_path: str = ""
    normalized_path: str = ""
    highres_path: str = ""
    
    # Vision metrics
    circularity: float = 0.0
    solidity: float = 0.0
    coin_likelihood: float = 0.0
    edge_support: float = 0.0
    
    # Bounding box
    bbox_x: int = 0
    bbox_y: int = 0
    bbox_w: int = 0
    bbox_h: int = 0
    
    # Classification
    final_classification: str = ""
    layer2_container: str = ""
    
    # Full metadata (JSON)
    vision_metadata: str = ""
    
    created_at: Optional[str] = None


@dataclass
class MLDatasetEntry:
    """ML-ready dataset entry with parsed labels."""
    
    id: Optional[int] = None
    coin_detection_id: int = 0
    
    # Image info
    image_path: str = ""
    image_hash: str = ""
    
    # Dataset split
    split: str = ""  # "train", "val", "test"
    
    # Parsed labels
    period: str = ""
    subperiod: str = ""
    authority: str = ""
    denomination: str = ""
    mint: str = ""
    material: str = ""
    
    # Raw label
    raw_label: str = ""
    label_confidence: float = 0.0
    
    # Flags
    needs_review: bool = False
    is_verified: bool = False
    
    created_at: Optional[str] = None


class CatalogDB:
    """
    MySQL catalog extending your existing auction_data database.
    
    Adds pipeline-specific tables while preserving your existing schema.
    """
    
    def __init__(self, config: MySQLConfig = None):
        self.config = config or MySQLConfig()
        self._ensure_schema()
    
    def _get_connection(self):
        """Get a new database connection."""
        return mysql.connector.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
        )
    
    def _ensure_schema(self):
        """
        Ensure pipeline tables exist.
        Does NOT modify your existing auction_data table.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Add columns to auction_data if they don't exist
            # (safe ALTER - ignores if column exists)
            alter_statements = [
                "ALTER TABLE auction_data ADD COLUMN site VARCHAR(50) DEFAULT ''",
                "ALTER TABLE auction_data ADD COLUMN auction_id VARCHAR(50) DEFAULT ''",
                "ALTER TABLE auction_data ADD COLUMN vision_processed TINYINT DEFAULT 0",
            ]
            
            for stmt in alter_statements:
                try:
                    cursor.execute(stmt)
                except MySQLError as e:
                    if e.errno != 1060:  # 1060 = Duplicate column name
                        raise
            
            # Create coin_detections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS coin_detections (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    auction_record_id INT NOT NULL,
                    
                    detection_index INT DEFAULT 0,
                    inferred_side VARCHAR(20),
                    
                    crop_path TEXT,
                    transparent_path TEXT,
                    normalized_path TEXT,
                    highres_path TEXT,
                    
                    circularity FLOAT,
                    solidity FLOAT,
                    coin_likelihood FLOAT,
                    edge_support FLOAT,
                    
                    bbox_x INT,
                    bbox_y INT,
                    bbox_w INT,
                    bbox_h INT,
                    
                    final_classification VARCHAR(100),
                    layer2_container VARCHAR(100),
                    
                    vision_metadata JSON,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    INDEX idx_auction_record (auction_record_id),
                    INDEX idx_coin_likelihood (coin_likelihood),
                    
                    FOREIGN KEY (auction_record_id) 
                        REFERENCES auction_data(id) 
                        ON DELETE CASCADE
                )
            """)
            
            # Create ml_dataset table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_dataset (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    coin_detection_id INT NOT NULL,
                    
                    image_path TEXT NOT NULL,
                    image_hash VARCHAR(64),
                    
                    split ENUM('train', 'val', 'test'),
                    
                    period VARCHAR(50),
                    subperiod VARCHAR(50),
                    authority VARCHAR(100),
                    denomination VARCHAR(50),
                    mint VARCHAR(100),
                    material VARCHAR(30),
                    
                    raw_label TEXT,
                    label_confidence FLOAT DEFAULT 0.0,
                    
                    needs_review TINYINT DEFAULT 0,
                    is_verified TINYINT DEFAULT 0,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    INDEX idx_split (split),
                    INDEX idx_period (period),
                    INDEX idx_hash (image_hash),
                    
                    UNIQUE KEY unique_hash (image_hash),
                    
                    FOREIGN KEY (coin_detection_id)
                        REFERENCES coin_detections(id)
                        ON DELETE CASCADE
                )
            """)
            
            conn.commit()
            
        finally:
            cursor.close()
            conn.close()
    
    # =========================================================================
    # AUCTION RECORDS (existing table)
    # =========================================================================
    
    def get_unprocessed_records(self, limit: int = 100) -> List[AuctionRecord]:
        """Get auction records that haven't been vision-processed."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("""
                SELECT * FROM auction_data
                WHERE vision_processed = 0
                AND image_path IS NOT NULL
                AND image_path != ''
                LIMIT %s
            """, (limit,))
            
            rows = cursor.fetchall()
            return [self._row_to_auction_record(r) for r in rows]
            
        finally:
            cursor.close()
            conn.close()
    
    def get_record_by_id(self, record_id: int) -> Optional[AuctionRecord]:
        """Get a single auction record by ID."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("SELECT * FROM auction_data WHERE id = %s", (record_id,))
            row = cursor.fetchone()
            return self._row_to_auction_record(row) if row else None
        finally:
            cursor.close()
            conn.close()
    
    def mark_vision_processed(self, record_id: int):
        """Mark an auction record as vision-processed."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "UPDATE auction_data SET vision_processed = 1 WHERE id = %s",
                (record_id,)
            )
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def update_record_metadata(self, record_id: int, site: str, auction_id: str):
        """Update site/auction metadata on existing record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "UPDATE auction_data SET site = %s, auction_id = %s WHERE id = %s",
                (site, auction_id, record_id)
            )
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def _row_to_auction_record(self, row: dict) -> AuctionRecord:
        return AuctionRecord(
            id=row.get('id'),
            lot_number=row.get('lot_number', 0),
            title=row.get('title', ''),
            description=row.get('description', ''),
            current_bid=row.get('current_bid', ''),
            image_url=row.get('image_url', ''),
            image_path=row.get('image_path', ''),
            closing_date=row.get('closing_date', ''),
            timestamp=str(row.get('timestamp', '')),
            site=row.get('site', ''),
            auction_id=row.get('auction_id', ''),
            vision_processed=bool(row.get('vision_processed', 0)),
        )
    
    # =========================================================================
    # COIN DETECTIONS
    # =========================================================================
    
    def insert_detection(self, detection: CoinDetection) -> int:
        """Insert a coin detection. Returns the new ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO coin_detections (
                    auction_record_id, detection_index, inferred_side,
                    crop_path, transparent_path, normalized_path, highres_path,
                    circularity, solidity, coin_likelihood, edge_support,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    final_classification, layer2_container, vision_metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                detection.auction_record_id,
                detection.detection_index,
                detection.inferred_side,
                detection.crop_path,
                detection.transparent_path,
                detection.normalized_path,
                detection.highres_path,
                detection.circularity,
                detection.solidity,
                detection.coin_likelihood,
                detection.edge_support,
                detection.bbox_x,
                detection.bbox_y,
                detection.bbox_w,
                detection.bbox_h,
                detection.final_classification,
                detection.layer2_container,
                detection.vision_metadata,
            ))
            
            conn.commit()
            return cursor.lastrowid
            
        finally:
            cursor.close()
            conn.close()
    
    def get_exportable_detections(self, min_likelihood: float = 0.5) -> List[Dict]:
        """
        Get detections ready for ML export.
        Joins with auction_data for label text.
        """
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("""
                SELECT 
                    d.*,
                    a.title,
                    a.description,
                    a.site,
                    a.auction_id,
                    a.lot_number
                FROM coin_detections d
                JOIN auction_data a ON d.auction_record_id = a.id
                WHERE d.coin_likelihood >= %s
                AND d.normalized_path IS NOT NULL
                AND d.id NOT IN (SELECT coin_detection_id FROM ml_dataset)
            """, (min_likelihood,))
            
            return cursor.fetchall()
            
        finally:
            cursor.close()
            conn.close()
    
    # =========================================================================
    # ML DATASET
    # =========================================================================
    
    def insert_ml_entry(self, entry: MLDatasetEntry) -> int:
        """Insert ML dataset entry. Returns -1 if duplicate hash."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO ml_dataset (
                    coin_detection_id, image_path, image_hash, split,
                    period, subperiod, authority, denomination, mint, material,
                    raw_label, label_confidence, needs_review, is_verified
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                entry.coin_detection_id,
                entry.image_path,
                entry.image_hash,
                entry.split,
                entry.period,
                entry.subperiod,
                entry.authority,
                entry.denomination,
                entry.mint,
                entry.material,
                entry.raw_label,
                entry.label_confidence,
                entry.needs_review,
                entry.is_verified,
            ))
            
            conn.commit()
            return cursor.lastrowid
            
        except MySQLError as e:
            if e.errno == 1062:  # Duplicate entry
                return -1
            raise
            
        finally:
            cursor.close()
            conn.close()
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the ML dataset."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            stats = {}
            
            # Split counts
            cursor.execute("""
                SELECT split, COUNT(*) as count 
                FROM ml_dataset 
                GROUP BY split
            """)
            for row in cursor.fetchall():
                stats[f"{row['split']}_count"] = row['count']
            
            # Period distribution
            cursor.execute("""
                SELECT period, COUNT(*) as count
                FROM ml_dataset
                WHERE period != ''
                GROUP BY period
                ORDER BY count DESC
            """)
            stats['periods'] = {r['period']: r['count'] for r in cursor.fetchall()}
            
            # Needs review count
            cursor.execute("SELECT COUNT(*) as count FROM ml_dataset WHERE needs_review = 1")
            stats['needs_review'] = cursor.fetchone()['count']
            
            # Total records
            cursor.execute("SELECT COUNT(*) as count FROM auction_data")
            stats['total_auction_records'] = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM auction_data WHERE vision_processed = 1")
            stats['vision_processed'] = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) as count FROM coin_detections")
            stats['total_detections'] = cursor.fetchone()['count']
            
            return stats
            
        finally:
            cursor.close()
            conn.close()
    
    def export_split_manifest(self, split: str) -> List[Dict]:
        """Export manifest for a dataset split."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("""
                SELECT 
                    image_path,
                    period, subperiod, authority, denomination, mint, material,
                    label_confidence, needs_review
                FROM ml_dataset
                WHERE split = %s
                AND period IS NOT NULL
                AND label_confidence >= 0.6  # Only include reasonable confidence
                -- If you want to prioritize verified:
                -- ORDER BY is_verified DESC, label_confidence DESC
            """, (split,))
            
            return cursor.fetchall()
            
        finally:
            cursor.close()
            conn.close()


def compute_image_hash(image_path: str, algorithm: str = "phash") -> str:
    """Compute perceptual hash for deduplication."""
    try:
        import cv2
        import numpy as np
        
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback to file hash
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        
        if algorithm == "phash":
            # Simple pHash
            resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(resized))
            dct_low = dct[:8, :8]
            median = np.median(dct_low)
            hash_bits = (dct_low > median).flatten()
            hash_str = ''.join(['1' if b else '0' for b in hash_bits])
            return hex(int(hash_str, 2))[2:].zfill(16)
        else:
            return hashlib.md5(img.tobytes()).hexdigest()
            
    except Exception:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
