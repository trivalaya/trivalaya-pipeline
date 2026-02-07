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
from typing import Optional, List, Dict, Any, Tuple

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
    sale_id: str = ""
    auction_house: str = ""
    sale_id: str = ""
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
# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class Coin:
    """Represents a physical coin specimen within an auction record."""
    id: Optional[int] = None
    auction_record_id: int = 0
    group_index: int = 0
    pairing_method: str = ""  # 'spatial', 'left_right', 'top_bottom', 'manual', 'model', 'legacy_backfill'
    pairing_confidence: float = 0.0
    created_at: Optional[str] = None


@dataclass 
class MLCoinDatasetEntry:
    """ML-ready dataset entry for paired (obv/rev) coin samples."""
    
    id: Optional[int] = None
    coin_id: int = 0
    
    obv_path: str = ""
    rev_path: str = ""
    has_obv: bool = False
    has_rev: bool = False
    
    obv_hash: str = ""
    rev_hash: str = ""
    pair_hash: str = ""
    
    # Dedup tracking
    is_duplicate: bool = False
    duplicate_of_coin_id: Optional[int] = None
    
    split: str = ""
    period: str = ""
    subperiod: str = ""
    authority: str = ""
    denomination: str = ""
    mint: str = ""
    material: str = ""
    raw_label: str = ""
    label_confidence: float = 0.0
    needs_review: bool = False
    is_verified: bool = False
    created_at: Optional[str] = None

# =============================================================================
# SPATIAL SORTING UTILITIES
# =============================================================================

def sort_detections_spatially(detections: List[Dict]) -> Tuple[List[Dict], str]:
    """
    Sort detections by spatial position to determine obverse/reverse.
    
    Convention:
    - Horizontal layout (most common): left = obverse, right = reverse
    - Vertical layout: top = obverse, bottom = reverse
    
    Args:
        detections: List of detection dicts with bbox_x, bbox_y, bbox_w, bbox_h
    
    Returns:
        (sorted_detections, layout_type) where layout_type is 'left_right', 'top_bottom', or 'ambiguous'
    """
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
    
    # Threshold: if centers are within 20% of average width, consider them aligned
    avg_width = (d1.get('bbox_w', 100) + d2.get('bbox_w', 100)) / 2
    avg_height = (d1.get('bbox_h', 100) + d2.get('bbox_h', 100)) / 2
    
    x_threshold = avg_width * 0.3
    y_threshold = avg_height * 0.3
    
    if x_diff > y_diff and x_diff > x_threshold:
        # Horizontal layout: sort left to right
        sorted_dets = sorted(detections, key=lambda d: d.get('bbox_x', 0) + d.get('bbox_w', 0) / 2)
        return sorted_dets, 'left_right'
    elif y_diff > x_diff and y_diff > y_threshold:
        # Vertical layout: sort top to bottom
        sorted_dets = sorted(detections, key=lambda d: d.get('bbox_y', 0) + d.get('bbox_h', 0) / 2)
        return sorted_dets, 'top_bottom'
    else:
        # Ambiguous - coins overlap or are diagonally arranged
        return detections, 'ambiguous'


def assign_sides_from_spatial_sort(
    detections: List[Dict],
    assume_obv_rev: bool = True
) -> List[Tuple[Dict, str, float]]:
    """
    Assign obverse/reverse sides based on spatial position.
    
    Args:
        detections: List of detection dicts
        assume_obv_rev: If True, assign obv/rev; if False, all become 'unknown'
    
    Returns:
        List of (detection, side, confidence) tuples
    """
    if len(detections) == 1:
        return [(detections[0], 'unknown', 0.0)]
    
    if len(detections) == 2 and assume_obv_rev:
        sorted_dets, layout = sort_detections_spatially(detections)
        
        if layout == 'ambiguous':
            # Don't guess - mark both as unknown
            return [
                (sorted_dets[0], 'unknown', 0.0),
                (sorted_dets[1], 'unknown', 0.0),
            ]
        
        # Confidence based on how clear the separation is
        confidence = 0.8 if layout in ('left_right', 'top_bottom') else 0.5
        
        return [
            (sorted_dets[0], 'obverse', confidence),
            (sorted_dets[1], 'reverse', confidence),
        ]
    
    # More than 2 detections or not assuming obv/rev
    return [(d, 'unknown', 0.0) for d in detections]

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
    # =========================================================================
    # ML COIN DATASET OPERATIONS (with dedup tracking)
    # =========================================================================
    
    def insert_ml_coin_entry(self, entry: 'MLCoinDatasetEntry') -> int:
        """
        Insert paired ML dataset entry.
        
        Unlike the old version, this always inserts (no UNIQUE constraint).
        Duplicates are tracked via is_duplicate flag.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO ml_coin_dataset (
                    coin_id, obv_path, rev_path, has_obv, has_rev,
                    obv_hash, rev_hash, pair_hash,
                    is_duplicate, duplicate_of_coin_id,
                    split, period, subperiod, authority, denomination, mint, material,
                    raw_label, label_confidence, needs_review, is_verified
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                entry.coin_id,
                entry.obv_path,
                entry.rev_path,
                entry.has_obv,
                entry.has_rev,
                entry.obv_hash,
                entry.rev_hash,
                entry.pair_hash,
                entry.is_duplicate,
                entry.duplicate_of_coin_id,
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
        finally:
            cursor.close()
            conn.close()
    
    def check_pair_hash_exists(self, pair_hash: str) -> Optional[int]:
        """
        Check if a pair_hash already exists in ml_coin_dataset.
        
        Returns the coin_id of the existing entry, or None if not found.
        """
        if not pair_hash:
            return None
            
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("""
                SELECT coin_id FROM ml_coin_dataset 
                WHERE pair_hash = %s AND is_duplicate = 0
                LIMIT 1
            """, (pair_hash,))
            
            row = cursor.fetchone()
            return row['coin_id'] if row else None
        finally:
            cursor.close()
            conn.close()
    
    def export_coin_pair_manifest(self, split: str) -> List[Dict]:
        """Export manifest for paired coin dataset split (excludes duplicates)."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("""
                SELECT 
                    coin_id,
                    obv_path,
                    rev_path,
                    has_obv,
                    has_rev,
                    period, 
                    subperiod, 
                    authority, 
                    denomination, 
                    mint, 
                    material,
                    label_confidence, 
                    needs_review
                FROM ml_coin_dataset
                WHERE split = %s
                    AND is_duplicate = 0
                    AND period IS NOT NULL
                    AND label_confidence >= 0.6
            """, (split,))
            
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
    
    def get_coin_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the paired ML coin dataset."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            stats = {}
            
            cursor.execute("""
                SELECT split, COUNT(*) as count 
                FROM ml_coin_dataset 
                WHERE is_duplicate = 0
                GROUP BY split
            """)
            for row in cursor.fetchall():
                stats[f"{row['split']}_count"] = row['count']
            
            cursor.execute("""
                SELECT period, COUNT(*) as count
                FROM ml_coin_dataset
                WHERE period != '' AND period IS NOT NULL AND is_duplicate = 0
                GROUP BY period
                ORDER BY count DESC
            """)
            stats['periods'] = {r['period']: r['count'] for r in cursor.fetchall()}
            
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN has_obv = 1 AND has_rev = 1 THEN 1 ELSE 0 END) as both_sides,
                    SUM(CASE WHEN has_obv = 1 AND has_rev = 0 THEN 1 ELSE 0 END) as obv_only,
                    SUM(CASE WHEN has_obv = 0 AND has_rev = 1 THEN 1 ELSE 0 END) as rev_only,
                    COUNT(*) as total
                FROM ml_coin_dataset
                WHERE is_duplicate = 0
            """)
            row = cursor.fetchone()
            stats['pairing'] = {
                'both_sides': row['both_sides'] or 0,
                'obv_only': row['obv_only'] or 0,
                'rev_only': row['rev_only'] or 0,
                'total': row['total'] or 0,
            }
            
            cursor.execute("SELECT COUNT(*) as count FROM ml_coin_dataset WHERE is_duplicate = 1")
            stats['duplicates_tracked'] = cursor.fetchone()['count']
            
            return stats
        finally:
            cursor.close()
            conn.close()


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
                "ALTER TABLE auction_data ADD COLUMN sale_id VARCHAR(50) DEFAULT ''",
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
    # PAIRED EXPORT QUERIES (DUPLICATE-SAFE)
    # =========================================================================
    
    def get_exportable_coins(self, min_likelihood: float = 0.5) -> List[Dict]:
        """
        Get coins ready for paired ML export.
        
        FIXED: Uses window functions to select exactly one detection per side,
        avoiding Cartesian explosion from multiple LEFT JOINs.
        
        Returns one row per coin with obv_path, rev_path, unk_path (any may be NULL).
        """
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # MySQL 8+ query with window functions
            cursor.execute("""
                WITH ranked AS (
                    SELECT
                        d.id,
                        d.coin_id,
                        d.side,
                        d.normalized_path,
                        d.coin_likelihood,
                        d.side_confidence,
                        ROW_NUMBER() OVER (
                            PARTITION BY d.coin_id, d.side
                            ORDER BY COALESCE(d.side_confidence, 0) DESC, d.id DESC
                        ) AS rn
                    FROM coin_detections d
                    WHERE d.normalized_path IS NOT NULL 
                        AND d.normalized_path != ''
                        AND d.coin_likelihood >= %s
                )
                SELECT
                    c.id AS coin_id,
                    c.auction_record_id,
                    c.pairing_method,
                    c.pairing_confidence,
                    
                    obv.id AS obv_detection_id,
                    obv.normalized_path AS obv_path,
                    obv.coin_likelihood AS obv_likelihood,
                    obv.side_confidence AS obv_side_confidence,
                    
                    rev.id AS rev_detection_id,
                    rev.normalized_path AS rev_path,
                    rev.coin_likelihood AS rev_likelihood,
                    rev.side_confidence AS rev_side_confidence,
                    
                    unk.id AS unk_detection_id,
                    unk.normalized_path AS unk_path,
                    unk.coin_likelihood AS unk_likelihood,
                    
                    a.title,
                    a.description,
                    a.site,
                    a.sale_id,
                    a.lot_number,
                    
                    COALESCE(ai.auction_house, a.site) AS auction_house,
                    ai.period AS auction_period
                    
                FROM coins c
                JOIN auction_data a ON a.id = c.auction_record_id
                
                LEFT JOIN auction_info ai
                    ON a.auction_house = ai.auction_house
                    AND a.sale_id = ai.sale_id
                    AND a.lot_number BETWEEN ai.lot_start AND ai.lot_end
                
                LEFT JOIN ranked obv 
                    ON obv.coin_id = c.id 
                    AND obv.side = 'obverse' 
                    AND obv.rn = 1
                
                LEFT JOIN ranked rev 
                    ON rev.coin_id = c.id 
                    AND rev.side = 'reverse' 
                    AND rev.rn = 1
                
                LEFT JOIN ranked unk 
                    ON unk.coin_id = c.id 
                    AND unk.side = 'unknown' 
                    AND unk.rn = 1
                
                WHERE 
                    (obv.id IS NOT NULL OR rev.id IS NOT NULL OR unk.id IS NOT NULL)
                    AND c.id NOT IN (SELECT coin_id FROM ml_coin_dataset WHERE is_duplicate = 0)
                
                ORDER BY c.id
            """, (min_likelihood,))
            
            return cursor.fetchall()
            
        except Exception as e:
            # Fallback for MySQL < 8 (no window functions)
            if 'syntax error' in str(e).lower() or 'function' in str(e).lower():
                return self._get_exportable_coins_mysql57(min_likelihood)
            raise
            
        finally:
            cursor.close()
            conn.close()
    
    def _get_exportable_coins_mysql57(self, min_likelihood: float = 0.5) -> List[Dict]:
        """
        Fallback query for MySQL 5.7 (no window functions).
        Uses subqueries with MAX(id) to select one detection per side.
        """
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("""
                SELECT
                    c.id AS coin_id,
                    c.auction_record_id,
                    c.pairing_method,
                    c.pairing_confidence,
                    
                    obv.id AS obv_detection_id,
                    obv.normalized_path AS obv_path,
                    obv.coin_likelihood AS obv_likelihood,
                    obv.side_confidence AS obv_side_confidence,
                    
                    rev.id AS rev_detection_id,
                    rev.normalized_path AS rev_path,
                    rev.coin_likelihood AS rev_likelihood,
                    rev.side_confidence AS rev_side_confidence,
                    
                    unk.id AS unk_detection_id,
                    unk.normalized_path AS unk_path,
                    unk.coin_likelihood AS unk_likelihood,
                    
                    a.title,
                    a.description,
                    a.site,
                    a.sale_id,
                    a.lot_number,
                    
                    COALESCE(ai.auction_house, a.site) AS auction_house,
                    ai.period AS auction_period
                    
                FROM coins c
                JOIN auction_data a ON a.id = c.auction_record_id
                
                LEFT JOIN auction_info ai
                    ON a.auction_house = ai.auction_house
                    AND a.sale_id = ai.sale_id
                    AND a.lot_number BETWEEN ai.lot_start AND ai.lot_end
                
                -- Subquery to get best obverse detection
                LEFT JOIN coin_detections obv ON obv.id = (
                    SELECT d.id FROM coin_detections d
                    WHERE d.coin_id = c.id 
                        AND d.side = 'obverse'
                        AND d.normalized_path IS NOT NULL
                        AND d.coin_likelihood >= %s
                    ORDER BY COALESCE(d.side_confidence, 0) DESC, d.id DESC
                    LIMIT 1
                )
                
                -- Subquery to get best reverse detection
                LEFT JOIN coin_detections rev ON rev.id = (
                    SELECT d.id FROM coin_detections d
                    WHERE d.coin_id = c.id 
                        AND d.side = 'reverse'
                        AND d.normalized_path IS NOT NULL
                        AND d.coin_likelihood >= %s
                    ORDER BY COALESCE(d.side_confidence, 0) DESC, d.id DESC
                    LIMIT 1
                )
                
                -- Subquery to get best unknown detection
                LEFT JOIN coin_detections unk ON unk.id = (
                    SELECT d.id FROM coin_detections d
                    WHERE d.coin_id = c.id 
                        AND d.side = 'unknown'
                        AND d.normalized_path IS NOT NULL
                        AND d.coin_likelihood >= %s
                    ORDER BY COALESCE(d.side_confidence, 0) DESC, d.id DESC
                    LIMIT 1
                )
                
                WHERE 
                    (obv.id IS NOT NULL OR rev.id IS NOT NULL OR unk.id IS NOT NULL)
                    AND c.id NOT IN (SELECT coin_id FROM ml_coin_dataset WHERE is_duplicate = 0)
                
                ORDER BY c.id
            """, (min_likelihood, min_likelihood, min_likelihood))
            
            return cursor.fetchall()
            
        finally:
            cursor.close()
            conn.close()
 # =========================================================================
    # COINS TABLE OPERATIONS
    # =========================================================================
    
    def create_coin(
        self, 
        auction_record_id: int, 
        group_index: int = 0,
        pairing_method: str = 'spatial',
        pairing_confidence: float = 0.0
    ) -> int:
        """Create a new coin identity. Returns the new coin_id."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO coins (
                    auction_record_id, group_index, pairing_method, pairing_confidence
                ) VALUES (%s, %s, %s, %s)
            """, (auction_record_id, group_index, pairing_method, pairing_confidence))
            
            conn.commit()
            return cursor.lastrowid
            
        finally:
            cursor.close()
            conn.close()
    
    def get_coin(self, coin_id: int) -> Optional[Coin]:
        """Get a coin by ID."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("SELECT * FROM coins WHERE id = %s", (coin_id,))
            row = cursor.fetchone()
            if not row:
                return None
            
            return Coin(
                id=row['id'],
                auction_record_id=row['auction_record_id'],
                group_index=row['group_index'],
                pairing_method=row['pairing_method'] or '',
                pairing_confidence=row['pairing_confidence'] or 0.0,
                created_at=str(row['created_at']) if row['created_at'] else None,
            )
        finally:
            cursor.close()
            conn.close()
    
    def get_coins_for_record(self, auction_record_id: int) -> List[Coin]:
        """Get all coins for an auction record."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("""
                SELECT * FROM coins 
                WHERE auction_record_id = %s
                ORDER BY group_index
            """, (auction_record_id,))
            
            return [
                Coin(
                    id=row['id'],
                    auction_record_id=row['auction_record_id'],
                    group_index=row['group_index'],
                    pairing_method=row['pairing_method'] or '',
                    pairing_confidence=row['pairing_confidence'] or 0.0,
                    created_at=str(row['created_at']) if row['created_at'] else None,
                )
                for row in cursor.fetchall()
            ]
        finally:
            cursor.close()
            conn.close()
    
    def get_detections_for_coin(self, coin_id: int) -> List[Dict]:
        """Get all detections for a coin, with side info."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # FIXED: Custom sort order for intuitive debugging (obverse first)
            cursor.execute("""
                SELECT * FROM coin_detections
                WHERE coin_id = %s
                ORDER BY FIELD(side, 'obverse', 'reverse', 'unknown'), id DESC
            """, (coin_id,))
            
            return cursor.fetchall()
        finally:
            cursor.close()
            conn.close()
    
    # =========================================================================
    # DETECTION INSERTION WITH COIN IDENTITY
    # =========================================================================
    
    def insert_detection_with_coin(
        self,
        detection: 'CoinDetection',
        coin_id: int,
        side: str = 'unknown',
        side_confidence: float = 0.0
    ) -> int:
        """Insert a coin detection linked to a coin identity."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO coin_detections (
                    auction_record_id, coin_id, side, side_confidence,
                    detection_index, inferred_side,
                    crop_path, transparent_path, normalized_path, highres_path,
                    circularity, solidity, coin_likelihood, edge_support,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    final_classification, layer2_container, vision_metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                detection.auction_record_id,
                coin_id,
                side,
                side_confidence,
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
    
    
    def get_coin_pair_stats(self) -> Dict[str, Any]:
        """Get statistics about coin pairing status."""
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            stats = {}
            
            cursor.execute("SELECT COUNT(*) as count FROM coins")
            stats['total_coins'] = cursor.fetchone()['count']
            
            cursor.execute("""
                SELECT pairing_method, COUNT(*) as count 
                FROM coins 
                GROUP BY pairing_method
            """)
            stats['by_pairing_method'] = {
                row['pairing_method'] or 'null': row['count'] 
                for row in cursor.fetchall()
            }
            
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN obv_count > 0 AND rev_count > 0 THEN 'both_sides'
                        WHEN obv_count > 0 THEN 'obverse_only'
                        WHEN rev_count > 0 THEN 'reverse_only'
                        ELSE 'unknown_only'
                    END as side_status,
                    COUNT(*) as count
                FROM (
                    SELECT 
                        c.id,
                        SUM(CASE WHEN d.side = 'obverse' THEN 1 ELSE 0 END) as obv_count,
                        SUM(CASE WHEN d.side = 'reverse' THEN 1 ELSE 0 END) as rev_count
                    FROM coins c
                    LEFT JOIN coin_detections d ON d.coin_id = c.id
                    GROUP BY c.id
                ) coin_sides
                GROUP BY side_status
            """)
            stats['by_side_status'] = {
                row['side_status']: row['count'] 
                for row in cursor.fetchall()
            }
            
            return stats
        finally:
            cursor.close()
            conn.close()
       
    # =========================================================================
    # DETECTION → COIN PAIRING
    # =========================================================================
    
    def get_unlinked_detections_grouped(self, min_likelihood: float = 0.5) -> Dict[int, List[Dict]]:
        """
        Get all detections not yet linked to a coin, grouped by auction_record_id.
        Returns {auction_record_id: [detection_dicts]}.
        """
        conn = self._get_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("""
                SELECT * FROM coin_detections
                WHERE coin_id IS NULL
                  AND coin_likelihood >= %s
                  AND normalized_path IS NOT NULL
                  AND normalized_path != ''
                ORDER BY auction_record_id, detection_index
            """, (min_likelihood,))
            
            groups: Dict[int, List[Dict]] = {}
            for row in cursor.fetchall():
                rid = row['auction_record_id']
                if rid not in groups:
                    groups[rid] = []
                groups[rid].append(row)
            
            return groups
        finally:
            cursor.close()
            conn.close()
    
    def link_detection_to_coin(self, detection_id: int, coin_id: int, side: str, side_confidence: float):
        """Update an existing detection to link it to a coin with side assignment."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE coin_detections
                SET coin_id = %s, side = %s, side_confidence = %s
                WHERE id = %s
            """, (coin_id, side, side_confidence, detection_id))
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def pair_unlinked_detections(self, min_likelihood: float = 0.5) -> Dict[str, int]:
        """
        Pair all unlinked detections into coin entities.
        
        For each auction_record_id group:
        - 2 detections: create one coin, assign obverse/reverse spatially
        - 1 detection: create one coin, side = unknown
        - 3+ detections: create one coin per pair (first two spatial), rest unknown
        
        Returns stats dict.
        """
        groups = self.get_unlinked_detections_grouped(min_likelihood)
        
        stats = {
            'records_processed': 0,
            'coins_created': 0,
            'detections_linked': 0,
            'pairs_assigned': 0,
            'singles': 0,
            'multi_detection': 0,
        }
        
        for auction_record_id, detections in groups.items():
            n = len(detections)
            
            if n == 2:
                # Standard case: create one coin, assign sides spatially
                sided = assign_sides_from_spatial_sort(detections, assume_obv_rev=True)
                
                # Determine pairing method from layout
                _, layout = sort_detections_spatially(detections)
                pairing_method = layout if layout in ('left_right', 'top_bottom') else 'spatial'
                confidence = sided[0][2] if sided else 0.0
                
                coin_id = self.create_coin(
                    auction_record_id=auction_record_id,
                    group_index=0,
                    pairing_method=pairing_method,
                    pairing_confidence=confidence,
                )
                
                for det, side, side_conf in sided:
                    self.link_detection_to_coin(det['id'], coin_id, side, side_conf)
                    stats['detections_linked'] += 1
                
                stats['coins_created'] += 1
                stats['pairs_assigned'] += 1
                
            elif n == 1:
                # Single detection: one coin, unknown side
                coin_id = self.create_coin(
                    auction_record_id=auction_record_id,
                    group_index=0,
                    pairing_method='spatial',
                    pairing_confidence=0.0,
                )
                self.link_detection_to_coin(detections[0]['id'], coin_id, 'unknown', 0.0)
                stats['coins_created'] += 1
                stats['detections_linked'] += 1
                stats['singles'] += 1
                
            else:
                # 3+ detections: pair first two, rest as unknown on same coin
                # (Most auction images have at most 2 coins per lot)
                sided = assign_sides_from_spatial_sort(detections[:2], assume_obv_rev=True)
                _, layout = sort_detections_spatially(detections[:2])
                pairing_method = layout if layout in ('left_right', 'top_bottom') else 'spatial'
                confidence = sided[0][2] if sided else 0.0
                
                coin_id = self.create_coin(
                    auction_record_id=auction_record_id,
                    group_index=0,
                    pairing_method=pairing_method,
                    pairing_confidence=confidence,
                )
                
                for det, side, side_conf in sided:
                    self.link_detection_to_coin(det['id'], coin_id, side, side_conf)
                    stats['detections_linked'] += 1
                
                # Extra detections go as unknown on same coin
                for det in detections[2:]:
                    self.link_detection_to_coin(det['id'], coin_id, 'unknown', 0.0)
                    stats['detections_linked'] += 1
                
                stats['coins_created'] += 1
                stats['multi_detection'] += 1
            
            stats['records_processed'] += 1
        
        return stats

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
    
    def update_record_metadata(self, record_id: int, site: str, sale_id: str):
        """Update site/auction metadata on existing record."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "UPDATE auction_data SET site = %s, sale_id = %s WHERE id = %s",
                (site, sale_id, record_id)
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
            auction_house=row.get('auction_house', ''),
            sale_id=row.get('sale_id', ''),
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
                            a.auction_house,
                            a.sale_id,
                            a.lot_number,
                            ai.period
                        FROM coin_detections d
                        JOIN auction_data a ON d.auction_record_id = a.id
                        LEFT JOIN auction_info ai 
                            ON a.auction_house = ai.auction_house 
                            AND a.sale_id = ai.sale_id
                            AND a.lot_number BETWEEN ai.lot_start AND ai.lot_end
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
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
            
    except Exception:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    
# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_pair_hash(obv_hash: str, rev_hash: str) -> str:
    """
    Compute a stable hash for a coin pair for deduplication.
    Uses sorted concatenation so order doesn't matter.
    """
    if not obv_hash and not rev_hash:
        return ""
    if not obv_hash:
        return rev_hash
    if not rev_hash:
        return obv_hash
    
    hashes = sorted([obv_hash, rev_hash])
    return f"{hashes[0]}:{hashes[1]}"
