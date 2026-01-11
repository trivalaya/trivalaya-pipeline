#!/usr/bin/env python3
"""
audit_data_integrity.py - Pre-scaling health check for Trivalaya dataset.

Checks for:
1. Missing image files on disk
2. Confidence score anomalies (0.0 with valid period)
3. Orphaned records (broken foreign keys)
4. Duplicate image hashes
5. Path/Period label mismatches
6. coin_detections metrics anomalies

Usage:
    python audit_data_integrity.py              # Full audit
    python audit_data_integrity.py --fix        # Attempt automatic fixes
    python audit_data_integrity.py --summary    # Quick summary only
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
DB_HOST = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
DB_USER = os.getenv("TRIVALAYA_DB_USER", "auction_user")
DB_PASSWORD = os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024")
DB_NAME = os.getenv("TRIVALAYA_DB_NAME", "auction_data")


def get_db_connection():
    """Create database connection with explicit parameters."""
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

# Where to look for images
POSSIBLE_ROOTS = [
    Path("."),
    Path(".."),
    Path("/root"),
    Path("/home"),
]


class AuditResult:
    """Container for audit findings."""
    
    def __init__(self):
        self.issues = defaultdict(list)
        self.stats = {}
    
    def add_issue(self, category: str, record_id: int, details: str):
        self.issues[category].append({
            'id': record_id,
            'details': details
        })
    
    def add_stat(self, name: str, value):
        self.stats[name] = value
    
    @property
    def total_issues(self) -> int:
        return sum(len(v) for v in self.issues.values())
    
    def print_report(self, max_per_category: int = 10):
        print("\n" + "=" * 60)
        print("📊 TRIVALAYA DATA INTEGRITY REPORT")
        print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Stats summary
        print("\n📈 DATASET STATISTICS")
        print("-" * 40)
        for name, value in self.stats.items():
            print(f"  {name}: {value}")
        
        # Issues by category
        print(f"\n⚠️  ISSUES FOUND: {self.total_issues}")
        print("-" * 40)
        
        if self.total_issues == 0:
            print("  ✅ No issues detected!")
            return
        
        for category, items in sorted(self.issues.items()):
            print(f"\n🔴 {category} ({len(items)} issues)")
            
            for item in items[:max_per_category]:
                print(f"    ID {item['id']}: {item['details']}")
            
            if len(items) > max_per_category:
                print(f"    ... and {len(items) - max_per_category} more")


def resolve_path(db_path: str) -> Path:
    """Try to find the actual file path."""
    if not db_path:
        return None
    
    p = Path(db_path)
    if p.exists():
        return p
    
    for root in POSSIBLE_ROOTS:
        candidate = root / p
        if candidate.exists():
            return candidate
    
    return None


def audit_missing_files(cursor, result: AuditResult):
    """Check for image files that don't exist on disk."""
    print("\n🔍 Checking for missing image files...")
    
    cursor.execute("""
        SELECT id, image_path FROM ml_dataset 
        WHERE image_path IS NOT NULL AND image_path != ''
    """)
    
    total = 0
    missing = 0
    
    for row in cursor.fetchall():
        total += 1
        path = resolve_path(row['image_path'])
        
        if not path:
            missing += 1
            result.add_issue(
                "MISSING_FILES",
                row['id'],
                f"File not found: {row['image_path']}"
            )
    
    result.add_stat("Total ML Dataset Records", total)
    result.add_stat("Missing Files", missing)
    print(f"  Scanned: {total}, Missing: {missing}")


def audit_confidence_anomalies(cursor, result: AuditResult):
    """Find records with suspicious confidence scores."""
    print("\n🔍 Checking confidence score anomalies...")
    
    # Confidence = 0 with valid period (should not happen)
    cursor.execute("""
        SELECT id, period, label_confidence, raw_label 
        FROM ml_dataset 
        WHERE label_confidence = 0 
        AND period IS NOT NULL
        AND period != ''
    """)
    
    for row in cursor.fetchall():
        result.add_issue(
            "CONFIDENCE_ZERO_WITH_PERIOD",
            row['id'],
            f"period='{row['period']}', conf=0.0, label='{row['raw_label'][:50]}...'"
        )
    
    # Period = NULL with high confidence (parser bug)
    cursor.execute("""
        SELECT id, period, label_confidence, raw_label 
        FROM ml_dataset 
        WHERE period IS NULL 
        AND label_confidence > 0.5
    """)
    
    for row in cursor.fetchall():
        result.add_issue(
            "NULL_PERIOD_HIGH_CONFIDENCE",
            row['id'],
            f"period=NULL, conf={row['label_confidence']}, label='{row['raw_label'][:50]}...'"
        )
    
    # Confidence distribution
    cursor.execute("""
        SELECT 
            CASE 
                WHEN label_confidence = 0 THEN 'zero'
                WHEN label_confidence < 0.6 THEN 'low'
                WHEN label_confidence < 0.8 THEN 'medium'
                WHEN label_confidence < 0.95 THEN 'high'
                ELSE 'very_high'
            END as bucket,
            COUNT(*) as count
        FROM ml_dataset
        GROUP BY bucket
    """)
    
    dist = {r['bucket']: r['count'] for r in cursor.fetchall()}
    result.add_stat("Confidence Distribution", dist)


def audit_orphaned_records(cursor, result: AuditResult):
    """Check for foreign key integrity issues."""
    print("\n🔍 Checking for orphaned records...")
    
    # ml_dataset with invalid coin_detection_id
    cursor.execute("""
        SELECT m.id, m.coin_detection_id
        FROM ml_dataset m
        LEFT JOIN coin_detections c ON m.coin_detection_id = c.id
        WHERE c.id IS NULL
    """)
    
    for row in cursor.fetchall():
        result.add_issue(
            "ORPHAN_ML_DATASET",
            row['id'],
            f"References non-existent coin_detection_id={row['coin_detection_id']}"
        )
    
    # coin_detections with invalid auction_record_id
    cursor.execute("""
        SELECT c.id, c.auction_record_id
        FROM coin_detections c
        LEFT JOIN auction_data a ON c.auction_record_id = a.id
        WHERE a.id IS NULL
        LIMIT 100
    """)
    
    for row in cursor.fetchall():
        result.add_issue(
            "ORPHAN_DETECTION",
            row['id'],
            f"References non-existent auction_record_id={row['auction_record_id']}"
        )


def audit_duplicate_hashes(cursor, result: AuditResult):
    """Check for duplicate image hashes (dedup failure)."""
    print("\n🔍 Checking for duplicate hashes...")
    
    cursor.execute("""
        SELECT image_hash, COUNT(*) as cnt, GROUP_CONCAT(id) as ids
        FROM ml_dataset 
        WHERE image_hash IS NOT NULL AND image_hash != ''
        GROUP BY image_hash 
        HAVING cnt > 1
        LIMIT 50
    """)
    
    dup_count = 0
    for row in cursor.fetchall():
        dup_count += row['cnt'] - 1  # Count extras, not originals
        result.add_issue(
            "DUPLICATE_HASH",
            0,  # Multiple IDs
            f"Hash {row['image_hash'][:16]}... appears {row['cnt']} times (IDs: {row['ids'][:50]})"
        )
    
    result.add_stat("Duplicate Images", dup_count)


def audit_path_period_mismatch(cursor, result: AuditResult):
    """Check for mismatches between folder path and period label."""
    print("\n🔍 Checking path/period mismatches...")
    
    cursor.execute("""
        SELECT 
            id, period, image_path,
            SUBSTRING_INDEX(SUBSTRING_INDEX(image_path, '/', -2), '/', 1) as path_folder
        FROM ml_dataset
        WHERE period IS NOT NULL
        LIMIT 10000
    """)
    
    mismatches = defaultdict(int)
    examples = []
    
    for row in cursor.fetchall():
        if row['period'] != row['path_folder']:
            key = f"{row['path_folder']} → {row['period']}"
            mismatches[key] += 1
            
            if len(examples) < 5:
                examples.append(row)
    
    # Report aggregated mismatches
    if mismatches:
        result.add_stat("Path/Period Mismatches", dict(mismatches))
        
        for row in examples:
            result.add_issue(
                "PATH_PERIOD_MISMATCH",
                row['id'],
                f"Path folder: {row['path_folder']}, DB period: {row['period']}"
            )


def audit_detection_metrics(cursor, result: AuditResult):
    """Check for anomalies in coin_detections metrics."""
    print("\n🔍 Checking detection metrics...")
    
    # coin_likelihood should be between 0 and 1
    cursor.execute("""
        SELECT id, coin_likelihood, circularity, solidity, edge_support
        FROM coin_detections
        WHERE coin_likelihood < 0 OR coin_likelihood > 1
           OR circularity < 0 OR circularity > 1
           OR solidity < 0 OR solidity > 1
           OR edge_support < 0 OR edge_support > 1
        LIMIT 50
    """)
    
    for row in cursor.fetchall():
        result.add_issue(
            "INVALID_DETECTION_METRICS",
            row['id'],
            f"likelihood={row['coin_likelihood']}, circ={row['circularity']}, "
            f"sol={row['solidity']}, edge={row['edge_support']}"
        )
    
    # Statistics on edge_support (you mentioned sometimes 0.9, sometimes 0)
    cursor.execute("""
        SELECT 
            CASE 
                WHEN edge_support = 0 THEN 'zero'
                WHEN edge_support < 0.2 THEN 'very_low'
                WHEN edge_support < 0.5 THEN 'low'
                WHEN edge_support < 0.8 THEN 'medium'
                ELSE 'high'
            END as bucket,
            COUNT(*) as count
        FROM coin_detections
        GROUP BY bucket
    """)
    
    dist = {r['bucket']: r['count'] for r in cursor.fetchall()}
    result.add_stat("Edge Support Distribution", dist)


def audit_needs_review_consistency(cursor, result: AuditResult):
    """Check needs_review flag consistency."""
    print("\n🔍 Checking needs_review consistency...")
    
    # needs_review=1 but is_verified=1 (conflicting flags)
    cursor.execute("""
        SELECT id, needs_review, is_verified, period, label_confidence
        FROM ml_dataset
        WHERE needs_review = 1 AND is_verified = 1
    """)
    
    for row in cursor.fetchall():
        result.add_issue(
            "CONFLICTING_FLAGS",
            row['id'],
            f"needs_review=1 but is_verified=1 (period={row['period']})"
        )
    
    # Statistics
    cursor.execute("""
        SELECT 
            needs_review,
            is_verified,
            COUNT(*) as count
        FROM ml_dataset
        GROUP BY needs_review, is_verified
    """)
    
    flags = {}
    for row in cursor.fetchall():
        key = f"review={row['needs_review']}, verified={row['is_verified']}"
        flags[key] = row['count']
    
    result.add_stat("Review/Verified Flags", flags)


def attempt_fixes(cursor, result: AuditResult, conn):
    """Attempt to fix some issues automatically."""
    print("\n🔧 ATTEMPTING AUTOMATIC FIXES")
    print("-" * 40)
    
    fixes_applied = 0
    
    # Fix 1: Clear conflicting flags (prefer is_verified)
    cursor.execute("""
        UPDATE ml_dataset 
        SET needs_review = 0 
        WHERE needs_review = 1 AND is_verified = 1
    """)
    fixed = cursor.rowcount
    if fixed > 0:
        print(f"  ✅ Cleared needs_review for {fixed} already-verified records")
        fixes_applied += fixed
    
    # Fix 2: Set needs_review=1 for NULL period records
    cursor.execute("""
        UPDATE ml_dataset 
        SET needs_review = 1 
        WHERE period IS NULL AND is_verified = 0
    """)
    fixed = cursor.rowcount
    if fixed > 0:
        print(f"  ✅ Flagged {fixed} NULL-period records for review")
        fixes_applied += fixed
    
    # Fix 3: Set minimum confidence for NULL period
    cursor.execute("""
        UPDATE ml_dataset 
        SET label_confidence = 0.3 
        WHERE period IS NULL AND label_confidence > 0.5
    """)
    fixed = cursor.rowcount
    if fixed > 0:
        print(f"  ✅ Reduced confidence for {fixed} NULL-period records")
        fixes_applied += fixed
    
    conn.commit()
    
    print(f"\n  Total fixes applied: {fixes_applied}")
    return fixes_applied


def print_summary(cursor):
    """Quick summary of dataset status."""
    print("\n📊 QUICK DATASET SUMMARY")
    print("=" * 50)
    
    # Total records
    cursor.execute("SELECT COUNT(*) as c FROM auction_data")
    print(f"Auction Records:      {cursor.fetchone()['c']}")
    
    cursor.execute("SELECT COUNT(*) as c FROM coin_detections")
    print(f"Coin Detections:      {cursor.fetchone()['c']}")
    
    cursor.execute("SELECT COUNT(*) as c FROM ml_dataset")
    print(f"ML Dataset Records:   {cursor.fetchone()['c']}")
    
    # Verification status
    cursor.execute("""
        SELECT 
            SUM(is_verified) as verified,
            SUM(needs_review) as needs_review
        FROM ml_dataset
    """)
    row = cursor.fetchone()
    print(f"  - Verified:         {row['verified']}")
    print(f"  - Needs Review:     {row['needs_review']}")
    
    # Period distribution
    cursor.execute("""
        SELECT period, COUNT(*) as c 
        FROM ml_dataset 
        WHERE period IS NOT NULL
        GROUP BY period 
        ORDER BY c DESC
    """)
    
    print("\nPeriod Distribution:")
    for row in cursor.fetchall():
        print(f"  {row['period']}: {row['c']}")


def main():
    parser = argparse.ArgumentParser(description="Audit Trivalaya dataset integrity")
    parser.add_argument('--fix', action='store_true', help="Attempt automatic fixes")
    parser.add_argument('--summary', action='store_true', help="Quick summary only")
    parser.add_argument('--max-issues', type=int, default=10, help="Max issues per category to show")
    args = parser.parse_args()
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        if args.summary:
            print_summary(cursor)
            return
        
        result = AuditResult()
        
        # Run all audits
        audit_missing_files(cursor, result)
        audit_confidence_anomalies(cursor, result)
        audit_orphaned_records(cursor, result)
        audit_duplicate_hashes(cursor, result)
        audit_path_period_mismatch(cursor, result)
        audit_detection_metrics(cursor, result)
        audit_needs_review_consistency(cursor, result)
        
        # Print report
        result.print_report(max_per_category=args.max_issues)
        
        # Attempt fixes if requested
        if args.fix and result.total_issues > 0:
            attempt_fixes(cursor, result, conn)
        
        # Health score
        health = max(0, 100 - result.total_issues)
        print(f"\n🏥 Dataset Health Score: {health}%")
        
        if result.total_issues > 0:
            print("\n💡 RECOMMENDATIONS:")
            if result.issues.get("MISSING_FILES"):
                print("  1. Run image re-sync or remove missing records")
            if result.issues.get("CONFIDENCE_ZERO_WITH_PERIOD"):
                print("  2. Update label_parser.py to fix confidence calculation")
            if result.issues.get("ORPHAN_ML_DATASET"):
                print("  3. Clean up orphaned ml_dataset records")
            if result.issues.get("DUPLICATE_HASH"):
                print("  4. Run deduplication pass")
            
            print("\n  Use --fix flag to attempt automatic repairs")
        
    except mysql.connector.Error as e:
        print(f"❌ Database error: {e}")
        raise
    finally:
        if 'cursor' in dir():
            cursor.close()
        if 'conn' in dir():
            conn.close()


if __name__ == "__main__":
    main()