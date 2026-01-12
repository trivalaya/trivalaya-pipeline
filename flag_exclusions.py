#!/usr/bin/env python3
"""
flag_exclusions.py - Flag records for exclusion based on text patterns and clusters

Usage:
    python flag_exclusions.py --flag-lots      # Flag group lots via text patterns
    python flag_exclusions.py --flag-modern    # Flag modern coins via text patterns  
    python flag_exclusions.py --flag-noise     # Flag noise cluster from clustering results
    python flag_exclusions.py --flag-cluster 1 group_lot  # Flag specific cluster
    python flag_exclusions.py --report         # Show exclusion statistics
"""

import os
import re
import argparse
import mysql.connector

# Database configuration
DB_HOST = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
DB_USER = os.getenv("TRIVALAYA_DB_USER", "auction_user")
DB_PASSWORD = os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024")
DB_NAME = os.getenv("TRIVALAYA_DB_NAME", "auction_data")

# Patterns for group lots
GROUP_LOT_PATTERNS = [
    r'\blot of\b',
    r'\bgroup of\b', 
    r'\brun of\b',
    r'\bcollection of\b',
    r'\b\d+ coins\b',
    r'\b\d+ pieces\b',
    r'\bmiscellaneous\b',
    r'\bvarious\b.*\bcoins\b',
    r'\bmixed lot\b',
]

# Patterns for modern coins (post-1800)
MODERN_PATTERNS = [
    r'\b1[89]\d{2}\b',  # Years 1800-1999
    r'\b20[012]\d\b',   # Years 2000-2029
    r'\bvictoria\b',
    r'\bgeorge [iv]+\b',
    r'\bwilliam iv\b',
    r'\bqueen elizabeth\b',
    r'\beuro\b',
    r'\bdollar\b',
    r'\bfranc\b.*\b19\d{2}\b',
    r'\bmark\b.*\b19\d{2}\b',
    r'\bpfennig\b.*\b19\d{2}\b',
    r'\bcommemorative\b',
    r'\bproof\b',
    r'\buncirculated\b.*\bmint\b',
]

def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def flag_by_patterns(patterns, exclude_reason, dry_run=False):
    """Flag records matching any of the given patterns."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Get all active records with their raw labels
    cursor.execute("""
        SELECT id, raw_label 
        FROM ml_dataset 
        WHERE is_active = 1 AND exclude_reason IS NULL
    """)
    records = cursor.fetchall()
    
    flagged = []
    for rec in records:
        text = (rec['raw_label'] or '').lower()
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                flagged.append((rec['id'], rec['raw_label'][:80], pattern))
                break
    
    print(f"\nFound {len(flagged)} records matching '{exclude_reason}' patterns")
    
    if flagged and not dry_run:
        ids = [f[0] for f in flagged]
        placeholders = ','.join(['%s'] * len(ids))
        cursor.execute(f"""
            UPDATE ml_dataset 
            SET is_active = 0, exclude_reason = %s 
            WHERE id IN ({placeholders})
        """, [exclude_reason] + ids)
        conn.commit()
        print(f"✅ Flagged {cursor.rowcount} records as '{exclude_reason}'")
    elif flagged and dry_run:
        print("\nDRY RUN - Would flag:")
        for id, label, pattern in flagged[:20]:
            print(f"  [{id}] {label}... (matched: {pattern})")
        if len(flagged) > 20:
            print(f"  ... and {len(flagged) - 20} more")
    
    cursor.close()
    conn.close()
    return len(flagged)

def flag_noise_cluster(cluster_csv='cluster_output/cluster_results.csv', dry_run=False):
    """Flag all records in noise cluster from clustering results."""
    import pandas as pd
    
    if not os.path.exists(cluster_csv):
        print(f"❌ Cluster results not found: {cluster_csv}")
        return 0
    
    df = pd.read_csv(cluster_csv)
    
    # Find noise records (cluster_id = -1 or 'noise')
    noise_mask = (df['cluster_id'].astype(str) == '-1') | (df['visual_cluster'].str.contains('noise', case=False, na=False))
    noise_paths = df[noise_mask]['image_path'].tolist()
    
    print(f"\nFound {len(noise_paths)} noise records in clustering results")
    
    if not noise_paths:
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Match by image path
    flagged = 0
    for path in noise_paths:
        if dry_run:
            continue
        cursor.execute("""
            UPDATE ml_dataset 
            SET is_active = 0, exclude_reason = 'visual_noise'
            WHERE image_path LIKE %s AND is_active = 1
        """, (f'%{os.path.basename(path)}',))
        flagged += cursor.rowcount
    
    if not dry_run:
        conn.commit()
        print(f"✅ Flagged {flagged} records as 'visual_noise'")
    else:
        print(f"DRY RUN - Would flag {len(noise_paths)} records")
    
    cursor.close()
    conn.close()
    return flagged

def flag_specific_cluster(cluster_id, exclude_reason, cluster_csv='cluster_output/cluster_results.csv', dry_run=False):
    """Flag all records in a specific cluster."""
    import pandas as pd
    
    if not os.path.exists(cluster_csv):
        print(f"❌ Cluster results not found: {cluster_csv}")
        return 0
    
    df = pd.read_csv(cluster_csv)
    
    # Find records in this cluster
    mask = df['visual_cluster'] == f'cluster_{cluster_id}'
    if mask.sum() == 0:
        # Try alternate format
        mask = df['cluster_id'].astype(str) == str(cluster_id)
    
    paths = df[mask]['image_path'].tolist()
    
    print(f"\nFound {len(paths)} records in cluster {cluster_id}")
    
    if not paths:
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    flagged = 0
    for path in paths:
        if dry_run:
            continue
        cursor.execute("""
            UPDATE ml_dataset 
            SET is_active = 0, exclude_reason = %s
            WHERE image_path LIKE %s AND is_active = 1
        """, (exclude_reason, f'%{os.path.basename(path)}'))
        flagged += cursor.rowcount
    
    if not dry_run:
        conn.commit()
        print(f"✅ Flagged {flagged} records as '{exclude_reason}'")
    else:
        print(f"DRY RUN - Would flag {len(paths)} records as '{exclude_reason}'")
    
    cursor.close()
    conn.close()
    return flagged

def exclusion_report():
    """Show exclusion statistics."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Overall stats
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(is_active) as active,
            SUM(CASE WHEN is_active = 0 THEN 1 ELSE 0 END) as excluded
        FROM ml_dataset
    """)
    stats = cursor.fetchone()
    
    print("\n" + "="*60)
    print("EXCLUSION REPORT")
    print("="*60)
    print(f"Total records:    {stats['total']}")
    print(f"Active:           {stats['active']} ({100*stats['active']/stats['total']:.1f}%)")
    print(f"Excluded:         {stats['excluded']} ({100*stats['excluded']/stats['total']:.1f}%)")
    
    # By reason
    cursor.execute("""
        SELECT 
            COALESCE(exclude_reason, 'active') as reason,
            COUNT(*) as count
        FROM ml_dataset
        GROUP BY exclude_reason
        ORDER BY count DESC
    """)
    
    print("\nBy Reason:")
    for row in cursor.fetchall():
        print(f"  {row['reason']}: {row['count']}")
    
    # By period (active only)
    cursor.execute("""
        SELECT period, COUNT(*) as count
        FROM ml_dataset
        WHERE is_active = 1
        GROUP BY period
        ORDER BY count DESC
    """)
    
    print("\nActive by Period:")
    for row in cursor.fetchall():
        print(f"  {row['period']}: {row['count']}")
    
    cursor.close()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description='Flag records for exclusion')
    parser.add_argument('--flag-lots', action='store_true', help='Flag group lots')
    parser.add_argument('--flag-modern', action='store_true', help='Flag modern coins')
    parser.add_argument('--flag-noise', action='store_true', help='Flag noise cluster')
    parser.add_argument('--flag-cluster', nargs=2, metavar=('CLUSTER_ID', 'REASON'),
                       help='Flag specific cluster with reason')
    parser.add_argument('--report', action='store_true', help='Show exclusion report')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be flagged')
    parser.add_argument('--password', help='Database password')
    
    args = parser.parse_args()
    
    if args.password:
        global DB_PASSWORD
        DB_PASSWORD = args.password
    
    if args.flag_lots:
        flag_by_patterns(GROUP_LOT_PATTERNS, 'group_lot', args.dry_run)
    
    if args.flag_modern:
        flag_by_patterns(MODERN_PATTERNS, 'modern', args.dry_run)
    
    if args.flag_noise:
        flag_noise_cluster(dry_run=args.dry_run)
    
    if args.flag_cluster:
        cluster_id, reason = args.flag_cluster
        flag_specific_cluster(cluster_id, reason, dry_run=args.dry_run)
    
    if args.report:
        exclusion_report()
    
    if not any([args.flag_lots, args.flag_modern, args.flag_noise, 
                args.flag_cluster, args.report]):
        parser.print_help()

if __name__ == "__main__":
    main()