#!/usr/bin/env python3
"""
Enrich cluster_output/cluster_results.csv with auction fields from MySQL.

Join path:
  cluster_results.image_path -> ml_dataset.image_path
  ml_dataset.coin_detection_id -> coin_detections.id
  coin_detections.auction_record_id -> auction_data.id
"""

import os
import sys
import math
import pandas as pd

# Use one of these connectors:
#   pip install mysql-connector-python
import mysql.connector


CSV_IN  = "cluster_output/cluster_results.csv"
CSV_OUT = "cluster_output/cluster_results.enriched.csv"

DB = {
    "host": os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1"),
    "user": os.getenv("TRIVALAYA_DB_USER", "auction_user"),
    "password": os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024"),
    "database": os.getenv("TRIVALAYA_DB_NAME", "auction_data"),
}

# Adjust if your paths have a known prefix difference (e.g. leading "./")
def normalize_path(p: str) -> str:
    if not isinstance(p, str):
        return p
    p = p.strip()
    if p.startswith("./"):
        p = p[2:]
    p = p.replace("\\", "/")
    return p

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def main():
    if not os.path.exists(CSV_IN):
        print(f"ERROR: Missing {CSV_IN}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_IN)
    if "image_path" not in df.columns:
        print("ERROR: cluster_results.csv must contain an image_path column", file=sys.stderr)
        sys.exit(1)

    df["image_path"] = df["image_path"].map(normalize_path)

    # Unique paths to reduce temp table size
    paths = df["image_path"].dropna().unique().tolist()
    print(f"Loaded {len(df):,} rows, {len(paths):,} unique image_path values")

    conn = mysql.connector.connect(**DB)
    conn.autocommit = True
    cur = conn.cursor()

    # Temp table for paths
    cur.execute("DROP TEMPORARY TABLE IF EXISTS tmp_cluster_paths")
    # Use VARCHAR if your paths are reasonably bounded; TEXT works too but indexing is limited.
    cur.execute("""
        CREATE TEMPORARY TABLE tmp_cluster_paths (
            image_path TEXT NOT NULL,
            PRIMARY KEY (image_path(255))
        ) ENGINE=InnoDB
    """)

    # Insert in chunks
    insert_sql = "INSERT IGNORE INTO tmp_cluster_paths (image_path) VALUES (%s)"
    for batch in chunked(paths, 2000):
        cur.executemany(insert_sql, [(p,) for p in batch])
        print(f"Inserted {len(batch):,} paths into tmp_cluster_paths...")

    # De-dup ml_dataset by image_path (pick latest id per path)
    # Then join across to auction_data.
    query = """
        SELECT
            p.image_path,

            md.raw_label AS raw_label,

            ad.auction_house AS auction_house,
            ad.sale_id      AS sale_id,
            ad.lot_number   AS lot_number,

            ad.title        AS lot_title,
            ad.description  AS lot_description,

            ad.image_url    AS image_url,
            ad.current_bid  AS current_bid,
            ad.closing_date AS closing_date,

            ad.site         AS site,
            ad.auction_id   AS auction_id,
            ad.lot_url AS lot_url
        FROM tmp_cluster_paths p
        LEFT JOIN (
            SELECT image_path, MAX(id) AS md_id
            FROM ml_dataset
            GROUP BY image_path
        ) mx
            ON mx.image_path = p.image_path
        LEFT JOIN ml_dataset md
            ON md.id = mx.md_id
        LEFT JOIN coin_detections cd
            ON cd.id = md.coin_detection_id
        LEFT JOIN auction_data ad
            ON ad.id = cd.auction_record_id
    """

    map_df = pd.read_sql(query, conn)
    print(f"Fetched mapping rows: {len(map_df):,}")

    # Merge back
    out = df.merge(map_df, on="image_path", how="left")

    # Optional: create a "view_lot_url" if you have a rule; placeholder here.
    # If you later add a real lot_url column, just include it in the SQL and remove this.
    # out["view_lot_url"] = None

    # QA summary
    matched = out["lot_number"].notna().sum()
    print(f"Matched auction rows: {matched:,} / {len(out):,} ({matched/len(out)*100:.1f}%)")

    out.to_csv(CSV_OUT, index=False)
    print(f"Wrote: {CSV_OUT}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
