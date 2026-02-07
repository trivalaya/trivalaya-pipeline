#!/usr/bin/env python3
"""
Enrich cluster_output/cluster_results.csv with auction fields from MySQL.

PAIR-safe join path (recommended):
  cluster_results.id -> ml_coin_dataset.id
  ml_coin_dataset.coin_id -> coin_detections.coin_id
  coin_detections.auction_record_id -> auction_data.id
"""

import os
import sys
import pandas as pd
import mysql.connector

CSV_IN  = "cluster_output_clip_spaces/cluster_results.csv"
CSV_OUT = "cluster_output_clip_spaces/cluster_results.enriched.csv"

DB = {
    "host": os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1"),
    "user": os.getenv("TRIVALAYA_DB_USER", "auction_user"),
    "password": os.getenv("TRIVALAYA_DB_PASSWORD", "Veritas@2024"),
    "database": os.getenv("TRIVALAYA_DB_NAME", "auction_data"),
}

def chunked(items, n):
    for i in range(0, len(items), n):
        yield items[i:i+n]

def main():
    if not os.path.exists(CSV_IN):
        print(f"ERROR: Missing {CSV_IN}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(CSV_IN)

    if "id" not in df.columns:
        print(f"ERROR: CSV must contain 'id'. Columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    # Normalize id as Int64 (nullable-safe), then list unique valid ints
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    ids = df["id"].dropna().astype(int).unique().tolist()

    print(f"Loaded {len(df):,} rows, {len(ids):,} unique ids")

    if not ids:
        print("ERROR: No usable ids found in CSV.", file=sys.stderr)
        sys.exit(1)

    conn = mysql.connector.connect(**DB)
    conn.autocommit = True
    cur = conn.cursor()

    # Temp table of CSV ids
    cur.execute("DROP TEMPORARY TABLE IF EXISTS tmp_cluster_ids")
    cur.execute("""
        CREATE TEMPORARY TABLE tmp_cluster_ids (
            id BIGINT NOT NULL,
            PRIMARY KEY (id)
        ) ENGINE=InnoDB
    """)

    ins = "INSERT IGNORE INTO tmp_cluster_ids (id) VALUES (%s)"
    for batch in chunked(ids, 5000):
        cur.executemany(ins, [(int(x),) for x in batch])
        print(f"Inserted {len(batch):,} ids into tmp_cluster_ids...")

    # Query:
    # 1) join CSV ids to ml_coin_dataset by m.id
    # 2) use m.coin_id to get latest coin_detections row for that coin_id
    # 3) join to auction_data
    query = """
    WITH m_pick AS (
        -- If multiple ml_coin_dataset rows share same coin_id, keep latest m.id
        SELECT
            CAST(TRIM(coin_id) AS CHAR) AS coin_id_key,
            MAX(id) AS max_m_id
        FROM ml_coin_dataset
        WHERE coin_id IS NOT NULL
        GROUP BY CAST(TRIM(coin_id) AS CHAR)
    ),
    cd_pick AS (
        -- If multiple detections per coin_id, keep latest cd.id
        SELECT
            CAST(TRIM(coin_id) AS CHAR) AS coin_id_key,
            MAX(id) AS max_cd_id
        FROM coin_detections
        WHERE coin_id IS NOT NULL
        GROUP BY CAST(TRIM(coin_id) AS CHAR)
    )
    SELECT
        t.id AS id,                    -- cluster_results.id (which is coin_id)
        m.coin_id AS coin_id,
        m.raw_label AS raw_label,
        m.period AS period,
        m.subperiod AS subperiod,
        m.authority AS authority,
        m.denomination AS denomination,
        m.material AS material,
        m.mint AS mint,
        m.obv_path AS obv_path,
        m.rev_path AS rev_path,

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
        ad.lot_url      AS lot_url
    FROM tmp_cluster_ids t
    LEFT JOIN m_pick mp
        ON mp.coin_id_key = CAST(t.id AS CHAR)
    LEFT JOIN ml_coin_dataset m
        ON m.id = mp.max_m_id
    LEFT JOIN cd_pick cdp
        ON cdp.coin_id_key = CAST(t.id AS CHAR)
    LEFT JOIN coin_detections cd
        ON cd.id = cdp.max_cd_id
    LEFT JOIN auction_data ad
        ON ad.id = cd.auction_record_id
"""

    map_df = pd.read_sql_query(query, conn)
    print(f"Fetched mapping rows: {len(map_df):,}")

    # Merge back on id
    out = df.merge(map_df, on="id", how="left")

    matched = out["lot_number"].notna().sum() if "lot_number" in out.columns else 0
    pct = (matched / len(out) * 100.0) if len(out) else 0.0
    print(f"Matched auction rows: {matched:,} / {len(out):,} ({pct:.1f}%)")

    out.to_csv(CSV_OUT, index=False)
    print(f"Wrote: {CSV_OUT}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
