from __future__ import annotations
from pathlib import Path
import os
import re
import sys
import time

import mysql.connector

from trivalaya_pipeline.storage.spaces_client import SpacesClient

# Matches folder like "cng_459" or "leu_20" -> site="cng", sale="459"
AUCTION_DIR_RE = re.compile(r"^(?P<site>[a-z0-9]+)_(?P<sale>\d+)$", re.I)

LOCAL_ROOT = Path("/root/trivalaya-pipeline")

def canonical_key(old_image_path: str) -> tuple[str, Path] | None:
    """
    Convert legacy DB path like 'cng_459/Lot_03547.jpg' to canonical Spaces key:
        'raw/auctions/cng/459/Lot_03547.jpg'
    Also returns the expected local file path.
    """
    p = (old_image_path or "").strip().lstrip("./").lstrip("/")
    if not p or p.startswith("raw/"):
        return None

    parts = p.split("/", 1)
    if len(parts) != 2:
        return None

    auction_dir, filename = parts
    # Accept both numeric and non-numeric sale ids (e.g., cng_459, cng_t23)
    if "_" not in auction_dir:
        return None

    site, sale = auction_dir.split("_", 1)
    site = site.lower().strip()
    sale = sale.lower().strip()

    if not site or not sale:
        return None

    new_key = f"raw/auctions/{site}/{sale}/{filename}"

    local_path = LOCAL_ROOT / auction_dir / filename
    return new_key, local_path

def main(dry_run: bool) -> None:
    # DB config (matches your droplet pattern)
    db_host = os.getenv("TRIVALAYA_DB_HOST", "127.0.0.1")
    db_user = os.getenv("TRIVALAYA_DB_USER", "auction_user")
    db_password = os.environ["TRIVALAYA_DB_PASSWORD"]
    db_name = os.getenv("TRIVALAYA_DB_NAME", "auction_data")  # your DB name from SHOW TABLES output

    spaces = SpacesClient()

    conn = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
    )
    cur = conn.cursor(buffered=True)   # <-- key change
    upd = conn.cursor() 

    migrated = skipped = missing = errors = 0
    t0 = time.time()

    try:
        # Only rows that are not already canonical
        cur.execute("""
            SELECT id, image_path
            FROM auction_data
            WHERE image_path IS NOT NULL
              AND image_path <> ''
              AND image_path NOT LIKE 'raw/%'
        """)

        for row_id, image_path in cur:
            result = canonical_key(image_path)
            if result is None:
                skipped += 1
                continue

            new_key, local_path = result

            if not local_path.exists():
                print(f"[missing] id={row_id} {image_path} -> {new_key} (local missing: {local_path})")
                missing += 1
                continue

            if dry_run:
                print(f"[DRY] id={row_id} {image_path} -> {new_key}")
                migrated += 1
                continue

            try:
                # Upload then update DB for that row
                spaces.upload_file(str(local_path), new_key, content_type="image/jpeg")

                upd.execute(
                    "UPDATE auction_data SET image_path=%s WHERE id=%s",
                    (new_key, row_id),
                )
                conn.commit()
                migrated += 1

                if migrated % 500 == 0:
                    elapsed = max(1e-6, time.time() - t0)
                    rate = migrated / elapsed
                    print(f"... migrated={migrated} skipped={skipped} missing={missing} errors={errors} ({rate:.1f}/s)")

            except Exception as e:
                conn.rollback()
                errors += 1
                print(f"[error] id={row_id} {image_path} -> {new_key}: {e}")

        print(f"Done. migrated={migrated} skipped={skipped} missing={missing} errors={errors}")

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    dry = ("--go" not in sys.argv)
    main(dry_run=dry)
