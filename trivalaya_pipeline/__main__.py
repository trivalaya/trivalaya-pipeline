"""
Trivalaya Pipeline CLI

Orchestrates trivalaya-data (scraper) and trivalaya-vision (CV) for ML dataset creation.

Usage:
    # Run full pipeline
    python -m trivalaya_pipeline run leu 31 --lots 1-100

    # Run individual steps
    python -m trivalaya_pipeline scrape leu 31 --lots 1-100
    python -m trivalaya_pipeline vision --batch 100

    # Export (detection mode - legacy)
    python -m trivalaya_pipeline export

    # Export (paired obv/rev mode - Option 1)
    python -m trivalaya_pipeline export --mode coin_pair --missing-side skip

    # Show statistics
    python -m trivalaya_pipeline stats

    # Validate setup
    python -m trivalaya_pipeline check
"""

import argparse
import sys
import os

from .pipeline import Pipeline


def parse_lot_range(lots_str: str) -> range:
    """Parse lot range string like '1-100' or '42'."""
    if "-" in lots_str:
        start, end = map(int, lots_str.split("-"))
        return range(start, end + 1)
    n = int(lots_str)
    return range(n, n + 1)


def cmd_check(args, pipeline: Pipeline) -> int:
    """Validate setup and connections."""
    print("\nValidating pipeline setup...")

    status = pipeline.validate()

    print(f"\n{'Component':<20} {'Status':<10}")
    print("-" * 30)

    for component, ok in status.items():
        icon = "✓" if ok else "✗"
        print(f"{component:<20} {icon}")

    if all(status.values()):
        print("\n✓ All components ready!")
        return 0

    print("\n⚠ Some components unavailable. Check paths and credentials.")
    return 1


def cmd_scrape(args, pipeline: Pipeline) -> int:
    """Scrape auction lots."""
    lots = parse_lot_range(args.lots)
    pipeline.scrape(
        site=args.site,
        sale_id=args.sale_id,
        lots=lots,
        closing_date=args.date,
        download_images=not args.no_images,
    )
    return 0


def cmd_vision(args, pipeline: Pipeline) -> int:
    """Process images through vision pipeline."""
    pipeline.process_vision(batch_size=args.batch)
    return 0


def cmd_pair(args, pipeline: Pipeline) -> int:
    """Pair detections into coin entities."""
    pipeline.pair_detections(min_likelihood=args.min_likelihood)
    return 0


def cmd_export(args, pipeline: Pipeline) -> int:
    """Export ML-ready dataset."""
    mode = getattr(args, "mode", "detection")

    # -------------------------------------------------------------------------
    # coin_pair export
    # -------------------------------------------------------------------------
    if mode == "coin_pair":
        exporter = getattr(pipeline, "exporter", None)
        if exporter is None or not hasattr(exporter, "export_coin_pair_dataset"):
            print("✗ Paired exporter not available.")
            print("  Ensure MLExporter.export_coin_pair_dataset is implemented and wired.")
            return 1

        from .ml_exporter import MissingSideRule

        stats = exporter.export_coin_pair_dataset(
            min_likelihood=args.min_likelihood,
            stratify_by=args.stratify,
            missing_side_rule=MissingSideRule(args.missing_side),
        )

        # Optional extra artifact for coin_pair mode only
        if args.pytorch and hasattr(exporter, "generate_pytorch_dataset"):
            path = exporter.generate_pytorch_dataset()
            print(f"PyTorch Pair Dataset: {path}")

        return 0

    # -------------------------------------------------------------------------
    # legacy detection export
    # -------------------------------------------------------------------------
    pipeline.export_ml_dataset(
        min_likelihood=args.min_likelihood,
        stratify_by=args.stratify,
        generate_pytorch=args.pytorch,
    )
    return 0

def cmd_run(args, pipeline: Pipeline) -> int:
    """Run full pipeline."""
    lots = parse_lot_range(args.lots)
    pipeline.run_full(
        site=args.site,
        sale_id=args.sale_id,
        lots=lots,
        closing_date=args.date,
    )
    return 0


def cmd_stats(args, pipeline: Pipeline) -> int:
    """Show catalog statistics."""
    pipeline.print_stats()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trivalaya Pipeline: Scraper → Vision → ML Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--scraper-path",
        default="../trivalaya-data",
        help="Path to trivalaya-data repo",
    )
    parser.add_argument(
        "--vision-path",
        default="../trivalaya-vision",
        help="Path to trivalaya-vision repo",
    )
    parser.add_argument(
        "--data-root",
        default="./trivalaya_data",
        help="Root directory for output data",
    )
    parser.add_argument("--mysql-host", default="127.0.0.1")
    parser.add_argument("--mysql-user", default="auction_user")
    parser.add_argument(
        "--mysql-password",
        default=os.environ.get("MYSQL_PASSWORD", ""),
        help="MySQL password (or set MYSQL_PASSWORD env var)",
    )
    parser.add_argument("--mysql-database", default="auction_data")

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # check
    subparsers.add_parser("check", help="Validate setup")

    # scrape
    scrape_parser = subparsers.add_parser("scrape", help="Scrape auction lots")
    scrape_parser.add_argument("site", help="Site name (leu, gorny, nomos, cng)")
    scrape_parser.add_argument("sale_id", help="Auction ID")
    scrape_parser.add_argument("--lots", required=True, help='Lot range (e.g., "1-100")')
    scrape_parser.add_argument("--date", help="Closing date")
    scrape_parser.add_argument("--no-images", action="store_true")

    # vision
    vision_parser = subparsers.add_parser("vision", help="Process through vision pipeline")
    vision_parser.add_argument("--batch", type=int, default=100, help="Batch size")

    # pair
    pair_parser = subparsers.add_parser("pair", help="Pair detections into coin entities (obv/rev)")
    pair_parser.add_argument("--min-likelihood", type=float, default=0.5)

    # export
    export_parser = subparsers.add_parser("export", help="Export ML dataset")
    export_parser.add_argument("--min-likelihood", type=float, default=0.5)
    export_parser.add_argument("--stratify", default="period")
    export_parser.add_argument(
    "--pytorch",
    action="store_true",
    default=False,
    help="Generate PyTorch dataset class (default: False)",
    )

    export_parser.add_argument(
        "--mode",
        choices=["detection", "coin_pair"],
        default="detection",
        help="Export mode: detection (legacy) or coin_pair (obv/rev paired)",
    )
    export_parser.add_argument(
        "--missing-side",
        choices=["skip", "placeholder", "duplicate"],
        default="skip",
        help="coin_pair mode only: what to do if obv/rev missing",
    )

    # run
    run_parser = subparsers.add_parser("run", help="Run full pipeline")
    run_parser.add_argument("site", help="Site name")
    run_parser.add_argument("sale_id", help="Auction ID")
    run_parser.add_argument("--lots", required=True, help="Lot range")
    run_parser.add_argument("--date", help="Closing date")

    # stats
    subparsers.add_parser("stats", help="Show statistics")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    pipeline = Pipeline(
        scraper_path=args.scraper_path,
        vision_path=args.vision_path,
        data_root=args.data_root,
        mysql_host=args.mysql_host,
        mysql_user=args.mysql_user,
        mysql_password=args.mysql_password,
        mysql_database=args.mysql_database,
    )

    commands = {
        "check": cmd_check,
        "scrape": cmd_scrape,
        "vision": cmd_vision,
        "pair": cmd_pair,
        "export": cmd_export,
        "run": cmd_run,
        "stats": cmd_stats,
    }

    rc = commands[args.command](args, pipeline)
    sys.exit(rc)


if __name__ == "__main__":
    main()
