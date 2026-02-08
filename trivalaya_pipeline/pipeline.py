"""
Pipeline: Main orchestrator connecting scraper, vision, and ML export.

This is the primary interface for running the full pipeline.
"""

import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .config import PipelineConfig, PathConfig, MySQLConfig, MLConfig, VisionConfig
from .catalog import CatalogDB, AuctionRecord
from .vision_adapter import VisionAdapter
from .ml_exporter import MLExporter, ExportStats
from .label_parser import LabelParser


class Pipeline:
    """
    Main orchestrator for the Trivalaya pipeline.
    
    Connects:
    - trivalaya-data (scraper) - your existing scraper module
    - trivalaya-vision (CV) - your existing vision module  
    - ML export - new functionality
    
    Usage:
        pipeline = Pipeline(
            scraper_path='../trivalaya-data',
            vision_path='../trivalaya-vision',
            mysql_password='your_password'
        )
        
        # Run full pipeline
        pipeline.run_full(site='leu', sale_id='31', lots=range(1, 100))
        
        # Or run steps individually
        pipeline.process_vision(batch_size=100)
        pipeline.export_ml_dataset()
    """
    
    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _temp_parent(path: Path) -> Optional[Path]:
        """Return the triv_vision_in_ temp directory containing *path*, or None."""
        for parent in path.parents:
            if parent.name.startswith("triv_vision_in_"):
                return parent
        return None
    
    def __init__(
        self,
        scraper_path: str = None,
        vision_path: str = None,
        data_root: str = None,
        mysql_host: str = "127.0.0.1",
        mysql_user: str = "auction_user",
        mysql_password: str = "",
        mysql_database: str = "auction_data",
    ):
        # Build config
        self.paths = PathConfig()
        if scraper_path:
            self.paths.scraper_module = Path(scraper_path)
        if vision_path:
            self.paths.vision_module = Path(vision_path)
        if data_root:
            self.paths.data_root = Path(data_root)
            self.paths.__post_init__()  # Recalculate derived paths
        
        self.mysql_config = MySQLConfig(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
        )
        
        self.ml_config = MLConfig()
        self.vision_config = VisionConfig()
        
        # Initialize components
        self.db = CatalogDB(self.mysql_config)
        self.vision = VisionAdapter(self.paths, self.vision_config)
        self.exporter = MLExporter(self.db, self.ml_config, self.paths)
        self.parser = LabelParser()
        
        # Lazy-load scraper
        self._scraper_module = None
        
        # Create directories
        self.paths.create_directories()
    
    @property
    def scraper(self):
        """Lazy-load the scraper module from trivalaya-data."""
        if self._scraper_module is None:
            self._scraper_module = self._load_scraper()
        return self._scraper_module
    
    def _load_scraper(self):
        """Import scraper from trivalaya-data."""
        scraper_path = self.paths.scraper_module
        
        if not scraper_path.exists():
            print(f"⚠ Scraper module not found at: {scraper_path}")
            return None
        
        # Add to path
        scraper_src = scraper_path / "scraper"
        if scraper_src.exists():
            sys.path.insert(0, str(scraper_src))
        else:
            sys.path.insert(0, str(scraper_path))
        
        try:
            # Import your existing modules
            from database_handler import DatabaseHandler
            from scraper import scrape_site
            from site_configs import SITE_CONFIGS
            
            print(f"✓ Scraper module loaded from: {scraper_path}")
            
            return {
                'DatabaseHandler': DatabaseHandler,
                'scrape_site': scrape_site,
                'SITE_CONFIGS': SITE_CONFIGS,
            }
            
        except ImportError as e:
            print(f"⚠ Failed to import scraper: {e}")
            return None
    
    def validate(self) -> Dict[str, bool]:
        """Check all components are available."""
        return {
            'scraper': self.scraper is not None,
            'vision': self.vision.is_available,
            'database': self._test_db_connection(),
        }
    
    def _test_db_connection(self) -> bool:
        """Test MySQL connection."""
        try:
            stats = self.db.get_dataset_stats()
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
    
    # =========================================================================
    # SCRAPING (uses your existing trivalaya-data)
    # =========================================================================
    
    def scrape(self,site: str,sale_id: str,lots: range,closing_date: str = None,download_images: bool = True) -> Dict[str, int]:
                if not self.scraper:
                    raise RuntimeError("Scraper module not loaded. Check scraper_path.")

                print(f"\n{'='*60}")
                print(f"SCRAPING: {site} sale {sale_id}, lots {lots.start}-{lots.stop-1}")
                print(f"{'='*60}")

                db_handler = self.scraper['DatabaseHandler'](
                    host=self.mysql_config.host,
                    user=self.mysql_config.user,
                    password=self.mysql_config.password,
                    database=self.mysql_config.database,
                )

                # NOTE: this assumes you refactored the scraper to accept sale_id
                self.scraper['scrape_site'](
                    site_name=site,
                    lot_range=lots,
                    sale_id=sale_id,
                    closing_date=closing_date,
                    db_handler=db_handler,
                    download_images=download_images,
                    site_configs=self.scraper['SITE_CONFIGS'],
                )
                    
                # Update site/auction metadata on scraped records
                # (Your scraper doesn't store these, so we add them)
                self._update_scraped_metadata(site, sale_id, lots)
                
                return {'lots': len(lots)}
    
    def _update_scraped_metadata(self, site: str, sale_id: str, lots: range):
        """Add site/auction metadata to scraped records."""
        # This assumes lot_number is unique enough to identify records
        # You might need to adjust based on your actual schema
        pass  # The catalog.py handles this via ALTER TABLE
    
    # =========================================================================
    # VISION PROCESSING
    # =========================================================================
    
    def process_vision(self, batch_size: int = 100) -> Dict[str, int]:
        """
        Process unprocessed images through vision pipeline.
        
        Returns:
            Statistics dict with processed/detected/error counts
        """
        print(f"\n{'='*60}")
        print(f"VISION PROCESSING (batch_size={batch_size})")
        print(f"{'='*60}")
        
        if not self.vision.is_available:
            print("⚠ Using fallback detection (full vision module not loaded)")
        
        stats = {'processed': 0, 'detections': 0, 'errors': 0, 'skipped': 0}
        
        # Get unprocessed records
        records = self.db.get_unprocessed_records(limit=batch_size)
        print(f"Found {len(records)} unprocessed images")
        
        from .vision_adapter import VisionJobContext

        for record in records:
            image_path = Path(record.image_path)

            # Resolve Spaces key → local temp file (or use local path as-is)
            local_path = self.vision._resolve_input_to_local(image_path)
            temp_dir = self._temp_parent(local_path)

            try:
                if not local_path.exists():
                    stats['skipped'] += 1
                    continue

                # Run vision
                result = self.vision.process_image(local_path)

                if result.status == "error":
                    print(f"  ✗ Error processing {image_path.name}: {result.error}")
                    stats['errors'] += 1
                    self.db.mark_vision_processed(record.id)
                    continue

                if result.status == "no_detections":
                    stats['skipped'] += 1
                    self.db.mark_vision_processed(record.id)
                    continue

                # Extract coins
                safe_lot = self._safe_int(record.lot_number)
                base_name = f"{record.auction_house or 'lot'}_{record.sale_id or '0'}_{safe_lot:05d}"
                output_dir = self.paths.extracted_coins / (record.auction_house or 'unknown') / (record.sale_id or '0')

                ctx = VisionJobContext(
                    site=record.auction_house or 'unknown',
                    sale_id=record.sale_id or '0',
                    lot_number=safe_lot,
                )
                extractions = self.vision.extract_coins(
                    local_path,
                    result.detections,
                    output_dir,
                    base_name,
                    ctx,
                )

                # Save to database
                detections = self.vision.result_to_detections(record.id, extractions)
                for det in detections:
                    self.db.insert_detection(det)
                    stats['detections'] += 1

                self.db.mark_vision_processed(record.id)
                stats['processed'] += 1

                print(f"  ✓ {image_path.name}: {len(extractions)} coins detected")
            finally:
                if temp_dir is not None:
                    shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"\nVision complete: {stats['processed']} images, {stats['detections']} coins")
        return stats
    
    # =========================================================================
    # COIN PAIRING
    # =========================================================================
    
    def pair_detections(self, min_likelihood: float = 0.5) -> Dict[str, int]:
        """
        Pair unlinked detections into coin entities with side assignments.
        
        Groups detections by auction_record_id, creates coin records,
        and assigns obverse/reverse based on spatial position.
        """
        print(f"\n{'='*60}")
        print(f"COIN PAIRING")
        print(f"{'='*60}")
        
        stats = self.db.pair_unlinked_detections(min_likelihood)
        
        print(f"\nPairing complete:")
        print(f"  Records processed: {stats['records_processed']}")
        print(f"  Coins created: {stats['coins_created']}")
        print(f"  Detections linked: {stats['detections_linked']}")
        print(f"  Pairs (obv+rev): {stats['pairs_assigned']}")
        print(f"  Singles: {stats['singles']}")
        print(f"  Multi-detection lots: {stats['multi_detection']}")
        
        return stats

    # =========================================================================
    # ML EXPORT
    # =========================================================================
    
    def export_ml_dataset(
        self,
        min_likelihood: float = 0.5,
        stratify_by: str = "period",
        generate_pytorch: bool = True
    ) -> ExportStats:
        """
        Export ML-ready dataset from processed detections.
        
        Returns:
            ExportStats with counts and distributions
        """
        print(f"\n{'='*60}")
        print(f"ML DATASET EXPORT")
        print(f"{'='*60}")
        
        stats = self.exporter.export_dataset(
            min_likelihood=min_likelihood,
            stratify_by=stratify_by,
        )
        
        print(f"\nDataset exported:")
        print(f"  Train: {stats.train_count}")
        print(f"  Val: {stats.val_count}")
        print(f"  Test: {stats.test_count}")
        print(f"  Total: {stats.total_images}")
        print(f"  Duplicates removed: {stats.duplicates_removed}")
        
        if stats.periods:
            print(f"\nPeriod distribution:")
            for period, count in sorted(stats.periods.items(), key=lambda x: -x[1])[:10]:
                print(f"  {period}: {count}")
        
        if generate_pytorch:
            pytorch_path = self.exporter.generate_pytorch_dataset()
            print(f"\nPyTorch Dataset: {pytorch_path}")
        
        print(f"\nOutput location: {self.paths.ml_dataset}")
        
        return stats
    
    # =========================================================================
    # FULL PIPELINE
    # =========================================================================
    
    def run_full(
        self,
        site: str,
        sale_id: str,
        lots: range,
        closing_date: str = None,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline: scrape → vision → export.
        """
        print("\n" + "="*60)
        print("TRIVALAYA FULL PIPELINE")
        print("="*60)
        
        results = {}
        
        # Step 1: Scrape
        print("\n[1/3] SCRAPING")
        scrape_stats = self.scrape(site, sale_id, lots, closing_date)
        results['scrape'] = scrape_stats
        
        # Step 2: Vision
        print("\n[2/3] VISION PROCESSING")
        vision_stats = self.process_vision(batch_size=len(lots) + 100)
        results['vision'] = vision_stats
        
        # Step 3: Export
        print("\n[3/3] ML EXPORT")
        export_stats = self.export_ml_dataset()
        results['export'] = export_stats.to_dict()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return results
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current catalog statistics."""
        return self.db.get_dataset_stats()
    
    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("CATALOG STATISTICS")
        print("="*60)
        
        print(f"\nAuction Records:")
        print(f"  Total: {stats.get('total_auction_records', 0)}")
        print(f"  Vision processed: {stats.get('vision_processed', 0)}")
        
        print(f"\nCoin Detections: {stats.get('total_detections', 0)}")
        
        print(f"\nML Dataset:")
        print(f"  Train: {stats.get('train_count', 0)}")
        print(f"  Val: {stats.get('val_count', 0)}")
        print(f"  Test: {stats.get('test_count', 0)}")
        print(f"  Needs review: {stats.get('needs_review', 0)}")
        
        if stats.get('periods'):
            print(f"\nTop periods:")
            for period, count in sorted(stats['periods'].items(), key=lambda x: -x[1])[:5]:
                print(f"  {period}: {count}")
