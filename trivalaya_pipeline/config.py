"""
Pipeline Configuration

Configures paths and settings for orchestrating:
- trivalaya-data (scraper)
- trivalaya-vision (computer vision)
- ML dataset export
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PathConfig:
    """
    Path configuration for the pipeline.
    
    External module paths should point to your local repos.
    """
    
    # === External Module Paths ===
    # Set these to your local repo locations, or use environment variables
    scraper_module: Path = field(default_factory=lambda: Path(
        os.environ.get("TRIVALAYA_SCRAPER_PATH", "../trivalaya-data")
    ))
    vision_module: Path = field(default_factory=lambda: Path(
        os.environ.get("TRIVALAYA_VISION_PATH", "../trivalaya-vision")
    ))
    
    # === Data Directories ===
    data_root: Path = field(default_factory=lambda: Path(
        os.environ.get("TRIVALAYA_DATA_ROOT", "./trivalaya_data")
    ))
    
    # Derived paths (set in __post_init__)
    raw_images: Path = field(init=False)
    extracted_coins: Path = field(init=False)
    ml_dataset: Path = field(init=False)
    train_split: Path = field(init=False)
    val_split: Path = field(init=False)
    test_split: Path = field(init=False)
    
    def __post_init__(self):
        # Convert strings to Paths
        self.scraper_module = Path(self.scraper_module)
        self.vision_module = Path(self.vision_module)
        self.data_root = Path(self.data_root)
        
        # Set derived paths
        self.raw_images = self.data_root / "01_raw" / "images"
        self.extracted_coins = self.data_root / "02_processed" / "extracted"
        self.ml_dataset = self.data_root / "03_ml_ready" / "dataset"
        self.train_split = self.ml_dataset / "train"
        self.val_split = self.ml_dataset / "val"
        self.test_split = self.ml_dataset / "test"
    
    def create_directories(self):
        """Create all data directories."""
        for path in [
            self.raw_images,
            self.extracted_coins,
            self.train_split,
            self.val_split,
            self.test_split,
        ]:
            path.mkdir(parents=True, exist_ok=True)
    
    def validate_modules(self) -> Dict[str, bool]:
        """Check if external modules are accessible."""
        return {
            'scraper': (self.scraper_module / "scraper").is_dir(),
            'vision': (self.vision_module / "src").is_dir(),
        }


@dataclass
class MySQLConfig:
    """MySQL connection settings — uses your existing database."""
    
    host: str = "127.0.0.1"
    port: int = 3306
    user: str = "auction_user"
    password: str = ""  # Set via environment or constructor
    database: str = "auction_data"
    
    def __post_init__(self):
        # Allow environment override
        self.host = os.environ.get("MYSQL_HOST", self.host)
        self.user = os.environ.get("MYSQL_USER", self.user)
        self.password = os.environ.get("MYSQL_PASSWORD", self.password)
        self.database = os.environ.get("MYSQL_DATABASE", self.database)


@dataclass
class MLConfig:
    """Machine learning dataset configuration."""
    
    # Image sizes
    target_size: tuple = (224, 224)
    high_res_size: tuple = (512, 512)
    
    # Data splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Quality filtering
    min_coin_likelihood: float = 0.50
    min_circularity: float = 0.60
    min_image_size: int = 100
    
    # Deduplication
    enable_dedup: bool = True
    hash_algorithm: str = "md5"
    
    # Label fields to extract
    classification_fields: List[str] = field(default_factory=lambda: [
        "period",
        "authority", 
        "denomination",
        "mint",
        "material",
    ])
    
    def __post_init__(self):
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 0.001, \
            "Split ratios must sum to 1.0"


@dataclass 
class VisionConfig:
    """Configuration passed to the vision adapter."""
    
    # Which sensitivity to try first
    default_sensitivity: str = "standard"
    fallback_sensitivity: str = "high"
    
    # Extraction settings
    crop_margin_ratio: float = 0.05
    save_transparent: bool = True
    save_normalized: bool = True
    
    # Two-coin scene handling
    assume_obv_rev_pair: bool = True


@dataclass
class PipelineConfig:
    """Master configuration combining all components."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    mysql: MySQLConfig = field(default_factory=MySQLConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    
    def validate(self) -> bool:
        """Run all validation checks."""
        module_status = self.paths.validate_modules()
        
        if not module_status['scraper']:
            print(f"⚠ Scraper module not found at: {self.paths.scraper_module}")
        if not module_status['vision']:
            print(f"⚠ Vision module not found at: {self.paths.vision_module}")
        
        return all(module_status.values())


# =============================================================================
# LABEL TAXONOMY (used by label_parser)
# =============================================================================

PERIOD_TAXONOMY = {
    "greek": {
        "display": "Greek",
        "subperiods": ["archaic", "classical", "hellenistic"],
        "date_range": (-800, -30),
    },
    "roman_republican": {
        "display": "Roman Republican", 
        "subperiods": ["early", "middle", "late"],
        "date_range": (-280, -27),
    },
    "roman_imperial": {
        "display": "Roman Imperial",
        "subperiods": [
            "julio-claudian", "flavian", "adoptive", "severan",
            "crisis", "tetrarchy", "constantinian", "valentinian", "theodosian"
        ],
        "date_range": (-27, 476),
    },
    "roman_provincial": {
        "display": "Roman Provincial",
        "subperiods": [],
        "date_range": (-27, 300),
    },
    "byzantine": {
        "display": "Byzantine",
        "subperiods": ["early", "middle", "late"],
        "date_range": (491, 1453),
    },
    "celtic": {
        "display": "Celtic",
        "subperiods": [],
        "date_range": (-400, 100),
    },
    "islamic": {
        "display": "Islamic",
        "subperiods": ["umayyad", "abbasid", "fatimid", "mamluk", "ottoman"],
        "date_range": (622, 1924),
    },
}

DENOMINATION_PATTERNS = {
    "denarius": r"\bdenari(?:us|i)\b",
    "aureus": r"\baureu[sm]\b",
    "sestertius": r"\bsesterti(?:us|i)\b",
    "as": r"\b(?:as|asses)\b",
    "antoninianus": r"\bantoninian(?:us|i)\b",
    "solidus": r"\bsolid(?:us|i)\b",
    "tremissis": r"\btremissis\b",
    "semissis": r"\bsemissis\b",
    "follis": r"\bfoll(?:is|es)\b",
    "nummus": r"\bnumm(?:us|i)\b",
    "tetradrachm": r"\btetradrachm?\b",
    "drachm": r"\bdrachm(?:a|s)?\b",
    "stater": r"\bstaters?\b",
    "obol": r"\bobols?\b",
}

MATERIAL_TYPES = ["gold", "electrum", "silver", "billon", "bronze", "copper", "lead"]
