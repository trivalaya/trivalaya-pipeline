"""
Label Parser: Extract structured numismatic labels from auction descriptions.

Converts free-text descriptions into ML-ready classifications.
"""

import re
from typing import Dict, Optional, List
from dataclasses import dataclass, field

from .config import PERIOD_TAXONOMY, DENOMINATION_PATTERNS, MATERIAL_TYPES


@dataclass
class ParsedLabel:
    """Structured label extracted from auction description."""
    
    period: str = ""
    subperiod: str = ""
    authority: str = ""
    denomination: str = ""
    mint: str = ""
    material: str = ""
    
    # Dating
    date_text: str = ""
    date_start: Optional[int] = None
    date_end: Optional[int] = None
    
    # Confidence scores
    confidence: Dict[str, float] = field(default_factory=dict)
    
    # Raw text
    raw_title: str = ""
    raw_description: str = ""
    
    # Diagnostics
    parsing_notes: List[str] = field(default_factory=list)
    
    @property
    def overall_confidence(self) -> float:
        """Average confidence across filled fields."""
        scores = [v for k, v in self.confidence.items() if getattr(self, k, "")]
        return sum(scores) / len(scores) if scores else 0.0
    
    @property
    def needs_review(self) -> bool:
        return self.overall_confidence < 0.5 or self.period == ""
    
    @property
    def raw_label(self) -> str:
        return f"{self.raw_title} | {self.raw_description}"


class LabelParser:
    """Parse auction descriptions into structured labels."""
    
    # Roman Emperors with period/subperiod mapping
    ROMAN_EMPERORS = {
        # Julio-Claudian
        r"\baugustus\b": ("Augustus", "roman_imperial", "julio-claudian"),
        r"\btiberius\b": ("Tiberius", "roman_imperial", "julio-claudian"),
        r"\bcaligula\b": ("Caligula", "roman_imperial", "julio-claudian"),
        r"\bclaudius\b(?!.*ii)": ("Claudius", "roman_imperial", "julio-claudian"),
        r"\bnero\b": ("Nero", "roman_imperial", "julio-claudian"),
        
        # Flavian
        r"\bvespasian\b": ("Vespasian", "roman_imperial", "flavian"),
        r"\btitus\b": ("Titus", "roman_imperial", "flavian"),
        r"\bdomitian\b": ("Domitian", "roman_imperial", "flavian"),
        
        # Adoptive/Antonine
        r"\bnerva\b": ("Nerva", "roman_imperial", "adoptive"),
        r"\btrajan\b": ("Trajan", "roman_imperial", "adoptive"),
        r"\bhadrian\b": ("Hadrian", "roman_imperial", "adoptive"),
        r"\bantoninus\s*pius\b": ("Antoninus Pius", "roman_imperial", "adoptive"),
        r"\bmarcus\s*aurelius\b": ("Marcus Aurelius", "roman_imperial", "adoptive"),
        r"\bfaustina\b": ("Faustina", "roman_imperial", "adoptive"),
        r"\bcommodus\b": ("Commodus", "roman_imperial", "adoptive"),
        
        # Severan
        r"\bseptimius\s*severus\b": ("Septimius Severus", "roman_imperial", "severan"),
        r"\bcaracalla\b": ("Caracalla", "roman_imperial", "severan"),
        r"\bgeta\b": ("Geta", "roman_imperial", "severan"),
        r"\belagabalus\b": ("Elagabalus", "roman_imperial", "severan"),
        r"\bseverus\s*alexander\b": ("Severus Alexander", "roman_imperial", "severan"),
        r"\bjulia\s*domna\b": ("Julia Domna", "roman_imperial", "severan"),
        r"\bjulia\s*maesa\b": ("Julia Maesa", "roman_imperial", "severan"),
        r"\bjulia\s*mamaea\b": ("Julia Mamaea", "roman_imperial", "severan"),
        
        # Crisis
        r"\bmaximinus\b(?!.*ii)": ("Maximinus I", "roman_imperial", "crisis"),
        r"\bgordian\s*iii\b": ("Gordian III", "roman_imperial", "crisis"),
        r"\bphilip\s*(?:i|the\s*arab)\b": ("Philip I", "roman_imperial", "crisis"),
        r"\btrajan\s*decius\b": ("Trajan Decius", "roman_imperial", "crisis"),
        r"\bvalerian\b": ("Valerian", "roman_imperial", "crisis"),
        r"\bgallienus\b": ("Gallienus", "roman_imperial", "crisis"),
        r"\bclaudius\s*(?:ii|gothicus)\b": ("Claudius II", "roman_imperial", "crisis"),
        r"\baurelian\b": ("Aurelian", "roman_imperial", "crisis"),
        r"\bprobus\b": ("Probus", "roman_imperial", "crisis"),
        
        # Tetrarchy & Constantinian
        r"\bdiocletian\b": ("Diocletian", "roman_imperial", "tetrarchy"),
        r"\bmaximian\b": ("Maximian", "roman_imperial", "tetrarchy"),
        r"\bconstantius\s*(?:i|chlorus)\b": ("Constantius I", "roman_imperial", "tetrarchy"),
        r"\bgalerius\b": ("Galerius", "roman_imperial", "tetrarchy"),
        r"\bconstantine\s*(?:i|the\s*great)\b": ("Constantine I", "roman_imperial", "constantinian"),
        r"\bhelena\b": ("Helena", "roman_imperial", "constantinian"),
        r"\bcrispus\b": ("Crispus", "roman_imperial", "constantinian"),
        r"\bconstantine\s*ii\b": ("Constantine II", "roman_imperial", "constantinian"),
        r"\bconstans\b(?!.*ii)": ("Constans", "roman_imperial", "constantinian"),
        r"\bconstantius\s*ii\b": ("Constantius II", "roman_imperial", "constantinian"),
        r"\bjulian\b(?!.*ii)": ("Julian II", "roman_imperial", "constantinian"),
        
        # Valentinian & Theodosian
        r"\bvalentinian\s*i\b": ("Valentinian I", "roman_imperial", "valentinian"),
        r"\bvalens\b": ("Valens", "roman_imperial", "valentinian"),
        r"\bgratian\b": ("Gratian", "roman_imperial", "valentinian"),
        r"\bvalentinian\s*ii\b": ("Valentinian II", "roman_imperial", "valentinian"),
        r"\btheodosius\s*(?:i|the\s*great)\b": ("Theodosius I", "roman_imperial", "theodosian"),
        r"\barcadius\b": ("Arcadius", "roman_imperial", "theodosian"),
        r"\bhonorius\b": ("Honorius", "roman_imperial", "theodosian"),
        
        # Republican
        r"\bjulius\s*caesar\b": ("Julius Caesar", "roman_republican", "late"),
        r"\bmark\s*antony\b": ("Mark Antony", "roman_republican", "late"),
        r"\bmarc\s*antony\b": ("Mark Antony", "roman_republican", "late"),
        r"\bbrutus\b": ("Brutus", "roman_republican", "late"),
        r"\bpompey\b": ("Pompey", "roman_republican", "late"),
    }
    
    GREEK_AUTHORITIES = {
        r"\bathens\b": ("Athens", "greek", "classical"),
        r"\bathenian\b": ("Athens", "greek", "classical"),
        r"\bcorinth\b": ("Corinth", "greek", "classical"),
        r"\bsyracuse\b": ("Syracuse", "greek", "classical"),
        r"\baegina\b": ("Aegina", "greek", "archaic"),
        r"\balexander\s*(?:iii|the\s*great)\b": ("Alexander III", "greek", "hellenistic"),
        r"\bphilip\s*ii\b(?!.*roman)": ("Philip II of Macedon", "greek", "classical"),
        r"\blysimachus\b": ("Lysimachus", "greek", "hellenistic"),
        r"\bseleuc": ("Seleucid", "greek", "hellenistic"),
        r"\bptolem": ("Ptolemaic", "greek", "hellenistic"),
        r"\bantiochus\b": ("Antiochos", "greek", "hellenistic"),
    }
    
    MINT_PATTERNS = {
        r"\brome\b": "Rome",
        r"\blugdunum\b": "Lugdunum",
        r"\blyon\b": "Lugdunum",
        r"\bantioch\b": "Antioch",
        r"\balexandria\b": "Alexandria",
        r"\bconstantinople\b": "Constantinople",
        r"\bsiscia\b": "Siscia",
        r"\bheraclea\b": "Heraclea",
        r"\bcyzicus\b": "Cyzicus",
        r"\bnicomedia\b": "Nicomedia",
        r"\bthessalonica\b": "Thessalonica",
        r"\btrier\b": "Trier",
        r"\btreveri\b": "Trier",
        r"\barles\b": "Arles",
        r"\barelate\b": "Arles",
        r"\bravenna\b": "Ravenna",
        r"\bmediolanum\b": "Milan",
        r"\bmilan\b": "Milan",
        r"\baquilea\b": "Aquileia",
        r"\bserdica\b": "Serdica",
        r"\bcarthage\b": "Carthage",
        r"\blondinium\b": "London",
        r"\blondon\b": "London",
    }
    
    def __init__(self):
        # Compile patterns
        self._emperor_patterns = [
            (re.compile(p, re.IGNORECASE), v)
            for p, v in self.ROMAN_EMPERORS.items()
        ]
        self._greek_patterns = [
            (re.compile(p, re.IGNORECASE), v)
            for p, v in self.GREEK_AUTHORITIES.items()
        ]
        self._mint_patterns = [
            (re.compile(p, re.IGNORECASE), v)
            for p, v in self.MINT_PATTERNS.items()
        ]
        self._denom_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for name, p in DENOMINATION_PATTERNS.items()
        ]
    
    def parse(self, title: str, description: str = "") -> ParsedLabel:
        """Parse title and description into structured label."""
        text = f"{title} {description}".lower()
        label = ParsedLabel(raw_title=title, raw_description=description)
        
        self._parse_authority(text, label)
        self._parse_denomination(text, label)
        self._parse_mint(text, label)
        self._parse_material(text, label)
        self._parse_dates(text, label)
        
        return label
    
    def _parse_authority(self, text: str, label: ParsedLabel):
        """Extract ruler/authority."""
        # Try Roman emperors
        for pattern, (authority, period, subperiod) in self._emperor_patterns:
            if pattern.search(text):
                label.authority = authority
                label.period = period
                label.subperiod = subperiod
                label.confidence['authority'] = 0.9
                label.confidence['period'] = 0.9
                return
        
        # Try Greek
        for pattern, (authority, period, subperiod) in self._greek_patterns:
            if pattern.search(text):
                label.authority = authority
                label.period = period
                label.subperiod = subperiod
                label.confidence['authority'] = 0.85
                label.confidence['period'] = 0.85
                return
        
        # Period keywords
        period_keywords = {
            r"\broman\s+imperial\b": "roman_imperial",
            r"\broman\s+republic": "roman_republican",
            r"\broman\s+provincial\b": "roman_provincial",
            r"\bbyzantin": "byzantine",
            r"\bgreek\b": "greek",
            r"\bceltic\b|\bbritain\b|\bgaul\b": "celtic",
            r"\bislamic\b|\bumayyad\b|\babbasid\b": "islamic",
        }
        
        for pattern, period in period_keywords.items():
            if re.search(pattern, text, re.IGNORECASE):
                label.period = period
                label.confidence['period'] = 0.7
                return
    
    def _parse_denomination(self, text: str, label: ParsedLabel):
        """Extract denomination."""
        for pattern, denom in self._denom_patterns:
            if pattern.search(text):
                label.denomination = denom
                label.confidence['denomination'] = 0.9
                return
        
        # Size hints
        size_hints = {
            r"\bae\s*1\b": "AE1",
            r"\bae\s*2\b": "AE2",
            r"\bae\s*3\b": "AE3",
            r"\bae\s*4\b": "AE4",
        }
        for pattern, denom in size_hints.items():
            if re.search(pattern, text, re.IGNORECASE):
                label.denomination = denom
                label.confidence['denomination'] = 0.6
                return
    
    def _parse_mint(self, text: str, label: ParsedLabel):
        """Extract mint."""
        for pattern, mint in self._mint_patterns:
            if pattern.search(text):
                label.mint = mint
                label.confidence['mint'] = 0.85
                return
    
    def _parse_material(self, text: str, label: ParsedLabel):
        """Extract material."""
        material_patterns = {
            "gold": r"\bgold\b|\bav\b|\baureus\b|\bsolidus\b",
            "electrum": r"\belectrum\b",
            "silver": r"\bsilver\b|\bar\b|\bdenari|\bdrachm",
            "billon": r"\bbillon\b",
            "bronze": r"\bbronze\b|\bae\b|\bsesterti",
            "copper": r"\bcopper\b",
        }
        
        for material, pattern in material_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                label.material = material
                label.confidence['material'] = 0.8
                return
    
    def _parse_dates(self, text: str, label: ParsedLabel):
        """Extract dating information."""
        patterns = [
            (r"(?:ad|a\.d\.?)\s*(\d{1,4})(?:\s*[-–]\s*(\d{1,4}))?", 1),
            (r"(\d{1,4})(?:\s*[-–]\s*(\d{1,4}))?\s*(?:ad|a\.d\.?)", 1),
            (r"(?:bc|b\.c\.?)\s*(\d{1,4})(?:\s*[-–]\s*(\d{1,4}))?", -1),
            (r"(\d{1,4})(?:\s*[-–]\s*(\d{1,4}))?\s*(?:bc|b\.c\.?)", -1),
        ]
        
        for pattern, multiplier in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                label.date_text = match.group(0)
                label.date_start = int(match.group(1)) * multiplier
                if match.group(2):
                    label.date_end = int(match.group(2)) * multiplier
                return


def parse_auction_label(title: str, description: str = "") -> ParsedLabel:
    """Convenience function to parse auction text."""
    return LabelParser().parse(title, description)
