import re
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class ParsedLabel:
    original_text: str
    period: Optional[str] = None
    denomination: Optional[str] = None
    material: Optional[str] = None
    mint: Optional[str] = None
    confidence: float = 0.0
    needs_review: bool = False
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        return self.confidence


class LabelParser:
    """
    Numismatic Logic Engine v2.8
    Coverage-first labeling pass.
    - Hanover+ => modern
    - Greek civic bronzes => roman_provincial w/out emperor if imperial-era geography present
    - Parthian normalized to Persian (for now)
    - Crusader stays medieval (split later)
    """

    # -----------------------------
    # 0. Non-coin / junk detection
    # -----------------------------
    JUNK_PATTERNS = [
        r"token", r"medal", r"jeton", r"replica",
        r"fantasy", r"\bcopy\b", r"reproduction",
        r"seal", r"\bweight\b", r"exonumia",
        r"banknote", r"note ", r"paper money"
    ]

    # -----------------------------
    # 1. Period patterns (regex)
    # NOTE: these are regex fragments; we wrap with word-ish boundaries
    # -----------------------------
    PERIOD_PATTERNS = {
        "roman_imperial": [
            r"imperial", r"augustus", r"caesar", r"\bric\b", r"cohen", r"\bbmcre\b", r"kaiser",
            r"sestertius", r"antoninianus", r"dupondius", r"\bsemis\b", r"quadrans",
            # emperors (provincial logic can piggyback on these)
            r"tiberius", r"caligula", r"claudius", r"nero", r"galba", r"otho", r"vitellius",
            r"vespasian", r"titus", r"domitian", r"nerva", r"trajan", r"hadrian",
            r"antoninus", r"aurelius", r"commodus", r"septimius", r"severus", r"caracalla",
            r"geta", r"macrinus", r"elagabalus", r"gordian", r"philip", r"decius",
            r"gallus", r"valerian", r"gallienus", r"aurelian", r"probus", r"diocletian",
            r"maximian", r"constantius", r"constantine", r"licinius",
        ],
        "roman_provincial": [
            r"provincial", r"\brpc\b",
            r"cappadocia", r"syria", r"alexandria",
            r"arabia", r"\bbostra\b",
            # These are strong provincial tells in auction text
            r"\bbillon tetradrachm\b",
        ],
        "roman_republican": [
            r"republic", r"republican", r"republik",
            r"crawford", r"sydenham", r"aes grave", r"quadrigatus", r"victoriatus",
            r"anonyme", r"prägungen",  # German: anonymous issues / coinages
        ],
        "greek": [
            r"\bgreek\b", r"magna graecia", r"sicily", r"attica", r"athens",
            r"ptolemaic", r"seleucid", r"\bsng\b", r"\bhgc\b",
            # expanded geography (can be greek OR provincial)
            r"thrace", r"moesia", r"pisidia", r"bithynia", r"pontus",
            r"paphlagonia", r"phrygia", r"lycia", r"pamphylia", r"cilicia",
            r"galatia", r"ionia", r"etruria", r"lucania", r"argolis",
            r"epirus", r"thessaly", r"elymais", r"kyrenaika", r"cyrenaica",
        ],
        "byzantine": [
            r"byzantine", r"\bdoc\b", r"\bsb\b", r"trachy",
            r"pentanummium", r"decanummium", r"hyperpyron", r"histamenon", r"tetarteron",
            r"nomisma",
        ],
        "islamic": [
            r"islamic", r"dirham", r"dinar", r"fals", r"umayyad", r"abbasid",
            r"artuqid", r"seljuq", r"ottoman",
        ],
        "medieval": [
            r"medieval", r"crusader",r"anglo-saxon", r"anglo saxon",  # crusader stays medieval for now
            r"grosso", r"denier", r"penny",
            r"sceat", r"sceatta", r"styca", r"bracteate",
            # some broad Europe signals (coverage-first)
            r"serbia", r"genoa", r"genova", r"venice", r"bulgaria", r"hungary",
            r"france", r"louis", r"austria", r"hanover",  # hanover will be upgraded to modern in resolver
            r"royal",
        ],
        "celtic": [
            r"celtic", r"gaul", r"danubian", r"stater", r"potin",
        ],
        "persian": [
            r"persian", r"sasanian", r"parthian", r"siglos", r"daric",
        ],
        "modern": [
            # you requested: Hanover and later => modern
            r"hanover", r"victoria", r"victorian",
            r"george iii", r"george iv", r"william iv",
            r"18th century", r"19th century", r"\bmodern\b",
        ],
    }

    # -----------------------------
    # 2. Denomination map
    # -----------------------------
    DENOM_MAP = {
        "nomisma": "solidus",
        "solidus": "solidus",
        "histamenon": "histamenon",
        "tetarteron": "tetarteron",
        "hyperpyron": "hyperpyron",
        "tremissis": "tremissis",
        "follis": "follis",
        "hexagram": "hexagram",
        "siliqua": "siliqua",

        "drachm": "drachm",
        "didrachm": "didrachm",
        "tridrachm": "tridrachm",
        "tetradrachm": "tetradrachm",
        "hemidrachm": "hemidrachm",
        "stater": "stater",
        "distater": "distater",
        "litra": "litra",
        "obol": "obol",
        "diobol": "diobol",
        "triobol": "triobol",
        "hemiobol": "hemiobol",

        "aureus": "aureus",
        "quinarius": "quinarius",
        "denarius": "denarius",
        "antoninianus": "antoninianus",
        "double sestertius": "double_sestertius",
        "sestertius": "sestertius",
        "dupondius": "dupondius",
        "as": "as",
        "semis": "semis",
        "quadrans": "quadrans",
        "aes grave": "aes_grave",

        "dirham": "dirham",
        "dinar": "dinar",
        "fals": "fals",

        "sceatta": "sceatta",
        "styca": "styca",
        "denier": "denier",
        "penny": "penny",
        "grosso": "grosso",
        "matapan": "grosso",
        "crown": "crown",
    }

    # -----------------------------
    # 3. Material map
    # -----------------------------
    MATERIAL_MAP = {
        "av": "gold", "gold": "gold", "aureus": "gold", "solidus": "gold", "el": "electrum", "electrum": "electrum",
        "ar": "silver", "silver": "silver", "denarius": "silver", "siliqua": "silver", "miliarense": "silver",
        "ae": "bronze", "bronze": "bronze", "copper": "bronze", "brass": "bronze", "orichalcum": "bronze",
        "bi": "billon", "billon": "billon", "potin": "potin",
    }

    # -----------------------------
    # 4. Denomination constraints (lightweight)
    # (coverage-first; used only to resolve obvious conflicts)
    # -----------------------------
    DENOM_CONSTRAINTS = {
        "solidus": {"roman_imperial", "byzantine", "medieval", "modern"},
        "histamenon": {"byzantine"},
        "tetarteron": {"byzantine"},
        "hyperpyron": {"byzantine"},
        "tremissis": {"roman_imperial", "byzantine", "medieval"},
        "siliqua": {"roman_imperial", "byzantine"},
        "follis": {"roman_imperial", "byzantine"},
        "aureus": {"roman_imperial", "roman_republican"},
        "denarius": {"roman_imperial", "roman_republican"},
        "quinarius": {"roman_imperial", "roman_republican"},
        "sestertius": {"roman_imperial", "roman_republican"},
        "dupondius": {"roman_imperial"},
        "antoninianus": {"roman_imperial", "roman_provincial"},
        "aes_grave": {"roman_republican"},
        "stater": {"greek", "persian", "celtic"},
        "tetradrachm": {"greek", "roman_provincial", "celtic", "persian"},
        "drachm": {"greek", "persian", "islamic"},
        "hemidrachm": {"greek", "persian"},
        "obol": {"greek"},
        "diobol": {"greek"},
        "triobol": {"greek"},
        "dirham": {"islamic"},
        "dinar": {"islamic", "medieval", "modern"},
        "sceatta": {"medieval"},
        "styca": {"medieval"},
        "denier": {"medieval"},
        "grosso": {"medieval", "byzantine"},
        "crown": {"medieval", "modern"},
    }

    # -----------------------------
    # 5. Provincial geography (imperial-era region bucket)
    # Used for rule: "Greek civic bronzes => provincial even without emperor names"
    # -----------------------------
    ROMAN_PROVINCIAL_REGIONS = {
        "thrace", "moesia", "bithynia", "pontus", "paphlagonia", "phrygia",
        "lycia", "pamphylia", "cilicia", "galatia", "pisidia", "cappadocia",
        "syria", "egypt", "alexandria", "arabia", "bostra", "antioch", "seleucia",
        "ionia",
    }
    #-----------------------------------
    # 6. Greek cities 
    #--------------------------------------
    GREEK_POLEIS = {
        "athens", "aegina", "corinth", "thebes", "argos",
        "syracuse", "gela", "akragas", "katane", "leontini",
        "tarentum", "metapontum", "croton", "sybaris",
        "miletos", "ephesos", "samos", "chios", "teos",
        "phokaia", "erythrai", "klazomenai",
        "rhodes", "knidos", "halikarnassos",
        "thasos", "abdera", "maroneia",
        "amphipolis", "olynthos",
    }
    #-----------------------------------------------
    # 7. Celtic - Germanic Tribes
    #-------------------------------------------------
    CELTIC_TRIBES = {
        "aedui", "sequani", "parisii", "nervii",
        "bituriges", "arverni", "helvetii",
        "boii", "boier","volcae", "remi",
        "iceni", "trinovantes", "catuvellauni",
    }


    def _has(self, text: str, pattern: str) -> bool:
        """
        Regex match with "word-ish" boundaries.
        Works with multi-word keys like 'magna graecia' and abbreviations like 'RIC'/'RPC'.
        """
        # If the pattern already contains explicit \b or other anchors, use as-is
        if r"\b" in pattern or pattern.startswith("^") or pattern.endswith("$"):
            return re.search(pattern, text) is not None

        # Otherwise wrap with non-word boundaries
        wrapped = rf"(?<!\w){pattern}(?!\w)"
        return re.search(wrapped, text) is not None

    def parse(self, title: str, description: str = "") -> ParsedLabel:
        text = (str(title) + " " + str(description)).lower()
        result = ParsedLabel(original_text=title)

        # A. Junk exclusion
        if any(self._has(text, p) for p in self.JUNK_PATTERNS):
            result.period = "non_coin"
            result.confidence = 0.99
            result.notes.append("Excluded non-coin object")
            return result

        # B. Detect period signals
        detected_periods: Set[str] = set()
        for period, patterns in self.PERIOD_PATTERNS.items():
            for p in patterns:
                if self._has(text, p):
                    detected_periods.add(period)
        
        # C. Denomination
        denom_hits = [k for k in self.DENOM_MAP if self._has(text, re.escape(k))]
        if denom_hits:
            best = max(denom_hits, key=len)
            result.denomination = self.DENOM_MAP[best]

        # D. Material
        mat_hits = [k for k in self.MATERIAL_MAP if self._has(text, re.escape(k))]
        if mat_hits:
            best = max(mat_hits, key=len)
            result.material = self.MATERIAL_MAP[best]

        primary_period: Optional[str] = None

        # E. Parthian -> Persian (explicit)
        if "persian" in detected_periods:
            primary_period = "persian"
            result.confidence = 0.95
            result.notes.append("Parthian normalized to Persian")

        # F. Provincial logic (coverage-first)
        # Rule 2 you gave:
        # - Greek civic bronzes / Greek regions -> roman_provincial without emperor names
        #   IF in imperial-era geography bucket
        if primary_period is None:
            has_prov_geo = any(r in text for r in self.ROMAN_PROVINCIAL_REGIONS)
            has_rpc = "roman_provincial" in detected_periods  # from patterns like RPC/billon/bostra
            has_imperial = "roman_imperial" in detected_periods
            has_greek = "greek" in detected_periods

            if (has_prov_geo or has_rpc) and (has_imperial or has_greek):
                primary_period = "roman_provincial"
                result.confidence = 0.95
                result.notes.append("Provincial Logic: imperial-era region (emperor not required)")

        # G. Modern override (Hanover and later)
        # If anything hits modern signals, we treat it as modern unless it's clearly ancient.
        if "modern" in detected_periods:
            # If it ALSO looks ancient (e.g., RIC + sestertius), keep ancient and flag.
            if any(p in detected_periods for p in {"roman_imperial", "roman_republican", "greek", "byzantine"}):
                result.warnings.append("Modern signal mixed with ancient signals")
            else:
                primary_period = "modern"
                result.confidence = max(result.confidence, 0.95)
                result.notes.append("Modern override: Hanover+ / 18th–19th c. signal")
        # B2. Detect Celtic Tribes
        if primary_period is None and any(re.search(rf'\b{t}\b', text) for t in self.CELTIC_TRIBES):
            primary_period = "celtic"
            result.confidence = 0.90
            result.notes.append("Celtic tribal authority detected")
        # H. Resolve simple detection if still None
        if primary_period is None:
            if len(detected_periods) == 1:
                primary_period = next(iter(detected_periods))
            elif len(detected_periods) > 1:
                # prefer modern already handled above; then byzantine; then roman_republican; then roman_imperial; then greek
                if "byzantine" in detected_periods:
                    primary_period = "byzantine"
                elif "roman_republican" in detected_periods:
                    primary_period = "roman_republican"
                elif "roman_imperial" in detected_periods:
                    primary_period = "roman_imperial"
                elif "greek" in detected_periods:
                    primary_period = "greek"
                else:
                    primary_period = next(iter(detected_periods))
                result.warnings.append("Multiple period signals")
        # H2. Greek City States
        if primary_period is None and any(city in text for city in self.GREEK_POLEIS):
            primary_period = "greek"
            result.confidence = 0.90
            result.notes.append("Greek civic polis detected")
        # Z. Unrecognized/Exotic regions (catch-all)
        EXOTIC_SIGNALS = ["kabul", "shahi", "kushano", "hephthalite", "local issues"]

        if primary_period is None:
            if any(sig in text for sig in EXOTIC_SIGNALS):
                primary_period = "central_asian"  # or create new "central_asian" category
                result.confidence = 0.60  # Low but non-zero
                result.needs_review = True
                result.notes.append("Exotic region - requires manual review")
        # I. Denomination-only inference (when no period signals)
        if primary_period is None:
            if result.denomination in {
                "obol", "diobol", "triobol", "hemiobol",
                "drachm", "hemidrachm", "tetradrachm",
                "stater", "distater"
            }:
                primary_period = "greek"
                result.confidence = 0.80
                result.notes.append("Inferred Greek from denomination")

            elif result.denomination in {"solidus", "histamenon", "tetarteron", "hyperpyron"}:
                primary_period = "byzantine"
                result.confidence = 0.85
                result.notes.append("Inferred Byzantine from denomination")

            elif result.denomination in {"denier", "penny", "sceatta", "styca", "grosso"}:
                primary_period = "medieval"
                result.confidence = 0.80
                result.notes.append("Inferred Medieval from denomination")

        # J. Constraint-based cleanup (lightweight)
        if result.denomination and result.denomination in self.DENOM_CONSTRAINTS and primary_period:
            allowed = self.DENOM_CONSTRAINTS[result.denomination]
            if primary_period not in allowed:
                # For coverage-first, we auto-correct only if the allowed set is single or the conflict is obvious.
                result.warnings.append(f"Conflict: '{result.denomination}' vs '{primary_period}'")
                if len(allowed) == 1:
                    primary_period = next(iter(allowed))
                    result.confidence = max(result.confidence, 0.92)
                    result.notes.append(f"Auto-corrected to {primary_period} via denomination constraint")

        # K. Finalize
        result.period = primary_period

        if result.confidence == 0.0:
            if primary_period:
                result.confidence = 0.9
            else:
                result.confidence = 0.3  # Non-zero but flagged
                result.needs_review = True
                result.warnings.append("No period detected - requires manual classification")

        if result.period is None or result.confidence < 0.6 or result.warnings:
            result.needs_review = True

        return result


def parse_auction_label(title: str, description: str = "") -> ParsedLabel:
    return LabelParser().parse(title, description)
