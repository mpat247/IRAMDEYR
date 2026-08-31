# Tier Lists

TIER_1 = [
    "CATAG3",
    "IRSEX",
    "IRMARIT_ABS",
    "IRPINC3_ABS",
    "SVYRDUDANY",
    "WRKDRGHLP_ABS",
]

TIER_2 = TIER_1 + [
    "IRSUICTHNK",
    "KSSLR6MAX_ABS",
    "RCVYMHPRB",
    "IRIMPGOUT_ABS",
]

TIER_3 = TIER_2 + [
    "ILLEMFLAG",
    "WHODASDASC_ABS",
    "MJCMOTHYR",
    "MJSMKYR",
    "ILLEMMON",
    "COCLALCUSE_ABS",
    "DIFOBTCRK",
    "MHNTENFCV",
    "MHNTINSCV",
]

TIERS = {
    "Tier_1_Basic": TIER_1,
    "Tier_2_Clinical": TIER_2,
    "Tier_3_Personalized": TIER_3,
}


# Export
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "tiers"
OUTPUT_DIR.mkdir(exist_ok=True)

for tier_name, features in TIERS.items():
    out_path = OUTPUT_DIR / f"{tier_name.lower()}_features.csv"
    pd.DataFrame({"feature": features}).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")