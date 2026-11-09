"""Rebuild data/annual_real_returns.csv from the raw monthly Shiller file.

The raw file data/raw/shiller_sp500_monthly.csv is data/data.csv of github.com/datasets/s-and-p-500 at commit
87a2284276f79d75041c71c8f6d9f2aa1c965e84 (ODC-PDDL-1.0; data prepared by Robert Shiller).
SHA-256 of the raw file: b797a3c8de4586af837b86a156b05f49be11ffab99e13171dae763b2a11bdf79
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcsize.data import ANNUAL_CSV, RAW_SHILLER, annual_real_returns  # noqa: E402

EXPECTED = "b797a3c8de4586af837b86a156b05f49be11ffab99e13171dae763b2a11bdf79"
digest = hashlib.sha256(RAW_SHILLER.read_bytes()).hexdigest()
if digest != EXPECTED:
    sys.exit(f"raw file changed: {digest}")
a = annual_real_returns()
a.to_csv(ANNUAL_CSV, index=False, float_format="%.10g")
print(f"wrote {ANNUAL_CSV.name}: {len(a)} years, {a.year.min()}-{a.year.max()}")
