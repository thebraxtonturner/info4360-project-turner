"""run_validation(): pass/fail checks on the cleaned sample (row counts, data descriptions, code logic).

Built on the three test habits from INFO 4340: count rows before and after each step, look at the
actual values, and recompute key numbers a second way. Returns a table so a failure is easy to see.
"""

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

from src.complaints import CATEGORIES
from src.data import NO_CONTENT, REPO_ROOT

CITIES = ["London", "Barcelona", "Paris", "Amsterdam", "Vienna", "Milan"]
CITY_BY_COUNTRY = {"Kingdom": "London", "Spain": "Barcelona", "France": "Paris",
                   "Netherlands": "Amsterdam", "Austria": "Vienna", "Italy": "Milan"}


def run_validation(df_raw, df_clean, labels_path, full_path=None):
    """df_raw = the sample file as read, df_clean = load_reviews() output, labels_path = Claude label cache.

    full_path is the full 515K CSV (kept outside the repo); without it the sampling checks are skipped.
    """
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})

    # Row counts
    check("Row count: only the 1 duplicate dropped", len(df_raw) - len(df_clean) == 1,
          f"{len(df_raw):,} >> {len(df_clean):,}")
    n_dup = df_clean.drop(columns="review_id").duplicated().sum()
    check("No duplicate rows or ids", n_dup == 0 and df_clean.review_id.is_unique, f"{n_dup} duplicate rows")

    # Data descriptions
    check("Scores in 2.5-10", df_clean.Reviewer_Score.between(2.5, 10).all(),
          f"{df_clean.Reviewer_Score.min()} to {df_clean.Reviewer_Score.max()}")
    check("Low_Score = score below 7", (df_clean.Low_Score == (df_clean.Reviewer_Score < 7)).all(), "recomputed from the score")
    check("Dates parsed, Aug 2015 to Aug 2017",
          df_clean.Review_Date.notna().all() and df_clean.Review_Date.between("2015-08-01", "2017-08-31").all(),
          f"{df_clean.Review_Date.isna().sum()} missing dates")
    check("City and Trip_Type valid",
          df_clean.City.isin(CITIES).all() and df_clean.Trip_Type.isin(["Leisure", "Business", "Unknown"]).all(),
          f"{df_clean.City.nunique()} cities, {df_clean.Trip_Type.nunique()} trip types")
    check("Nights: 10 missing, all others >= 1",
          df_clean.Nights.isna().sum() == 10 and (df_clean.Nights.dropna() >= 1).all(),
          f"{df_clean.Nights.isna().sum()} missing, max {df_clean.Nights.max()}")

    # Placeholders
    empty = df_clean.has_neg == 0
    check("Placeholder boxes are empty text with 0 words",
          (df_clean.neg_text[empty] == "").all() and (df_clean.neg_words[empty] == 0).all(),
          f"{empty.sum():,} negative placeholders")
    leaks = df_clean.Negative_Review[~empty].str.strip().str.lower().isin(NO_CONTENT).sum()
    check("No placeholder counted as a complaint", leaks == 0, f"{leaks} leaks")

    # Sampling: does the 20K sample match the full data?
    if full_path and Path(full_path).exists():
        full = pd.read_csv(full_path, usecols=["Hotel_Address", "Tags", "Reviewer_Score"])
        full["City"] = full.Hotel_Address.str.split().str[-1].map(CITY_BY_COUNTRY)
        full["Trip_Type"] = "Unknown"
        full.loc[full.Tags.str.contains("Leisure trip"), "Trip_Type"] = "Leisure"
        full.loc[full.Tags.str.contains("Business trip"), "Trip_Type"] = "Business"
        share_full = full.groupby(["City", "Trip_Type"]).size() / len(full)
        share_sample = df_clean.groupby(["City", "Trip_Type"]).size() / len(df_clean)
        gap = (share_full - share_sample).abs().max() * 100
        check("Every city x trip group within 0.1 pp of the full data", gap < 0.1, f"largest gap {gap:.3f} pp")
        low_gap = abs((full.Reviewer_Score < 7).mean() - df_clean.Low_Score.mean()) * 100
        check("Low-score share within 0.5 pp of the full data", low_gap < 0.5, f"{low_gap:.2f} pp")
    else:
        for name in ["Every city x trip group within 0.1 pp of the full data", "Low-score share within 0.5 pp of the full data"]:
            checks.append({"check": name, "result": "SKIP", "detail": "full 515K file not available"})

    # Claude labels
    cache = [json.loads(line) for line in open(labels_path, encoding="utf-8")]
    trial = df_clean[df_clean.review_id.isin([r["review_id"] for r in cache])]
    check("12 cached labels, all on real complaints", len(cache) == 12 and len(trial) == 12 and (trial.has_neg == 1).all(),
          f"{trial.Low_Score.sum()} low, {12 - trial.Low_Score.sum()} high")
    tags = [c for r in cache for c in r["complaints"]]
    check("Every label uses a defined category", all(c["category"] in CATEGORIES for c in tags), f"{len(tags)} labels")
    check("No empty evidence quotes", all(c["evidence"].strip() for c in tags), f"{len(tags)} quotes")

    # API key safety
    tracked = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files"], capture_output=True, text=True).stdout.split()
    # A real key is "sk-" plus a long random part; a set ANTHROPIC_API_KEY= line also counts
    key_pattern = re.compile(r"\bsk-\w{3}-\w{5}-[\w-]{40,}|ANTHROPIC_API_KEY\s*=\s*['\"]?[\w-]{20,}")
    leaked = [f for f in tracked if key_pattern.search((REPO_ROOT / f).read_text(encoding="utf-8", errors="ignore"))]
    check("No API key in any tracked file", bool(tracked) and not leaked, f"{len(tracked)} files scanned")

    return pd.DataFrame(checks)
