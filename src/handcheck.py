"""Hand check of the Claude complaint labels (Section 4.6 of the Phase 2 plan).

1. make_sheet() writes data/handcheck_100.csv: 100 negative boxes with ONE EMPTY COLUMN PER CATEGORY.
   Claude's labels are not in the sheet, so my labels are not anchored on Claude's (a blind check).
2. I open it in Excel and put a 1 in every category the complaint really contains (blank = 0).
3. score_sheet() compares my labels with Claude's: precision and recall per category, plus a table of
   every disagreement so I can look at why.

How the 100 are picked: up to 4 reviews for each category Claude used (rarest first), so rare categories
get checked at all, then random reviews (half low-score, half high-score) to fill up to 100. Picking on
Claude's labels inflates recall, so recall is computed on the random part only; precision uses all 100.
"""

import numpy as np
import pandas as pd

from src.batch_labels import COMPLAINTS, label_flags

SHEET_COLS = ["review_id", "score", "complaint_text"]   # "picked_for" would reveal Claude's labels: kept in a key file


def make_sheet(records, df, clean, path, n=100, per_category=4, seed=4360):
    flags = label_flags(records, df, "neg")
    text = clean.set_index("review_id").neg_clean
    has = df.loc[df.has_neg == 1].set_index("review_id")
    rng = np.random.default_rng(seed)
    picked = {}
    for cat in flags.sum().sort_values().index:                    # rarest category first
        pool = [i for i in flags.index[flags[cat] == 1] if i not in picked and i in has.index]
        for i in rng.permutation(pool)[:per_category]:
            picked[int(i)] = cat
    rest = has[~has.index.isin(list(picked))]
    need = max(0, n - len(picked))
    for low, k in [(1, need // 2), (0, need - need // 2)]:
        ids = rest.index[rest.Low_Score == low]
        if k == 0 or len(ids) == 0:
            continue
        for i in rng.choice(ids, size=min(k, len(ids)), replace=False):
            picked[int(i)] = "random"

    sheet = pd.DataFrame({"review_id": list(picked), "picked_for": list(picked.values())})
    sheet["score"] = sheet.review_id.map(has.Reviewer_Score)
    sheet["complaint_text"] = sheet.review_id.map(text)
    sheet = sheet.sample(frac=1, random_state=seed)               # shuffle so the order gives nothing away
    sheet[["review_id", "picked_for"]].to_csv(key_path(path), index=False)
    sheet = sheet[SHEET_COLS].copy()
    for cat in COMPLAINTS:
        sheet[cat] = ""
    sheet["notes"] = ""
    sheet.to_csv(path, index=False, encoding="utf-8-sig")         # utf-8-sig so Excel opens it cleanly
    return sheet


def key_path(path):
    """data/handcheck_100.csv -> data/handcheck_100_key.csv (why each review was picked; don't open while checking)."""
    path = str(path)
    return path[:-4] + "_key.csv"


def read_sheet(path):
    sheet = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    for cat in COMPLAINTS:
        sheet[cat] = pd.to_numeric(sheet[cat].replace("", 0), errors="coerce").fillna(0).astype(int).clip(0, 1)
    return sheet


def score_sheet(path, records, df):
    """Returns (per-category table, overall row, disagreements)."""
    sheet = read_sheet(path)
    me = sheet.set_index("review_id")[list(COMPLAINTS)]
    claude = label_flags(records, df, "neg").loc[me.index]
    key = pd.read_csv(key_path(path)).set_index("review_id").picked_for
    random_part = key.loc[me.index] == "random"

    rows = []
    for cat in COMPLAINTS:
        both = int(((me[cat] == 1) & (claude[cat] == 1)).sum())
        rows.append({
            "category": cat,
            "Claude tagged": int(claude[cat].sum()),
            "I tagged": int(me[cat].sum()),
            "both": both,
            "precision": both / claude[cat].sum() if claude[cat].sum() else np.nan,
            "recall (random part)": (((me[cat] == 1) & (claude[cat] == 1) & random_part).sum()
                                     / ((me[cat] == 1) & random_part).sum()) if ((me[cat] == 1) & random_part).sum() else np.nan,
        })
    table = pd.DataFrame(rows)
    tp = ((me == 1) & (claude == 1)).to_numpy().sum()
    overall = {
        "precision (all labels)": tp / claude.to_numpy().sum(),
        "recall (random part)": ((me == 1) & (claude == 1))[random_part].to_numpy().sum() / me[random_part].to_numpy().sum(),
        "reviews where every category matches": (me == claude).all(axis=1).mean(),
    }

    evid = {(r["review_id"], lab["category"]): lab["evidence"] for r in records
            if r["box"] == "neg" and not r["repeat"] for lab in r["labels"]}
    dis = []
    for rid in me.index:
        for cat in COMPLAINTS:
            if me.at[rid, cat] != claude.at[rid, cat]:
                dis.append({"review_id": rid, "category": cat,
                            "who": "Claude only" if claude.at[rid, cat] else "me only",
                            "Claude's evidence": evid.get((rid, cat), ""),
                            "complaint_text": sheet.set_index("review_id").at[rid, "complaint_text"]})
    return table, overall, pd.DataFrame(dis)
