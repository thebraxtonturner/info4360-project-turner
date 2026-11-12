"""Phase 3 labeling: complaint and praise categories for every review, sent through the Batch API.

The API is only called from scripts/run_batch.py, which I run by hand. The notebook never calls
the API; it only reads data/labels_final.jsonl. That way re-running the notebook can't resend
anything, and the batch script skips every request that already has a saved result.

Changes from the Phase 2 prompt (Section 3 of the Phase 2 notebook):
- The positive box is labeled too, with its own praise categories.
- Rules for the overlapping definitions found in the 12-review test ("2 singles pushed together",
  "air con broken in the bar") and for negation.
"""

import json
from pathlib import Path

import pandas as pd

from src.complaints import CATEGORIES
from src.preprocessing import clean_review

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
CLEAN_PATH = DATA / "reviews_clean_20k.csv"          # NER-cleaned text (what is sent to the API)
LABELS_PATH = DATA / "labels_final.jsonl"            # one line per finished request (API results)
PILOT_PATH = DATA / "pilot_labels.jsonl"             # labels made in Claude Code, same format, for testing code
BATCH_DIR = DATA / "batch"                           # request files and batch ids (not in git)

MODEL = "claude-haiku-5-5"
EFFORT = "low"
MAX_TOKENS = 4096                      # 1024 cut off 8 of 32,106 long reviews in the full run
PRICE_IN, PRICE_OUT = 0.10, 0.50       # USD per 1M tokens, standard price
BATCH_DISCOUNT = 0.5                   # the Batch API charges half

COMPLAINTS = CATEGORIES                # same 14 complaint categories as Phase 2

PRAISE = {
    "location": "good location: close to transport, sights, shops or restaurants, or a nice or safe area",
    "staff": "friendly, helpful, or attentive staff or service",
    "cleanliness": "clean room, bathroom, or hotel",
    "room": "nice, spacious, modern, or well-equipped room or bathroom, or a good view",
    "bed_comfort": "comfortable bed, mattress, or pillows, or slept well",
    "breakfast": "good breakfast, restaurant, or other food and drink at the hotel",
    "facilities": "good hotel facilities: bar, pool, gym, spa, lounge, parking, rooftop, free extras",
    "value": "good value for money or a reasonable price",
    "quiet": "quiet room or hotel",
    "design": "nice decor, design, building, or atmosphere",
    "checkin": "smooth or early check-in or check-out, an upgrade, or the booking handled well",
    "wifi": "good Wi-Fi or internet",
    "other": "real praise that fits none of the categories above",
}

BOXES = {
    "neg": {"categories": COMPLAINTS, "column": "neg_clean",
            "question": '"What did you dislike?"', "kind": "complaint"},
    "pos": {"categories": PRAISE, "column": "pos_clean",
            "question": '"What did you like?"', "kind": "praise"},
}

RULES = {
    "neg": """- Tag only complaints the guest actually states. Do not infer a complaint from praise or from the hotel type.
- A review can have zero, one, or several categories. Statements like "nothing" or "all was fine" have zero.
- Negation matters: "not clean" is a cleanliness complaint, "nothing dirty" is not.
- Mild wishes ("breakfast could have more choice") are still complaints; tag them.
- A price complaint about one thing ("breakfast is overpriced") gets both that thing's category and value.
- Bed type: "two singles pushed together" is checkin only if the guest says they booked or asked for a different bed; otherwise it is bed_comfort.
- temperature is only about the guest's own room. Broken air conditioning in the bar or lobby is facilities.
- A noisy machine in the room (fridge, air conditioning) is noise, plus temperature or maintenance if it also does not work.""",
    "pos": """- Tag only praise the guest actually states. If the guest wrote a complaint in this box, do not tag it as praise.
- A review can have zero, one, or several categories. Statements like "nothing" or "n a" have zero.
- General praise with no specific aspect ("everything", "great hotel") is other.""",
}

PROMPT = '''You are labeling hotel guest reviews for a hotel group's guest experience manager.
Below is what one guest wrote in the {question} box. Tag every {kind} category that applies.

Categories:
{categories}

Rules:
{rules}
- For each category, quote the exact words from the review that show it ("evidence"). Copy them, don't paraphrase.

<review>
{review}
</review>'''


def make_schema(categories):
    return {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": list(categories)},
                        "evidence": {"type": "string"},
                    },
                    "required": ["category", "evidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["labels"],
        "additionalProperties": False,
    }


def make_prompt(box, text):
    spec = BOXES[box]
    return PROMPT.format(question=spec["question"], kind=spec["kind"], rules=RULES[box], review=text,
                         categories="\n".join(f"- {k}: {v}" for k, v in spec["categories"].items()))


# ---------------------------------------------------------------- ids

def custom_id(box, review_id, repeat=False):
    """'neg-123', or 'neg-123-r2' for the second copy used in the consistency check."""
    return f"{box}-{review_id}" + ("-r2" if repeat else "")


def parse_id(cid):
    parts = cid.split("-")
    return parts[0], int(parts[1]), len(parts) == 3


# ---------------------------------------------------------------- cleaned text

def build_clean_text(df, path=CLEAN_PATH):
    """NER-clean both boxes for all reviews once and save them (about 2 minutes), so the batch script,
    the pilot and the notebook all use exactly the same text."""
    from src.preprocessing import load_ner, denoise, common_lowercase_words
    nlp = load_ner()
    common = common_lowercase_words(list(df.neg_text) + list(df.pos_text))
    out = pd.DataFrame({"review_id": df.review_id})
    for side in ["neg", "pos"]:
        cleaned, _, _ = denoise(df, f"{side}_text", nlp, common)
        out[f"{side}_clean"] = cleaned
        out.loc[df[f"has_{side}"].values == 0, f"{side}_clean"] = ""
    out.to_csv(path, index=False)
    return out


def load_clean_text(path=CLEAN_PATH):
    return pd.read_csv(path, keep_default_na=False)


# ---------------------------------------------------------------- requests

def make_request(box, review_id, text, repeat=False):
    return {
        "custom_id": custom_id(box, review_id, repeat),
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": make_prompt(box, text)}],
            "output_config": {"effort": EFFORT,
                              "format": {"type": "json_schema", "schema": make_schema(BOXES[box]["categories"])}},
        },
    }


def test_sample_ids(df, n=100, seed=4360):
    """Review ids for the small test batch: half low-score, half high-score, all with a real complaint."""
    has = df[df.has_neg == 1]
    return sorted(has.groupby("Low_Score").sample(n // 2, random_state=seed).review_id)


def consistency_ids(df, n=50, seed=43):
    """Reviews whose negative box is sent twice, to measure how often Claude changes its answer."""
    return sorted(df[df.has_neg == 1].sample(n, random_state=seed).review_id)


def build_requests(df, clean, review_ids=None, repeats=(), done=frozenset()):
    """Requests for every non-empty box of the given reviews, minus the custom_ids already in `done`."""
    text = clean.set_index("review_id")
    ids = df.review_id if review_ids is None else pd.Series(review_ids)
    reqs = []
    for rid in ids:
        for box in BOXES:
            t = text.at[rid, f"{box}_clean"]
            if t.strip():
                reqs.append(make_request(box, rid, t))
    for rid in repeats:
        reqs.append(make_request("neg", rid, text.at[rid, "neg_clean"], repeat=True))
    return [r for r in reqs if r["custom_id"] not in done]


def estimate_cost(requests, out_tokens_per_request=200):
    """Rough cost before sending: about 4 characters per token for input (no API call needed).
    The Phase 2 test cost ($0.00015 per review at full price) implies about 200 output tokens per request,
    because Haiku 5.5 thinks before answering and thinking counts as output."""
    chars = sum(len(r["params"]["messages"][0]["content"]) for r in requests)
    tokens_in = chars / 4
    tokens_out = out_tokens_per_request * len(requests)
    cost = (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1e6 * BATCH_DISCOUNT
    return {"requests": len(requests), "input_tokens_est": int(tokens_in),
            "output_tokens_est": tokens_out, "cost_usd_est": round(cost, 2)}


# ---------------------------------------------------------------- results

def done_ids(path=None):
    """custom_ids that already have a successful result (these are never sent again)."""
    return {r["custom_id"] for r in read_labels(path) if r["status"] == "succeeded"}


def read_labels(path=None):
    path = Path(path or LABELS_PATH)
    if not path.exists():
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def result_record(cid, status, labels=None, usage=None, stop_reason=None, error=None):
    box, rid, repeat = parse_id(cid)
    rec = {"custom_id": cid, "review_id": rid, "box": box, "repeat": repeat, "status": status,
           "labels": labels or [], "stop_reason": stop_reason, "error": error}
    if usage:
        rec["input_tokens"], rec["output_tokens"] = usage
        rec["cost_usd"] = (usage[0] * PRICE_IN + usage[1] * PRICE_OUT) / 1e6 * BATCH_DISCOUNT
    return rec


def label_flags(records, df, box, categories=None):
    """0/1 table, one row per review in df and one column per category, from the label records.

    Reviews with an empty box get all zeros (they were never sent). Repeat copies are ignored.
    """
    categories = list(categories or BOXES[box]["categories"])
    flags = pd.DataFrame(0, index=df.review_id.values, columns=categories)
    for r in records:
        if r["box"] == box and not r["repeat"] and r["status"] == "succeeded" and r["review_id"] in flags.index:
            for lab in r["labels"]:
                flags.at[r["review_id"], lab["category"]] = 1
    flags.index.name = "review_id"
    return flags


def norm(s):
    return " ".join(clean_review(s).split())


def evidence_table(records, clean):
    """One row per label: is the quote really in the review? (An empty quote counts as not found.)"""
    text = clean.set_index("review_id")
    rows = []
    for r in records:
        if r["status"] != "succeeded" or r["repeat"]:
            continue
        review = norm(text.at[r["review_id"], f"{r['box']}_clean"])
        for lab in r["labels"]:
            q = norm(lab["evidence"])
            rows.append({"review_id": r["review_id"], "box": r["box"], "category": lab["category"],
                         "evidence": lab["evidence"], "found": bool(q) and q in review})
    return pd.DataFrame(rows, columns=["review_id", "box", "category", "evidence", "found"])


def consistency_table(records):
    """For reviews sent twice: did both runs give the same set of categories?"""
    by_id = {}
    for r in records:
        if r["box"] == "neg" and r["status"] == "succeeded":
            by_id.setdefault(r["review_id"], {})["second" if r["repeat"] else "first"] = \
                sorted({lab["category"] for lab in r["labels"]})
    rows = [{"review_id": rid, "first": v["first"], "second": v["second"], "same": v["first"] == v["second"]}
            for rid, v in by_id.items() if "first" in v and "second" in v]
    return pd.DataFrame(rows, columns=["review_id", "first", "second", "same"])


def validate_labels(records, df, clean):
    """Pass/fail checks on the label file (same style as run_validation in src/validation.py)."""
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})

    main = [r for r in records if not r["repeat"]]
    ids = [r["custom_id"] for r in records]
    check("No custom_id saved twice", len(ids) == len(set(ids)), f"{len(ids):,} records")
    in_df = set(df.review_id)
    expected = {custom_id(b, rid) for b in BOXES
                for rid, t in zip(clean.review_id, clean[f"{b}_clean"]) if rid in in_df and t.strip()}
    got = {r["custom_id"] for r in main if r["status"] == "succeeded"}
    check("Every non-empty box has a successful label", expected <= got,
          f"{len(expected & got):,} of {len(expected):,} boxes; {len(expected - got):,} missing")
    check("No request sent for an empty box", not (got - expected), f"{len(got - expected)} extra")
    bad = [r["custom_id"] for r in main if r["status"] == "succeeded" and r["stop_reason"] != "end_turn"]
    check("Every response finished normally (end_turn)", not bad, f"{len(bad)} cut off or refused")
    wrong = sum(lab["category"] not in BOXES[r["box"]]["categories"] for r in main for lab in r["labels"])
    check("Every label uses a defined category", wrong == 0, f"{wrong} unknown categories")
    ev = evidence_table(records, clean)
    rate = ev.found.mean() if len(ev) else float("nan")
    check("At least 98% of evidence quotes appear in the review", len(ev) > 0 and rate >= 0.98,
          f"{rate:.1%} of {len(ev):,} quotes")
    return pd.DataFrame(checks)
