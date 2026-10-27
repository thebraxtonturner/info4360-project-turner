"""Complaint categories: a keyword lexicon for exploration, and Claude labeling for Phase 3.

The keyword lexicon is a rough first pass (it counts "clean" in "not clean" and "hot" in
"no hot water"). Claude labels are the real measurement; the lexicon is used to explore
the data before spending anything on the API, and later as a sanity check on Claude.
"""

import json
import os
import re
import time
from pathlib import Path

import pandas as pd

from src.preprocessing import clean_review

REPO_ROOT = Path(__file__).resolve().parents[1]

# Category -> definition given to Claude (also the column names used in the models)
CATEGORIES = {
    "cleanliness": "dirt, stains, dust, hair, bad smells, mould, or poor housekeeping",
    "noise": "noise from the street, other guests, thin walls, construction, or the building",
    "staff": "rude, unhelpful, or slow staff, or poor service from reception or other staff",
    "room_size": "the room or bathroom is small, cramped, or has no space for luggage",
    "breakfast": "breakfast, restaurant, or other food and drink served at the hotel",
    "facilities": "hotel facilities missing, closed, or extra cost: bar, pool, gym, spa, minibar, parking, kettle",
    "value": "price too high for what you get, unexpected charges or extra fees, or not up to the hotel's star rating",
    "maintenance": "broken, worn, or outdated things: shower, plumbing, lift, furniture, decor",
    "temperature": "room too hot or too cold, air conditioning or heating not working",
    "wifi": "Wi-Fi or internet is slow, unreliable, or costs extra",
    "location": "the location or neighbourhood is inconvenient, unsafe, or far from things",
    "checkin": "check-in or check-out problems, or the booking was not honored: waiting, room not ready, "
               "a different room or bed type than booked",
    "bed_comfort": "uncomfortable bed, mattress, or pillows, or poor sleep because of the bed",
    "other": "a real complaint that fits none of the categories above",
}

KEYWORDS = {
    "cleanliness": ["dirty", "clean", "cleaning", "stain", "stains", "dust", "dusty", "hair", "smell",
                    "smelly", "smelled", "mould", "mold", "filthy", "hygiene"],
    "noise": ["noise", "noisy", "loud", "thin walls", "hear", "heard", "traffic", "construction"],
    "staff": ["staff", "rude", "reception", "receptionist", "unfriendly", "unhelpful", "service",
              "manager", "attitude"],
    "room_size": ["small", "tiny", "cramped", "size", "space"],
    "breakfast": ["breakfast", "food", "coffee", "buffet", "restaurant"],
    "facilities": ["bar", "pool", "gym", "spa", "minibar", "mini bar", "parking", "kettle", "sauna"],
    "value": ["expensive", "price", "overpriced", "value", "money", "charge", "charged", "pay", "paid", "cost"],
    "maintenance": ["broken", "old", "dated", "repair", "worn", "leak", "leaking", "lift", "elevator",
                    "tired", "renovation", "shower"],
    "temperature": ["air conditioning", "aircon", "air con", "hot", "cold", "heating", "heater",
                    "temperature", "warm"],
    "wifi": ["wifi", "wi fi", "internet"],
    "location": ["location", "area", "far", "distance", "neighbourhood", "neighborhood"],
    "checkin": ["check in", "checkin", "check out", "checkout", "waiting", "queue", "booked", "booking"],
    "bed_comfort": ["bed", "beds", "mattress", "pillow", "pillows", "uncomfortable"],
}


def keyword_flags(texts):
    """0/1 column per category: 1 if any of its keywords appears as a whole word."""
    cleaned = pd.Series(texts).map(clean_review)
    flags = {}
    for cat, words in KEYWORDS.items():
        pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
        flags[cat] = cleaned.str.contains(pattern, regex=True).astype(int)
    return pd.DataFrame(flags, index=getattr(texts, "index", None))


# ---------------------------------------------------------------- Claude labeling

MODEL = "claude-haiku-5-5"   # small and cheap; tagging is a simple, high-volume task
EFFORT = "low"
PRICES = {"claude-haiku-5-5": (0.10, 0.50)}   # USD per 1M tokens (input, output)

label_schema = {
    "type": "object",
    "properties": {
        "complaints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "evidence": {"type": "string"},   # exact words from the review
                },
                "required": ["category", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["complaints"],
    "additionalProperties": False,
}

prompt_label = '''You are labeling hotel guest reviews for a hotel group's guest experience manager.
Below is what one guest wrote in the "What did you dislike?" box. Tag every complaint category that applies.

Categories:
{categories}

Rules:
- Tag only complaints the guest actually states. Do not infer a complaint from praise or from the hotel type.
- A review can have zero, one, or several categories. Statements like "nothing" or "all was fine" have zero.
- If a sentence is mild ("breakfast could have more choice"), it is still a complaint; tag it.
- For each category, quote the exact words from the review that show it ("evidence").

<review>
{review}
</review>'''


def get_client():
    """Anthropic client with the key read from .env in the repo root (never printed)."""
    import anthropic
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("No ANTHROPIC_API_KEY. Put ANTHROPIC_API_KEY=sk-ant-... in a .env file in the repo root.")
    return anthropic.Anthropic()


def label_review(client, text, model=MODEL):
    """Label one negative review. Returns (list of {category, evidence}, cost in USD)."""
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt_label.format(
            categories="\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items()), review=text)}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": label_schema}},
    )
    price_in, price_out = PRICES[model]
    cost = response.usage.input_tokens / 1e6 * price_in + response.usage.output_tokens / 1e6 * price_out
    if response.stop_reason != "end_turn":
        print("Unexpected stop:", response.stop_reason)
        return [], cost
    answer = next(b.text for b in response.content if b.type == "text")
    return json.loads(answer)["complaints"], cost


def label_with_cache(df, cache_path, text_col="neg_text", id_col="review_id"):
    """Label each row once; results are appended to a JSONL cache so re-runs cost nothing.

    Saved labels are reused only if they were made with the current categories and definitions.
    """
    import anthropic
    category_list = [[name, meaning] for name, meaning in sorted(CATEGORIES.items())]
    done = {}
    if Path(cache_path).exists():
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("category_list") == category_list:
                done[r[id_col]] = r
    todo = df[~df[id_col].isin(done)]
    if len(todo):
        client = get_client()
        for row in todo.itertuples():
            for attempt in range(4):
                try:
                    complaints, cost = label_review(client, getattr(row, text_col))
                    break
                except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
                    time.sleep(10 * (attempt + 1))
            else:
                print("gave up on", getattr(row, id_col))
                continue
            rec = {id_col: getattr(row, id_col), "complaints": complaints, "cost_usd": cost,
                   "category_list": category_list}
            done[rec[id_col]] = rec
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
    print(f"{len(df) - len(todo)} labels reused from cache, {len(todo)} new API calls")
    return [done.get(i) for i in df[id_col]]
