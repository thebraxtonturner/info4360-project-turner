"""Load the hotel review sample and add the text fields used throughout the project."""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO_ROOT / "data" / "hotel_reviews_20k.csv"

# What guests type when they have nothing to say. Booking.com fills an empty box with
# "No Negative" / "No Positive"; guests also write "nothing", "n a", "none" and so on.
# These are not complaints (or praise), so they become empty text.
NO_CONTENT = {
    "no negative", "no positive", "nothing", "n a", "na", "none", "nil", "nothing really",
    "nothing at all", "no complaints", "all good", "nothing to complain about",
    "nothing to report", "nothing to say", "non", "no", "",
}


def is_no_content(text):
    """True if the review box holds a placeholder rather than real content."""
    return str(text).strip().lower() in NO_CONTENT


def word_count(text):
    return len(str(text).split())


def load_reviews(path=DEFAULT_PATH):
    """Read the sample and add cleaned text columns, flags, and word counts."""
    df = pd.read_csv(path)
    df["review_id"] = range(len(df))   # stable id (row number in the sample file) for caching labels
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], format="%m/%d/%Y")
    df["Reviewer_Nationality"] = df["Reviewer_Nationality"].str.strip()

    for side, col in [("neg", "Negative_Review"), ("pos", "Positive_Review")]:
        empty = df[col].map(is_no_content)
        df[f"{side}_text"] = df[col].str.strip().where(~empty, "")
        df[f"has_{side}"] = (~empty).astype(int)
        df[f"{side}_words"] = df[f"{side}_text"].map(word_count)
    return df
