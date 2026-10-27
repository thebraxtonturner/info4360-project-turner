"""
Build small samples of the 515K Hotel Reviews dataset for the repo.

The full dataset (~238 MB) is too large for GitHub, so it stays outside the repo.
Download it from Kaggle ("515K Hotel Reviews Data in Europe") or the Hugging Face
mirror (Dricz/515k-Hotel-Reviews-In-Europe), then run:

    python scripts/make_sample.py path/to/Hotel_Reviews.csv

This writes two files:
- data/hotel_reviews_sample.csv  balanced 2,000 rows (1,000 low, 1,000 high), from Phase 1
- data/hotel_reviews_20k.csv     20,000 rows stratified by City x Trip_Type, keeping the
                                 real share of low scores (used in Phase 2 and later)
"""

import sys
import pandas as pd

LOW_SCORE_CUTOFF = 7.0   # reviews scored below 7 are labeled "low"
SAMPLE_PER_CLASS = 1000  # 1,000 low + 1,000 high = 2,000 rows
STRATIFIED_N = 20000     # size of the stratified sample
RANDOM_STATE = 42

CITY_BY_COUNTRY = {
    "Kingdom": "London",
    "Spain": "Barcelona",
    "France": "Paris",
    "Netherlands": "Amsterdam",
    "Austria": "Vienna",
    "Italy": "Milan",
}


def add_city(df):
    # The last word of the address is the country, and each country has one city.
    country = df["Hotel_Address"].str.split().str[-1]
    df["City"] = country.map(CITY_BY_COUNTRY)
    return df


def add_trip_type(df):
    # Tags look like "[' Leisure trip ', ' Couple ', ...]"
    df["Trip_Type"] = "Unknown"
    df.loc[df["Tags"].str.contains("Leisure trip"), "Trip_Type"] = "Leisure"
    df.loc[df["Tags"].str.contains("Business trip"), "Trip_Type"] = "Business"
    return df


def add_nights(df):
    # Length of stay is also in Tags, e.g. ' Stayed 3 nights '
    df["Nights"] = df["Tags"].str.extract(r"Stayed (\d+) night", expand=False).astype("Int64")
    return df


def add_target(df):
    df["Low_Score"] = (df["Reviewer_Score"] < LOW_SCORE_CUTOFF).astype(int)
    return df


def main(path):
    df = pd.read_csv(path)
    df = add_city(df)
    df = add_trip_type(df)
    df = add_nights(df)
    df = add_target(df)
    print(f"Full dataset: {len(df):,} rows, {df['Low_Score'].mean():.1%} low scores")

    # Balanced sample so both classes are visible in the small file (Phase 1)
    balanced = (
        df.groupby("Low_Score", group_keys=False)
        .sample(n=SAMPLE_PER_CLASS, random_state=RANDOM_STATE)
    )
    keep = [
        "Hotel_Name", "City", "Review_Date", "Reviewer_Nationality", "Trip_Type",
        "Tags", "Negative_Review", "Positive_Review", "Reviewer_Score", "Low_Score",
    ]
    balanced[keep].to_csv("data/hotel_reviews_sample.csv", index=False)
    print(f"Wrote {len(balanced)} rows to data/hotel_reviews_sample.csv")

    # Stratified sample: each City x Trip_Type cell keeps its share of the full data,
    # and the low-score rate stays close to the real 16.8%
    frac = STRATIFIED_N / len(df)
    stratified = (
        df.groupby(["City", "Trip_Type"], group_keys=False)
        .sample(frac=frac, random_state=RANDOM_STATE)
    )
    keep_20k = keep[:6] + ["Nights", "Average_Score"] + keep[6:]  # Average_Score kept only for the leakage check
    stratified[keep_20k].to_csv("data/hotel_reviews_20k.csv", index=False)
    print(f"Wrote {len(stratified):,} rows to data/hotel_reviews_20k.csv "
          f"({stratified['Low_Score'].mean():.1%} low scores)")


if __name__ == "__main__":
    main(sys.argv[1])
