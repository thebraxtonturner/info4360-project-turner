"""
Build a small, stratified sample of the 515K Hotel Reviews dataset for the repo.

The full dataset (~238 MB) is too large for GitHub, so it stays outside the repo.
Download it from Kaggle ("515K Hotel Reviews Data in Europe") or the Hugging Face
mirror (Dricz/515k-Hotel-Reviews-In-Europe), then run:

    python scripts/make_sample.py path/to/Hotel_Reviews.csv
"""

import sys
import pandas as pd

LOW_SCORE_CUTOFF = 7.0   # reviews scored below 7 are labeled "low"
SAMPLE_PER_CLASS = 1000  # 1,000 low + 1,000 high = 2,000 rows
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


def add_target(df):
    df["Low_Score"] = (df["Reviewer_Score"] < LOW_SCORE_CUTOFF).astype(int)
    return df


def main(path):
    df = pd.read_csv(path)
    df = add_city(df)
    df = add_trip_type(df)
    df = add_target(df)

    # Balanced sample so both classes are visible in the small file
    sample = (
        df.groupby("Low_Score", group_keys=False)
        .sample(n=SAMPLE_PER_CLASS, random_state=RANDOM_STATE)
    )

    keep = [
        "Hotel_Name", "City", "Review_Date", "Reviewer_Nationality", "Trip_Type",
        "Tags", "Negative_Review", "Positive_Review", "Reviewer_Score", "Low_Score",
    ]
    sample[keep].to_csv("data/hotel_reviews_sample.csv", index=False)
    print(f"Wrote {len(sample)} rows to data/hotel_reviews_sample.csv")


if __name__ == "__main__":
    main(sys.argv[1])
