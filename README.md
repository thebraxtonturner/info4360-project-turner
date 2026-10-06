# What Drives Low Hotel Review Scores?

INFO 4360 Complex Data Analytics, course project (Braxton Turner, Fall 2026)

## Problem

A hotel group's **guest experience manager** sees thousands of guest reviews each month across its properties. A drop in review scores pushes hotels down in booking-site search results and costs bookings, but no one can read every review to figure out *which* problems are actually behind the bad scores.

This project builds a model that predicts whether a review is a **low score (below 7 out of 10)** from the review text. It then looks at which complaints (for example noise, cleanliness, staff attitude, room size or breakfast) push a review toward a low score, and whether those drivers differ by **city** or **trip type** (business vs. leisure). The goal is a ranked, actionable list of fixes for the manager, not just a prediction.

## Data

[515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) (Booking.com, scraped by Jiashen Liu)

| | |
|---|---|
| Rows | 515,738 reviews |
| Hotels | 1,492 hotels in 6 cities (London, Barcelona, Paris, Amsterdam, Vienna, Milan) |
| Time span | Aug 2015 to Aug 2017 |
| Text fields | `Negative_Review`, `Positive_Review` (guests write likes and dislikes separately) |
| Target | `Low_Score` = 1 if `Reviewer_Score` < 7, else 0 (16.8% of reviews are low) |
| Segments | `City`, `Trip_Type` (from `Tags`), `Review_Date`, `Reviewer_Nationality` |

The full file (~238 MB) is too big for GitHub. `data/hotel_reviews_sample.csv` is a balanced sample of 2,000 reviews (1,000 low, 1,000 high) for Phase 1.

## Approach (Path A: Classification)

1. **Label complaint types with Claude.** Send a sample of about 500 negative reviews to the Claude API and tag each with complaint categories (cleanliness, noise, staff, room size, breakfast, value, maintenance, Wi-Fi, location, check-in).
2. **Build features with classical NLP.** Preprocess the text, then build TF-IDF features plus the complaint-category features.
3. **Train and evaluate.** Fit a logistic regression and score it on a held-out test set. Because low scores are the minority class, report recall and F1 for that class, not just accuracy.
4. **Interpret.** Look at which features drive low scores, whether those drivers are actionable, and whether they hold up across city and trip type.

## Repo Structure

```
data/                     sample data (full dataset kept outside the repo)
scripts/make_sample.py    builds the sample from the full dataset
```

## How to Run

```bash
pip install pandas
# download Hotel_Reviews.csv from Kaggle (link above), then:
python scripts/make_sample.py path/to/Hotel_Reviews.csv
```

The Claude API key will go in a `.env` file, which `.gitignore` keeps out of the repo.
