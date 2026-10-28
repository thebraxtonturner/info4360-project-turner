# What Drives Low Hotel Review Scores?

INFO 4360 Complex Data Analytics, course project (Braxton Turner, Fall 2026)

## Problem

A hotel group's guest experience manager sees thousands of guest reviews each month across its properties. A drop in review scores pushes hotels down in booking-site search results and costs bookings, but no one can read every review to figure out which problems are actually behind the bad scores.

This project finds which complaints cost the most review points (for example noise, cleanliness, staff attitude, room size or breakfast), and whether that differs by city or trip type (business vs. leisure). The goal is a ranked, actionable fix list for the manager: how many points each complaint costs, times how often it occurs.

## Data

[515K Hotel Reviews Data in Europe](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) (Booking.com, scraped by Jiashen Liu)

| | |
|---|---|
| Rows | 515,738 reviews |
| Hotels | 1,492 hotels in 6 cities (London, Barcelona, Paris, Amsterdam, Vienna, Milan) |
| Time span | Aug 2015 to Aug 2017 |
| Text fields | `Negative_Review`, `Positive_Review` (guests write likes and dislikes separately) |
| Target | `Reviewer_Score` (2.5 to 10); `Low_Score` = 1 if below 7 (16.8% of reviews) |
| Segments | `City`, `Trip_Type` and `Nights` (from `Tags`), `Review_Date`, `Reviewer_Nationality` |

The full file (~238 MB) is too big for GitHub, so the repo has two samples built by `scripts/make_sample.py`:

| File | Rows | Used for |
|---|---|---|
| `data/hotel_reviews_20k.csv` | 19,998 (19,999 in the file, minus 1 exact duplicate), stratified by City x Trip_Type, real share of low scores (17.1%) | Phase 2 exploration and onward |
| `data/hotel_reviews_sample.csv` | 2,000, balanced (1,000 low, 1,000 high) | Phase 1 |
| `data/claude_sample_labels.jsonl` | 12 | cached Claude labels from the Phase 2 sample prompts |

## Approach

Path A, focused on explaining the score rather than predicting it (revised after Phase 1 feedback):

1. Clean the text. Treat placeholders like "No Negative" as empty, remove hotel and staff names with spaCy NER (word by word, so complaint words that spaCy mistakes for names are kept), and pass `clean_review` to the vectorizers as `preprocessor=`.
2. Label complaints and praise with Claude. A sample of about 3,000 reviews, stratified by city and trip type, is sent through the Batch API (Claude Haiku 5.5, JSON schema). Each review can have several complaint categories (negative box) and praise categories (positive box), each with an evidence quote. About 100 labels are checked by hand.
3. Regress the score on the flags with city, trip type, nationality and length of stay as controls. Coefficient x frequency gives the ranked fix list. Check that the ranking holds for business vs. leisure travelers and across cities.
4. Compare with a TF-IDF model on the same split, using out-of-sample R² and cross-validation, to see how much of what the words explain the named categories capture.

See Section 4 of the Phase 2 notebook for the full plan and the reasoning.

## Progress

- Phase 1: proposal, data sample, README.
- Phase 2 (this check-in): `notebooks/phase2_exploration.ipynb` covers data exploration, text mining (NER noise removal, keyword frequency, word clouds, TF-IDF, a keyword-based first look at complaint categories, VADER sentiment checked against the true score, LDA topics), Claude sample prompts on 12 reviews, and the Phase 3 plan. A data validation step (Section 1.7, `src/validation.py`) runs 15 pass/fail checks on row counts, value ranges, placeholders, the sample vs. the full data, the Claude labels and the API key. All pass.
- Phase 3 (next): Claude batch labeling, regression models, fix list, and the comparison with TF-IDF.

## Repo Structure

```
data/                        samples (full dataset kept outside the repo) and cached Claude labels
notebooks/phase2_exploration.ipynb   Phase 2: exploration, text mining, Phase 3 plan
scripts/make_sample.py       builds the samples from the full dataset
src/data.py                  load the sample, handle "No Negative" placeholders
src/preprocessing.py         clean_review (vectorizer preprocessor) and NER noise removal
src/sentiment.py             VADER scores and why a score is neutral
src/complaints.py            complaint categories, keyword lexicon, cached Claude labeler
src/validation.py            run_validation(): pass/fail data checks
```

## How to Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter nbconvert --to notebook --execute notebooks/phase2_exploration.ipynb   # or open it and Run All
```

The notebook runs from the files in `data/`. The Claude labels are cached, so no API key is needed to re-run it. To label new reviews, put `ANTHROPIC_API_KEY=sk-ant-...` in a `.env` file in the repo root; only reviews that aren't in the cache are sent to the API. `.gitignore` keeps it out of the repo.

To rebuild the samples from the full dataset:

```bash
python scripts/make_sample.py path/to/Hotel_Reviews.csv
```

## AI Assistance

I used Claude Code (an AI coding assistant) to help write and debug code for this project.

## License

MIT, see `LICENSE`.
