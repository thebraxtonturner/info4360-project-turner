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
| `data/reviews_clean_20k.csv` | 19,998 | NER-cleaned text of both boxes (what is sent to Claude) |
| `data/labels_final.jsonl` | one line per box | Claude labels from the Batch API (after the final run) |
| `data/pilot_labels.jsonl` | 40 reviews | labels made in Claude Code to test the code before the paid run |

## Approach

Path A, focused on explaining the score rather than predicting it (revised after Phase 1 feedback):

1. Clean the text. Treat placeholders like "No Negative" as empty, remove hotel and staff names with spaCy NER (word by word, so complaint words that spaCy mistakes for names are kept), and pass `clean_review` to the vectorizers as `preprocessor=`.
2. Label complaints and praise with Claude. Every review in the 19,998-review sample (about 14 per hotel) is sent through the Batch API (Claude Haiku 5.5, JSON schema): 14 complaint categories for the negative box and 13 praise categories for the positive box, each with an evidence quote. A 100-review test batch goes first. 100 labels are checked by hand, blind.
3. Regress the score on the flags with city, trip type, nationality and length of stay as controls. Coefficient x frequency gives the ranked fix list. Check that the ranking holds for business vs. leisure travelers and across cities.
4. Compare with a TF-IDF model on the same split, using out-of-sample R² and cross-validation, to see how much of what the words explain the named categories capture.

See Section 4 of the Phase 2 notebook for the full plan and the reasoning.

## Progress

- Phase 1: proposal, data sample, README.
- Phase 2 (this check-in): `notebooks/phase2_exploration.ipynb` covers data exploration, text mining (NER noise removal, keyword frequency, word clouds, TF-IDF, a keyword-based first look at complaint categories, VADER sentiment checked against the true score, LDA topics), Claude sample prompts on 12 reviews, and the Phase 3 plan. A data validation step (Section 1.7, `src/validation.py`) runs 15 pass/fail checks on row counts, value ranges, placeholders, the sample vs. the full data, the Claude labels and the API key. All pass.
- Phase 3: `notebooks/phase3_analysis.ipynb` applies the Phase 2 feedback (top words without negations, lemmas, counts vs. TF-IDF), labels all reviews with Claude through the Batch API, checks the labels (validation, evidence quotes, a repeat-request consistency test, keyword agreement, a blind hand check), compares models M0 to M3 with 5-fold cross-validation, and builds the fix list, with business vs. leisure, city and robustness checks.

## Repo Structure

```
data/                        samples (full dataset kept outside the repo) and cached Claude labels
notebooks/phase2_exploration.ipynb   Phase 2: exploration, text mining, Phase 3 plan
notebooks/phase3_analysis.ipynb      Phase 3: labels, label checks, models, fix list
scripts/make_sample.py       builds the samples from the full dataset
scripts/run_batch.py         the only code that calls the API: test batch, full batch, fetch results
scripts/make_pilot_labels.py 40 pilot labels made in Claude Code to test the code (not reported)
src/data.py                  load the sample, handle "No Negative" placeholders
src/preprocessing.py         clean_review (vectorizer preprocessor) and NER noise removal
src/sentiment.py             VADER scores and why a score is neutral
src/complaints.py            complaint categories, keyword lexicon, cached Claude labeler
src/validation.py            run_validation(): pass/fail data checks
src/batch_labels.py          Phase 3 prompts, categories, batch requests, label checks
src/handcheck.py             blind hand-check sheet and precision/recall scoring
src/models.py                controls, M0-M3 comparison, fix list, segments, robustness
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

## Final labeling run

The notebook never calls the API. All API calls go through `scripts/run_batch.py`, run by hand from the repo root, with the API key in `.env`:

```bash
python scripts/run_batch.py prepare                 # NER-clean the text once (no API; already done)
python scripts/run_batch.py dry-run --scope test    # builds requests + cost estimate, sends nothing
python scripts/run_batch.py submit  --scope test    # 100 reviews (184 requests); type yes to send
python scripts/run_batch.py status  --name test     # repeat until it says ended
python scripts/run_batch.py fetch   --name test     # saves results to data/labels_final.jsonl
#   -> open the notebook, check Section 3 on the test labels, then:
python scripts/run_batch.py dry-run --scope full    # everything not labeled yet (~32,100 requests, ~$2.40)
python scripts/run_batch.py submit  --scope full
python scripts/run_batch.py status  --name full
python scripts/run_batch.py fetch   --name full
```

A request with a saved result is never built again, and a batch name can only be submitted once. If some requests fail, `submit --scope full --name full-retry` sends only those.

## AI Assistance

I used Claude Code (an AI coding assistant) to help write and debug code for this project.

## License

MIT, see `LICENSE`.
