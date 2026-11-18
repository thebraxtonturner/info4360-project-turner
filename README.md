# What Drives Low Hotel Review Scores?

INFO 4360 Complex Data Analytics, course project, Fall 2026

**Team:** Braxton Turner (solo project)

## Problem

A hotel group's guest experience manager sees thousands of guest reviews each month across its properties. A drop in review scores pushes hotels down in booking-site search results and costs bookings, but no one can read every review to figure out which problems are actually behind the bad scores.

This project finds which complaints cost the most review points (for example noise, cleanliness, staff attitude, room size or breakfast), and whether that differs by city or trip type (business vs. leisure). The goal is a ranked, actionable fix list for the manager: how many points each complaint costs, times how often it occurs.

## Main Results

From `notebooks/phase3_analysis.ipynb` (19,998 reviews, every review labeled by Claude):

- **The fix list.** Staff (11.0 points lost per 100 reviews), maintenance (9.3), value (8.4), room size (8.1) and cleanliness (7.8) cost the most: 61% of all review points lost to complaints.
- **Common is not the same as costly.** Breakfast complaints are as common as staff complaints but cost 0.17 points each vs. 1.05.
- **Business vs. leisure.** Same top problems, but business guests punish maintenance, value, Wi-Fi and temperature harder. Vienna is the one city with a different order (value first).
- **Categories vs. words.** 27 named categories capture 95% of what about 2,900 TF-IDF words explain (out-of-sample R² 0.472 vs. 0.494). As a low-score classifier, the categories reach 87.2% test accuracy (F1 0.54) vs. an 82.9% majority baseline, with no overfitting.
- **Label quality.** Against a blind hand check of 100 reviews, Claude's precision is 64% overall and 74 to 90% for the top complaints. Facilities (22%) and check-in (31%) are unreliable and flagged.

## API Costs

| Run | Requests | Cost |
|---|---|---|
| Phase 2 sample prompts | 12 | under $0.01 |
| Phase 3 test batch | 184 | $0.015 |
| Phase 3 full batch + 8 retries | 32,106 + 8 | $2.47 |
| **Total Phase 3** | **32,290** | **$2.48** (35.7M input + 2.8M output tokens, Claude Haiku 5.5, Batch API at half price) |

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
- Phase 2: `notebooks/phase2_exploration.ipynb` covers data exploration, text mining (NER noise removal, keyword frequency, word clouds, TF-IDF, a keyword-based first look at complaint categories, VADER sentiment checked against the true score, LDA topics), Claude sample prompts on 12 reviews, and the Phase 3 plan. A data validation step (Section 1.7, `src/validation.py`) runs 15 pass/fail checks on row counts, value ranges, placeholders, the sample vs. the full data, the Claude labels and the API key. All pass.
- Phase 3 (final): `notebooks/phase3_analysis.ipynb` applies the Phase 2 feedback (top words without negations, lemmas, counts vs. TF-IDF), labels all reviews with Claude through the Batch API, checks the labels (validation, evidence quotes, a repeat-request consistency test, keyword agreement, a blind hand check), compares models M0 to M3 with 5-fold cross-validation, predicts low scores against a majority baseline, and builds the fix list, with business vs. leisure, city and robustness checks. It ends with key findings, ethical and privacy considerations, and an AI usage log.

## Repo Structure

```
data/                        samples (full dataset kept outside the repo) and cached Claude labels
notebooks/phase2_exploration.ipynb   Phase 2: exploration, text mining, Phase 3 plan (+ .html export)
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

1. **Clone the repo and move into it**

   ```bash
   git clone https://github.com/thebraxtonturner/info4360-project-turner.git
   cd info4360-project-turner
   ```

2. **Create a virtual environment and install the packages**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Mac: source .venv/bin/activate
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Add your API key** (only needed to label new reviews; skip it to reproduce the results)

   Create a file named `.env` in the project folder containing one line:

   ```
   ANTHROPIC_API_KEY=your-key-here
   ```

   `.env` is listed in `.gitignore`, so it is never committed.

4. **Get the data**

   Nothing to download to reproduce the results: the 19,998-review sample (`data/hotel_reviews_20k.csv`), the cleaned text and all Claude labels (`data/labels_final.jsonl`) are in the repo. To rebuild the samples from scratch, download `Hotel_Reviews.csv` from [Kaggle](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) (about 238 MB, too big for GitHub) and run:

   ```bash
   python scripts/make_sample.py path/to/Hotel_Reviews.csv
   ```

5. **Run the notebooks**

   Open `notebooks/phase3_analysis.ipynb` (the final report) and choose **Run All**, or run it from the command line:

   ```bash
   cd notebooks
   jupyter nbconvert --to notebook --execute --inplace phase3_analysis.ipynb
   ```

   It takes about 10 minutes on a laptop (the TF-IDF cross-validation is the slow part). The notebook never calls the API: it reads the saved labels, so it costs nothing to run. `phase2_exploration.ipynb` (exploration and text mining) runs the same way. HTML exports of both are in `notebooks/`.

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

I used Claude Code (an AI coding assistant) to help write and debug code for this project, and Claude Haiku 5.5 to label the reviews. The AI usage log at the end of `notebooks/phase3_analysis.ipynb` (Section 10) lists each use, how I verified it, and where Claude got something wrong.

## License

MIT, see `LICENSE`.
