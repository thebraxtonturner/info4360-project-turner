"""Phase 3 models: explain the review score with complaint and praise flags (Section 4 of the Phase 2 plan).

M0  controls only                                    baseline
M1  controls + complaint and praise flags (OLS)      the fix list
M2  controls + TF-IDF on both boxes (ridge)          benchmark: how much can the words explain?
M3  M1 + M2 (ridge)                                  what the categories miss

Prediction comparisons use the same 5 folds for every model (out-of-sample R^2 and RMSE).
The fix list comes from M1 fit on all reviews with standard errors clustered by hotel, because
reviews of the same hotel are not independent.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.preprocessing import clean_review, STOP_WORDS

TOP_NATIONALITIES = 15


# ---------------------------------------------------------------- development stand-in

PRAISE_KEYWORDS = {
    "location": ["location", "located", "close", "walking distance", "metro", "station"],
    "staff": ["staff", "friendly", "helpful", "reception", "receptionist", "service"],
    "cleanliness": ["clean", "spotless", "cleanliness"],
    "room": ["room", "rooms", "spacious", "view", "bathroom", "shower", "balcony"],
    "bed_comfort": ["bed", "beds", "comfy", "comfortable", "pillow", "pillows"],
    "breakfast": ["breakfast", "food", "restaurant", "coffee"],
    "facilities": ["bar", "pool", "gym", "spa", "lounge", "parking", "rooftop"],
    "value": ["value", "price", "cheap", "reasonable"],
    "quiet": ["quiet"],
    "design": ["decor", "design", "modern", "beautiful", "stylish"],
    "checkin": ["check in", "checkin", "early check", "upgrade", "upgraded"],
    "wifi": ["wifi", "wi fi", "internet"],
}


def keyword_stand_in(clean):
    """DEVELOPMENT ONLY: keyword flags for both boxes, used to run the models before the Claude labels
    exist. Indexed by review_id, same shape as label_flags(). Never used for reported results."""
    import re
    from src.complaints import KEYWORDS

    def flags(texts, lexicon):
        t = texts.map(clean_review)
        wb = "\\" + "b"   # regex word boundary, written this way to keep the source plain ASCII
        patterns = {cat: wb + "(?:" + "|".join(map(re.escape, w)) + ")" + wb for cat, w in lexicon.items()}
        return pd.DataFrame({cat: t.str.contains(p).astype(int).to_numpy() for cat, p in patterns.items()})

    neg = flags(clean.neg_clean, KEYWORDS).set_index(clean.review_id)
    pos = flags(clean.pos_clean, PRAISE_KEYWORDS).set_index(clean.review_id)
    return neg, pos


# ---------------------------------------------------------------- design

def controls(df):
    """Dummy-coded controls: city, trip type, nationality (top 15, rest 'Other'), nights, review quarter."""
    top = df.Reviewer_Nationality.value_counts().index[:TOP_NATIONALITIES]
    nights = pd.cut(df.Nights, [0, 1, 2, 3, 6, np.inf], labels=["1", "2", "3", "4-6", "7+"]).astype(str)
    c = pd.DataFrame({
        "City": df.City,
        "Trip": df.Trip_Type,
        "Nat": df.Reviewer_Nationality.where(df.Reviewer_Nationality.isin(top), "Other"),
        "Nights": nights.replace("nan", "unknown"),
        "Quarter": df.Review_Date.dt.to_period("Q").astype(str),
    }, index=df.index)
    # Reference levels: London, Leisure, United Kingdom, 1 night, first quarter
    out = pd.get_dummies(c, drop_first=False, dtype=float)
    ref = ["City_London", "Trip_Leisure", "Nat_United Kingdom", "Nights_1", f"Quarter_{c.Quarter.min()}"]
    return out.drop(columns=[r for r in ref if r in out.columns])


def flag_matrix(neg_flags, pos_flags, index):
    """c_<category> for complaints, p_<category> for praise, aligned to the reviews table."""
    f = pd.concat([neg_flags.add_prefix("c_"), pos_flags.add_prefix("p_")], axis=1)
    f = f.reindex(index=index).fillna(0)
    return f.loc[:, f.sum() > 0]          # a category nobody used can't be estimated


def length_features(df):
    return pd.DataFrame({"log_neg_words": np.log1p(df.neg_words), "log_pos_words": np.log1p(df.pos_words)},
                        index=df.index)


# ---------------------------------------------------------------- OLS with clustered errors

def ols(y, X, groups):
    """OLS with standard errors clustered by hotel."""
    Xc = sm.add_constant(X, has_constant="add")
    codes = pd.factorize(groups)[0]
    return sm.OLS(y.to_numpy(), Xc).fit(cov_type="cluster", cov_kwds={"groups": codes})


def coef_table(fit, prefix):
    names = [n for n in fit.params.index if n.startswith(prefix)]
    ci = fit.conf_int().loc[names]
    return pd.DataFrame({"coef": fit.params[names], "low95": ci[0], "high95": ci[1],
                         "p": fit.pvalues[names]}).rename(index=lambda n: n[len(prefix):])


def fix_list(fit, flags, prefix="c_"):
    """Points lost per 100 reviews = -coefficient x share of reviews with the complaint x 100."""
    t = coef_table(fit, prefix)
    t["share of reviews"] = flags[[prefix + c for c in t.index]].mean().to_numpy()
    t["points lost per 100 reviews"] = -t.coef * t["share of reviews"] * 100
    t["rank"] = t["points lost per 100 reviews"].rank(ascending=False).astype(int)
    return t.sort_values("points lost per 100 reviews", ascending=False)


def praise_list(fit, flags):
    t = coef_table(fit, "p_")
    t["share of reviews"] = flags[["p_" + c for c in t.index]].mean().to_numpy()
    t["points gained per 100 reviews"] = t.coef * t["share of reviews"] * 100
    return t.sort_values("points gained per 100 reviews", ascending=False)


def segment_fix_lists(df, X_controls, flags, by):
    """Fit M1 separately in each segment (e.g. Trip_Type or City); one fix list per segment.
    The segment's own dummies are dropped (they are constant inside the segment)."""
    out = {}
    for seg, idx in df.groupby(by).groups.items():
        Xc = X_controls.loc[idx]
        Xc = Xc.loc[:, Xc.std() > 0]
        # If a segment has no reviews in a control's reference level (e.g. no UK guests among
        # non-native-English reviewers), that control's dummies sum to 1 and duplicate the constant:
        # drop its most common level, which becomes the reference inside this segment.
        for prefix in ["City_", "Trip_", "Nat_", "Nights_", "Quarter_"]:
            cols = [c for c in Xc.columns if c.startswith(prefix)]
            if cols and (Xc[cols].sum(axis=1) == 1).all():
                Xc = Xc.drop(columns=Xc[cols].sum().idxmax())
        f = flags.loc[idx]
        f = f.loc[:, f.sum() >= 10]          # too few cases for a stable coefficient otherwise
        fit = ols(df.loc[idx, "Reviewer_Score"], pd.concat([Xc, f], axis=1), df.loc[idx, "Hotel_Name"])
        out[seg] = fix_list(fit, f)
    return out


def rank_agreement(base, others):
    """Spearman correlation between the overall fix-list ranking and each segment's ranking."""
    rows = []
    for name, t in others.items():
        common = base.index.intersection(t.index)
        rho = spearmanr(base.loc[common, "points lost per 100 reviews"],
                        t.loc[common, "points lost per 100 reviews"]).statistic
        top3 = ", ".join(t.index[:3])
        rows.append({"segment": name, "Spearman vs. overall": rho, "top 3 complaints": top3})
    return pd.DataFrame(rows)


def hotel_fixed_effects(df, X, groups):
    """Within-hotel OLS: subtract each hotel's mean from y and X, so only differences between guests
    of the same hotel are used. City drops out (a hotel is in one city)."""
    keep = X.loc[:, X.groupby(groups).transform("std").fillna(0).sum() > 0]
    yd = df.Reviewer_Score - df.groupby(groups).Reviewer_Score.transform("mean")
    Xd = keep - keep.groupby(groups).transform("mean")
    codes = pd.factorize(groups)[0]
    return sm.OLS(yd.to_numpy(), Xd).fit(cov_type="cluster", cov_kwds={"groups": codes})


def low_score_logit(df, X):
    """Logistic regression on Low_Score (below 7); average marginal effects in percentage points."""
    Xc = sm.add_constant(X, has_constant="add")
    fit = sm.Logit(df.Low_Score.to_numpy(), Xc).fit(disp=0, maxiter=200)
    me = fit.get_margeff(at="overall").summary_frame()
    return fit, me


# ---------------------------------------------------------------- prediction comparison

def tfidf_features(train_df, test_df, min_df=10):
    """Two separate vectorizers, one per box, fit on the training fold only."""
    mats = []
    for col in ["neg_clean", "pos_clean"]:
        vec = TfidfVectorizer(preprocessor=clean_review, stop_words=STOP_WORDS, min_df=min_df, sublinear_tf=True)
        tr = vec.fit_transform(train_df[col])
        mats.append((tr, vec.transform(test_df[col]), len(vec.vocabulary_)))
    Xtr = sparse.hstack([m[0] for m in mats]).tocsr()
    Xte = sparse.hstack([m[1] for m in mats]).tocsr()
    return Xtr, Xte, sum(m[2] for m in mats)


def compare_models(df, X_controls, flags, n_splits=5, seed=4360):
    """Out-of-sample R^2 and RMSE of M0-M3 on the same folds."""
    y = df.Reviewer_Score.to_numpy()
    alphas = np.logspace(-1, 3, 7)
    rows = []
    for fold, (tr, te) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(df)):
        Ctr, Cte = X_controls.iloc[tr].to_numpy(), X_controls.iloc[te].to_numpy()
        Ftr, Fte = flags.iloc[tr].to_numpy(), flags.iloc[te].to_numpy()
        Ttr, Tte, n_words = tfidf_features(df.iloc[tr], df.iloc[te])
        models = {
            "M0 controls": (LinearRegression(), Ctr, Cte),
            "M1 + categories": (LinearRegression(), np.hstack([Ctr, Ftr]), np.hstack([Cte, Fte])),
            "M2 + TF-IDF words": (RidgeCV(alphas=alphas), sparse.hstack([Ctr, Ttr]).tocsr(), sparse.hstack([Cte, Tte]).tocsr()),
            "M3 categories + words": (RidgeCV(alphas=alphas), sparse.hstack([Ctr, Ftr, Ttr]).tocsr(),
                                      sparse.hstack([Cte, Fte, Tte]).tocsr()),
        }
        for name, (model, Xtr, Xte) in models.items():
            pred = model.fit(Xtr, y[tr]).predict(Xte)
            rows.append({"fold": fold, "model": name, "R2": r2_score(y[te], pred),
                         "RMSE": mean_squared_error(y[te], pred) ** 0.5,
                         "features": Xtr.shape[1]})
    res = pd.DataFrame(rows)
    summary = res.groupby("model", sort=False).agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                                                   RMSE_mean=("RMSE", "mean"), features=("features", "mean"))
    r = summary.R2_mean
    share = (r["M1 + categories"] - r["M0 controls"]) / (r["M2 + TF-IDF words"] - r["M0 controls"])
    return res, summary, share


def fold_stability(df, X_controls, flags, base, n_splits=5, seed=4360):
    """Refit M1 on each training fold; Spearman of each fold's fix-list ranking with the full-data one."""
    rows = []
    for fold, (tr, _) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(df)):
        idx = df.index[tr]
        fit = ols(df.loc[idx, "Reviewer_Score"], pd.concat([X_controls.loc[idx], flags.loc[idx]], axis=1),
                  df.loc[idx, "Hotel_Name"])
        t = fix_list(fit, flags.loc[idx])
        common = base.index.intersection(t.index)
        rows.append({"fold": fold, "Spearman vs. full data": spearmanr(
            base.loc[common, "points lost per 100 reviews"], t.loc[common, "points lost per 100 reviews"]).statistic,
            "top 3": ", ".join(t.index[:3])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Path A: predict a low score

def classify_low_score(df, X_controls, flags, test_size=0.2, seed=4360):
    """Predict Low_Score (below 7) on a held-out 20% test set, stratified so both sets keep the 17% share.

    Compares a majority-class baseline, controls only, controls + categories (plain and with balanced
    class weights) and controls + TF-IDF words. Returns the score table (train and test side by side),
    the test confusion matrix of the categories model, and that model's coefficients.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split

    y = df.Low_Score.astype(int)
    tr, te = train_test_split(df.index, test_size=test_size, stratify=y, random_state=seed)
    Xc = X_controls.astype(float)
    Xf = pd.concat([Xc, flags], axis=1).astype(float)
    words_tr, words_te, _ = tfidf_features(df.loc[tr], df.loc[te])
    Xw_tr = sparse.hstack([sparse.csr_matrix(Xc.loc[tr].to_numpy()), words_tr]).tocsr()
    Xw_te = sparse.hstack([sparse.csr_matrix(Xc.loc[te].to_numpy()), words_te]).tocsr()

    logit = lambda **kw: LogisticRegression(max_iter=3000, **kw)
    models = [
        ("Majority class (always 'not low')", DummyClassifier(strategy="most_frequent"), Xc.loc[tr], Xc.loc[te]),
        ("Controls only", logit(), Xc.loc[tr], Xc.loc[te]),
        ("Controls + categories", logit(), Xf.loc[tr], Xf.loc[te]),
        ("Controls + categories, balanced weights", logit(class_weight="balanced"), Xf.loc[tr], Xf.loc[te]),
        ("Controls + TF-IDF words", logit(), Xw_tr, Xw_te),
    ]
    rows, fitted = [], {}
    for name, model, Xtr, Xte in models:
        model.fit(Xtr, y.loc[tr])
        row = {"model": name}
        for part, X, yy in (("train", Xtr, y.loc[tr]), ("test", Xte, y.loc[te])):
            p = model.predict(X)
            row[f"{part} accuracy"] = accuracy_score(yy, p)
            row[f"{part} F1"] = f1_score(yy, p, zero_division=0)
            if part == "test":
                row["test precision"] = precision_score(yy, p, zero_division=0)
                row["test recall"] = recall_score(yy, p, zero_division=0)
        rows.append(row)
        fitted[name] = (model, Xte)
    table = pd.DataFrame(rows).set_index("model")
    model, Xte = fitted["Controls + categories"]
    cm = pd.DataFrame(confusion_matrix(y.loc[te], model.predict(Xte)),
                      index=["actual: not low", "actual: low"], columns=["predicted: not low", "predicted: low"])
    coefs = pd.Series(model.coef_[0], index=Xf.columns)
    coefs = coefs[[c for c in coefs.index if c.startswith(("c_", "p_"))]].sort_values()
    return table, cm, coefs
