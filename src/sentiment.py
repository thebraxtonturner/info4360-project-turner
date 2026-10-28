"""VADER baseline sentiment, plus a check on WHY a review scores neutral.

VADER only adds up the scores of words in its lexicon. A compound score of 0 can mean
two different things: no lexicon word was found, or positive and negative words
cancelled out. vader_table() separates the two.
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def lexicon_hits(text):
    """Number of words in the text that VADER has a score for."""
    return sum(word in analyzer.lexicon for word in str(text).lower().split())


def neutral_reason(compound, hits):
    if compound != 0:
        return "not neutral"
    return "no lexicon words" if hits == 0 else "words cancel out"


def vader_table(texts):
    """Compound score, lexicon hit count, and neutral reason for each text."""
    rows = []
    for text in texts:
        compound = analyzer.polarity_scores(str(text))["compound"]
        hits = lexicon_hits(text)
        rows.append({"compound": compound, "lexicon_hits": hits,
                     "neutral_reason": neutral_reason(compound, hits)})
    return pd.DataFrame(rows, index=texts.index)
