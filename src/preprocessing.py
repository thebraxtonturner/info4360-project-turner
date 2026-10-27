"""Text cleaning for the keyword and TF-IDF analysis.

clean_review() is written so it can be passed straight to a vectorizer:

    TfidfVectorizer(preprocessor=clean_review, stop_words=STOP_WORDS)

The vectorizer then cleans, tokenizes, and drops stop words in one step.
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations carry the meaning of a complaint ("not clean", "no hot water"), so keep them.
# "hotel", "room" and "staff" are kept too: whether they show up in a complaint is informative.
NEGATIONS = {"no", "not", "nor", "never", "nothing", "none", "cannot"}
STOP_WORDS = sorted((set(ENGLISH_STOP_WORDS) - NEGATIONS) | {"did", "didn", "don", "wasn", "t", "s", "really", "bit"})

# spaCy entity labels to remove: hotel and brand names (ORG) are noise for "what went wrong",
# and staff names (PERSON) are personal information we do not need
ENTITIES_TO_REMOVE = ("ORG", "PERSON")

# Amenity acronyms guests write in capitals ("AC not working"); spaCy tags them ORG, but they are complaints
KEEP_ACRONYMS = {"ac", "tv", "wc"}


def clean_review(text):
    """Lowercase, keep letters only, collapse spaces. Used as a vectorizer preprocessor."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)   # Drop digits and any leftover punctuation
    return re.sub(r"\s+", " ", text).strip()


def remove_hotel_name(text, hotel_name):
    """Remove the reviewed hotel's own name (any case), e.g. 'The Park Grand' in its own review."""
    name = re.sub(r"^(the|hotel)\s+", "", str(hotel_name).strip(), flags=re.IGNORECASE)
    return re.sub(re.escape(name), " ", str(text), flags=re.IGNORECASE)


def load_ner():
    """spaCy small English model with only the NER step (faster)."""
    import spacy
    return spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])


def common_lowercase_words(texts, min_count=5):
    """Words that appear in lowercase at least min_count times in the corpus.

    Booking.com strips punctuation, so a capitalized word at the start of a sentence
    ("Breakfast was cold", "Pillows too soft") looks like a name to spaCy. A word that guests
    also write in lowercase many times is an ordinary word, not a hotel or a person.
    """
    counts = {}
    for text in texts:
        for word in re.findall(r"\b[a-z]+\b", str(text)):
            counts[word] = counts.get(word, 0) + 1
    return {w for w, n in counts.items() if n >= min_count}


def remove_entities(texts, nlp, common_words=frozenset(), labels=ENTITIES_TO_REMOVE, batch_size=500):
    """Remove named entities of the given types from each text, word by word.

    Inside an entity, a word is kept if it is a common lowercase word (see common_lowercase_words)
    or an amenity acronym. So "Kensington High Street Bed" loses only "Kensington", and
    "Breakfast" or "AC" tagged as ORG are kept. Returns the cleaned texts and the lists of
    words removed and kept, as (word, label), so the notebook can show what was taken out.
    """
    cleaned, removed, kept = [], [], []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        drop = set()   # token positions to remove
        for ent in doc.ents:
            if ent.label_ not in labels:
                continue
            for tok in ent:
                word = tok.text.lower()
                # Names are capitalized, so a lowercase word inside an entity is an ordinary word
                if tok.text[0].islower() or word in common_words or word in KEEP_ACRONYMS or not tok.is_alpha:
                    kept.append((tok.text, ent.label_))
                else:
                    drop.add(tok.i)
                    removed.append((tok.text, ent.label_))
        words = [tok.text_with_ws for tok in doc if tok.i not in drop]
        cleaned.append(re.sub(r"\s+", " ", "".join(words)).strip())
    return cleaned, removed, kept


def denoise(df, text_col, nlp, common_words):
    """Hotel name removal + NER removal for one text column of the reviews table."""
    texts = [remove_hotel_name(t, h) for t, h in zip(df[text_col], df["Hotel_Name"])]
    return remove_entities(texts, nlp, common_words)
