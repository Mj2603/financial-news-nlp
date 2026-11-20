import math

import pandas as pd

from newsnlp.text import clean_text, score_articles, score_text


def test_clean_text_handles_missing_and_whitespace():
    assert clean_text("A  b", None) == "A b"
    assert clean_text("A", float("nan")) == "A"
    assert clean_text(None, None) == ""
    assert clean_text("Title", " desc ") == "Title desc"


def test_negation_is_preserved():
    # Removing stop words (which include "not") would turn this positive.
    pol, _ = score_text("Profits were not good this quarter")
    assert pol < 0


def test_score_bounds_and_signs():
    pos, s1 = score_text("Acme Corp reports excellent quarterly results")
    neg, s2 = score_text("Acme Corp reports terrible quarterly results")
    assert pos > 0 > neg
    assert 0 <= s1 <= 1 and 0 <= s2 <= 1
    assert score_text("") == (0.0, 0.0)


def test_score_articles_without_description_column():
    df = pd.DataFrame({"title": ["great results", "bad results"]})
    out = score_articles(df)
    assert list(out.columns) == ["title", "text", "polarity", "subjectivity"]
    assert out["polarity"].iloc[0] > 0 > out["polarity"].iloc[1]
    assert not math.isnan(out["polarity"].sum())
