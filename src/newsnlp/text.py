"""Headline text cleaning and lexicon sentiment scoring.

Scoring uses TextBlob's general-purpose pattern lexicon on the raw text.
Stop-word removal and lemmatisation are deliberately *not* applied before
scoring: standard English stop-word lists contain negators such as "not" and
"no", and removing them flips the sign of phrases like "profits were not good".
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import pandas as pd


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def clean_text(title: Optional[str], description: Optional[str] = None) -> str:
    """Join title and description, tolerating missing values and extra whitespace."""
    parts = [str(p) for p in (title, description) if not _is_missing(p)]
    return " ".join(" ".join(parts).split())


@lru_cache(maxsize=100_000)
def score_text(text: str) -> tuple[float, float]:
    """Return (polarity in [-1, 1], subjectivity in [0, 1]) for one string."""
    from textblob import TextBlob

    if not text:
        return 0.0, 0.0
    s = TextBlob(text).sentiment
    return float(s.polarity), float(s.subjectivity)


def score_articles(news: pd.DataFrame) -> pd.DataFrame:
    """Add `text`, `polarity` and `subjectivity` columns to a news frame.

    Expects `title` and optionally `description` columns. Returns a copy.
    """
    out = news.copy()
    desc = out["description"] if "description" in out else pd.Series(None, index=out.index)
    out["text"] = [clean_text(t, d) for t, d in zip(out["title"], desc)]
    scores = [score_text(t) for t in out["text"]]
    out["polarity"] = [s[0] for s in scores]
    out["subjectivity"] = [s[1] for s in scores]
    return out
