"""News headline sentiment vs. forward returns, with point-in-time alignment."""

from .align import align, assign_calendar_date, assign_session, daily_sentiment, forward_returns
from .stats import evaluate
from .text import clean_text, score_articles, score_text

__all__ = [
    "align",
    "assign_calendar_date",
    "assign_session",
    "clean_text",
    "daily_sentiment",
    "evaluate",
    "forward_returns",
    "score_articles",
    "score_text",
]
__version__ = "0.2.0"
