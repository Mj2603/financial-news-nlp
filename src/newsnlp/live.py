"""Optional live data loaders (NewsAPI and Yahoo Finance via yfinance).

These need network access and the `live` extra (`pip install -e ".[live]"`).
The NewsAPI key is read from the NEWS_API_KEY environment variable; it is
never stored in code. These functions are not exercised by the experiment
in this repository, which uses synthetic data only.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import pandas as pd


def articles_to_frame(articles: Iterable[dict]) -> pd.DataFrame:
    """Convert NewsAPI `articles` JSON objects into a news frame.

    `published_at` is parsed as UTC (NewsAPI timestamps end in 'Z').
    """
    rows = [
        {
            "published_at": a.get("publishedAt"),
            "title": a.get("title"),
            "description": a.get("description"),
            "source": (a.get("source") or {}).get("name"),
            "url": a.get("url"),
        }
        for a in articles
    ]
    df = pd.DataFrame(rows, columns=["published_at", "title", "description", "source", "url"])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df.dropna(subset=["published_at", "title"]).reset_index(drop=True)


def fetch_news(query: str, days: int = 30, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch English articles matching `query` from NewsAPI's /everything endpoint."""
    from newsapi import NewsApiClient

    key = api_key or os.environ.get("NEWS_API_KEY")
    if not key:
        raise RuntimeError("Set the NEWS_API_KEY environment variable.")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    resp = NewsApiClient(api_key=key).get_everything(
        q=query, from_param=since, language="en", sort_by="publishedAt", page_size=100
    )
    return articles_to_frame(resp.get("articles", []))


def fetch_close(symbol: str, start: str, end: Optional[str] = None) -> pd.Series:
    """Daily adjusted close prices from Yahoo Finance, indexed by date."""
    import yfinance as yf

    hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d", auto_adjust=True)
    close = hist["Close"].copy()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close.name = "close"
    return close
