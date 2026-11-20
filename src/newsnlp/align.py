"""Point-in-time alignment of news with daily closing prices.

Convention: an article can only inform a trade placed at the first market
close at or after its publication time. The signal for session t therefore
contains articles published in (close_{t-1}, close_t], and it is compared with
the forward return close_t -> close_{t+h}. Bucketing articles by calendar date
instead leaks news published after the close into that day's signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXCHANGE_TZ = "America/New_York"
CLOSE_TIME = "16:00"


def _to_exchange_naive(ts: pd.Series) -> pd.Series:
    ts = pd.to_datetime(ts)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(EXCHANGE_TZ).dt.tz_localize(None)
    return ts


def session_closes(trading_days: pd.DatetimeIndex, close_time: str = CLOSE_TIME) -> pd.DatetimeIndex:
    """Closing timestamps (exchange local time, tz-naive) for each trading day."""
    days = pd.DatetimeIndex(trading_days).normalize()
    return days + pd.Timedelta(close_time + ":00")


def assign_session(
    published_at: pd.Series,
    trading_days: pd.DatetimeIndex,
    close_time: str = CLOSE_TIME,
) -> pd.Series:
    """Map each publication time to the first trading day whose close is >= it.

    tz-aware inputs are converted to exchange time; tz-naive inputs are assumed
    to already be in exchange time. Articles after the last close get NaT.
    """
    ts = _to_exchange_naive(published_at)
    closes = session_closes(trading_days, close_time)
    days = pd.DatetimeIndex(trading_days).normalize()
    pos = np.searchsorted(closes.values, ts.values, side="left")
    valid = pos < len(closes)
    out = pd.Series(pd.NaT, index=published_at.index, dtype="datetime64[ns]")
    out[valid] = days.values[pos[valid]]
    return out


def assign_calendar_date(published_at: pd.Series) -> pd.Series:
    """Naive (leaky) assignment by calendar date. Kept only for comparison."""
    return _to_exchange_naive(published_at).dt.normalize()


def daily_sentiment(news: pd.DataFrame, session_col: str = "session") -> pd.DataFrame:
    """Mean polarity and article count per session."""
    g = news.dropna(subset=[session_col]).groupby(session_col)["polarity"]
    out = pd.DataFrame({"sentiment": g.mean(), "n_articles": g.size()})
    out.index.name = "date"
    return out


def forward_returns(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Simple return from close_t to close_{t+horizon}, indexed at t."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return close.shift(-horizon) / close - 1.0


def align(sentiment: pd.Series, close: pd.Series, horizon: int = 1) -> pd.DataFrame:
    """Join a daily signal (known at close t) with the forward return from t.

    Days without a signal or without a full forward window are dropped.
    """
    close = close.copy()
    close.index = pd.DatetimeIndex(close.index).normalize()
    fwd = forward_returns(close, horizon).rename("fwd_ret")
    df = pd.concat([sentiment.rename("signal"), fwd], axis=1, join="inner")
    return df.dropna()
