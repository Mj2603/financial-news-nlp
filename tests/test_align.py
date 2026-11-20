import numpy as np
import pandas as pd
import pytest

from newsnlp.align import (
    align,
    assign_calendar_date,
    assign_session,
    daily_sentiment,
    forward_returns,
)

DAYS = pd.DatetimeIndex(["2024-03-01", "2024-03-04", "2024-03-05"])  # Fri, Mon, Tue


def test_assign_session_cutoff_rules():
    ts = pd.Series(pd.to_datetime([
        "2024-03-01 09:00:00",   # before Friday close -> Fri
        "2024-03-01 16:00:00",   # exactly at close -> Fri
        "2024-03-01 16:00:01",  # after close -> Mon
        "2024-03-02 12:00:00",   # Saturday -> Mon
        "2024-03-04 23:00:00",   # Monday evening -> Tue
        "2024-03-05 17:00:00",   # after last close -> NaT
    ]))
    got = assign_session(ts, DAYS)
    exp = pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-04", "2024-03-04", "2024-03-05", None])
    assert got.iloc[:5].tolist() == list(exp[:5])
    assert pd.isna(got.iloc[5])


def test_assign_session_converts_utc():
    # 20:30 UTC on 4 Mar 2024 is 15:30 New York (EST, UTC-5) -> Monday session.
    # 21:30 UTC is 16:30 New York -> Tuesday session.
    ts = pd.Series(pd.to_datetime(["2024-03-04T20:30:00Z", "2024-03-04T21:30:00Z"]))
    got = assign_session(ts, DAYS)
    assert list(got) == list(pd.to_datetime(["2024-03-04", "2024-03-05"]))


def test_calendar_date_differs_after_close():
    ts = pd.Series(pd.to_datetime(["2024-03-04 23:00:00"]))
    assert assign_calendar_date(ts).iloc[0] == pd.Timestamp("2024-03-04")
    assert assign_session(ts, DAYS).iloc[0] == pd.Timestamp("2024-03-05")


def test_forward_returns_known_values():
    close = pd.Series([100.0, 110.0, 99.0, 99.0], index=pd.bdate_range("2024-01-01", periods=4))
    f1 = forward_returns(close, 1)
    np.testing.assert_allclose(f1.iloc[:3], [0.10, -0.10, 0.0])
    assert np.isnan(f1.iloc[3])
    f2 = forward_returns(close, 2)
    np.testing.assert_allclose(f2.iloc[:2], [-0.01, -0.10])
    with pytest.raises(ValueError):
        forward_returns(close, 0)


def test_align_uses_forward_not_same_day_return():
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([100.0, 110.0, 99.0, 99.0], index=idx)
    signal = pd.Series([1.0, -1.0, 0.5, 2.0], index=idx)
    df = align(signal, close)
    assert len(df) == 3  # last day has no forward return
    np.testing.assert_allclose(df["fwd_ret"], [0.10, -0.10, 0.0])


def test_daily_sentiment_mean_and_count():
    news = pd.DataFrame({
        "session": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-04", None]),
        "polarity": [0.5, -0.1, 0.3, 0.9],
    })
    d = daily_sentiment(news)
    np.testing.assert_allclose(d["sentiment"], [0.2, 0.3])
    assert list(d["n_articles"]) == [2, 1]
