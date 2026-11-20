"""Ground-truth checks on SYNTHETIC data with a known data-generating process."""
import math

import numpy as np
import pandas as pd

from newsnlp.align import align, assign_calendar_date, assign_session, daily_sentiment
from newsnlp.stats import ols_slope, pearson_ic
from newsnlp.synthetic import SyntheticConfig, generate
from newsnlp.text import score_articles


def test_generator_is_deterministic():
    a, b = generate(SyntheticConfig(n_days=50, seed=7)), generate(SyntheticConfig(n_days=50, seed=7))
    pd.testing.assert_series_equal(a.close, b.close)
    pd.testing.assert_frame_equal(a.news, b.news)


def test_session_assignment_recovers_true_session():
    d = generate(SyntheticConfig(n_days=200, seed=3))
    got = assign_session(d.news["published_at"], d.close.index)
    assert (got.values == d.news["true_session"].values).all()
    # some headlines land after the previous close, so calendar bucketing differs
    cal = assign_calendar_date(d.news["published_at"])
    assert (cal.values != d.news["true_session"].values).mean() > 0.2


def test_latent_factor_recovers_known_slope_and_ic():
    c = SyntheticConfig(n_days=20000, a=0.5, b=0.3, seed=11)
    d = generate(c)
    df = align(d.z, d.close)
    # forward simple return = scale*(a z_{t+1} + b z_t + eps); slope on z_t is scale*b
    assert math.isclose(ols_slope(df["signal"], df["fwd_ret"]), c.scale * c.b, rel_tol=0.1)
    se = 1 / math.sqrt(len(df))
    assert abs(pearson_ic(df["signal"], df["fwd_ret"]) - d.true_forward_ic()) < 4 * se


def test_null_model_has_no_predictive_ic():
    d = generate(SyntheticConfig(n_days=5000, b=0.0, seed=5))
    news = score_articles(d.news)
    news["session"] = assign_session(news["published_at"], d.close.index)
    df = align(daily_sentiment(news)["sentiment"], d.close)
    assert abs(pearson_ic(df["signal"], df["fwd_ret"])) < 4 / math.sqrt(len(df))


def test_calendar_bucketing_leaks_future_information():
    # With b=0 nothing is predictable, but calendar-date bucketing puts
    # after-close headlines (about tomorrow's move) into today's signal.
    d = generate(SyntheticConfig(n_days=5000, a=0.8, b=0.0, seed=9))
    news = score_articles(d.news)
    news["session"] = assign_calendar_date(news["published_at"])
    leaky = align(daily_sentiment(news)["sentiment"], d.close)
    assert pearson_ic(leaky["signal"], leaky["fwd_ret"]) > 6 / math.sqrt(len(leaky))
