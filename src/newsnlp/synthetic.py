"""Seeded SYNTHETIC news and price data with a known ground truth.

Model (all quantities are synthetic; nothing here is market data):

* Each trading session t has a latent news factor z_t ~ N(0, 1). It describes
  information released in the window (close_{t-1}, close_t].
* Close-to-close return into session t:
      r_t = scale * (a * z_t + b * z_{t-1} + eps_t),   eps_t ~ N(0, 1)
  `a` is the immediate reaction, `b` is a one-day drift (the only part a
  signal known at close t can predict).
* So the forward return from close_t is r_{t+1} and
      corr(z_t, r_{t+1}) = b / sqrt(a^2 + b^2 + 1)     (closed form).
* Each session gets Poisson(articles_per_day) headlines with publication
  times uniform over its window, so some land after the previous close.
  A headline is neutral with probability p_neutral, otherwise positive with
  probability sigmoid(k * z_t) and negative otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .align import session_closes

COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella Inc", "Stark Industries"]

POSITIVE = [
    "{co} reports excellent quarterly results",
    "{co} posts strong revenue growth",
    "Analysts upgrade {co} after great earnings",
    "{co} beats estimates with impressive margins",
    "{co} wins major contract, outlook positive",
]
NEGATIVE = [
    "{co} reports terrible quarterly results",
    "{co} posts weak revenue and poor guidance",
    "Analysts downgrade {co} after bad earnings",
    "{co} misses estimates, margins disappointing",
    "{co} loses major contract, outlook negative",
]
NEUTRAL = [
    "{co} to hold annual shareholder meeting",
    "{co} announces date of earnings call",
    "{co} files quarterly report with regulator",
]


@dataclass
class SyntheticConfig:
    n_days: int = 1000
    a: float = 0.5
    b: float = 0.1
    scale: float = 0.01
    articles_per_day: float = 3.0
    p_neutral: float = 0.3
    k: float = 2.0
    start: str = "2020-01-02"
    seed: int = 0


@dataclass
class SyntheticData:
    close: pd.Series        # synthetic close prices indexed by trading day
    news: pd.DataFrame      # columns: published_at, title, description, true_session
    z: pd.Series            # latent factor per session
    config: SyntheticConfig

    def true_forward_ic(self) -> float:
        c = self.config
        return c.b / np.sqrt(c.a ** 2 + c.b ** 2 + 1.0)


def generate(config: SyntheticConfig | None = None) -> SyntheticData:
    c = config or SyntheticConfig()
    rng = np.random.default_rng(c.seed)
    # one extra leading day so session 0 has a defined opening window
    days = pd.bdate_range(c.start, periods=c.n_days + 1)
    closes = session_closes(days)
    n = len(days)

    z = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    z_lag = np.concatenate([[0.0], z[:-1]])
    r = c.scale * (c.a * z + c.b * z_lag + eps)
    r[0] = 0.0
    close = pd.Series(100.0 * np.cumprod(1.0 + r), index=days, name="close")

    rows = []
    for t in range(1, n):
        window_start, window_end = closes[t - 1], closes[t]
        span = (window_end - window_start).total_seconds()
        p_pos = 1.0 / (1.0 + np.exp(-c.k * z[t]))
        for _ in range(rng.poisson(c.articles_per_day)):
            # uniform on (start, end]; rounding to whole seconds keeps it inside
            offset = np.ceil(rng.uniform(0.0, span))
            offset = max(offset, 1.0)
            u = rng.random()
            if u < c.p_neutral:
                pool = NEUTRAL
            elif rng.random() < p_pos:
                pool = POSITIVE
            else:
                pool = NEGATIVE
            title = pool[rng.integers(len(pool))].format(co=COMPANIES[rng.integers(len(COMPANIES))])
            rows.append(
                {
                    "published_at": window_start + pd.Timedelta(seconds=offset),
                    "title": title,
                    "description": None,
                    "true_session": days[t],
                }
            )
    news = pd.DataFrame(rows)
    return SyntheticData(close=close, news=news, z=pd.Series(z, index=days, name="z"), config=c)
