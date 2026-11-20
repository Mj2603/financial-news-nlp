"""Signal evaluation statistics for a single-asset daily time series."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def pearson_ic(x: pd.Series, y: pd.Series) -> float:
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def spearman_ic(x: pd.Series, y: pd.Series) -> float:
    return pearson_ic(pd.Series(x).rank(), pd.Series(y).rank())


def corr_tstat(r: float, n: int) -> float:
    """t-statistic for H0: correlation = 0, t = r * sqrt((n - 2) / (1 - r^2))."""
    if n <= 2:
        return float("nan")
    if abs(r) >= 1:
        return math.copysign(float("inf"), r)
    return r * math.sqrt((n - 2) / (1 - r * r))


def ols_slope(x: pd.Series, y: pd.Series) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def annualised_sharpe(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    r = np.asarray(returns, float)
    sd = r.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(math.sqrt(periods) * r.mean() / sd)


@dataclass
class SignalReport:
    n: int
    pearson_ic: float
    spearman_ic: float
    t_stat: float
    hit_rate: float
    sharpe: float


def evaluate(aligned: pd.DataFrame) -> SignalReport:
    """Evaluate `signal` against `fwd_ret` (output of newsnlp.align.align).

    The trading rule is sign(signal): long on positive, short on negative,
    flat on zero. No transaction costs. `hit_rate` is over non-flat days.
    """
    s, r = aligned["signal"], aligned["fwd_ret"]
    ic = pearson_ic(s, r)
    pos = np.sign(s)
    strat = pos * r
    active = pos != 0
    hit = float((np.sign(r[active]) == pos[active]).mean()) if active.any() else float("nan")
    return SignalReport(
        n=len(aligned),
        pearson_ic=ic,
        spearman_ic=spearman_ic(s, r),
        t_stat=corr_tstat(ic, len(aligned)),
        hit_rate=hit,
        sharpe=annualised_sharpe(strat),
    )
