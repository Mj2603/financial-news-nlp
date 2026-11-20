import math

import numpy as np
import pandas as pd

from newsnlp.stats import annualised_sharpe, corr_tstat, evaluate, ols_slope, pearson_ic, spearman_ic


def test_corr_tstat_closed_form():
    assert math.isclose(corr_tstat(0.5, 27), 0.5 * math.sqrt(25 / 0.75))
    assert corr_tstat(0.0, 100) == 0.0
    assert math.isnan(corr_tstat(0.3, 2))


def test_ic_matches_numpy_and_is_rank_invariant():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal(200), rng.standard_normal(200)
    assert math.isclose(pearson_ic(x, y), np.corrcoef(x, y)[0, 1])
    # Spearman is invariant to strictly increasing transforms
    assert math.isclose(spearman_ic(x, y), spearman_ic(np.exp(x), y ** 3))


def test_ols_slope_exact_line():
    x = np.arange(10.0)
    assert math.isclose(ols_slope(x, 3.0 * x - 2.0), 3.0)


def test_sharpe_known_value():
    r = pd.Series([0.01, -0.01, 0.02, 0.0])
    exp = math.sqrt(252) * r.mean() / r.std(ddof=1)
    assert math.isclose(annualised_sharpe(r), exp)
    assert math.isnan(annualised_sharpe(pd.Series([0.01, 0.01])))


def test_evaluate_hit_rate_and_flat_days():
    df = pd.DataFrame({"signal": [1.0, -1.0, 0.0, 2.0], "fwd_ret": [0.01, 0.02, 0.05, 0.03]})
    rep = evaluate(df)
    assert rep.n == 4
    assert math.isclose(rep.hit_rate, 2 / 3)  # flat day excluded
