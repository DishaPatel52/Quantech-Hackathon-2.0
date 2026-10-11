"""
QUANTECH Unit Tests - Risk & Quantitative Metrics
==================================================
Deterministic unit tests using verified mathematical calculations and synthetic series.
"""

import numpy as np
import pandas as pd
import pytest
from analytics import (
    calculate_annualized_volatility,
    calculate_max_drawdown,
    calculate_var_cvar,
    calculate_sharpe_ratio,
    calculate_hhi,
    calculate_beta,
    calculate_portfolio_returns,
    calculate_annualized_return,
)


def test_annualized_volatility_known_series():
    """
    Test annualized volatility with a deterministic return series.
    If daily return has sample standard deviation s,
    annualized volatility must be s * sqrt(252).
    """
    # Deterministic series alternating +1% and -1%
    # With mean 0, sample variance for [-0.01, 0.01, -0.01, 0.01] (n=4):
    returns = pd.Series([-0.01, 0.01, -0.01, 0.01])
    sample_std = returns.std(ddof=1)
    expected_ann_vol = sample_std * np.sqrt(252)

    computed_vol = calculate_annualized_volatility(returns, periods_per_year=252)
    assert np.isclose(computed_vol, expected_ann_vol, atol=1e-6)

    # Constant return series has zero volatility
    flat_returns = pd.Series([0.005] * 100)
    assert np.isclose(calculate_annualized_volatility(flat_returns), 0.0, atol=1e-8)


def test_maximum_drawdown_hand_calculated():
    """
    Hand-calculated example:
    Day 0: Wealth = 100
    Day 1: Return = +0.10 -> Wealth = 110 (Peak = 110)
    Day 2: Return = -0.20 -> Wealth = 88  (Drawdown = (88 - 110) / 110 = -20.0%)
    Day 3: Return = +0.10 -> Wealth = 96.8 (Drawdown = (96.8 - 110) / 110 = -12.0%)
    Max Drawdown must be exactly 0.20 (20.0%).
    """
    returns = pd.Series([0.10, -0.20, 0.10])
    computed_mdd = calculate_max_drawdown(returns)
    assert np.isclose(computed_mdd, 0.20, atol=1e-6)


def test_var_and_cvar():
    """
    VaR and CVaR calculations on a controlled distribution.
    100 observations: 1 to 100 with known percentiles.
    """
    # Construct 100 daily returns from -0.050 to +0.049 in increments of 0.001
    returns = pd.Series(np.linspace(-0.05, 0.049, 100))
    var_95, cvar_95 = calculate_var_cvar(returns, confidence_level=0.95)

    # 5th percentile is at index around 5 -> approx 4.5% to 5.0% loss
    assert var_95 > 0
    assert cvar_95 >= var_95, "CVaR must always be greater than or equal to VaR in the tail"
    assert np.isclose(var_95, 0.04505, atol=0.005)


def test_sharpe_ratio():
    """
    Sharpe Ratio test:
    Annualized return = 16.5%
    Risk-free rate = 6.5%
    Annualized vol = 20.0%
    Sharpe = (0.165 - 0.065) / 0.20 = 0.50
    """
    sharpe = calculate_sharpe_ratio(ann_return=0.165, ann_vol=0.20, risk_free_rate=0.065)
    assert np.isclose(sharpe, 0.50, atol=1e-6)

    # Negative excess return
    sharpe_neg = calculate_sharpe_ratio(ann_return=0.045, ann_vol=0.20, risk_free_rate=0.065)
    assert np.isclose(sharpe_neg, -0.10, atol=1e-6)


def test_hhi_concentration():
    """
    Herfindahl-Hirschman Index:
    - Equal weight across 4 assets: HHI = 4 * (0.25)^2 = 0.25
    - Single asset 100%: HHI = 1.0
    - Equal weight across N assets: HHI = 1 / N
    """
    w_equal_4 = np.array([0.25, 0.25, 0.25, 0.25])
    assert np.isclose(calculate_hhi(w_equal_4), 0.25, atol=1e-6)

    w_single = np.array([1.0, 0.0, 0.0])
    assert np.isclose(calculate_hhi(w_single), 1.0, atol=1e-6)

    w_equal_5 = np.ones(5) / 5.0
    assert np.isclose(calculate_hhi(w_equal_5), 0.20, atol=1e-6)


def test_beta_calculation():
    """
    Portfolio with identical returns to benchmark has beta = 1.0.
    Portfolio with 2x returns to benchmark has beta = 2.0.
    """
    bm = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02, -0.01])
    port_same = bm.copy()
    port_double = 2.0 * bm

    assert np.isclose(calculate_beta(port_same, bm), 1.0, atol=1e-6)
    assert np.isclose(calculate_beta(port_double, bm), 2.0, atol=1e-6)
