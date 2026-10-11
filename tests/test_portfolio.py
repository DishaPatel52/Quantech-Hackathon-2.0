"""
QUANTECH Unit Tests - Portfolio Construction & Optimization
===========================================================
Tests weight normalization, Ledoit-Wolf minimum-variance optimization,
turnover costs, risk classification, and missing data fallbacks.
"""

import numpy as np
import pandas as pd
import pytest
from analytics import (
    optimize_minimum_variance,
    calculate_turnover_and_cost,
    classify_portfolio_risk,
    fetch_market_data,
    generate_synthetic_data,
    clean_ticker,
)


def test_weights_sum_to_one():
    """Verify that portfolio optimization output weights sum strictly to 1.0."""
    np.random.seed(101)
    returns = pd.DataFrame(
        np.random.normal(0.0005, 0.015, (252, 4)),
        columns=["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"]
    )
    weights, fallback_used, msg = optimize_minimum_variance(returns, max_asset_cap=0.40)
    assert not fallback_used
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
    assert np.all(weights >= 0.0)
    # Check max cap constraint
    assert np.all(weights <= 0.4001)


def test_optimization_cap_feasibility():
    """
    If N=2, a strict 40% cap is mathematically impossible (sum can at most be 80%).
    The optimizer should dynamically handle this by relaxing cap to max(cap, 1/N) = 0.50.
    """
    np.random.seed(102)
    returns = pd.DataFrame(
        np.random.normal(0.0005, 0.015, (252, 2)),
        columns=["STOCK_A", "STOCK_B"]
    )
    weights, fallback_used, msg = optimize_minimum_variance(returns, max_asset_cap=0.40)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
    assert np.all(weights >= 0.0)
    assert np.all(weights <= 0.5001)


def test_turnover_and_transaction_cost():
    """
    Test turnover calculation and 15 bps transaction cost:
    Original: [0.5, 0.5]
    New: [0.7, 0.3]
    Turnover = 0.5 * (|0.7 - 0.5| + |0.3 - 0.5|) = 0.5 * (0.2 + 0.2) = 0.20 (20%)
    For portfolio value of INR 1,000,000 at 15 bps (0.0015):
    Tx Cost = 1,000,000 * 0.20 * 0.0015 = INR 300.
    """
    w_orig = np.array([0.50, 0.50])
    w_new = np.array([0.70, 0.30])
    port_val = 1_000_000.0

    res = calculate_turnover_and_cost(w_orig, w_new, port_val, cost_bps=15.0)
    assert np.isclose(res["turnover"], 0.20, atol=1e-6)
    assert np.isclose(res["cost_inr"], 300.0, atol=1e-4)


def test_risk_classification_rules():
    """
    Test transparent rule-based risk classification tiers:
    - Low: vol <= 15%, max DD <= 18%, HHI <= 0.22
    - High: vol > 25%, max DD > 30%, HHI > 0.35
    """
    low_res = classify_portfolio_risk(ann_vol=0.12, max_drawdown=0.14, hhi=0.20)
    assert low_res["tier"] == "Low"

    high_res = classify_portfolio_risk(ann_vol=0.32, max_drawdown=0.38, hhi=0.45)
    assert high_res["tier"] == "High"

    med_res = classify_portfolio_risk(ann_vol=0.19, max_drawdown=0.22, hhi=0.28)
    assert med_res["tier"] == "Medium"


def test_ticker_cleaning():
    """Ensure Indian tickers are properly normalized."""
    assert clean_ticker("reliance") == "RELIANCE.NS"
    assert clean_ticker("TCS.NS") == "TCS.NS"
    assert clean_ticker("INFY.BO") == "INFY.BO"
    assert clean_ticker("^NSEI") == "^NSEI"


def test_missing_market_data_fallback():
    """
    When market data fetching fails or demo mode is requested,
    system falls back safely to synthetic data without crashing.
    """
    prices, bm, status = fetch_market_data(["NON_EXISTENT_XYZ123"], force_demo=True)
    assert status["is_synthetic"] is True
    assert len(prices) > 100
    assert not prices.empty
    assert not bm.empty
