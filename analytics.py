"""
QUANTECH - Core Quantitative Portfolio Analytics Engine
========================================================
Implements financial mathematics, risk analytics, Ledoit-Wolf shrinkage optimization,
rule-based risk classification, backtesting, stress testing, and explainability.
Designed for Indian-market equity portfolios.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


# ==============================================================================
# DATA INGESTION & CACHING
# ==============================================================================

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def clean_ticker(ticker: str) -> str:
    """Normalize Indian ticker symbols, adding .NS if not already suffixed."""
    t = ticker.strip().upper()
    if not t:
        return ""
    if t.startswith("^") or "." in t:
        return t
    return f"{t}.NS"


def fetch_ticker_data(
    ticker: str,
    period_years: int = 3,
    use_cache: bool = True,
    cache_ttl_hours: int = 24
) -> Optional[pd.Series]:
    """
    Fetch historical adjusted close series for a single ticker with disk caching.
    Returns pd.Series indexed by Date or None on failure.
    """
    import yfinance as yf

    safe_name = ticker.replace("^", "IDX_").replace(".", "_")
    cache_file = os.path.join(CACHE_DIR, f"{safe_name}_{period_years}y.csv")

    # Check cache freshness
    if use_cache and os.path.exists(cache_file):
        file_age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600.0
        if file_age_hours < cache_ttl_hours:
            try:
                df = pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")
                if "Close" in df.columns and len(df) > 20:
                    return df["Close"].dropna()
            except Exception:
                pass  # Fall back to live fetch

    # Download from yfinance
    try:
        data = yf.download(
            ticker,
            period=f"{period_years}y",
            auto_adjust=True,
            progress=False,
            multi_level_index=False
        )
        if data is not None and not data.empty and len(data) > 20:
            col = "Close" if "Close" in data.columns else data.columns[0]
            series = data[col].dropna()
            series.name = "Close"
            # Save to cache
            series.to_frame().to_csv(cache_file)
            return series
    except Exception:
        pass

    return None


def generate_synthetic_data(
    tickers: List[str],
    benchmark: str = "^NSEI",
    days: int = 756,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates realistic synthetic daily prices for offline/demo operation.
    Calibrated with realistic Indian equity market parameters (12-18% drift, 18-28% vol).
    """
    np.random.seed(seed)
    end_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_date, periods=days)

    # Benchmark: 13% drift, 16% vol
    mu_b = 0.13 / 252
    sigma_b = 0.16 / np.sqrt(252)
    bm_rets = np.random.normal(mu_b, sigma_b, days)
    bm_prices = 18000.0 * np.exp(np.cumsum(bm_rets))
    benchmark_series = pd.Series(bm_prices, index=dates, name=benchmark)

    # Tickers with correlated returns
    data_dict = {}
    for i, ticker in enumerate(tickers):
        beta = 0.75 + 0.45 * (i % 3)
        idio_vol = (0.16 + 0.05 * (i % 4)) / np.sqrt(252)
        stock_rets = beta * bm_rets + np.random.normal(0, idio_vol, days)
        start_price = 1000.0 * (1.5 ** (i % 5))
        prices = start_price * np.exp(np.cumsum(stock_rets))
        data_dict[ticker] = prices

    prices_df = pd.DataFrame(data_dict, index=dates)
    return prices_df, benchmark_series


def fetch_market_data(
    tickers: List[str],
    benchmark: str = "^NSEI",
    period_years: int = 3,
    use_cache: bool = True,
    force_demo: bool = False
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Fetch market data for portfolio tickers and benchmark.
    Falls back gracefully to synthetic demo data if network fails or tickers are invalid.
    """
    cleaned_tickers = [clean_ticker(t) for t in tickers if clean_ticker(t)]
    cleaned_benchmark = clean_ticker(benchmark) if benchmark else "^NSEI"

    if force_demo:
        prices, bm = generate_synthetic_data(cleaned_tickers, cleaned_benchmark, days=period_years * 252)
        status = {
            "is_synthetic": True,
            "loaded_tickers": cleaned_tickers,
            "failed_tickers": [],
            "benchmark_loaded": True,
            "data_points": len(prices),
            "start_date": str(prices.index[0].date()),
            "end_date": str(prices.index[-1].date()),
            "message": "Demo Mode active: Using calibrated synthetic data."
        }
        return prices, bm, status

    loaded_series = {}
    failed_tickers = []

    for t in cleaned_tickers:
        series = fetch_ticker_data(t, period_years=period_years, use_cache=use_cache)
        if series is not None and len(series) > 20:
            loaded_series[t] = series
        else:
            failed_tickers.append(t)

    bm_series = fetch_ticker_data(cleaned_benchmark, period_years=period_years, use_cache=use_cache)

    # If all tickers failed or network is down, fall back to synthetic
    if len(loaded_series) == 0:
        prices, bm = generate_synthetic_data(cleaned_tickers, cleaned_benchmark, days=period_years * 252)
        status = {
            "is_synthetic": True,
            "loaded_tickers": cleaned_tickers,
            "failed_tickers": failed_tickers,
            "benchmark_loaded": True,
            "data_points": len(prices),
            "start_date": str(prices.index[0].date()),
            "end_date": str(prices.index[-1].date()),
            "message": "Live market fetch unavailable. Automatically running in Synthetic Demo Mode."
        }
        return prices, bm, status

    # Align dates
    prices_df = pd.DataFrame(loaded_series).dropna()

    if bm_series is not None and not bm_series.empty:
        # Align benchmark with prices
        combined = pd.concat([prices_df, bm_series.rename(cleaned_benchmark)], axis=1, join="inner").dropna()
        prices_df = combined[list(loaded_series.keys())]
        bm_series = combined[cleaned_benchmark]
    else:
        # Synthetic benchmark fallback if benchmark failed
        _, synth_bm = generate_synthetic_data(list(loaded_series.keys()), cleaned_benchmark, days=len(prices_df))
        synth_bm.index = prices_df.index
        bm_series = synth_bm

    status = {
        "is_synthetic": False,
        "loaded_tickers": list(prices_df.columns),
        "failed_tickers": failed_tickers,
        "benchmark_loaded": True,
        "data_points": len(prices_df),
        "start_date": str(prices_df.index[0].date()),
        "end_date": str(prices_df.index[-1].date()),
        "message": "Live market data successfully loaded."
    }

    return prices_df, bm_series, status


# ==============================================================================
# CORE QUANTITATIVE METRICS
# ==============================================================================

def calculate_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Computes percentage daily returns from prices, dropping the initial NaN."""
    return prices_df.pct_change().dropna()


def calculate_portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Computes daily weighted portfolio returns."""
    w = np.asarray(weights, dtype=float)
    return (returns_df * w).sum(axis=1)


def calculate_annualized_return(portfolio_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Computes Compound Annual Growth Rate (CAGR) from daily returns.
    Falls back to arithmetic annualized return if compound calculation is degenerate.
    """
    n_days = len(portfolio_returns)
    if n_days < 2:
        return 0.0
    cumulative_return = (1.0 + portfolio_returns).prod()
    if cumulative_return > 0:
        cagr = (cumulative_return ** (periods_per_year / n_days)) - 1.0
        return float(cagr)
    return float(portfolio_returns.mean() * periods_per_year)


def calculate_annualized_volatility(portfolio_returns: pd.Series, periods_per_year: int = 252) -> float:
    """Computes annualized standard deviation of daily returns."""
    if len(portfolio_returns) < 2:
        return 0.0
    daily_vol = portfolio_returns.std(ddof=1)
    ann_vol = float(daily_vol * np.sqrt(periods_per_year))
    return 0.0 if ann_vol < 1e-12 else ann_vol


def calculate_sharpe_ratio(
    ann_return: float,
    ann_vol: float,
    risk_free_rate: float = 0.065
) -> float:
    """
    Computes annualized Sharpe ratio: (R_p - R_f) / Vol.
    Default R_f = 6.5% (approximate Indian 91-day T-Bill / RBI repo benchmark).
    """
    if ann_vol <= 1e-6:
        return 0.0
    return float((ann_return - risk_free_rate) / ann_vol)


def calculate_max_drawdown(portfolio_returns: pd.Series) -> float:
    """
    Computes maximum peak-to-trough drawdown from daily returns as a positive float.
    E.g. 0.224 represents a 22.4% maximum drawdown.
    """
    if len(portfolio_returns) == 0:
        return 0.0
    cum_returns = (1.0 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return float(abs(drawdown.min()))


def calculate_drawdown_series(portfolio_returns: pd.Series) -> pd.Series:
    """Computes the full historical drawdown percentage series."""
    if len(portfolio_returns) == 0:
        return pd.Series(dtype=float)
    cum_returns = (1.0 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    return (cum_returns - running_max) / running_max


def calculate_var_cvar(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Calculates historical 1-day Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    Returns both as positive loss fractions (e.g. 0.021 = 2.1% 1-day loss).
    """
    if len(portfolio_returns) == 0:
        return 0.0, 0.0
    alpha = 1.0 - confidence_level
    var_threshold = np.percentile(portfolio_returns, alpha * 100)
    var = float(-var_threshold)

    # CVaR is the mean of losses strictly worse than or equal to the VaR threshold
    tail_losses = portfolio_returns[portfolio_returns <= var_threshold]
    if len(tail_losses) > 0:
        cvar = float(-tail_losses.mean())
    else:
        cvar = var

    return max(0.0, var), max(0.0, cvar)


def calculate_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Computes portfolio beta relative to the benchmark: Cov(R_p, R_b) / Var(R_b).
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return 1.0
    bm_var = aligned.iloc[:, 1].var()
    if bm_var <= 1e-8:
        return 1.0
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    return float(cov / bm_var)


def calculate_hhi(weights: np.ndarray) -> float:
    """
    Herfindahl-Hirschman Index (HHI) for asset concentration: sum(w_i^2).
    Range: [1/N, 1.0]. Lower values indicate superior diversification.
    """
    w = np.asarray(weights, dtype=float)
    w_sum = w.sum()
    if w_sum <= 0:
        return 1.0
    w_norm = w / w_sum
    return float(np.sum(w_norm ** 2))


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Computes Pearson asset return correlation matrix."""
    return returns_df.corr()


# ==============================================================================
# RISK EXPLAINABILITY & MARGINAL CONTRIBUTIONS
# ==============================================================================

def calculate_risk_contributions(returns_df: pd.DataFrame, weights: np.ndarray) -> Dict[str, Any]:
    """
    Decomposes portfolio volatility into Marginal Contribution to Risk (MCR)
    and Percentage Contribution to Risk (PCR) for each constituent.
    """
    tickers = list(returns_df.columns)
    w = np.asarray(weights, dtype=float)
    cov_matrix = returns_df.cov().values * 252.0  # Annualized covariance
    port_var = float(w.T @ cov_matrix @ w)
    port_vol = np.sqrt(max(port_var, 1e-8))

    # Marginal Contribution to Risk: d(sigma)/d(w_i) = (cov @ w) / sigma
    mcr = (cov_matrix @ w) / port_vol
    # Absolute risk contribution of asset i: w_i * MCR_i
    abs_rc = w * mcr
    # Percentage contribution to total risk
    pcr = abs_rc / port_vol

    asset_vols = np.sqrt(np.diag(cov_matrix))
    weighted_vol_sum = float(np.sum(w * asset_vols))
    # Diversification Ratio = Weighted individual vol / Portfolio vol
    div_ratio = weighted_vol_sum / port_vol if port_vol > 0 else 1.0

    return {
        "tickers": tickers,
        "asset_annual_vols": dict(zip(tickers, asset_vols)),
        "marginal_contributions": dict(zip(tickers, mcr)),
        "risk_contributions_pct": dict(zip(tickers, pcr)),
        "portfolio_vol": port_vol,
        "diversification_ratio": div_ratio,
        "covariance_matrix": cov_matrix
    }


def generate_explainability_narrative(
    weights_dict: Dict[str, float],
    risk_contributions: Dict[str, Any],
    hhi: float,
    beta: float,
    ann_vol: float
) -> List[str]:
    """Generates transparent, deterministic risk drivers in plain English."""
    pcr = risk_contributions["risk_contributions_pct"]
    tickers = risk_contributions["tickers"]
    sorted_risk = sorted(pcr.items(), key=lambda x: x[1], reverse=True)
    top_risk_ticker, top_risk_pct = sorted_risk[0]

    narrative = []

    # 1. Primary Risk Driver
    orig_w = weights_dict.get(top_risk_ticker, 0.0)
    narrative.append(
        f"**Primary Risk Driver:** `{top_risk_ticker}` is responsible for "
        f"**{top_risk_pct * 100:.1f}%** of total portfolio volatility while having a weight of "
        f"**{orig_w * 100:.1f}%**."
    )

    # 2. Diversification Benefit
    div_ratio = risk_contributions["diversification_ratio"]
    vol_reduction = (1.0 - (1.0 / div_ratio)) * 100 if div_ratio > 1.0 else 0.0
    narrative.append(
        f"**Diversification Effect:** Imperfect cross-asset correlations reduce total portfolio risk "
        f"by **{vol_reduction:.1f}%** compared to an un-diversified linear sum of individual stock volatilities "
        f"(Diversification Ratio: {div_ratio:.2f}x)."
    )

    # 3. Market Sensitivity (Beta)
    if beta > 1.15:
        narrative.append(
            f"**High Market Sensitivity:** Portfolio Beta of **{beta:.2f}** indicates aggressive co-movement; "
            f"it will amplify broader NIFTY 50 drawdowns by ~{(beta - 1.0) * 100:.0f}%."
        )
    elif beta < 0.85:
        narrative.append(
            f"**Defensive Exposure:** Portfolio Beta of **{beta:.2f}** provides defensive downside insulation "
            f"relative to the benchmark index."
        )
    else:
        narrative.append(
            f"**Market Tracking:** Portfolio Beta of **{beta:.2f}** indicates near-parity sensitivity "
            f"with the broader NIFTY 50 index."
        )

    # 4. Concentration Warning
    if hhi > 0.35:
        narrative.append(
            f"**Concentration Alert (HHI {hhi:.3f}):** Portfolio capital is heavily clustered in few names, "
            f"elevating idiosyncratic stock-specific risk."
        )
    elif hhi < 0.22:
        narrative.append(
            f"**Balanced Concentration (HHI {hhi:.3f}):** Capital distribution is broadly decentralized across assets."
        )

    return narrative


# ==============================================================================
# RULE-BASED RISK CLASSIFICATION
# ==============================================================================

def classify_portfolio_risk(ann_vol: float, max_drawdown: float, hhi: float) -> Dict[str, Any]:
    """
    Transparent, explainable rule-based risk classification (Low / Medium / High).
    Thresholds:
    - Volatility: Low (<= 15%), Medium (15-25%), High (> 25%)
    - Max Drawdown: Low (< 18%), Medium (18-30%), High (> 30%)
    - Concentration (HHI): Low (< 0.22), Medium (0.22-0.35), High (> 0.35)
    """
    # Factor 1: Volatility
    if ann_vol <= 0.15:
        vol_score, vol_tier = 1, "Low"
    elif ann_vol <= 0.25:
        vol_score, vol_tier = 2, "Medium"
    else:
        vol_score, vol_tier = 3, "High"

    # Factor 2: Drawdown
    if max_drawdown <= 0.18:
        dd_score, dd_tier = 1, "Low"
    elif max_drawdown <= 0.30:
        dd_score, dd_tier = 2, "Medium"
    else:
        dd_score, dd_tier = 3, "High"

    # Factor 3: Concentration
    if hhi <= 0.22:
        hhi_score, hhi_tier = 1, "Low"
    elif hhi <= 0.35:
        hhi_score, hhi_tier = 2, "Medium"
    else:
        hhi_score, hhi_tier = 3, "High"

    # Weighted Composite Score (Vol: 40%, Drawdown: 35%, Concentration: 25%)
    composite_score = 0.40 * vol_score + 0.35 * dd_score + 0.25 * hhi_score

    if composite_score < 1.60:
        overall_tier = "Low"
        color = "#10b981"  # Emerald green
    elif composite_score < 2.40:
        overall_tier = "Medium"
        color = "#f59e0b"  # Amber
    else:
        overall_tier = "High"
        color = "#ef4444"  # Crimson

    factors = [
        {
            "Factor": "Annualized Volatility",
            "Value": f"{ann_vol * 100:.1f}%",
            "Tier": vol_tier,
            "Threshold": "Low: ≤15% | Med: 15-25% | High: >25%"
        },
        {
            "Factor": "Max Historical Drawdown",
            "Value": f"{max_drawdown * 100:.1f}%",
            "Tier": dd_tier,
            "Threshold": "Low: ≤18% | Med: 18-30% | High: >30%"
        },
        {
            "Factor": "Concentration (HHI)",
            "Value": f"{hhi:.3f}",
            "Tier": hhi_tier,
            "Threshold": "Low: ≤0.22 | Med: 0.22-0.35 | High: >0.35"
        }
    ]

    return {
        "tier": overall_tier,
        "score": composite_score,
        "color": color,
        "factors": factors,
        "rationale": (
            f"Classified as {overall_tier} Risk based on composite factor score of {composite_score:.2f}/3.00. "
            f"Volatility contributed {vol_tier}, Drawdown contributed {dd_tier}, and Concentration contributed {hhi_tier}."
        )
    }


# ==============================================================================
# ALLOCATION RECOMMENDATIONS (LEDOIT-WOLF MIN-VARIANCE & EQUAL WEIGHT)
# ==============================================================================

def optimize_minimum_variance(
    returns_df: pd.DataFrame,
    max_asset_cap: float = 0.40
) -> Tuple[np.ndarray, bool, str]:
    """
    Computes long-only minimum-variance weights using Ledoit-Wolf shrinkage covariance.
    Enforces non-negative weights and single-asset cap (default 40%, adjusted if N is small).
    Falls back to equal weights if optimization fails.
    """
    n_assets = returns_df.shape[1]
    if n_assets == 0:
        return np.array([]), False, "Empty asset universe."

    if n_assets == 1:
        return np.array([1.0]), False, "Single asset trivial allocation."

    # Guarantee feasibility: single asset cap cannot be strictly less than 1/N
    effective_cap = max(max_asset_cap, 1.0 / n_assets)

    # Ledoit-Wolf shrinkage covariance
    try:
        lw = LedoitWolf().fit(returns_df.values)
        cov = lw.covariance_ * 252.0
        method_desc = "Ledoit-Wolf shrinkage covariance estimate"
    except Exception:
        cov = returns_df.cov().values * 252.0
        method_desc = "Standard sample covariance estimate (fallback)"

    # Objective: Minimize portfolio variance w^T @ Sigma @ w
    def objective(w: np.ndarray) -> float:
        return float(w.T @ cov @ w)

    # Gradient for efficiency
    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * (cov @ w)

    init_weights = np.ones(n_assets) / n_assets
    bounds = [(0.0, effective_cap) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        objective,
        init_weights,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 200}
    )

    if res.success:
        weights = res.x
        weights = np.maximum(0.0, weights)
        weights = weights / np.sum(weights)
        return weights, False, f"Optimized successfully using {method_desc} with single-asset cap of {effective_cap*100:.0f}%."
    else:
        # Fallback to equal weight
        fallback_w = np.ones(n_assets) / n_assets
        return fallback_w, True, f"Optimization solver did not converge ({res.message}). Fallback to Equal Weighting applied."


def calculate_turnover_and_cost(
    orig_weights: np.ndarray,
    new_weights: np.ndarray,
    portfolio_value: float,
    cost_bps: float = 15.0
) -> Dict[str, float]:
    """
    Calculates one-way portfolio turnover and estimated transaction costs.
    cost_bps: 15 basis points = 0.0015 (0.15% per unit of one-way turnover).
    """
    w_orig = np.asarray(orig_weights, dtype=float)
    w_new = np.asarray(new_weights, dtype=float)
    turnover = float(0.5 * np.sum(np.abs(w_new - w_orig)))
    cost_rate = cost_bps / 10000.0
    cost_inr = float(turnover * portfolio_value * cost_rate)
    return {
        "turnover": turnover,
        "cost_rate": cost_rate,
        "cost_inr": cost_inr,
        "cost_bps": cost_bps
    }


# ==============================================================================
# BACKTESTING & STRESS TESTING
# ==============================================================================

def run_portfolio_backtest(
    returns_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    orig_weights: np.ndarray,
    rec_weights: np.ndarray,
    portfolio_value: float = 1000000.0,
    cost_bps: float = 15.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backtests Original, Recommended, Equal-Weight, and Benchmark over historical window.
    Applies initial rebalancing transaction cost to Recommended portfolio.
    Returns:
    1. Wealth trajectories (Base ₹100)
    2. Drawdown series for all strategies
    """
    dates = returns_df.index
    n_assets = returns_df.shape[1]
    eq_weights = np.ones(n_assets) / n_assets

    # Daily returns
    r_orig = calculate_portfolio_returns(returns_df, orig_weights)
    r_rec = calculate_portfolio_returns(returns_df, rec_weights)
    r_eq = calculate_portfolio_returns(returns_df, eq_weights)
    r_bm = benchmark_returns.reindex(dates).fillna(0.0)

    # Initial rebalance cost deduction on Recommended
    turnover_data = calculate_turnover_and_cost(orig_weights, rec_weights, portfolio_value, cost_bps)
    initial_rec_penalty = turnover_data["turnover"] * turnover_data["cost_rate"]
    starting_rec_capital = 100.0 * (1.0 - initial_rec_penalty)

    # Cumulative wealth (Base ₹100)
    wealth_orig = 100.0 * (1.0 + r_orig).cumprod()
    wealth_rec = starting_rec_capital * (1.0 + r_rec).cumprod()
    wealth_eq = 100.0 * (1.0 + r_eq).cumprod()
    wealth_bm = 100.0 * (1.0 + r_bm).cumprod()

    wealth_df = pd.DataFrame({
        "Original Allocation": wealth_orig,
        "Recommended (Min-Var)": wealth_rec,
        "Equal-Weight": wealth_eq,
        "Benchmark (NIFTY 50)": wealth_bm
    }, index=dates)

    # Drawdowns
    dd_orig = calculate_drawdown_series(r_orig) * 100.0
    dd_rec = calculate_drawdown_series(r_rec) * 100.0
    dd_eq = calculate_drawdown_series(r_eq) * 100.0
    dd_bm = calculate_drawdown_series(r_bm) * 100.0

    dd_df = pd.DataFrame({
        "Original Allocation": dd_orig,
        "Recommended (Min-Var)": dd_rec,
        "Equal-Weight": dd_eq,
        "Benchmark (NIFTY 50)": dd_bm
    }, index=dates)

    return wealth_df, dd_df


def execute_crash_stress_test(
    returns_df: pd.DataFrame,
    benchmark_returns: pd.Series,
    orig_weights: np.ndarray,
    rec_weights: np.ndarray,
    shock_pct: float = -0.20,
    portfolio_value: float = 1000000.0
) -> Dict[str, Any]:
    """
    Stress tests portfolio under an instantaneous market crash scenario.
    Asset impact = Asset Beta * Market Shock.
    """
    tickers = list(returns_df.columns)
    asset_betas = {}
    for t in tickers:
        asset_betas[t] = calculate_beta(returns_df[t], benchmark_returns)

    beta_orig = float(np.sum(orig_weights * [asset_betas[t] for t in tickers]))
    beta_rec = float(np.sum(rec_weights * [asset_betas[t] for t in tickers]))

    # Projected returns under shock
    port_shock_orig = beta_orig * shock_pct
    port_shock_rec = beta_rec * shock_pct

    loss_orig_inr = portfolio_value * abs(port_shock_orig)
    loss_rec_inr = portfolio_value * abs(port_shock_rec)
    inr_saved = loss_orig_inr - loss_rec_inr

    asset_impacts = []
    for i, t in enumerate(tickers):
        b = asset_betas[t]
        asset_impact = b * shock_pct
        asset_impacts.append({
            "Ticker": t,
            "Beta": round(b, 2),
            "Original Weight": f"{orig_weights[i] * 100:.1f}%",
            "Rec Weight": f"{rec_weights[i] * 100:.1f}%",
            "Hypothetical Drop": f"{asset_impact * 100:.1f}%",
            "Orig Loss (INR)": f"₹{portfolio_value * orig_weights[i] * abs(asset_impact):,.0f}"
        })

    return {
        "shock_pct": shock_pct,
        "portfolio_value": portfolio_value,
        "beta_orig": beta_orig,
        "beta_rec": beta_rec,
        "orig_loss_pct": port_shock_orig,
        "rec_loss_pct": port_shock_rec,
        "orig_loss_inr": loss_orig_inr,
        "rec_loss_inr": loss_rec_inr,
        "inr_saved": inr_saved,
        "asset_impacts": asset_impacts
    }
