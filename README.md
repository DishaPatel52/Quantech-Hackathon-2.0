# QUANTECH — Explainable Portfolio Risk Analysis & Recommendation Platform

> **Hackathon MVP for Indian-Market Equity Investors**  
> Built with Python, Streamlit, pandas, NumPy, yfinance, Plotly, and scikit-learn.

---

## ⚡ Overview

**QUANTECH** is an explainable quantitative portfolio risk analysis and recommendation platform tailored for Indian capital market investors. It analyzes portfolios of National Stock Exchange (NSE) equities, benchmarks them against the **NIFTY 50** (`^NSEI`), classifies overall risk using an explicit rule-based framework, attributes risk drivers using Marginal Contribution to Risk (MCR), and computes a disciplined, long-only **Minimum-Variance Allocation** using **Ledoit-Wolf shrinkage covariance estimation**.

---

## 🚀 Key Features

### 1. Portfolio Input & Validation
- **Indian Market Focus**: Supports all liquid NSE equity tickers (`.NS`) and BSE tickers (`.BO`).
- **Default Presets**: Pre-configured with liquid NSE bluechips (*Reliance, TCS, HDFC Bank, Infosys, ICICI Bank*) as well as thematic presets (*High Beta*, *Defensive FMCG/Pharma*).
- **Interactive Weight Editing**: Custom capital weights with automated normalization and feedback.
- **Configurable Portfolio Capital**: Denominated in Indian Rupees (₹ INR) for real-world turnover and loss calculations.
- **Benchmark Selection**: NIFTY 50 Index (`^NSEI`) or Nippon India Nifty 50 ETF proxy (`NIFTYBEES.NS`).

### 2. Market Data & Local Caching
- **Automated Ingestion**: Historical adjusted close data fetched via `yfinance`.
- **Local Disk Cache**: Downloads are saved in `cache/` as CSV files with a 24-hour TTL for instant re-runs and offline resilience.
- **Graceful Error Handling**: Detects missing tickers and trading suspensions without application crashes.
- **Synthetic Demo Mode**: A toggleable simulation mode powered by calibrated Geometric Brownian Motion (12–18% drift, 18–28% vol) with prominent badge labeling when active. **Never fabricates synthetic prices as real data.**

### 3. Quantitative Risk Analytics
Calculated from aligned historical daily return series:
- **Annualized Return (CAGR)**: Compounded annual growth rate.
- **Annualized Volatility ($\sigma$)**: Sample standard deviation scaled by $\sqrt{252}$.
- **Sharpe Ratio**: Excess return over the Indian risk-free rate assumption (default **6.5%**, benchmarked against RBI 91-day T-Bill yields).
- **1-Day 95% Historical Value at Risk (VaR)**: Tail quantile loss expressed in percentage and ₹ INR.
- **1-Day 95% Conditional VaR (CVaR / Expected Shortfall)**: Average loss in the worst 5% tail.
- **Maximum Drawdown (MDD)**: Peak-to-trough historical decline with underwater equity curve.
- **Portfolio Beta ($\beta$)**: Systematic sensitivity relative to NIFTY 50 ($\frac{\text{Cov}(R_p, R_b)}{\text{Var}(R_b)}$).
- **Herfindahl-Hirschman Index (HHI)**: Asset concentration metric ($\sum w_i^2$).
- **Correlation Heatmap**: Interactive Plotly heatmap with teal gradient displaying pairwise Pearson correlations.

### 4. Rule-Based Risk Classification
- Transparent, deterministic classification: **Low Risk**, **Medium Risk**, or **High Risk**.
- Evaluates three explicit factor scores:
  - **Volatility**: Low ($\le 15\%$), Medium ($15\% - 25\%$), High ($> 25\%$)
  - **Max Drawdown**: Low ($\le 18\%$), Medium ($18\% - 30\%$), High ($> 30\%$)
  - **Concentration (HHI)**: Low ($\le 0.22$), Medium ($0.22 - 0.35$), High ($> 0.35$)
- Displays full factor breakdown and rationale.
- Clearly documented as a deterministic rule-based classifier that can be replaced with an ML model in subsequent phases.

### 5. Disciplined Allocation Recommendations
- **Equal-Weight Portfolio**: $1/N$ benchmark allocation.
- **Long-Only Minimum-Variance Portfolio**:
  - Solves $\min_w w^T \Sigma w$ subject to $\sum w_i = 1$ and $0 \le w_i \le \text{Cap}$ (default 40%).
  - Uses **Ledoit-Wolf shrinkage covariance estimation** (`sklearn.covariance.LedoitWolf`) to suppress empirical estimation noise.
  - Automatically handles small asset universes ($N < 3$) with dynamic feasibility relaxation.
  - Falls back transparently to equal-weighting if optimization fails, with explicit user notification.
- **Turnover & Transaction Costs**: Computes one-way turnover ($\frac{1}{2} \sum |w^{\text{new}} - w^{\text{orig}}|$) and estimates execution friction assuming **15 basis points (0.15%)** per unit of one-way turnover.

### 6. Multi-Strategy Backtesting & Stress Testing
- **Growth of ₹100 Chart**: Comparative historical equity curves for Original, Recommended, Equal-Weight, and Benchmark portfolios.
- **Rebalancing Friction**: Recommended portfolio accounts for initial rebalancing transaction cost.
- **Hypothetical Market Crash Simulator**: Configurable market shock slider (-5% to -50%, default -20%). Estimates portfolio loss in ₹ INR and provides stock-by-stock beta breakdown.

### 7. Explainability & Risk Attribution
- **Marginal Contribution to Risk (MCR)**: $\frac{(\Sigma w)_i}{\sigma_p}$.
- **Percentage Contribution to Risk (PCR)**: $\frac{w_i \cdot \text{MCR}_i}{\sigma_p}$.
- Contrasts capital allocation weights against actual percentage risk contributions.
- Computes **Diversification Ratio** ($\frac{\sum w_i \sigma_i}{\sigma_p}$) explaining volatility dampening from non-correlated assets.
- Deterministic plain-English narratives explaining primary risk drivers and market sensitivity.

---

## 🛠️ Project Structure

```
Quantech/
├── app.py                 # Main Streamlit dashboard application
├── analytics.py           # Core quantitative math, optimization & data engine
├── requirements.txt       # Project dependencies
├── README.md              # Documentation & launch guide
├── cache/                 # Local market data cache directory
│   ├── RELIANCE_NS_3y.csv
│   ├── TCS_NS_3y.csv
│   └── ...
└── tests/                 # Unit test suite
    ├── test_metrics.py    # Math & quantitative metrics verification
    └── test_portfolio.py  # Optimization constraints & fallback tests
```

---

## ⚡ Installation & Launch

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or with Python launcher: `py -3.13 -m pip install -r requirements.txt`)*

### 2. Run the Unit Test Suite
```bash
pytest tests/ -v
```
*(Or: `py -3.13 -m pytest tests/ -v`)*

All 12 unit tests verify mathematical correctness (annualized volatility, max drawdown, VaR/CVaR, Sharpe ratio, HHI, turnover costs, and optimization bounds).

### 3. Launch the Application
```bash
streamlit run app.py
```
*(Or: `py -3.13 -m streamlit run app.py`)*

Open your browser at `http://localhost:8501`.

---

## 📐 Assumptions & Methodological Notes

| Parameter | Value | Rationale |
|:---|:---|:---|
| **Trading Days / Year** | 252 | Standard NSE trading calendar |
| **Risk-Free Rate ($R_f$)** | 6.5% Annual | Benchmarked against Indian 91-day Treasury Bill / RBI repo rate |
| **Transaction Cost** | 15 bps (0.0015) | Indian brokerage, STT, exchange turnover fees, and slippage estimate |
| **Default Lookback** | 3 Years (~756 days) | Balances regime representation with asset survival bias |
| **Covariance Estimator** | Ledoit-Wolf Shrinkage | Shrinks sample covariance toward an identity target to dampen estimation error |
| **Single-Asset Cap** | 40% (Configurable) | Prevents single-stock concentration while allowing optimization room |

---

## ⚠️ Decision Support Disclaimer

QUANTECH is an analytical and educational decision-support platform for quantitative portfolio risk analysis. Historical performance, statistical covariances, Value at Risk, and Ledoit-Wolf optimizations are historical estimates and do not guarantee future returns or protect against market losses. This software is not registered with SEBI as an Investment Adviser (RIA) or Research Analyst (RA) and does not constitute financial or investment advice.
