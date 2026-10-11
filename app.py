"""
QUANTECH — Portfolio Risk & Allocation Platform
Indian equity analysis tool built with Streamlit + yfinance + scikit-learn
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    clean_ticker,
    fetch_market_data,
    calculate_daily_returns,
    calculate_portfolio_returns,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_var_cvar,
    calculate_beta,
    calculate_hhi,
    calculate_correlation_matrix,
    calculate_risk_contributions,
    generate_explainability_narrative,
    classify_portfolio_risk,
    optimize_minimum_variance,
    calculate_turnover_and_cost,
    run_portfolio_backtest,
    execute_crash_stress_test,
)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QUANTECH | Portfolio Risk Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] hr { border-color: #334155; }

    .kpi-container {
        background: #131c2e;
        border-left: 4px solid #0ea5e9;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .kpi-container::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #0ea5e9, #14b8a6);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.1;
    }
    .kpi-sub {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 5px;
    }
    .badge-live {
        display: inline-block;
        background-color: rgba(16,185,129,0.12);
        color: #10b981;
        border: 1px solid rgba(16,185,129,0.35);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-demo {
        display: inline-block;
        background-color: rgba(245,158,11,0.12);
        color: #f59e0b;
        border: 1px solid rgba(245,158,11,0.35);
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .insight-card {
        background-color: #131c2e;
        border: 1px solid #1e293b;
        border-left: 4px solid #14b8a6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.91rem;
        color: #e2e8f0;
        line-height: 1.55;
    }
    .qt-footer {
        background-color: #0f172a;
        border-left: 3px solid #334155;
        padding: 14px 18px;
        border-radius: 4px;
        font-size: 0.80rem;
        color: #64748b;
        margin-top: 36px;
        line-height: 1.6;
    }

    /* Table styling */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    div[data-testid="stDataFrameResizable"] { border: 1px solid #1e293b; border-radius: 8px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ── Portfolio presets ─────────────────────────────────────────────────────────

PRESETS = {
    "NSE Large-Cap Basket": {
        "tickers": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"],
        "weights": [0.25, 0.20, 0.25, 0.15, 0.15]
    },
    "High-Beta Growth": {
        "tickers": ["TATAMOTORS.NS", "BAJFINANCE.NS", "TITAN.NS", "ADANIENT.NS"],
        "weights": [0.30, 0.25, 0.25, 0.20]
    },
    "Defensive — FMCG & Pharma": {
        "tickers": ["ITC.NS", "HINDUNILVR.NS", "SUNPHARMA.NS", "TCS.NS"],
        "weights": [0.35, 0.30, 0.20, 0.15]
    }
}


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ QUANTECH")
    st.markdown(
        "<span style='font-size:0.83rem; color:#64748b;'>Indian equity portfolio analyser</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("**Portfolio**")
    preset_choice = st.selectbox(
        "Start from a preset",
        ["Custom"] + list(PRESETS.keys()),
        index=1
    )

    portfolio_value = st.number_input(
        "Portfolio size (₹)",
        min_value=10_000,
        max_value=1_000_000_000,
        value=1_000_000,
        step=50_000,
        help="Used for ₹ rupee loss estimates and transaction cost calculations."
    )

    if preset_choice in PRESETS:
        default_tickers_text = ", ".join(PRESETS[preset_choice]["tickers"])
        default_weights_text = ", ".join([str(int(w * 100)) for w in PRESETS[preset_choice]["weights"]])
    else:
        default_tickers_text = "RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS"
        default_weights_text = "25, 20, 25, 15, 15"

    tickers_input = st.text_area(
        "Tickers (comma separated)",
        value=default_tickers_text,
        help="NSE symbols with .NS suffix. Plain names like RELIANCE also work — .NS is added automatically."
    )

    weights_input = st.text_area(
        "Weights % (comma separated)",
        value=default_weights_text,
        help="Doesn't need to sum to 100 — it'll be normalised automatically."
    )

    # Parse tickers
    parsed_tickers = [clean_ticker(t.strip()) for t in tickers_input.split(",") if t.strip()]
    raw_weights = []
    for w in weights_input.split(","):
        w_s = w.strip()
        if w_s:
            try:
                raw_weights.append(float(w_s))
            except ValueError:
                raw_weights.append(0.0)

    # Align lengths
    if len(parsed_tickers) != len(raw_weights):
        if len(raw_weights) < len(parsed_tickers):
            raw_weights.extend([0.0] * (len(parsed_tickers) - len(raw_weights)))
        else:
            raw_weights = raw_weights[:len(parsed_tickers)]

    raw_weights_arr = np.array(raw_weights, dtype=float)
    total_raw_weight = float(np.sum(raw_weights_arr))

    if not np.isclose(total_raw_weight, 100.0, atol=0.5) and total_raw_weight > 0:
        norm_weights = raw_weights_arr / total_raw_weight
        st.warning(f"Weights summed to {total_raw_weight:.1f}% — normalised to 100%.")
    elif total_raw_weight > 0:
        norm_weights = raw_weights_arr / 100.0
    else:
        norm_weights = np.ones(len(parsed_tickers)) / max(1, len(parsed_tickers))
        st.error("All weights are zero — defaulting to equal weight.")

    st.markdown("---")
    st.markdown("**Settings**")

    benchmark = st.selectbox(
        "Benchmark",
        ["^NSEI", "NIFTYBEES.NS"],
        index=0,
        help="NIFTY 50 index (^NSEI) or the Nippon India ETF proxy."
    )

    period_years = st.select_slider(
        "History (years)",
        options=[1, 2, 3, 5],
        value=3,
        help="Longer windows are more statistically stable but may include stale market regimes."
    )

    risk_free_rate = st.number_input(
        "Risk-free rate (%)",
        min_value=0.0, max_value=15.0, value=6.5, step=0.1,
        help="Used in Sharpe ratio. Default is 6.5%, close to the current RBI 91-day T-bill yield."
    ) / 100.0

    max_asset_cap = st.slider(
        "Max weight per stock (%)",
        min_value=20, max_value=60, value=40, step=5,
        help="Upper limit applied when building the minimum-variance portfolio."
    ) / 100.0

    force_demo_mode = st.checkbox(
        "Use synthetic data (offline mode)",
        value=False,
        help="Replaces live prices with simulated data — useful for demos without internet."
    )

    st.markdown("---")
    st.caption("QUANTECH v1.0 · For educational use only")


# ── Data fetch ────────────────────────────────────────────────────────────────

with st.spinner("Loading price data..."):
    prices_df, bm_series, data_status = fetch_market_data(
        tickers=parsed_tickers,
        benchmark=benchmark,
        period_years=period_years,
        use_cache=True,
        force_demo=force_demo_mode
    )

active_tickers = list(prices_df.columns)
if len(active_tickers) == 0:
    st.error("Couldn't load data for any of the tickers. Check the symbols or enable offline mode.")
    st.stop()

active_indices = [parsed_tickers.index(t) for t in active_tickers if t in parsed_tickers]
active_weights = np.array([norm_weights[i] for i in active_indices], dtype=float)
if np.sum(active_weights) > 0:
    active_weights = active_weights / np.sum(active_weights)
else:
    active_weights = np.ones(len(active_tickers)) / len(active_tickers)

returns_df      = calculate_daily_returns(prices_df)
bm_returns      = calculate_daily_returns(bm_series.to_frame()).iloc[:, 0]
orig_port_returns = calculate_portfolio_returns(returns_df, active_weights)


# ── Compute all metrics ───────────────────────────────────────────────────────

ann_return  = calculate_annualized_return(orig_port_returns)
ann_vol     = calculate_annualized_volatility(orig_port_returns)
sharpe      = calculate_sharpe_ratio(ann_return, ann_vol, risk_free_rate)
max_dd      = calculate_max_drawdown(orig_port_returns)
var_95, cvar_95 = calculate_var_cvar(orig_port_returns)
beta        = calculate_beta(orig_port_returns, bm_returns)
hhi         = calculate_hhi(active_weights)
bm_ann_return = calculate_annualized_return(bm_returns)
bm_ann_vol    = calculate_annualized_volatility(bm_returns)
bm_sharpe     = calculate_sharpe_ratio(bm_ann_return, bm_ann_vol, risk_free_rate)
bm_max_dd     = calculate_max_drawdown(bm_returns)

risk_profile      = classify_portfolio_risk(ann_vol, max_dd, hhi)
risk_decomp       = calculate_risk_contributions(returns_df, active_weights)
weights_map       = dict(zip(active_tickers, active_weights))
explainability    = generate_explainability_narrative(weights_map, risk_decomp, hhi, beta, ann_vol)

rec_weights, opt_fallback, opt_msg = optimize_minimum_variance(returns_df, max_asset_cap=max_asset_cap)
rec_returns   = calculate_portfolio_returns(returns_df, rec_weights)
rec_ann_return = calculate_annualized_return(rec_returns)
rec_ann_vol   = calculate_annualized_volatility(rec_returns)
rec_sharpe    = calculate_sharpe_ratio(rec_ann_return, rec_ann_vol, risk_free_rate)
rec_max_dd    = calculate_max_drawdown(rec_returns)

turnover_info = calculate_turnover_and_cost(active_weights, rec_weights, portfolio_value, cost_bps=15.0)

# Per-stock metrics
stock_metrics = []
for i, t in enumerate(active_tickers):
    s_ret = returns_df[t]
    s_ann_r = calculate_annualized_return(s_ret)
    s_ann_v = calculate_annualized_volatility(s_ret)
    s_sharpe = calculate_sharpe_ratio(s_ann_r, s_ann_v, risk_free_rate)
    s_mdd    = calculate_max_drawdown(s_ret)
    s_beta   = calculate_beta(s_ret, bm_returns)
    stock_metrics.append({
        "Ticker":        t.replace(".NS", "").replace(".BO", ""),
        "Weight":        f"{active_weights[i]*100:.1f}%",
        "Return (ann.)": f"{s_ann_r*100:.1f}%",
        "Volatility":    f"{s_ann_v*100:.1f}%",
        "Sharpe":        f"{s_sharpe:.2f}",
        "Max DD":        f"-{s_mdd*100:.1f}%",
        "Beta":          f"{s_beta:.2f}",
    })
stock_metrics_df = pd.DataFrame(stock_metrics)

# Rolling 90-day Sharpe
ROLL = 90
roll_excess = orig_port_returns - risk_free_rate / 252.0
rolling_sharpe = (
    roll_excess.rolling(ROLL).mean() /
    orig_port_returns.rolling(ROLL).std(ddof=1)
) * np.sqrt(252)

bm_roll_excess = bm_returns - risk_free_rate / 252.0
bm_rolling_sharpe = (
    bm_roll_excess.rolling(ROLL).mean() /
    bm_returns.rolling(ROLL).std(ddof=1)
) * np.sqrt(252)

# Calendar-year returns
def calendar_year_returns(ret_series: pd.Series) -> dict:
    yearly = {}
    for yr, grp in ret_series.groupby(ret_series.index.year):
        yearly[yr] = round(((1 + grp).prod() - 1) * 100, 1)
    return yearly

cal_orig = calendar_year_returns(orig_port_returns)
cal_bm   = calendar_year_returns(bm_returns)


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class='qt-hero'>
        <div class='qt-logo'>⚡</div>
        <div class='qt-title'>QUANTECH Portfolio Risk Platform</div>
    </div>
    <div style='color:#64748b; font-size:0.88rem; margin-top:4px;'>
        {len(active_tickers)} stocks • {benchmark} • {data_status['start_date']} to {data_status['end_date']} ({data_status['data_points']} trading days)
    </div>
    <div style='text-align:right;'>
        {'<span class=\'badge-demo\'>⚠️ SYNTHETIC DATA · DEMO MODE</span>' if data_status["is_synthetic"] else '<span class=\'badge-live\'>● Live · NSE via Yahoo Finance</span>'}
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

hdr_left, hdr_right = st.columns([3, 1])
with hdr_left:
    st.markdown(
        "<h2 style='margin-bottom:3px; font-size:1.6rem;'>QUANTECH Portfolio Risk Platform</h2>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<span style='color:#64748b; font-size:0.88rem;'>"
        f"{len(active_tickers)} stocks · {benchmark} benchmark · "
        f"{data_status['start_date']} to {data_status['end_date']} "
        f"({data_status['data_points']} trading days)"
        f"</span>",
        unsafe_allow_html=True
    )
with hdr_right:
    st.markdown("<div style='text-align:right; padding-top:12px;'>", unsafe_allow_html=True)
    if data_status["is_synthetic"]:
        st.markdown("<span class='badge-demo'>⚠️ SYNTHETIC DATA · DEMO MODE</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-live'>● Live · NSE via Yahoo Finance</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if data_status["failed_tickers"]:
    st.warning(f"Couldn't load {', '.join(data_status['failed_tickers'])} — excluded from analysis.")

st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


# ── Risk classification banner ────────────────────────────────────────────────

tier_color = risk_profile["color"]
tier_name  = risk_profile["tier"]
score_val  = risk_profile["score"]

st.markdown(
    f"""
    <div style='background:#131c2e; border-left:5px solid {tier_color};
                border-radius:8px; padding:14px 20px; margin-bottom:20px;'>
        <span style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:#64748b;'>
            Risk assessment (rule-based)
        </span>
        <div style='font-size:1.35rem; font-weight:700; color:{tier_color}; margin-top:3px;'>
            {tier_name} Risk &nbsp;
            <span style='font-size:0.88rem; color:#94a3b8; font-weight:400;'>
                — composite score {score_val:.2f} / 3.00
            </span>
        </div>
        <div style='font-size:0.83rem; color:#94a3b8; margin-top:4px;'>
            {risk_profile["rationale"]}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_dash, tab_stocks, tab_explain, tab_recom, tab_backtest, tab_stress = st.tabs([
    "📊 Dashboard",
    "📋 Stock Breakdown",
    "🔍 Risk Drivers",
    "⚖️ Recommendations",
    "📈 Backtest",
    "⚡ Crash Simulator"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_dash:

    # ── 8 KPI cards in 2 rows ──
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)

    with r1c1:
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Annualised Volatility</div>
                <div class='kpi-value'>{ann_vol*100:.1f}%</div>
                <div class='kpi-sub'>Benchmark: {bm_ann_vol*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    with r1c2:
        sharpe_color = "#10b981" if sharpe > 0 else "#f43f5e"
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Sharpe Ratio (Rf {risk_free_rate*100:.1f}%)</div>
                <div class='kpi-value' style='color:{sharpe_color};'>{sharpe:.2f}</div>
                <div class='kpi-sub'>Benchmark: {bm_sharpe:.2f}</div>
            </div>""", unsafe_allow_html=True)

    with r1c3:
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Max Drawdown</div>
                <div class='kpi-value' style='color:#f43f5e;'>-{max_dd*100:.1f}%</div>
                <div class='kpi-sub'>Benchmark: -{bm_max_dd*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    with r1c4:
        ret_color = "#10b981" if ann_return >= 0 else "#f43f5e"
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Annual Return (CAGR)</div>
                <div class='kpi-value' style='color:{ret_color};'>{ann_return*100:.1f}%</div>
                <div class='kpi-sub'>Benchmark: {bm_ann_return*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    with r2c1:
        var_inr = portfolio_value * var_95
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>1-Day VaR (95%)</div>
                <div class='kpi-value'>{var_95*100:.2f}%</div>
                <div class='kpi-sub'>≈ ₹{var_inr:,.0f} on ₹{portfolio_value:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    with r2c2:
        cvar_inr = portfolio_value * cvar_95
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>1-Day CVaR (95%)</div>
                <div class='kpi-value'>{cvar_95*100:.2f}%</div>
                <div class='kpi-sub'>Avg. loss in worst 5% days · ₹{cvar_inr:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    with r2c3:
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Beta vs {benchmark}</div>
                <div class='kpi-value'>{beta:.2f}</div>
                <div class='kpi-sub'>{"Aggressive — moves more than market" if beta>1.1 else ("Defensive" if beta<0.9 else "Near-market sensitivity")}</div>
            </div>""", unsafe_allow_html=True)

    with r2c4:
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Concentration (HHI)</div>
                <div class='kpi-value'>{hhi:.3f}</div>
                <div class='kpi-sub'>Fully equal: {1/len(active_tickers):.3f} · Max: 1.0</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # ── Charts: Allocation donut + Correlation heatmap ──
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("#### Allocation")
        alloc_df = pd.DataFrame({
            "Ticker": [t.replace(".NS","").replace(".BO","") for t in active_tickers],
            "Weight": active_weights * 100.0,
            "INR":    active_weights * portfolio_value
        })
        fig_pie = px.pie(
            alloc_df, values="Weight", names="Ticker",
            hole=0.52,
            color_discrete_sequence=px.colors.sequential.Teal_r
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{percent}<br>₹%{customdata[0]:,.0f}<extra></extra>",
            customdata=alloc_df[["INR"]]
        )
        fig_pie.update_layout(
            paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
            font_color="#cbd5e1", margin=dict(t=10,b=10,l=10,r=10),
            legend=dict(orientation="h", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        st.markdown("#### Return correlations")
        corr_matrix = calculate_correlation_matrix(returns_df)
        short_labels = [t.replace(".NS","").replace(".BO","") for t in active_tickers]
        corr_matrix.index = short_labels
        corr_matrix.columns = short_labels

        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale="Teal",
            aspect="auto", zmin=-0.2, zmax=1.0
        )
        fig_corr.update_layout(
            paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
            font_color="#cbd5e1", margin=dict(t=10,b=10,l=10,r=10),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    # ── Rolling 90-day Sharpe ──
    st.markdown("#### Rolling 90-day Sharpe ratio")
    st.caption("Shows whether risk-adjusted performance has been improving or deteriorating over time.")

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        mode="lines", name="Your Portfolio",
        line=dict(color="#0ea5e9", width=2)
    ))
    fig_roll.add_trace(go.Scatter(
        x=bm_rolling_sharpe.index, y=bm_rolling_sharpe.values,
        mode="lines", name=f"Benchmark ({benchmark})",
        line=dict(color="#64748b", width=1.5, dash="dot")
    ))
    fig_roll.add_hline(y=0, line_dash="dash", line_color="#334155")
    fig_roll.update_layout(
        paper_bgcolor="#0b0f19", plot_bgcolor="#0f172a",
        font_color="#cbd5e1",
        hovermode="x unified",
        margin=dict(t=10,b=20,l=10,r=10),
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1),
        yaxis_title="Sharpe (trailing 90 days)"
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # ── Risk factor table ──
    st.markdown("#### What drove this risk classification")
    factors_df = pd.DataFrame(risk_profile["factors"])
    st.dataframe(factors_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PER-STOCK BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

with tab_stocks:
    st.markdown("### Stock-by-stock metrics")
    st.caption(
        "These are individual stock numbers — not portfolio-level. "
        "The portfolio's overall volatility is lower than any single stock's because correlations dampen the combined risk."
    )

    st.dataframe(stock_metrics_df, use_container_width=True, hide_index=True)

    # ── Calendar year returns ──
    st.markdown("### Calendar year returns")
    st.caption("How each calendar year played out for your portfolio vs. the NIFTY 50 benchmark.")

    years = sorted(set(list(cal_orig.keys()) + list(cal_bm.keys())))
    cal_data = []
    for yr in years:
        cal_data.append({
            "Year": str(yr),
            "Your Portfolio": f"{cal_orig.get(yr, 0):+.1f}%",
            f"Benchmark ({benchmark})": f"{cal_bm.get(yr, 0):+.1f}%",
            "Outperformed?": "✅" if cal_orig.get(yr, 0) > cal_bm.get(yr, 0) else "❌"
        })
    st.dataframe(pd.DataFrame(cal_data), use_container_width=True, hide_index=True)

    # ── Waterfall: year-by-year portfolio return chart ──
    port_years  = [str(y) for y in years if y in cal_orig]
    port_values = [cal_orig[y] for y in years if y in cal_orig]
    bm_values   = [cal_bm.get(y, 0) for y in years if y in cal_orig]

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Bar(
        name="Your Portfolio", x=port_years, y=port_values,
        marker_color=["#10b981" if v >= 0 else "#f43f5e" for v in port_values]
    ))
    fig_cal.add_trace(go.Scatter(
        name=f"Benchmark", x=port_years, y=bm_values,
        mode="lines+markers",
        line=dict(color="#64748b", width=2, dash="dot"),
        marker=dict(size=6)
    ))
    fig_cal.add_hline(y=0, line_color="#334155")
    fig_cal.update_layout(
        paper_bgcolor="#0b0f19", plot_bgcolor="#0f172a",
        font_color="#cbd5e1",
        margin=dict(t=20,b=20,l=10,r=10),
        yaxis_title="Return (%)",
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_cal, use_container_width=True)

    # ── Download button ──
    st.markdown("### Export")
    csv_data = stock_metrics_df.to_csv(index=False)
    st.download_button(
        label="Download stock metrics as CSV",
        data=csv_data,
        file_name="quantech_stock_metrics.csv",
        mime="text/csv"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK DRIVERS / EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_explain:
    st.markdown("### What's actually driving your risk?")
    st.markdown(
        "A stock's weight in your portfolio doesn't equal its contribution to risk. "
        "If one stock is highly volatile or correlated with the others, it can dominate the risk "
        "even with a relatively small allocation. Here's the breakdown."
    )

    for note in explainability:
        st.markdown(f"<div class='insight-card'>{note}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    # ── Weight vs Risk contribution chart ──
    st.markdown("#### Weight vs. actual risk contribution")
    st.caption(
        "Blue bars = how much capital you put in. Red bars = how much of total portfolio "
        "volatility comes from that stock. A big gap means concentration risk is hidden."
    )

    rc_data = []
    pcr_dict = risk_decomp["risk_contributions_pct"]
    for i, t in enumerate(active_tickers):
        short = t.replace(".NS","").replace(".BO","")
        rc_data.append({"Ticker": short, "Type": "Capital weight", "Value": active_weights[i]*100.0})
        rc_data.append({"Ticker": short, "Type": "Risk contribution", "Value": pcr_dict[t]*100.0})
    rc_df = pd.DataFrame(rc_data)

    fig_rc = px.bar(
        rc_df, x="Ticker", y="Value", color="Type", barmode="group",
        color_discrete_map={"Capital weight": "#0ea5e9", "Risk contribution": "#f43f5e"}
    )
    fig_rc.update_layout(
        paper_bgcolor="#0b0f19", plot_bgcolor="#0f172a", font_color="#cbd5e1",
        margin=dict(t=20,b=20,l=20,r=20),
        yaxis_title="%",
        legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_rc, use_container_width=True)

    # ── Plain English metric guide ──
    with st.expander("What do all these numbers actually mean?"):
        st.markdown(r"""
**Volatility (σ)** — How much your portfolio's daily value bounces around, expressed annually.
High volatility means bigger swings in both directions. Calculated as the standard deviation
of daily returns × √252 (trading days per year).

**Sharpe Ratio** — How much extra return you're getting for each unit of risk you're taking.
A Sharpe of 1.0 is generally considered decent. Below 0.5 means you're not being well-compensated
for the risk. The formula is: (Portfolio Return − Risk-Free Rate) ÷ Volatility.

**Value at Risk (VaR, 95%)** — On a typical bad day (the worst ~5% of days historically),
your portfolio would lose at least this much. It's a floor, not an average.

**CVaR / Expected Shortfall** — When things go beyond that VaR threshold — on the really bad days —
this is how much you'd typically lose on average. Always worse than VaR.

**Max Drawdown** — The biggest fall from a previous peak, before recovering. If your portfolio
hit ₹150 then fell to ₹110 before recovering, that's a -26.7% drawdown.

**Beta** — How closely your portfolio tracks the NIFTY 50. Beta 1.2 means if NIFTY drops 10%,
you'd historically expect to drop ~12%. Beta below 1.0 means your portfolio is more defensive.

**HHI (Herfindahl-Hirschman Index)** — A concentration score. Sum of squared weights.
With 5 equal stocks it's 0.20. The closer to 1.0, the more concentrated in a single name.

**Diversification Ratio** — How much correlation between your stocks is reducing overall risk.
A ratio of 1.5 means correlations are cutting your risk by ~33% vs. holding each stock individually.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_recom:
    st.markdown("### Suggested rebalancing")
    st.markdown(
        "The suggested weights below come from a **minimum-variance optimisation** — a mathematical "
        "technique that finds the combination of your existing stocks that historically produced the "
        "lowest possible volatility. It uses Ledoit-Wolf covariance shrinkage to avoid overfitting to "
        "noisy short-term correlations, and constrains each stock to a maximum of "
        f"**{max_asset_cap*100:.0f}%**."
    )

    if opt_fallback:
        st.warning(f"⚠️ {opt_msg}")
    else:
        st.success(f"Optimisation converged. {opt_msg}")

    # ── Comparison KPIs ──
    cc1, cc2, cc3, cc4 = st.columns(4)

    vol_chg = (rec_ann_vol - ann_vol) * 100
    with cc1:
        color = "#10b981" if vol_chg <= 0 else "#f43f5e"
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Volatility</div>
                <div class='kpi-value' style='color:{color};'>{rec_ann_vol*100:.1f}%</div>
                <div class='kpi-sub'>Currently {ann_vol*100:.1f}% ({vol_chg:+.1f}pp)</div>
            </div>""", unsafe_allow_html=True)

    with cc2:
        shp_chg = rec_sharpe - sharpe
        color2 = "#10b981" if shp_chg >= 0 else "#f43f5e"
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Sharpe Ratio</div>
                <div class='kpi-value' style='color:{color2};'>{rec_sharpe:.2f}</div>
                <div class='kpi-sub'>Currently {sharpe:.2f} ({shp_chg:+.2f})</div>
            </div>""", unsafe_allow_html=True)

    with cc3:
        dd_chg = (rec_max_dd - max_dd) * 100
        color3 = "#10b981" if dd_chg <= 0 else "#f43f5e"
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Max Drawdown</div>
                <div class='kpi-value' style='color:{color3};'>-{rec_max_dd*100:.1f}%</div>
                <div class='kpi-sub'>Currently -{max_dd*100:.1f}% ({dd_chg:+.1f}pp)</div>
            </div>""", unsafe_allow_html=True)

    with cc4:
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Rebalancing cost</div>
                <div class='kpi-value'>{turnover_info['turnover']*100:.1f}%</div>
                <div class='kpi-sub'>~₹{turnover_info['cost_inr']:,.0f} at 15 bps one-way</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── Comparison table + bar chart ──
    col_tbl, col_chart = st.columns(2)

    with col_tbl:
        st.markdown("#### Weight changes")
        comp_rows = []
        for i, t in enumerate(active_tickers):
            short = t.replace(".NS","").replace(".BO","")
            orig_w = active_weights[i]
            new_w  = rec_weights[i]
            comp_rows.append({
                "Stock":     short,
                "Current":   f"{orig_w*100:.1f}%",
                "Suggested": f"{new_w*100:.1f}%",
                "Change":    f"{(new_w-orig_w)*100:+.1f}pp",
                "Target ₹":  f"₹{portfolio_value*new_w:,.0f}"
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    with col_chart:
        st.markdown("#### Visual comparison")
        shift_data = []
        for i, t in enumerate(active_tickers):
            short = t.replace(".NS","").replace(".BO","")
            shift_data.append({"Stock": short, "Portfolio": "Current",   "Weight": active_weights[i]*100})
            shift_data.append({"Stock": short, "Portfolio": "Suggested", "Weight": rec_weights[i]*100})
        shift_df = pd.DataFrame(shift_data)

        fig_shift = px.bar(
            shift_df, x="Stock", y="Weight", color="Portfolio", barmode="group",
            color_discrete_map={"Current": "#475569", "Suggested": "#14b8a6"}
        )
        fig_shift.update_layout(
            paper_bgcolor="#0b0f19", plot_bgcolor="#0f172a", font_color="#cbd5e1",
            margin=dict(t=20,b=20,l=20,r=20),
            yaxis_title="Weight (%)",
            legend=dict(orientation="h", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_shift, use_container_width=True)

    # Download suggested weights
    dl_df = pd.DataFrame({
        "Ticker":   active_tickers,
        "Current%": [round(w*100,2) for w in active_weights],
        "Suggested%": [round(w*100,2) for w in rec_weights],
        "Target_INR": [round(portfolio_value*w,0) for w in rec_weights]
    })
    st.download_button(
        "Download suggested allocation as CSV",
        data=dl_df.to_csv(index=False),
        file_name="quantech_suggested_weights.csv",
        mime="text/csv"
    )

    st.caption(
        "The suggested weights are derived purely from historical return covariances. "
        "They don't factor in analyst forecasts, fundamentals, or future market conditions. "
        "Past covariance patterns don't guarantee future results."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    st.markdown("### Historical performance comparison")
    st.caption(
        "Replays all four strategies over the same historical window. The suggested (min-var) "
        "portfolio has the one-time 15 bps rebalancing cost deducted at the start. "
        "No ongoing rebalancing is modelled — this is a buy-and-hold replay."
    )

    wealth_df, dd_df = run_portfolio_backtest(
        returns_df, bm_returns, active_weights, rec_weights,
        portfolio_value=portfolio_value, cost_bps=15.0
    )

    COLORS = {
        "Original Allocation":    "#0ea5e9",
        "Recommended (Min-Var)":  "#10b981",
        "Equal-Weight":           "#f59e0b",
        "Benchmark (NIFTY 50)":   "#64748b"
    }

    fig_wealth = go.Figure()
    for col in wealth_df.columns:
        fig_wealth.add_trace(go.Scatter(
            x=wealth_df.index, y=wealth_df[col],
            mode="lines", name=col,
            line=dict(color=COLORS.get(col,"#fff"), width=2.2 if "Recommended" in col else 1.8)
        ))
    fig_wealth.update_layout(
        title="Growth of ₹100 invested",
        xaxis_title="Date", yaxis_title="Portfolio value (₹)",
        paper_bgcolor="#0b0f19", plot_bgcolor="#0f172a", font_color="#cbd5e1",
        hovermode="x unified", margin=dict(t=40,b=20,l=20,r=20),
        legend=dict(orientation="h", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_wealth, use_container_width=True)

    st.markdown("#### Drawdown over time")
    fig_dd = go.Figure()
    for col in dd_df.columns:
        fig_dd.add_trace(go.Scatter(
            x=dd_df.index, y=dd_df[col],
            mode="lines", name=col,
            line=dict(color=COLORS.get(col,"#fff"), width=1.6),
            fill="tozeroy", fillcolor=COLORS.get(col,"#fff").replace("#","rgba(") + ",0.07)"
            if False else None  # clean fill disabled — too noisy with 4 series
        ))
    fig_dd.update_layout(
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        paper_bgcolor="#0b0f19", plot_bgcolor="#0f172a", font_color="#cbd5e1",
        hovermode="x unified", margin=dict(t=20,b=20,l=20,r=20),
        legend=dict(orientation="h", y=1.02, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Strategy summary table with Sharpe added ──
    st.markdown("#### Summary")
    summary_rows = []
    for col in wealth_df.columns:
        end_val  = wealth_df[col].iloc[-1]
        cagr_s   = ((end_val/100.0) ** (252.0/len(wealth_df))) - 1.0
        mdd_s    = abs(dd_df[col].min())
        # Get daily returns for Sharpe
        raw_rets = wealth_df[col].pct_change().dropna()
        vol_s    = raw_rets.std(ddof=1) * np.sqrt(252)
        shp_s    = (cagr_s - risk_free_rate) / vol_s if vol_s > 0 else 0.0
        summary_rows.append({
            "Strategy":          col,
            "Final ₹ (from ₹100)": f"₹{end_val:.2f}",
            "CAGR":              f"{cagr_s*100:.1f}%",
            "Volatility":        f"{vol_s*100:.1f}%",
            "Sharpe":            f"{shp_s:.2f}",
            "Max Drawdown":      f"-{mdd_s:.1f}%"
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — CRASH SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

with tab_stress:
    st.markdown("### Market crash simulator")
    st.markdown(
        "Estimates what would happen to your portfolio if the NIFTY 50 dropped by a specific amount "
        "in a single trading session. Each stock's loss is estimated using its historical beta "
        "— so a stock with beta 1.5 would fall roughly 1.5× as much as the index."
    )
    st.caption("⚠️ This is a simplified linear model. Real crashes involve liquidity gaps, circuit breakers, and non-linear panic selling — actual losses could be worse or different in character.")

    crash_pct = st.slider(
        "NIFTY 50 shock (%)",
        min_value=-50, max_value=-5, value=-20, step=1,
        help="Drag left to simulate a more severe crash."
    )
    shock_fraction = crash_pct / 100.0

    stress = execute_crash_stress_test(
        returns_df, bm_returns, active_weights, rec_weights,
        shock_pct=shock_fraction, portfolio_value=portfolio_value
    )

    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Your portfolio — estimated loss</div>
                <div class='kpi-value' style='color:#f43f5e;'>{stress['orig_loss_pct']*100:.1f}%</div>
                <div class='kpi-sub'>≈ −₹{stress['orig_loss_inr']:,.0f} · Beta {stress['beta_orig']:.2f}</div>
            </div>""", unsafe_allow_html=True)

    with sc2:
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Suggested portfolio — estimated loss</div>
                <div class='kpi-value' style='color:#f59e0b;'>{stress['rec_loss_pct']*100:.1f}%</div>
                <div class='kpi-sub'>≈ −₹{stress['rec_loss_inr']:,.0f} · Beta {stress['beta_rec']:.2f}</div>
            </div>""", unsafe_allow_html=True)

    with sc3:
        saved = stress["inr_saved"]
        color = "#10b981" if saved >= 0 else "#94a3b8"
        st.markdown(
            f"""<div class='kpi-container'>
                <div class='kpi-label'>Capital preserved (vs. current)</div>
                <div class='kpi-value' style='color:{color};'>₹{saved:,.0f}</div>
                <div class='kpi-sub'>Difference in estimated crash loss</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    st.markdown(f"#### Stock-by-stock impact under a {crash_pct}% NIFTY drop")
    st.dataframe(pd.DataFrame(stress["asset_impacts"]), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class='qt-footer'>
        <b>For educational use only.</b> QUANTECH is a quantitative analysis tool, not a SEBI-registered
        investment adviser. All metrics are historical estimates computed from publicly available price data
        via Yahoo Finance. Historical performance does not guarantee future results. Nothing on this platform
        constitutes financial advice or a recommendation to buy or sell any security.
        Data source: Yahoo Finance (yfinance) · Risk-free rate: RBI 91-day T-bill proxy · Transaction cost assumption: 15 bps one-way.
    </div>
    """,
    unsafe_allow_html=True
)
