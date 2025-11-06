"""Streamlit UI for the SPY hedge optimizer with an editable quote grid."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit as st

from hedge_app import (
    Prefs,
    prepare_regime_data,
    select_regime_pool,
    suggested_regime_today,
    validate_quotes_df_bidask_strict,
    run_hedge_workflow,
    fetch_intraday_quotes,
    build_regime_explanation,
)

st.set_page_config(
    page_title="SPY Hedge Cockpit",
    page_icon="🛡️",
    layout="wide",
)


def _inject_robinhood_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        .stApp {
            background: radial-gradient(circle at top, #082012 0%, #030906 70%);
            color: #E6F4EA;
            font-family: 'Inter', sans-serif;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, h4, h5 {
            font-weight: 600;
            color: #E6F4EA;
        }
        .metric-card {
            background: linear-gradient(145deg, rgba(33,206,153,0.2), rgba(3,9,6,0.6));
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            border: 1px solid rgba(33,206,153,0.2);
            box-shadow: 0 25px 40px -20px rgba(33,206,153,0.4);
        }
        .metric-card h3 {
            font-size: 0.95rem;
            letter-spacing: 0.08rem;
            color: rgba(230,244,234,0.75);
            text-transform: uppercase;
        }
        .metric-card p {
            font-size: 2rem;
            margin: 0;
            font-weight: 700;
            color: #21CE99;
        }
        .shadow-card {
            background: rgba(6,18,12,0.8);
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            border: 1px solid rgba(33,206,153,0.15);
            box-shadow: 0 25px 45px -25px rgba(0,0,0,0.65);
        }
        .stDataFrame, .stTable {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(33,206,153,0.1);
        }
        .stCheckbox > label, .stRadio > label, .stSelectbox > label, .stSlider > label {
            color: rgba(230,244,234,0.85) !important;
            font-weight: 500;
        }
        .stSlider [data-baseweb="slider"] div {
            background-color: rgba(33,206,153,0.35);
        }
        .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
            background-color: rgba(33,206,153,0.35);
        }
        .st-emotion-cache-1y4p8pa, .st-emotion-cache-10trblm {
            color: rgba(230,244,234,0.9) !important;
        }
        .stButton button {
            background: linear-gradient(135deg, #21CE99, #16A57A);
            border: none;
            color: #02140B;
            font-weight: 700;
            letter-spacing: 0.02em;
            border-radius: 999px;
            padding: 0.65rem 1.5rem;
            box-shadow: 0 20px 35px -15px rgba(33,206,153,0.5);
        }
        .stButton button:hover {
            box-shadow: 0 25px 40px -12px rgba(33,206,153,0.65);
            transform: translateY(-1px);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


_inject_robinhood_styles()


REFRESH_SECONDS = 60
st_autorefresh(interval=REFRESH_SECONDS * 1000, key="intraday_autorefresh")


@st.cache_data(show_spinner=False)
def load_history(years_back: int = 10):
    df_reg, lo, hi = prepare_regime_data(years_back=years_back)
    return df_reg, float(lo), float(hi)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_intraday(period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    data = fetch_intraday_quotes(symbols=("SPY", "^VIX"), period=period, interval=interval)
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.copy()
        try:
            data.index = data.index.tz_localize(None)
        except TypeError:
            pass
    return data


def _default_quotes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "kind": ["C", "C", "P", "P"],
            "strike": [665, 673, 670, 677],
            "bid": [11.39, 5.72, 4.51, 7.45],
            "ask": [11.44, 5.76, 4.53, 7.49],
            "expiry": ["2025-10-31"] * 4,
            "label": ["C665", "C673", "P670", "P677"],
        }
    )


def _format_currency(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _build_payoff_curve(quotes: pd.DataFrame, buy, sell, n_shares: int, S0: float) -> pd.DataFrame:
    S_grid = np.linspace(0.7 * S0, 1.3 * S0, 181)
    mult = 100.0
    ask = quotes["ask"].to_numpy(float)
    bid = quotes["bid"].to_numpy(float)
    strikes = quotes["strike"].to_numpy(float)
    kinds = quotes["kind"].tolist()
    init_cost = float((ask * mult) @ buy + (-bid * mult) @ sell)

    payoff = []
    for S in S_grid:
        value = 0.0
        for j, kind in enumerate(kinds):
            intrinsic = max(S - strikes[j], 0.0) if kind == "C" else max(strikes[j] - S, 0.0)
            value += buy[j] * intrinsic * mult - sell[j] * intrinsic * mult
        payoff.append(value)
    shares_leg = n_shares * (S_grid - S0)
    total = shares_leg + np.array(payoff) - init_cost
    return pd.DataFrame({"SPY": S_grid, "Total": total})


def _build_intraday_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    color_spy = "#21CE99"
    color_vix = "#F5A623"

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["SPY"],
            name="SPY",
            line=dict(color=color_spy, width=2),
            hovertemplate="%{x|%b %d %H:%M}<br>SPY: $%{y:.2f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["^VIX"],
            name="VIX",
            line=dict(color=color_vix, width=2, dash="dot"),
            hovertemplate="%{x|%b %d %H:%M}<br>VIX: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
        plot_bgcolor="rgba(6,18,12,0.8)",
        paper_bgcolor="rgba(6,18,12,0.0)",
        font=dict(family="Inter", color="#E6F4EA"),
    )
    fig.update_yaxes(title_text="SPY", secondary_y=False, color=color_spy)
    fig.update_yaxes(title_text="VIX", secondary_y=True, color=color_vix)
    fig.update_xaxes(title_text="Intraday (Eastern Time)")
    return fig


def _plot_payoff_curve(data: pd.DataFrame, S0: float):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(data["SPY"], data["Total"], color="#21CE99", linewidth=2.2)
    ax.axhline(0, color="#0B291B", linestyle="--", linewidth=1)
    ax.axvline(S0, color="#0B291B", linestyle=":", linewidth=1)
    ax.set_title("Payoff at Expiry")
    ax.set_xlabel("SPY at Expiry")
    ax.set_ylabel("Portfolio P&L (USD)")
    fig.tight_layout()
    return fig


def _plot_pnl_hist(unhedged, hedged):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bins = max(30, int(math.sqrt(len(unhedged))))
    ax.hist(unhedged, bins=bins, alpha=0.45, color="#FF7F50", label="Unhedged")
    ax.hist(hedged, bins=bins, alpha=0.55, color="#21CE99", label="Hedged")
    ax.axvline(0, color="#0B291B", linestyle="--", linewidth=1)
    ax.legend()
    ax.set_title("Scenario P&L Distribution")
    ax.set_xlabel("P&L (USD)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def _clean_quotes(df: pd.DataFrame) -> pd.DataFrame:
    required = ["kind", "strike", "bid", "ask", "expiry"]
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=required, how="any")
    cleaned["kind"] = cleaned["kind"].astype(str).str.upper().str.strip()
    cleaned["expiry"] = cleaned["expiry"].astype(str).str.strip()
    for col in ["strike", "bid", "ask"]:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.dropna(subset=["strike", "bid", "ask"], how="any")
    return cleaned


def main() -> None:
    df_reg, lo, hi = load_history()
    latest_spy = float(df_reg["SPY"].iloc[-1])
    latest_vix = float(df_reg["VIX"].iloc[-1])
    suggested = suggested_regime_today(df_reg)
    regime_reason = build_regime_explanation(df_reg, lo, hi)

    st.title("🛡️ SPY Hedge Cockpit")
    st.markdown(
        """
        <p style='color:rgba(230,244,234,0.75); font-size:1.05rem;'>
        Upload or edit option quotes, configure hedging preferences, and compare
        Robinhood-inspired risk metrics before sharing a live URL on Streamlit Community Cloud.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        col1, col2, col3 = st.columns([1.15, 1.15, 1])
        with col1:
            st.markdown(
                f"<div class='metric-card'><h3>SPY Last</h3><p>${latest_spy:,.2f}</p></div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"<div class='metric-card'><h3>VIX Last</h3><p>{latest_vix:,.2f}</p></div>",
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f"<div class='metric-card'><h3>Suggested Regime</h3><p>{suggested}</p></div>",
                unsafe_allow_html=True,
            )

    st.caption(
        f"Regime cut-offs • LOW ≤ {lo:.2f} • MID in ({lo:.2f}, {hi:.2f}) • HIGH ≥ {hi:.2f}"
    )

    try:
        intraday_df = load_intraday()
    except ValueError:
        intraday_df = pd.DataFrame()

    if not intraday_df.empty:
        st.plotly_chart(_build_intraday_chart(intraday_df), use_container_width=True, theme=None)
        st.caption("Live SPY/VIX quotes refresh automatically every minute.")
    else:
        st.warning("Intraday quotes are temporarily unavailable; retry in a moment.")

    st.markdown(
        f"""
        <div class='shadow-card' style='margin-top:1rem;'>
            <h4 style='margin-bottom:0.4rem;'>Why {suggested} today?</h4>
            <p style='margin-bottom:0; color:rgba(230,244,234,0.8);'>{regime_reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='shadow-card' style='margin-top:1rem;'>
            <h4 style='margin-bottom:0.4rem;'>How to hedge with this cockpit</h4>
            <ol style='padding-left:1.2rem; color:rgba(230,244,234,0.8);'>
                <li>Use the live SPY/VIX context above to anchor your view on volatility and choose (or override) the regime.</li>
                <li>Paste or edit your SPY option quotes in the table, matching the contracts you can trade.</li>
                <li>Configure scenario assumptions and portfolio permissions on the next tab using the hover tips for guidance.</li>
                <li>Launch <strong>Simulate / Optimize</strong> to compare fractional vs. rounded hedges and review payoff &amp; CVaR.</li>
            </ol>
            <p style='margin-bottom:0; color:rgba(230,244,234,0.65);'>Adjust quotes or constraints and rerun as market conditions evolve.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    tabs = st.tabs(["Option Quotes", "Scenario & Constraints", "Results"])

    with tabs[0]:
        st.markdown("<div class='shadow-card'>", unsafe_allow_html=True)
        st.subheader("Editable Option Quotes")
        st.caption("Paste values directly or upload a CSV. Bid/ask and a single expiry are required.")

        uploaded = st.file_uploader(
            "Upload CSV with option quotes",
            type=["csv"],
            label_visibility="collapsed",
            help="Include columns for option kind (C/P), strike, bid, ask, expiry, and optionally a label column.",
        )
        uploaded = st.file_uploader("Upload CSV with option quotes", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)
            working_df = uploaded_df
        else:
            working_df = _default_quotes()

        edited_df = st.data_editor(
            working_df,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "kind": st.column_config.SelectboxColumn(
                    "Kind",
                    options=["C", "P"],
                    help="Call (C) or Put (P)",
                    width="small",
                ),
                "strike": st.column_config.NumberColumn("Strike", format="%d", step=1),
                "bid": st.column_config.NumberColumn("Bid", format="%.2f", step=0.01),
                "ask": st.column_config.NumberColumn("Ask", format="%.2f", step=0.01),
                "expiry": st.column_config.TextColumn("Expiry (YYYY-MM-DD)", width="medium"),
                "label": st.column_config.TextColumn(
                    "Label",
                    help="Optional nickname for the option leg",
                ),
            },
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='shadow-card'>", unsafe_allow_html=True)
        st.subheader("Scenario Engine & Permissions")

        scenario_col, prefs_col = st.columns(2, gap="large")
        with scenario_col:
            regime_mode = st.radio(
                "Regime Selection",
                options=("Auto (use suggested)", "Manual"),
                horizontal=True,
                help="Follow the app's suggested volatility regime or choose your own bucket for stress testing.",
            )
            override = None
            if regime_mode == "Manual":
                options = ["LOW", "MID", "HIGH"]
                default_idx = options.index(suggested) if suggested in options else 1
                override = st.selectbox(
                    "Choose regime",
                    options,
                    index=default_idx,
                    help="Pick the historical volatility bucket used to bootstrap SPY/VIX scenarios.",
                )
            horizon = st.selectbox(
                "Scenario horizon",
                ["1w", "1m", "3m", "1y"],
                index=0,
                help="Length of each bootstrapped path driving hedge outcomes (weekly to yearly horizons).",
            )
            n_scen = st.slider(
                "Simulations",
                1000,
                10000,
                2000,
                step=500,
                help="Number of Monte Carlo draws sampled from the chosen regime history.",
            )
            alpha = st.slider(
                "Tail level (α)",
                0.80,
                0.99,
                0.95,
                step=0.01,
                help="Confidence level for CVaR: lower α focuses on extreme losses, higher α on more typical drawdowns.",
            )

        with prefs_col:
            n_shares = st.number_input(
                "SPY shares to hedge",
                min_value=0,
                value=20,
                step=1,
                help="Underlying shares (or delta-equivalent exposure) that require protection.",
            )
            retail_mode = st.toggle(
                "Retail mode (buy-only)",
                value=True,
                help="Restrict hedges to long option legs for retail account compliance.",
            )
            allow_selling = st.toggle(
                "Allow selling legs",
                value=False,
                help="Enable short option legs when running institutional or margin strategies.",
            )
            zero_cost = st.toggle(
                "Enforce zero-cost structure",
                value=False,
                help="Match purchased and sold premium so the hedge has no upfront cost before rounding.",
            )
            budget_usd = st.number_input(
                "Premium budget (USD)",
                min_value=0.0,
                value=200.0,
                step=50.0,
                help="Cap the cash outlay allocated to buying protection.",
            )
            allow_net_credit = st.toggle(
                "Allow net credit when not zero-cost",
                value=False,
                help="Permit receiving net credit if zero-cost is off; useful for collar-style overlays.",
            )
            max_buy = st.slider(
                "Max buy contracts per leg",
                0.0,
                200.0,
                10.0,
                step=1.0,
                help="Upper bound for long contracts on any single option leg.",
            )
            max_sell = st.slider(
                "Max sell contracts per leg",
                0.0,
                200.0,
                10.0,
                step=1.0,
                help="Upper bound for short contracts on any single option leg.",
            )
            integer_round = st.toggle(
                "Round to whole contracts",
                value=True,
                help="Force rounded trades to whole-number contracts (disable for fractional overlays).",
            )
            step = st.select_slider(
                "Rounding step",
                options=[1.0, 0.5, 0.1],
                value=1.0,
                help="Smallest contract increment allowed when rounding the optimized solution.",
            )
            keep_budget = st.toggle(
                "Budget enforced after rounding",
                value=True,
                help="Reapply the premium budget to the rounded trade list to avoid overspending.",
            )
            zc_tol = st.number_input(
                "Zero-cost tolerance after rounding (USD)",
                min_value=0.0,
                value=5.0,
                step=1.0,
                help="Acceptable drift from the zero-cost target once trades are rounded.",
            )
                override = st.selectbox("Choose regime", options, index=default_idx)
            horizon = st.selectbox("Scenario horizon", ["1w", "1m", "3m", "1y"], index=0)
            n_scen = st.slider("Simulations", 1000, 10000, 2000, step=500)
            alpha = st.slider("Tail level (α)", 0.80, 0.99, 0.95, step=0.01)

        with prefs_col:
            n_shares = st.number_input("SPY shares to hedge", min_value=0, value=20, step=1)
            retail_mode = st.toggle("Retail mode (buy-only)", value=True)
            allow_selling = st.toggle("Allow selling legs", value=False)
            zero_cost = st.toggle("Enforce zero-cost structure", value=False)
            budget_usd = st.number_input("Premium budget (USD)", min_value=0.0, value=200.0, step=50.0)
            allow_net_credit = st.toggle("Allow net credit when not zero-cost", value=False)
            max_buy = st.slider("Max buy contracts per leg", 0.0, 200.0, 10.0, step=1.0)
            max_sell = st.slider("Max sell contracts per leg", 0.0, 200.0, 10.0, step=1.0)
            integer_round = st.toggle("Round to whole contracts", value=True)
            step = st.select_slider("Rounding step", options=[1.0, 0.5, 0.1], value=1.0)
            keep_budget = st.toggle("Budget enforced after rounding", value=True)
            zc_tol = st.number_input("Zero-cost tolerance after rounding (USD)", min_value=0.0, value=5.0, step=1.0)

        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='shadow-card'>", unsafe_allow_html=True)
        st.subheader("Simulation Results")
        run_clicked = st.button("🚀 Simulate / Optimize", use_container_width=False)

        if run_clicked:
            with st.spinner("Running CVaR optimization..."):
                try:
                    quotes_clean = _clean_quotes(edited_df)
                    if quotes_clean.empty:
                        raise ValueError("Add at least one complete option row before running the optimizer.")

                    quotes_validated = validate_quotes_df_bidask_strict(quotes_clean)

                    if regime_mode.startswith("Auto"):
                        chosen, pool = select_regime_pool(df_reg, mode="auto")
                    else:
                        chosen, pool = select_regime_pool(df_reg, mode="manual", override=override)

                    prefs = Prefs(
                        horizon=horizon,
                        n_scen=n_scen,
                        seed=123,
                        alpha=alpha,
                        n_shares=int(n_shares),
                        risk_free=0.00,
                        retail_mode=retail_mode,
                        allow_selling=allow_selling,
                        zero_cost=zero_cost,
                        allow_net_credit=allow_net_credit,
                        budget_usd=float(budget_usd),
                        max_buy_contracts=float(max_buy),
                        max_sell_contracts=float(max_sell),
                        integer_round=integer_round,
                        step=float(step),
                        budget_enforced_after_rounding=keep_budget,
                        zero_cost_tolerance=float(zc_tol),
                        make_plots=False,
                    )

                    result = run_hedge_workflow(quotes_validated, df_reg, pool, prefs)

                    lp = result["lp_result"]
                    rounded = result["rounded"]
                    pnl = result["pnl"]
                    labels = result["labels"]

                    st.success(f"Regime used: {chosen} | pool size: {len(pool)} days")

                    lp_df = pd.DataFrame(
                        {
                            "label": labels,
                            "buy": np.round(lp["weights"]["buy"], 4),
                            "sell": np.round(lp["weights"]["sell"], 4),
                            "net": np.round(lp["weights"]["net"], 4),
                        }
                    )
                    rounded_df = pd.DataFrame(
                        {
                            "label": labels,
                            "buy": rounded["buy"],
                            "sell": rounded["sell"],
                            "net": rounded["net"],
                        }
                    )

                    spend_lp = lp["spend_usd"]
                    spend_rounded = rounded["spend"]

                    st.markdown("### Optimal Allocation")
                    col_lp, col_round = st.columns(2)
                    with col_lp:
                        st.caption("Fractional LP solution")
                        st.dataframe(lp_df, use_container_width=True)
                        st.metric("LP spend", _format_currency(spend_lp))
                    with col_round:
                        st.caption("Rounded (executable) portfolio")
                        st.dataframe(rounded_df, use_container_width=True)
                        st.metric("Rounded spend", _format_currency(spend_rounded))

                    alpha_label = f"{alpha:.2f}"
                    var_unh = np.quantile(-pnl["unhedged"], alpha)
                    var_hd = np.quantile(-pnl["hedged"], alpha)
                    def cvar(series):
                        losses = -series
                        cutoff = np.quantile(losses, alpha)
                        return float(losses[losses >= cutoff].mean())
                    cvar_unh = cvar(pnl["unhedged"])
                    cvar_hd = cvar(pnl["hedged"])
                    metrics_df = pd.DataFrame(
                        {
                            "": ["Unhedged", "Hedged", "Improvement"],
                            f"VaR@{alpha_label}": [var_unh, var_hd, var_unh - var_hd],
                            f"CVaR@{alpha_label}": [cvar_unh, cvar_hd, cvar_unh - cvar_hd],
                        }
                    )
                    st.markdown("### Risk Snapshot")
                    st.dataframe(metrics_df, use_container_width=True)

                    payoff_df = _build_payoff_curve(
                        quotes_validated,
                        rounded["buy"],
                        rounded["sell"],
                        prefs.n_shares,
                        float(df_reg["SPY"].iloc[-1]),
                    )
                    fig_payoff = _plot_payoff_curve(payoff_df, float(df_reg["SPY"].iloc[-1]))
                    fig_hist = _plot_pnl_hist(pnl["unhedged"], pnl["hedged"])

                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.pyplot(fig_payoff, use_container_width=True)
                    with chart_col2:
                        st.pyplot(fig_hist, use_container_width=True)

                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"{type(exc).__name__}: {exc}")
        else:
            st.info("Configure inputs and click **Simulate / Optimize** to populate this panel.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.title("Need a public link?")
    st.sidebar.markdown(
        "Deploy to **Streamlit Community Cloud** (free) to generate a shareable URL."
        " Push this repo to GitHub, sign into streamlit.io, and point a new app at `app.py`."
        " Details in `DEPLOY.md`."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Made with ❤️ for interactive hedging research.")


if __name__ == "__main__":
    main()
