from __future__ import annotations

import inspect
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

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

# -----------------------------------------------------------------------------
# Page & layout config
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="SPY Hedge Cockpit",
    page_icon="🛡️",
    layout="wide",
)

# Dynamic kwargs for dataframe / data_editor depending on Streamlit version
if "width" in inspect.signature(st.dataframe).parameters:
    _DATAFRAME_KWARGS = {"width": "stretch"}
else:
    _DATAFRAME_KWARGS = {"use_container_width": True}

if "width" in inspect.signature(st.data_editor).parameters:
    _DATA_EDITOR_KWARGS = {"width": "stretch"}
else:
    _DATA_EDITOR_KWARGS = {"use_container_width": True}


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
        .regime-card p {
            font-size: 2rem;
            margin-bottom: 0.35rem;
            color: #21CE99;
        }
        .regime-card .regime-reason {
            font-size: 0.95rem;
            line-height: 1.45;
            color: rgba(230,244,234,0.78);
            margin-top: 0.35rem;
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

# -----------------------------------------------------------------------------
# Data loading & helpers
# -----------------------------------------------------------------------------

REFRESH_SECONDS = 60
st_autorefresh(interval=REFRESH_SECONDS * 1000, key="intraday_autorefresh")

# Yahoo-style timeframe options for the price chart
TIMEFRAME_OPTIONS = {
    "1D": ("1d", "5m"),
    "5D": ("5d", "15m"),
    "1M": ("1mo", "60m"),
    "6M": ("6mo", "1d"),
    "1Y": ("1y", "1d"),
}


@st.cache_data(show_spinner=False)
def load_history(years_back: int = 10):
    df_reg, lo, hi = prepare_regime_data(years_back=years_back)
    return df_reg, float(lo), float(hi)


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def load_intraday(period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetch intraday data and convert timestamps to US/Eastern, then drop tz
    so the x-axis shows ET wall-clock times instead of UTC.
    """
    data = fetch_intraday_quotes(symbols=("SPY", "^VIX"), period=period, interval=interval)

    if not isinstance(data.index, pd.DatetimeIndex):
        return data

    idx = data.index
    try:
        # If index is tz-naive, assume UTC then convert; otherwise just convert.
        if idx.tz is None:
            idx = idx.tz_localize("UTC").tz_convert("US/Eastern")
        else:
            idx = idx.tz_convert("US/Eastern")

        data = data.copy()
        # Strip timezone but keep ET wall-clock
        data.index = idx.tz_localize(None)
    except Exception:
        # Fallback: at least ensure index is tz-naive
        data = data.copy()
        try:
            data.index = data.index.tz_localize(None)
        except Exception:
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


def _build_payoff_curve(
    quotes: pd.DataFrame,
    buy: np.ndarray,
    sell: np.ndarray,
    n_shares: int,
    S0: float,
) -> pd.DataFrame:
    S_grid = np.linspace(0.7 * S0, 1.3 * S0, 181)
    mult = 100.0
    ask = quotes["ask"].to_numpy(dtype=float)
    bid = quotes["bid"].to_numpy(dtype=float)
    strikes = quotes["strike"].to_numpy(dtype=float)
    kinds = quotes["kind"].tolist()

    # Initial cost of options (buy at ask, sell at bid)
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


def _build_price_sparkline(
    df: pd.DataFrame,
    column: str,
    color: str,
    fill: str,
    hover_label: str,
) -> go.Figure:
    fig = go.Figure()
    if df.empty or column not in df:
        return fig

    series = df[column].astype(float)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=series,
            mode="lines",
            line=dict(color=color, width=2.4),
            fill="tozeroy",
            fillcolor=fill,
            hovertemplate="%{x|%b %d %I:%M %p}<br>"
            f"{hover_label}: "
            "%{y:.2f}<extra></extra>",
            name=hover_label,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[df.index[-1]],
            y=[series.iloc[-1]],
            mode="markers",
            marker=dict(size=9, color="#E6F4EA", line=dict(color=color, width=2)),
            hovertemplate="%{x|%b %d %I:%M %p}<br>"
            f"{hover_label}: "
            "%{y:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        margin=dict(l=40, r=10, t=10, b=40),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#E6F4EA"),
        height=220,
        hoverlabel=dict(
            bgcolor="#02140B",
            font_color="#E6F4EA",
            bordercolor="#21CE99",
        ),
        xaxis=dict(
            showgrid=False,
            showticklabels=True,
            zeroline=False,
            title="Time (ET)",
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=True,
            zeroline=False,
            title=f"{hover_label} price",
        ),
    )
    return fig


def _build_intraday_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty:
        return fig

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
        margin=dict(l=60, r=40, t=10, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(6,18,12,0.8)",
        paper_bgcolor="rgba(6,18,12,0.0)",
        font=dict(family="Inter", color="#E6F4EA"),
        hoverlabel=dict(
            bgcolor="#02140B",
            font_color="#E6F4EA",
            bordercolor="#21CE99",
        ),
    )
    fig.update_yaxes(title_text="SPY", secondary_y=False, color=color_spy)
    fig.update_yaxes(title_text="VIX", secondary_y=True, color=color_vix)
    fig.update_xaxes(title_text="Time (ET)")

    return fig


def _render_price_card(
    label: str,
    ticker: str,
    price_display: str,
    df: pd.DataFrame,
    column: str,
    color: str,
    fill: str,
    key: str,  # kept for API compatibility
) -> None:
    if df.empty or column not in df:
        chart_html = """
            <div class='chart-unavailable'>
                <p>Intraday data temporarily unavailable.</p>
            </div>
        """
        height = 160
    else:
        fig = _build_price_sparkline(df, column, color=color, fill=fill, hover_label=ticker)
        chart_html = pio.to_html(
            fig,
            include_plotlyjs="cdn",
            full_html=False,
            config={"displayModeBar": False, "responsive": True},
        )
        height = 280

    components.html(
        f"""
        <style>
            :root {{
                color-scheme: dark;
                font-family: 'Inter', sans-serif;
            }}
            body {{
                background: transparent;
                margin: 0;
                color: #E6F4EA;
                font-family: 'Inter', sans-serif;
            }}
            .price-card-wrapper {{
                background: linear-gradient(145deg, rgba(33,206,153,0.22), rgba(3,9,6,0.68));
                border-radius: 18px;
                border: 1px solid rgba(33,206,153,0.22);
                box-shadow: 0 30px 45px -28px rgba(33,206,153,0.55);
                padding: 1.15rem 1.25rem 0.55rem;
            }}
            .price-card-header {{
                display: flex;
                flex-direction: column;
                gap: 0.3rem;
            }}
            .price-card-header .label {{
                font-size: 0.85rem;
                letter-spacing: 0.08rem;
                text-transform: uppercase;
                color: rgba(230,244,234,0.75);
            }}
            .price-card-header .value {{
                font-size: 2.35rem;
                font-weight: 700;
                color: #21CE99;
                line-height: 1.1;
            }}
            .price-card-header .ticker {{
                font-size: 0.92rem;
                color: rgba(230,244,234,0.6);
            }}
            .chart-unavailable {{
                margin-top: 0.75rem;
                padding: 0.75rem 0.85rem;
                background: rgba(3,9,6,0.65);
                border-radius: 14px;
                font-size: 0.9rem;
                color: rgba(230,244,234,0.75);
            }}
            .chart-unavailable p {{
                margin: 0;
            }}
        </style>
        <div class='price-card-wrapper'>
            <div class='price-card-header'>
                <span class='label'>{label}</span>
                <span class='value'>{price_display}</span>
                <span class='ticker'>{ticker}</span>
            </div>
            {chart_html}
        </div>
        """,
        height=height,
        scrolling=False,
    )


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


def _plot_pnl_hist(unhedged: np.ndarray, hedged: np.ndarray):
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


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------

def main() -> None:
    df_reg, lo, hi = load_history()
    latest_spy = float(df_reg["SPY"].iloc[-1])
    latest_vix = float(df_reg["VIX"].iloc[-1])
    suggested = suggested_regime_today(df_reg)

    # Base explanation from hedge_app, with small cleanups
    regime_reason_raw = build_regime_explanation(df_reg, lo, hi)
    regime_reason = (
        regime_reason_raw.replace("up -", "-")
        .replace("down -", "-")
        .replace("over the last 5 sessions", "over the last 5 trading days")
    )

    # Header
    st.title("🛡️ SPY Hedge Cockpit")
    st.markdown(
        """
        <p style='color:rgba(230,244,234,0.75); font-size:1.05rem;'>
        This app lets you design and evaluate SPY option hedges using live SPY/VIX data
        and regime-based scenario simulations. Use the tabs below to load your option
        quotes, set hedging constraints, and compare risk with and without the hedge.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # How-to card (up-front)
    st.markdown(
        """
        <div class='shadow-card' style='margin-top:0.75rem;'>
            <h4 style='margin-bottom:0.4rem;'>How to hedge with this cockpit</h4>
            <ol style='padding-left:1.2rem; color:rgba(230,244,234,0.8);'>
                <li>Glance at the live SPY/VIX context and suggested regime to anchor your view on volatility.</li>
                <li>In <em>Option Quotes</em>, paste or edit your SPY option quotes, matching the contracts you can trade.</li>
                <li>In <em>Scenario &amp; Constraints</em>, configure the scenario horizon, account permissions, and budget/position limits.</li>
                <li>In <em>Results</em>, click <strong>Simulate / Optimize</strong> to compare fractional vs. rounded hedges and review payoff &amp; CVaR.</li>
            </ol>
            <p style='margin-bottom:0; color:rgba(230,244,234,0.65);'>
                Adjust quotes or constraints and rerun as market conditions evolve.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Timeframe selector (Yahoo-style) for SPY/VIX charts ---
    timeframe = st.radio(
        "Price chart window",
        list(TIMEFRAME_OPTIONS.keys()),
        index=0,
        horizontal=True,
    )
    period, interval = TIMEFRAME_OPTIONS[timeframe]

    # Intraday / price data
    try:
        intraday_df = load_intraday(period=period, interval=interval)
    except ValueError:
        intraday_df = pd.DataFrame()

    last_refresh_display = None
    if not intraday_df.empty:
        latest_spy = float(intraday_df["SPY"].iloc[-1])
        latest_vix = float(intraday_df["^VIX"].iloc[-1])
        last_refresh = pd.Timestamp(intraday_df.index[-1])
        last_refresh_display = last_refresh.strftime("%b %d %I:%M %p ET")

    # Top cards: SPY, VIX, regime
    with st.container():
        col1, col2, col3 = st.columns([1.15, 1.15, 1], gap="large")
        with col1:
            _render_price_card(
                label="SPDR S&P 500 ETF",
                ticker="SPY",
                price_display=f"${latest_spy:,.2f}",
                df=intraday_df,
                column="SPY",
                color="#21CE99",
                fill="rgba(33,206,153,0.28)",
                key="spy-price-card",
            )
        with col2:
            _render_price_card(
                label="CBOE Volatility Index",
                ticker="VIX",
                price_display=f"{latest_vix:,.2f}",
                df=intraday_df,
                column="^VIX",
                color="#F5A623",
                fill="rgba(245,166,35,0.28)",
                key="vix-price-card",
            )
        with col3:
            st.markdown(
                f"""
                <div class='metric-card regime-card'>
                    <h3>Suggested Regime</h3>
                    <p>{suggested}</p>
                    <div class='regime-reason'>{regime_reason}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    caption = f"Regime cut-offs • LOW ≤ {lo:.2f} • MID in ({lo:.2f}, {hi:.2f}) • HIGH ≥ {hi:.2f}"
    if last_refresh_display:
        caption += f" • Price data refreshed {last_refresh_display}"
    st.caption(caption)
    st.caption("Live SPY and VIX prices update every minute while this session remains open.")

    # Combined SPY/VIX chart
    if not intraday_df.empty:
        st.plotly_chart(_build_intraday_chart(intraday_df), use_container_width=True, theme=None)
        st.caption("Live SPY/VIX quotes refresh automatically every minute.")
    else:
        st.warning("Price quotes are temporarily unavailable; retry in a moment.")

    # Explanation card
    st.markdown(
        f"""
        <div class='shadow-card' style='margin-top:1rem;'>
            <h4 style='margin-bottom:0.4rem;'>Why {suggested} today?</h4>
            <p style='margin-bottom:0; color:rgba(230,244,234,0.8);'>{regime_reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    tabs = st.tabs(["Option Quotes", "Scenario & Constraints", "Results"])

    # -------------------------------------------------------------------------
    # Tab 0: Option Quotes
    # -------------------------------------------------------------------------
    with tabs[0]:
        st.markdown("<div class='shadow-card'>", unsafe_allow_html=True)
        st.subheader("Editable Option Quotes")
        st.caption("Paste values directly or upload a CSV. Bid/ask and a single expiry are required.")

        uploaded = st.file_uploader(
            "Upload CSV with option quotes",
            type=["csv"],
            label_visibility="collapsed",
            help=(
                "Include columns for option kind (C/P), strike, bid, ask, expiry, "
                "and optionally a label column."
            ),
        )

        if uploaded is not None:
            uploaded_df = pd.read_csv(uploaded)
            working_df = uploaded_df
        else:
            working_df = _default_quotes()

        edited_df = st.data_editor(
            working_df,
            num_rows="dynamic",
            hide_index=True,
            **_DATA_EDITOR_KWARGS,
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

        # Download current quotes as CSV
        st.download_button(
            "Download current quotes as CSV",
            data=edited_df.to_csv(index=False),
            file_name="spy_quotes.csv",
            mime="text/csv",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Tab 1: Scenario & Constraints
    # -------------------------------------------------------------------------
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
                regime_options = ["LOW", "MID", "HIGH"]
                default_idx = regime_options.index(suggested) if suggested in regime_options else 1
                override = st.selectbox(
                    "Choose regime",
                    regime_options,
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
            rf_annual = st.number_input(
                "Risk-free rate (annual, %)",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.25,
                help="Used for Black–Scholes repricing of option values at the scenario horizon.",
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

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Tab 2: Results (with persistence across auto-refresh)
    # -------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("<div class='shadow-card'>", unsafe_allow_html=True)
        st.subheader("Simulation Results")

        run_clicked = st.button("🚀 Simulate / Optimize")

        def _render_results(res_obj: dict):
            """Render results from stored hedge_result (session or fresh)."""

            lp = res_obj["lp"]
            rounded = res_obj["rounded"]
            pnl = res_obj["pnl"]
            labels = res_obj["labels"]
            chosen_regime = res_obj.get("regime_used")
            pool_size = res_obj.get("pool_size")
            alpha_val = res_obj.get("alpha", 0.95)
            prefs_obj: Prefs | None = res_obj.get("prefs")
            quotes_validated = res_obj.get("quotes_validated")

            # Scenario summary line
            scen_bits = []
            if prefs_obj is not None:
                scen_bits.append(f"Horizon: {prefs_obj.horizon}")
                scen_bits.append(f"Scenarios: {prefs_obj.n_scen}")
                scen_bits.append(f"α = {alpha_val:.2f}")
                scen_bits.append(f"r_f ≈ {prefs_obj.risk_free * 100:.2f}%")
            if chosen_regime is not None:
                scen_bits.append(f"Regime: {chosen_regime}")
            if pool_size is not None:
                scen_bits.append(f"Pool: {pool_size} days")
            if scen_bits:
                st.caption(" | ".join(scen_bits))

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

            spend_lp = float(lp["spend_usd"])
            spend_rounded = float(rounded["spend"])

            st.markdown("### Optimal Allocation")
            col_lp, col_round = st.columns(2)
            with col_lp:
                st.caption("Fractional LP solution")
                st.dataframe(lp_df, **_DATAFRAME_KWARGS)
                st.metric("LP spend", _format_currency(spend_lp))
            with col_round:
                st.caption("Rounded (executable) portfolio")
                st.dataframe(rounded_df, **_DATAFRAME_KWARGS)
                st.metric("Rounded spend", _format_currency(spend_rounded))

            # Per-leg summary (quotes + net positions + approximate notional)
            if isinstance(quotes_validated, pd.DataFrame):
                leg_df = quotes_validated.copy()
                leg_df["buy"] = rounded["buy"]
                leg_df["sell"] = rounded["sell"]
                leg_df["net"] = rounded["net"]
                S0_current = float(df_reg["SPY"].iloc[-1])
                leg_df["approx_notional_usd"] = (leg_df["net"].abs() * 100.0 * S0_current).round(2)

                st.markdown("#### Per-leg summary")
                show_cols = [
                    "label",
                    "kind",
                    "strike",
                    "bid",
                    "ask",
                    "mid",
                    "buy",
                    "sell",
                    "net",
                    "approx_notional_usd",
                ]
                show_cols = [c for c in show_cols if c in leg_df.columns]
                st.dataframe(leg_df[show_cols], **_DATAFRAME_KWARGS)

            # Risk metrics
            alpha_label = f"{alpha_val:.2f}"
            var_unh = float(np.quantile(-pnl["unhedged"], alpha_val))
            var_hd = float(np.quantile(-pnl["hedged"], alpha_val))

            def cvar(series: np.ndarray, a: float) -> float:
                losses = -series
                cutoff = np.quantile(losses, a)
                return float(losses[losses >= cutoff].mean())

            cvar_unh = cvar(pnl["unhedged"], alpha_val)
            cvar_hd = cvar(pnl["hedged"], alpha_val)

            metrics_df = pd.DataFrame(
                {
                    "": ["Unhedged", "Hedged", "Improvement"],
                    f"VaR@{alpha_label}": [var_unh, var_hd, var_unh - var_hd],
                    f"CVaR@{alpha_label}": [cvar_unh, cvar_hd, cvar_unh - cvar_hd],
                }
            )
            st.markdown("### Risk Snapshot")
            st.dataframe(metrics_df, **_DATAFRAME_KWARGS)

            # Tail-loss reduction per dollar
            tail_improvement = cvar_unh - cvar_hd
            if spend_rounded != 0:
                ratio = tail_improvement / spend_rounded
                st.caption(
                    f"This hedge reduces CVaR by approximately {_format_currency(tail_improvement)} "
                    f"for an upfront cost of {_format_currency(spend_rounded)}, "
                    f"≈ {ratio:.3f} dollars of tail-loss reduction per dollar of premium."
                )
            else:
                st.caption(
                    f"This hedge changes CVaR by approximately {_format_currency(tail_improvement)} "
                    "with zero net premium outlay (zero-cost structure)."
                )
            if prefs_obj is not None:
                st.caption(
                    f"Note: CVaR is computed over bootstrapped SPY/VIX scenarios at horizon "
                    f"{prefs_obj.horizon} using an annual risk-free rate of {prefs_obj.risk_free * 100:.2f}% "
                    "and European-style Black–Scholes repricing."
                )

            # Payoff & PnL distribution
            if isinstance(quotes_validated, pd.DataFrame):
                payoff_df = _build_payoff_curve(
                    quotes_validated,
                    rounded["buy"],
                    rounded["sell"],
                    prefs_obj.n_shares if prefs_obj is not None else 0,
                    float(df_reg["SPY"].iloc[-1]),
                )
                fig_payoff = _plot_payoff_curve(payoff_df, float(df_reg["SPY"].iloc[-1]))
                fig_hist = _plot_pnl_hist(pnl["unhedged"], pnl["hedged"])

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.pyplot(fig_payoff, use_container_width=True)
                    st.caption(
                        "Payoff of SPY + hedge at option expiry across terminal SPY prices, "
                        "net of initial option cost."
                    )
                with chart_col2:
                    st.pyplot(fig_hist, use_container_width=True)
                    st.caption(
                        "Histogram of simulated portfolio P&L over the selected horizon. "
                        "Green = hedged, orange = unhedged; the vertical line at 0 marks break-even."
                    )

        if run_clicked:
            with st.spinner("Running CVaR optimization..."):
                try:
                    quotes_clean = _clean_quotes(edited_df)
                    if quotes_clean.empty:
                        raise ValueError("Add at least one complete option row before running the optimizer.")

                    quotes_validated = validate_quotes_df_bidask_strict(quotes_clean)

                    # Regime pool selection
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
                        risk_free=rf_annual / 100.0,
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

                    # Persist result in session_state so auto-refresh doesn't wipe it
                    st.session_state["hedge_result"] = {
                        "lp": lp,
                        "rounded": rounded,
                        "pnl": pnl,
                        "labels": labels,
                        "regime_used": chosen,
                        "pool_size": len(pool),
                        "quotes_validated": quotes_validated,
                        "prefs": prefs,
                        "alpha": alpha,
                    }

                    st.success(f"Regime used: {chosen} | pool size: {len(pool)} days")
                    _render_results(st.session_state["hedge_result"])

                except Exception as exc:  # pylint: disable=broad-except
                    st.session_state["hedge_result"] = None
                    st.error(f"{type(exc).__name__}: {exc}")
        else:
            # If user hasn't clicked this run, try to show last stored result
            if "hedge_result" in st.session_state and st.session_state["hedge_result"] is not None:
                st.info(
                    "Showing the most recent hedge result. "
                    "Adjust inputs and click **Simulate / Optimize** to recompute."
                )
                _render_results(st.session_state["hedge_result"])
            else:
                st.info("Configure inputs and click **Simulate / Optimize** to populate this panel.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar
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
