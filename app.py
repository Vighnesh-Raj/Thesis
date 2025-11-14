"""Streamlit CVaR hedging cockpit (simplified).

This app keeps the original thesis idea—hedging a spot SPY position with
listed options—while trimming the code to a single, readable module.  The app
fetches live market context, lets the user edit a small option board, and
solves a CVaR optimization via linear programming over bootstrap scenarios.
"""
from __future__ import annotations

import math
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linprog
import yfinance as yf

CONTRACT_SIZE = 100
DEFAULT_SPOT_SYMBOL = "SPY"
DEFAULT_VOL_SYMBOL = "^VIX"
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"
CACHE_TTL = 300  # seconds


@dataclass
class ScenarioResults:
    prices: pd.Series
    baseline_pnl: pd.Series
    option_pnl: Dict[str, pd.Series]

    @property
    def combined(self) -> pd.DataFrame:
        df = pd.DataFrame({"Price": self.prices, "Baseline PnL": self.baseline_pnl})
        for name, pnl in self.option_pnl.items():
            df[f"{name} PnL"] = pnl
        return df


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_reference_data(
    spot_symbol: str = DEFAULT_SPOT_SYMBOL,
    vol_symbol: str = DEFAULT_VOL_SYMBOL,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return OHLCV data for SPY and VIX."""
    spot = yf.download(spot_symbol, period=period, interval=interval, auto_adjust=True)
    vol = yf.download(vol_symbol, period=period, interval=interval, auto_adjust=False)
    for frame in (spot, vol):
        if isinstance(frame.index, pd.DatetimeIndex):
            frame.index = frame.index.tz_localize(None)
    return spot, vol


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_intraday_quotes(
    symbols: Iterable[str] = (DEFAULT_SPOT_SYMBOL, DEFAULT_VOL_SYMBOL),
    period: str = "1d",
    interval: str = "1m",
) -> pd.DataFrame:
    data = yf.download(list(symbols), period=period, interval=interval, auto_adjust=False)
    if data.empty and (period, interval) != ("5d", "5m"):
        data = yf.download(list(symbols), period="5d", interval="5m", auto_adjust=False)
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
        data.index = data.index.tz_localize(None)
    return data


def latest_price(data: pd.DataFrame, symbol: str) -> float:
    try:
        if isinstance(data.columns, pd.MultiIndex):
            return float(data.xs(symbol, level=1)["Close"].iloc[-1])
        if (symbol, "Close") in data.columns:
            return float(data[(symbol, "Close")].iloc[-1])
        if "Close" in data.columns:
            return float(data["Close"].iloc[-1])
        fallback = data.iloc[-1].dropna()
        if not fallback.empty:
            return float(fallback.iloc[-1])
    except Exception:
        return float("nan")


def _extract_close_series(data: pd.DataFrame, symbol: str) -> pd.Series:
    if data.empty:
        return pd.Series(dtype="float64")

    series: pd.Series | None
    series = None

    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", symbol) in data.columns:
            series = data[("Close", symbol)]
        elif (symbol, "Close") in data.columns:
            series = data[(symbol, "Close")]
        else:
            try:
                subset = data.xs(symbol, axis=1, level=-1)
                series = subset.get("Close")
            except Exception:
                series = None
    elif (symbol, "Close") in data.columns:
        series = data[(symbol, "Close")]
    elif "Close" in data.columns:
        series = data["Close"]
    else:
        series = data.iloc[:, 0]

    if series is None:
        return pd.Series(dtype="float64")

    series = pd.Series(series).dropna()

    if isinstance(series.index, pd.MultiIndex):
        try:
            series = series.xs("Close", level=-1)
        except Exception:
            try:
                series = series.droplevel(list(range(series.index.nlevels - 1)))
            except Exception:
                pass

    if isinstance(series.index, pd.DatetimeIndex):
        series = series.sort_index()
        if series.index.tz is not None:
            series.index = series.index.tz_convert(None)

    return series


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def sample_quotes() -> pd.DataFrame:
    now = datetime.now(timezone.utc).date()
    return pd.DataFrame(
        [
            {"Contract": f"SPY {now:%y%m%d} 500C", "Type": "Call", "Strike": 500.0, "Mid": 6.20},
            {"Contract": f"SPY {now:%y%m%d} 480P", "Type": "Put", "Strike": 480.0, "Mid": 5.10},
            {"Contract": f"SPY {now:%y%m%d} 470P", "Type": "Put", "Strike": 470.0, "Mid": 3.40},
        ]
    )


def suggest_regime(current_vix: float, history: pd.Series) -> Tuple[str, str]:
    if history.empty or math.isnan(current_vix):
        return "Normal", "Unable to determine regime from incomplete VIX data."
    q_low, q_high = history.quantile([0.3, 0.7])
    if current_vix <= q_low:
        return (
            "Calm",
            f"VIX at {current_vix:.1f} sits in the lower 30% of the past year (≤ {q_low:.1f}).",
        )
    if current_vix >= q_high:
        return (
            "Stressed",
            f"VIX at {current_vix:.1f} is above the upper 30% threshold ({q_high:.1f}).",
        )
    return (
        "Normal",
        f"VIX at {current_vix:.1f} is between {q_low:.1f} and {q_high:.1f}; treat the tape as mid-cycle.",
    )


def generate_scenarios(
    prices: pd.Series,
    n_paths: int,
    horizon_days: int,
    seed: int | None = None,
) -> pd.Series:
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Not enough history to build scenarios.")
    rng = np.random.default_rng(seed)
    samples = rng.choice(returns.values, size=(n_paths, horizon_days), replace=True)
    cumulative = (1 + samples).prod(axis=1)
    last_price = prices.iloc[-1]
    simulated = last_price * cumulative
    return pd.Series(simulated, name="Simulated Price")


def build_option_pnl(
    scenarios: pd.Series, quotes: pd.DataFrame
) -> Tuple[Dict[str, pd.Series], List[str], np.ndarray]:
    pnl: Dict[str, pd.Series] = {}
    labels: List[str] = []
    costs: List[float] = []
    for _, row in quotes.iterrows():
        if any(pd.isna(row.get(col)) for col in ("Type", "Strike", "Mid")):
            continue
        option_type = str(row["Type"]).strip().lower()
        if option_type not in {"call", "put"}:
            continue
        strike = float(row["Strike"])
        premium = float(row["Mid"]) * CONTRACT_SIZE
        name = str(row.get("Contract")) if isinstance(row.get("Contract"), str) else f"Strike {strike:.0f}"
        payouts = []
        for price in scenarios.values:
            if option_type == "call":
                intrinsic = max(price - strike, 0.0)
            else:
                intrinsic = max(strike - price, 0.0)
            payouts.append((intrinsic * CONTRACT_SIZE) - premium)
        pnl[name] = pd.Series(payouts, index=scenarios.index)
        labels.append(name)
        costs.append(premium)
    return pnl, labels, np.array(costs)


def baseline_pnl(scenarios: pd.Series, shares: int, spot: float) -> pd.Series:
    diffs = scenarios - spot
    return diffs * shares


def solve_cvar(
    base_losses: np.ndarray,
    option_matrix: np.ndarray,
    cost: np.ndarray,
    budget: float,
    alpha: float,
    max_contracts: float,
) -> Tuple[np.ndarray, float, float]:
    n_opts = option_matrix.shape[1]
    n_scen = option_matrix.shape[0]

    # Decision variables: [w_1 .. w_n, t, z_1 .. z_n]
    c = np.concatenate(
        [np.zeros(n_opts), np.array([1.0]), np.full(n_scen, 1.0 / ((1 - alpha) * n_scen))]
    )

    A_ub = []
    b_ub = []

    # Budget constraint
    budget_row = np.concatenate([cost, np.zeros(1 + n_scen)])
    A_ub.append(budget_row)
    b_ub.append(budget)

    # Scenario CVaR constraints
    for s in range(n_scen):
        row = np.zeros(n_opts + 1 + n_scen)
        row[:n_opts] = -option_matrix[s]
        row[n_opts] = -1.0  # t coefficient
        row[n_opts + 1 + s] = -1.0  # z_s coefficient
        A_ub.append(row)
        b_ub.append(-base_losses[s])

    # z_s >= 0 -> -z_s <= 0
    for s in range(n_scen):
        row = np.zeros(n_opts + 1 + n_scen)
        row[n_opts + 1 + s] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    bounds: List[Tuple[float | None, float | None]] = []
    for _ in range(n_opts):
        bounds.append((0.0, max_contracts))
    bounds.append((None, None))  # t
    bounds.extend([(0.0, None) for _ in range(n_scen)])

    result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(result.message)

    weights = result.x[:n_opts]
    t_value = result.x[n_opts]
    z_values = result.x[n_opts + 1 :]
    cvar = t_value + z_values.sum() / ((1 - alpha) * n_scen)
    var = t_value
    return weights, var, cvar


def _format_delta(series: pd.Series) -> str | None:
    if series.empty:
        return None
    first = float(series.iloc[0])
    last = float(series.iloc[-1])
    if math.isnan(first) or first == 0:
        change_value = last - first
        return f"{change_value:+.2f}"
    change_value = last - first
    change_pct = change_value / first
    return f"{change_value:+.2f} ({change_pct:+.2%})"


def _render_price_card(
    label: str,
    ticker: str,
    series: pd.Series,
    *,
    prefix: str,
) -> float:
    if series.empty:
        st.metric(f"{label} ({ticker})", "n/a")
        st.caption("Real-time data unavailable.")
        return float("nan")

    last = float(series.iloc[-1])
    delta_text = _format_delta(series)

    metric_value = f"{prefix}{last:,.2f}" if prefix else f"{last:,.2f}"
    st.metric(f"{label} ({ticker})", metric_value, delta=delta_text)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line=dict(color="#21ce99", width=2),
            hovertemplate="%{x|%H:%M} — %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=10),
        height=220,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
    )
    _plotly_stretch(fig)

    return last


def render_price_context(intraday: pd.DataFrame) -> Tuple[float, float]:
    st.subheader("Market context", divider="rainbow")
    cols = st.columns(2)

    spy_series = _extract_close_series(intraday, DEFAULT_SPOT_SYMBOL)
    vix_series = _extract_close_series(intraday, DEFAULT_VOL_SYMBOL)

    with cols[0]:
        spy_last = _render_price_card(
            label="SPDR S&P 500 ETF",
            ticker=DEFAULT_SPOT_SYMBOL,
            series=spy_series,
            prefix="$",
        )

    with cols[1]:
        vix_last = _render_price_card(
            label="CBOE Volatility Index",
            ticker="VIX",
            series=vix_series,
            prefix="",
        )

    return spy_last, vix_last


def _dataframe_stretch(df: pd.DataFrame) -> None:
    params = inspect.signature(st.dataframe).parameters
    if "width" in params:
        st.dataframe(df, width="stretch")
    else:  # pragma: no cover - compatibility
        st.dataframe(df, use_container_width=True)


def _plotly_stretch(fig: go.Figure) -> None:
    params = inspect.signature(st.plotly_chart).parameters
    if "width" in params:
        st.plotly_chart(fig, width="stretch")
    else:  # pragma: no cover - compatibility
        st.plotly_chart(fig, use_container_width=True)


def render_regime_box(vix_level: float, vix_history: pd.Series) -> None:
    regime, explanation = suggest_regime(vix_level, vix_history)
    st.subheader("Suggested regime")
    st.info(f"**{regime}** — {explanation}")
    st.write(
        "Use the suggested regime as a sanity check before launching simulations. "
        "Higher VIX regimes generally justify deeper protection and larger budgets."
    )


def render_instructions() -> None:
    st.subheader("How to use the hedger")
    st.markdown(
        """
1. Review or edit the SPY option board below. Prices use mid quotes (USD).  
2. Set your share exposure, hedging budget, and scenario assumptions.  
3. Press **Run optimization** to size each contract.  
4. Inspect the resulting trade list and the scenario distribution plot.
"""
    )


def run_app() -> None:
    st.set_page_config(page_title="SPY CVaR Hedge", layout="wide")
    st.title("SPY Hedging Workbench (Simplified)")
    st.caption("Optimize a CVaR hedge using liquid SPY options and bootstrap scenarios.")

    spot_hist, vix_hist = load_reference_data()
    intraday = load_intraday_quotes()

    spot_price, vix_price = render_price_context(intraday)
    if math.isnan(spot_price) and "Close" in spot_hist:
        spot_price = float(spot_hist["Close"].iloc[-1])
    render_regime_box(vix_price, vix_hist["Close"] if "Close" in vix_hist else vix_hist)
    render_instructions()

    st.subheader("Option quotes")
    quotes = st.data_editor(sample_quotes(), num_rows="dynamic", hide_index=True)

    st.subheader("Scenario & portfolio settings")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        shares = st.number_input("Underlying shares", value=100, step=50)
        horizon = st.slider("Scenario horizon (days)", min_value=5, max_value=60, value=21, step=1)
    with col_b:
        budget = st.number_input("Hedging budget (USD)", value=1000.0, min_value=0.0, step=100.0)
        paths = st.slider("Simulations", min_value=500, max_value=5000, value=2000, step=100)
    with col_c:
        alpha = st.slider("CVaR confidence", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
        max_contracts = st.slider("Max contracts per strike", min_value=1, max_value=20, value=5, step=1)

    st.divider()
    if st.button("Run optimization", type="primary"):
        with st.spinner("Sampling scenarios and solving CVaR..."):
            try:
                scenarios = generate_scenarios(spot_hist["Close"], paths, horizon)
                option_pnl, labels, cost = build_option_pnl(scenarios, quotes)
                base = baseline_pnl(scenarios, shares, spot_price)

                if not option_pnl:
                    st.error("Please provide at least one option contract with a valid mid price.")
                    return

                base_losses = -base.values
                option_matrix = np.column_stack([series.values for series in option_pnl.values()])

                weights, var, cvar = solve_cvar(
                    base_losses=base_losses,
                    option_matrix=option_matrix,
                    cost=cost,
                    budget=budget,
                    alpha=alpha,
                    max_contracts=float(max_contracts),
                )

            except Exception as exc:  # noqa: BLE001
                st.error(f"Optimization failed: {exc}")
                return

        st.success("Optimization complete")
        results = pd.DataFrame(
            {
                "Contract": labels,
                "Contracts": weights,
                "Notional ($)": cost * weights,
            }
        )
        results["Notional ($)"] = results["Notional ($)"].round(2)
        _dataframe_stretch(results)

        st.metric("Estimated VaR", f"${var:,.0f}")
        st.metric("Estimated CVaR", f"${cvar:,.0f}")

        combined = ScenarioResults(
            prices=scenarios,
            baseline_pnl=base,
            option_pnl=option_pnl,
        ).combined
        hedged = base.copy()
        for series, weight in zip(option_pnl.values(), weights):
            hedged = hedged + series * weight
        combined["Hedged PnL"] = hedged

        fig = go.Figure()
        fig.add_histogram(x=base, name="Unhedged", opacity=0.6)
        fig.add_histogram(x=combined["Hedged PnL"], name="Hedged", opacity=0.6)
        fig.update_layout(barmode="overlay", title="Scenario PnL distribution")
        _plotly_stretch(fig)

        st.caption(
            "The histogram compares the simulated distribution of the unhedged shares versus the optimized hedge."
        )


def main() -> None:  # pragma: no cover - entrypoint
    run_app()


if __name__ == "__main__":  # pragma: no cover
    main()
