"""Data loading and regime preparation utilities for the hedge optimizer."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

WINDOW_DAYS = 20


def get_dynamic_date_range(years_back: int = 10) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Return timezone-aware start/end dates covering ``years_back`` years."""
    today_ny = pd.Timestamp.today(tz="America/New_York").normalize()
    start = today_ny - pd.DateOffset(years=years_back)
    return start.tz_localize(None), today_ny.tz_localize(None)


def _safe_close(df: pd.DataFrame, ticker_col_hint: str) -> pd.Series:
    """Robustly extract a close column from yfinance output."""
    if isinstance(df.columns, pd.MultiIndex):
        for key in [("Close", ticker_col_hint), ("Adj Close", ticker_col_hint)]:
            if key in df.columns:
                return df[key]
        if "Close" in df.columns.get_level_values(0):
            return df.xs("Close", axis=1, level=0).iloc[:, 0]
    for key in ["Close", "Adj Close"]:
        if key in df.columns:
            return df[key]
    raise KeyError("Could not locate a Close column in the downloaded data.")


def load_market_data_simple(
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    window_days: int = WINDOW_DAYS,
) -> pd.DataFrame:
    """Download SPY/VIX history and compute helper columns."""
    spy = yf.download("SPY", start=start_dt, end=end_dt, progress=False, auto_adjust=True)
    vix = yf.download("^VIX", start=start_dt, end=end_dt, progress=False, auto_adjust=False)

    spy_close = _safe_close(spy, "SPY").rename("SPY")
    vix_close = _safe_close(vix, "^VIX").rename("VIX")

    df = pd.concat([spy_close, vix_close], axis=1).dropna()
    df["ret_1d"] = df["SPY"].pct_change()
    df["rv_rolling"] = df["ret_1d"].rolling(window_days).std() * np.sqrt(252)
    return df.dropna()


def compute_regime_thresholds(
    df: pd.DataFrame, q_low: float = 0.33, q_high: float = 0.66
) -> Tuple[float, float]:
    """Return VIX quantile cutoffs used to bucket volatility regimes."""
    vix = df["VIX"].dropna().astype(float)
    lo = float(vix.quantile(q_low))
    hi = float(vix.quantile(q_high))
    if lo >= hi:
        med = float(vix.median())
        lo, hi = med, med + 0.01
    if not np.isfinite(lo) or not np.isfinite(hi):
        med = float(vix.median())
        lo, hi = med, med
    return lo, hi


def add_regime_labels(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Attach LOW/MID/HIGH regime labels based on VIX quantiles."""
    def _label(vix_val: float) -> str:
        if vix_val <= lo:
            return "LOW"
        if vix_val >= hi:
            return "HIGH"
        return "MID"

    out = df.copy()
    out["regime"] = out["VIX"].astype(float).apply(_label)
    return out


def suggested_regime_today(df_reg: pd.DataFrame) -> str:
    return str(df_reg["regime"].iloc[-1])


def fetch_intraday_quotes(
    symbols: Iterable[str] = ("SPY", "^VIX"),
    period: str = "5d",
    interval: str = "5m",
) -> pd.DataFrame:
    """Download near real-time quotes for the provided symbols.

    The request is split per symbol to avoid ambiguous MultiIndex handling when
    multiple tickers are passed to ``yfinance`` at once.
    """

    frames = []
    for symbol in symbols:
        auto_adjust = symbol != "^VIX"
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=auto_adjust,
        )
        if raw.empty:
            continue
        close = _safe_close(raw, symbol).rename(symbol)
        frames.append(close)

    if not frames:
        raise ValueError("No intraday data returned for the requested symbols.")

    return pd.concat(frames, axis=1).dropna()


def build_regime_explanation(df_reg: pd.DataFrame, lo: float, hi: float) -> str:
    """Return a short narrative explaining the suggested regime."""

    current_regime = suggested_regime_today(df_reg)
    vix_now = float(df_reg["VIX"].iloc[-1])
    spy_now = float(df_reg["SPY"].iloc[-1])

    vix_prev = float(df_reg["VIX"].iloc[-2]) if len(df_reg) > 1 else np.nan
    vix_change = vix_now - vix_prev if np.isfinite(vix_prev) else np.nan
    vix_pct_rank = float(df_reg["VIX"].rank(pct=True).iloc[-1] * 100.0)
    trailing_mean = float(df_reg["VIX"].tail(WINDOW_DAYS).mean())

    lookback = min(5, len(df_reg) - 1)
    spy_return = np.nan
    if lookback > 0:
        spy_return = (spy_now / float(df_reg["SPY"].iloc[-1 - lookback]) - 1.0) * 100.0

    change_str = "flat"
    if np.isfinite(vix_change):
        change_str = f"up {vix_change:+.2f} pts" if vix_change else "unchanged"

    if current_regime == "LOW":
        reason = (
            f"VIX at {vix_now:.2f} sits beneath the calm threshold of {lo:.2f} "
            f"({vix_pct_rank:.0f}th percentile) and is near the {WINDOW_DAYS}-day average "
            f"of {trailing_mean:.2f}."
        )
    elif current_regime == "HIGH":
        reason = (
            f"VIX at {vix_now:.2f} breaks above the stress threshold of {hi:.2f} "
            f"({vix_pct_rank:.0f}th percentile), signalling elevated hedging demand."
        )
    else:
        reason = (
            f"VIX at {vix_now:.2f} is between the calm ({lo:.2f}) and stress ({hi:.2f}) "
            f"cut-offs ({vix_pct_rank:.0f}th percentile), pointing to a neutral regime."
        )

    if np.isfinite(vix_change) and vix_change:
        reason += f" Today's move is {change_str}."

    if np.isfinite(spy_return):
        direction = "higher" if spy_return >= 0 else "lower"
        reason += f" SPY traded {direction} by {abs(spy_return):.1f}% over the last {lookback} sessions."

    return reason


def select_regime_pool(
    df_reg: pd.DataFrame, mode: str = "auto", override: str | None = None
) -> Tuple[str, pd.DataFrame]:
    """Return the regime name and subset of rows used for bootstrapping."""
    valid = {"LOW", "MID", "HIGH"}
    if mode == "auto":
        chosen = suggested_regime_today(df_reg)
    else:
        if override is None or str(override).upper() not in valid:
            raise ValueError("Override must be one of: LOW, MID, HIGH")
        chosen = str(override).upper()
    pool = df_reg[df_reg["regime"] == chosen].copy()
    if pool.empty:
        raise ValueError(f"No rows found for regime '{chosen}'.")
    return chosen, pool


def prepare_regime_data(
    years_back: int = 10, q_low: float = 0.33, q_high: float = 0.66
) -> Tuple[pd.DataFrame, float, float]:
    """Load history, compute thresholds, and attach regime labels."""
    start_dt, end_dt = get_dynamic_date_range(years_back)
    df = load_market_data_simple(start_dt, end_dt)
    lo, hi = compute_regime_thresholds(df, q_low=q_low, q_high=q_high)
    df_reg = add_regime_labels(df, lo, hi)
    return df_reg, lo, hi


__all__ = [
    "WINDOW_DAYS",
    "get_dynamic_date_range",
    "load_market_data_simple",
    "compute_regime_thresholds",
    "add_regime_labels",
    "suggested_regime_today",
    "select_regime_pool",
    "prepare_regime_data",
    "fetch_intraday_quotes",
    "build_regime_explanation",
]
