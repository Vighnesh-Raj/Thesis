"""Core optimization primitives extracted from the research notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog

@dataclass
class Prefs:
    """User-configurable settings that feed the hedge workflow."""

    # Scenario/portfolio
    horizon: str = "1w"  # '1w'|'1m'|'3m'|'1y'
    n_scen: int = 2000
    seed: int = 123
    alpha: float = 0.95
    n_shares: int = 20
    risk_free: float = 0.00

    # Permissions (toggles)
    retail_mode: bool = True
    allow_selling: bool = False
    zero_cost: bool = False
    allow_net_credit: bool = False
    budget_usd: float = 200.0

    # Bounds
    max_buy_contracts: float = 10.0
    max_sell_contracts: float = 10.0

    # Rounding & evaluation
    integer_round: bool = True
    step: float = 1.0
    budget_enforced_after_rounding: bool = True
    zero_cost_tolerance: float = 5.0
    make_plots: bool = False  # handled by the Streamlit layer


def simulate_terminal_from_pool(
    df_reg: pd.DataFrame,
    pool: pd.DataFrame,
    days: int = 5,
    n_scen: int = 2000,
    seed: int = 42,
    vix_floor: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    S0 = float(df_reg["SPY"].iloc[-1])
    V0 = float(df_reg["VIX"].iloc[-1])
    ret_pool = pool["ret_1d"].dropna().values
    dvix_pool = pool["VIX"].pct_change().dropna().values
    if len(ret_pool) < 50 or len(dvix_pool) < 50:
        raise ValueError("Regime pool too small after cleaning; adjust thresholds or use AUTO.")
    S_T = np.empty(n_scen, dtype=float)
    V_T = np.empty(n_scen, dtype=float)
    for k in range(n_scen):
        r = rng.choice(ret_pool, size=days, replace=True)
        dv = rng.choice(dvix_pool, size=days, replace=True)
        S_T[k] = S0 * np.prod(1.0 + r)
        V_T[k] = max(vix_floor, V0 * np.prod(1.0 + dv))
    return S_T, V_T


def simulate_terminal(
    df_reg: pd.DataFrame,
    pool: pd.DataFrame,
    horizon: str = "1w",
    n_scen: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    horizon = str(horizon).lower()
    map_days = {"1w": 5, "1m": 21, "3m": 63, "1y": 252}
    if horizon not in map_days:
        raise ValueError(f"horizon must be one of {list(map_days.keys())}")
    days = map_days[horizon]
    if days <= 5:
        return simulate_terminal_from_pool(df_reg, pool, days=days, n_scen=n_scen, seed=seed)

    rng = np.random.default_rng(seed)
    S0 = float(df_reg["SPY"].iloc[-1])
    V0 = float(df_reg["VIX"].iloc[-1])
    ret_pool = pool["ret_1d"].dropna().values
    dvix_pool = pool["VIX"].pct_change().dropna().values
    if len(ret_pool) < 50:
        raise ValueError("Regime pool too small.")
    block = 5
    n_blocks = max(1, days // block)
    S_T, V_T = np.empty(n_scen), np.empty(n_scen)
    for k in range(n_scen):
        r_seq, dv_seq = [], []
        for _ in range(n_blocks):
            r_seq.extend(rng.choice(ret_pool, size=block, replace=True))
            dv_seq.extend(rng.choice(dvix_pool, size=block, replace=True))
        S_T[k] = S0 * np.prod(1.0 + np.array(r_seq)[:days])
        V_T[k] = max(8.0, V0 * np.prod(1.0 + np.array(dv_seq)[:days]))
    return S_T, V_T


def validate_quotes_df_bidask_strict(quotes: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize option quotes containing bid/ask data."""

    req = {"kind", "strike", "bid", "ask", "expiry"}
    cols_lower = {c.lower() for c in quotes.columns}
    missing = req - cols_lower
    if missing:
        raise ValueError(f"quotes must include: {sorted(req)}; missing {sorted(missing)}")

    q = quotes.copy()
    q.columns = [c.lower() for c in q.columns]

    q["kind"] = q["kind"].str.upper().str.strip()
    q["strike"] = pd.to_numeric(q["strike"], errors="coerce")
    q["bid"] = pd.to_numeric(q["bid"], errors="coerce")
    q["ask"] = pd.to_numeric(q["ask"], errors="coerce")
    q["expiry"] = q["expiry"].astype(str).str.strip()

    if not q["kind"].isin(["C", "P"]).all():
        bad = q.loc[~q["kind"].isin(["C", "P"]), "kind"].unique().tolist()
        raise ValueError(f"kind must be 'C' or 'P'; got {bad}")
    if (q["strike"] <= 0).any():
        raise ValueError("strike must be positive")
    if (q["ask"] <= 0).any() or (q["bid"] < 0).any():
        raise ValueError("ask must be >0 and bid must be >=0")
    if (q["ask"] < q["bid"]).any():
        bad = q[q["ask"] < q["bid"]][["kind", "strike", "bid", "ask"]]
        raise ValueError(f"ask < bid for rows:\n{bad}")
    exps = q["expiry"].unique()
    if len(exps) != 1:
        raise ValueError(f"single expiry required for now; got: {exps}")

    if "label" not in q.columns:
        q["label"] = q["kind"] + q["strike"].map(lambda x: f"{x:g}")

    q["mid"] = 0.5 * (q["bid"] + q["ask"])

    q = q.sort_values(["kind", "strike"]).reset_index(drop=True)
    return q


def optimize_hedge(
    quotes_df: pd.DataFrame,
    df_reg: pd.DataFrame,
    pool: pd.DataFrame,
    *,
    horizon: str = "1w",
    n_scen: int = 2000,
    seed: int = 123,
    alpha: float = 0.95,
    n_shares: int = 20,
    risk_free: float = 0.00,
    zero_cost: bool = False,
    budget_usd: float = 200.0,
    allow_net_credit: bool = False,
    allow_buy_calls: bool = True,
    allow_sell_calls: bool = False,
    allow_buy_puts: bool = True,
    allow_sell_puts: bool = False,
    max_buy_contracts: float = 10.0,
    max_sell_contracts: float = 10.0,
    round_step: Optional[float] = None,
) -> dict:
    """Solve the Rockafellar–Uryasev CVaR linear program."""

    def _validate_quotes(q: pd.DataFrame) -> pd.DataFrame:
        req = {"kind", "strike", "mid", "expiry"}
        if req - set(q.columns.str.lower()):
            raise ValueError("quotes_df must include: kind, strike, mid, expiry")
        qq = q.copy()
        qq.columns = [c.lower() for c in qq.columns]
        qq["kind"] = qq["kind"].str.upper().str.strip()
        qq["strike"] = pd.to_numeric(qq["strike"], errors="coerce")
        qq["mid"] = pd.to_numeric(qq["mid"], errors="coerce")
        qq["expiry"] = qq["expiry"].astype(str)
        if not qq["kind"].isin(["C", "P"]).all():
            raise ValueError("kind must be C or P")
        if (qq["mid"] <= 0).any():
            raise ValueError("mid must be positive")
        if len(qq["expiry"].unique()) != 1:
            raise ValueError("one expiry only for now")
        for col in ["bid", "ask"]:
            if col in qq.columns:
                qq[col] = pd.to_numeric(qq[col], errors="coerce")
        if "label" not in qq.columns:
            qq["label"] = qq["kind"] + qq["strike"].map(lambda x: f"{x:g}")
        return qq.sort_values(["kind", "strike"]).reset_index(drop=True)

    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _bs_price(S, K, tau, r, sigma, kind):
        tau = max(tau, 1e-8)
        sigma = max(sigma, 1e-8)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        Nd1 = _norm_cdf(d1)
        Nd2 = _norm_cdf(d2)
        if kind == "C":
            return S * Nd1 - K * np.exp(-r * tau) * Nd2
        return K * np.exp(-r * tau) * (1.0 - Nd2) - S * (1.0 - Nd1)

    def _iv_from_vix(VT, floor=0.08, cap=1.00):
        return np.clip(VT / 100.0, floor, cap)

    def _days_to_expiry(expiry_str, today_ts):
        return max(int((pd.Timestamp(expiry_str) - today_ts).days), 0)

    def _build_mats(q, ST, VT, today_ts, horizon_days, r=0.0, mult=100.0):
        N, m = len(ST), len(q)
        X_buy = np.empty((N, m))
        X_sell = np.empty((N, m))
        c_buy = np.empty(m)
        c_sell = np.empty(m)
        for j, rr in q.iterrows():
            kind = rr["kind"]
            K = float(rr["strike"])
            mid0 = float(rr["mid"])
            bid0 = float(rr.get("bid", np.nan))
            ask0 = float(rr.get("ask", np.nan))
            if not np.isfinite(bid0) or not np.isfinite(ask0):
                spr = 0.02 * mid0
                bid0 = max(mid0 - spr, mid0 * 0.5)
                ask0 = max(mid0 + spr, bid0 + 1e-6)
            tauT = max((_days_to_expiry(rr["expiry"], today_ts) - horizon_days) / 252.0, 1e-6)
            sigT = _iv_from_vix(VT)
            priceT = np.vectorize(_bs_price, otypes=[float])(ST, K, tauT, r, sigT, kind)
            X_buy[:, j] = (priceT - ask0) * mult
            X_sell[:, j] = (bid0 - priceT) * mult
            c_buy[j] = ask0 * mult
            c_sell[j] = -bid0 * mult
        return X_buy, X_sell, c_buy, c_sell

    def _var_cvar(pnl, a=0.95):
        loss = -np.asarray(pnl)
        var = float(np.quantile(loss, a))
        tail = loss[loss >= var]
        cvar = float(np.mean(tail)) if tail.size else var
        return var, cvar

    days_map = {"1w": 5, "1m": 21, "3m": 63, "1y": 252}
    if horizon not in days_map:
        raise ValueError("horizon must be one of '1w','1m','3m','1y'")
    h_days = days_map[horizon]

    ST, VT = simulate_terminal(df_reg, pool, horizon=horizon, n_scen=n_scen, seed=seed)
    S0 = float(df_reg["SPY"].iloc[-1])
    today_ts = df_reg.index[-1]
    y_scaled = (ST - S0) * float(n_shares)

    q = _validate_quotes(quotes_df)
    Xp, Xm, c_plus, c_minus = _build_mats(q, ST, VT, today_ts, h_days, r=risk_free)

    N, m = Xp.shape
    nvar = 2 * m + 1 + N
    idx_wp = slice(0, m)
    idx_wm = slice(m, 2 * m)
    idx_z = 2 * m
    idx_u = slice(2 * m + 1, 2 * m + 1 + N)

    c_vec = np.zeros(nvar)
    c_vec[idx_z] = 1.0
    c_vec[idx_u] = 1.0 / ((1.0 - alpha) * N)
    A_ub, b_ub = [], []

    for i in range(N):
        row = np.zeros(nvar)
        row[idx_wp] = -Xp[i]
        row[idx_wm] = -Xm[i]
        row[idx_z] = -1.0
        row[2 * m + 1 + i] = -1.0
        A_ub.append(row)
        b_ub.append(y_scaled[i])

    for i in range(N):
        row = np.zeros(nvar)
        row[2 * m + 1 + i] = -1.0
        A_ub.append(row)
        b_ub.append(0.0)

    row_cost = np.zeros(nvar)
    row_cost[idx_wp] = c_plus
    row_cost[idx_wm] = c_minus
    if zero_cost:
        A_ub.append(row_cost.copy())
        b_ub.append(0.0)
        A_ub.append((-row_cost).copy())
        b_ub.append(0.0)
    else:
        B = float(budget_usd)
        if allow_net_credit:
            A_ub.append(row_cost.copy())
            b_ub.append(B)
            A_ub.append((-row_cost).copy())
            b_ub.append(B)
        else:
            A_ub.append(row_cost.copy())
            b_ub.append(B)
            A_ub.append((-row_cost).copy())
            b_ub.append(0.0)

    A_ub = np.vstack(A_ub)
    b_ub = np.asarray(b_ub)

    bounds = []
    kinds = q["kind"].tolist()
    for k in kinds:
        if (k == "C" and not allow_buy_calls) or (k == "P" and not allow_buy_puts):
            bounds.append((0.0, 0.0))
        else:
            bounds.append((0.0, max_buy_contracts))
    for k in kinds:
        if (k == "C" and not allow_sell_calls) or (k == "P" and not allow_sell_puts):
            bounds.append((0.0, 0.0))
        else:
            bounds.append((0.0, max_sell_contracts))
    bounds.append((None, None))
    bounds.extend([(0.0, None)] * N)

    res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CVaR LP failed: {res.message}")

    x = res.x
    wp = x[idx_wp]
    wm = x[idx_wm]
    pnl_unhedged = y_scaled
    pnl_hedged = y_scaled + (Xp @ wp) + (Xm @ wm)

    if round_step is not None and round_step > 0:
        wp = np.round(wp / round_step) * round_step
        wm = np.round(wm / round_step) * round_step
        pnl_hedged = y_scaled + (Xp @ wp) + (Xm @ wm)

    spend = float(c_plus @ wp + c_minus @ wm)
    var_emp, cvar_emp = _var_cvar(pnl_hedged, a=alpha)

    return {
        "weights": {
            "buy": wp.copy(),
            "sell": wm.copy(),
            "net": (wp - wm).copy(),
            "labels": q["label"].tolist(),
        },
        "spend_usd": spend,
        "pnl": {"unhedged": pnl_unhedged, "hedged": pnl_hedged},
        "risk": {"alpha": alpha, "var": var_emp, "cvar": cvar_emp},
        "meta": {
            "horizon": horizon,
            "n_scen": int(n_scen),
            "n_shares": int(n_shares),
            "zero_cost": bool(zero_cost),
            "allow_net_credit": bool(allow_net_credit),
        },
        "_design": {
            "Xp": Xp,
            "Xm": Xm,
            "c_plus": c_plus,
            "c_minus": c_minus,
            "ST": ST,
            "S0": S0,
        },
    }


def institutional_zero_cost_integer_rounding(
    wp,
    wm,
    c_plus,
    c_minus,
    max_buy,
    max_sell,
    step: float = 1.0,
    tol: float = 5.0,
):
    """Round buy/sell weights while keeping cost close to zero."""

    wp = np.asarray(wp, float)
    wm = np.asarray(wm, float)
    wp_r = np.clip(np.round(wp / step) * step, 0, max_buy)
    wm_r = np.clip(np.round(wm / step) * step, 0, max_sell)

    def cost(wp_, wm_):
        return float(c_plus @ wp_ + c_minus @ wm_)

    curr_cost = cost(wp_r, wm_r)
    if abs(curr_cost) <= tol:
        return wp_r, wm_r, curr_cost

    for _ in range(10000):
        curr_cost = cost(wp_r, wm_r)
        if abs(curr_cost) <= tol:
            break
        if curr_cost < -tol:
            best_gain, action = -np.inf, None
            for j in range(len(wm_r)):
                if wm_r[j] >= step:
                    gain = -c_minus[j]
                    if gain > best_gain:
                        best_gain, action = gain, ("dec_sell", j)
            for j in range(len(wp_r)):
                if wp_r[j] + step <= max_buy:
                    gain = c_plus[j]
                    if gain > best_gain:
                        best_gain, action = gain, ("inc_buy", j)
            if action is None:
                break
            kind, j = action
            if kind == "dec_sell":
                wm_r[j] -= step
            else:
                wp_r[j] += step
        else:
            best_drop, action = -np.inf, None
            for j in range(len(wm_r)):
                if wm_r[j] + step <= max_sell:
                    drop = -c_minus[j]
                    if drop > best_drop:
                        best_drop, action = drop, ("inc_sell", j)
            for j in range(len(wp_r)):
                if wp_r[j] >= step:
                    drop = c_plus[j]
                    if drop > best_drop:
                        best_drop, action = drop, ("dec_buy", j)
            if action is None:
                break
            kind, j = action
            if kind == "inc_sell":
                wm_r[j] += step
            else:
                wp_r[j] -= step

    return wp_r, wm_r, cost(wp_r, wm_r)


def retail_budget_aware_integer_rounding(wp, c_plus, budget_usd, step: float = 1.0):
    wp = np.asarray(wp, float)
    w = np.round(wp / step) * step
    spend = float(c_plus @ w)
    if spend <= budget_usd:
        return w, spend
    if spend > 0:
        scale = budget_usd / spend
        w = np.floor((w * scale) / step) * step
    spend = float(c_plus @ w)
    if spend <= budget_usd:
        return w, spend
    order = np.argsort(-c_plus)
    while spend > budget_usd + 1e-9 and w.sum() > 0:
        for j in order:
            if w[j] >= step:
                w[j] -= step
                spend = float(c_plus @ w)
                if spend <= budget_usd + 1e-9:
                    break
    return w, float(c_plus @ w)


def run_hedge_workflow(
    quotes_df: pd.DataFrame,
    df_reg: pd.DataFrame,
    pool: pd.DataFrame,
    prefs: Prefs,
    *,
    verbose: bool = False,
) -> dict:
    """High-level router used by the Streamlit UI."""

    if prefs.retail_mode:
        allow_buy_calls = allow_buy_puts = True
        allow_sell_calls = allow_sell_puts = bool(prefs.allow_selling)
    else:
        allow_buy_calls = allow_buy_puts = allow_sell_calls = allow_sell_puts = True

    zero_cost = bool(prefs.zero_cost)
    allow_net_credit = bool(prefs.allow_net_credit)

    res = optimize_hedge(
        quotes_df=quotes_df,
        df_reg=df_reg,
        pool=pool,
        horizon=prefs.horizon,
        n_scen=prefs.n_scen,
        seed=prefs.seed,
        alpha=prefs.alpha,
        n_shares=prefs.n_shares,
        risk_free=prefs.risk_free,
        zero_cost=zero_cost,
        budget_usd=prefs.budget_usd,
        allow_net_credit=allow_net_credit,
        allow_buy_calls=allow_buy_calls,
        allow_sell_calls=allow_sell_calls,
        allow_buy_puts=allow_buy_puts,
        allow_sell_puts=allow_sell_puts,
        max_buy_contracts=prefs.max_buy_contracts,
        max_sell_contracts=prefs.max_sell_contracts,
        round_step=None,
    )

    labels = res["weights"]["labels"]
    wp = res["weights"]["buy"].copy()
    wm = res["weights"]["sell"].copy()
    spend_lp = res["spend_usd"]

    if verbose:
        print("\n=== LP Solution (fractional) ===")
        print(f"Initial spend from LP: ${spend_lp:.2f} | zero_cost={zero_cost}")
        for lbl, b, s, n in zip(labels, wp, wm, (wp - wm)):
            print(f"{lbl:>8s}: buy={b:6.3f}, sell={s:6.3f}, net={n:6.3f}")

    Xp = res["_design"]["Xp"]
    Xm = res["_design"]["Xm"]
    c_plus = res["_design"]["c_plus"]
    c_minus = res["_design"]["c_minus"]
    y_unh = res["pnl"]["unhedged"]

    if prefs.integer_round:
        if not allow_sell_calls and not allow_sell_puts:
            wp_r, spend_r = retail_budget_aware_integer_rounding(
                wp,
                c_plus,
                budget_usd=(0.0 if zero_cost else prefs.budget_usd)
                if prefs.budget_enforced_after_rounding
                else 1e18,
                step=prefs.step,
            )
            wm_r = np.zeros_like(wp_r)
        else:
            wp_r, wm_r, cost_r = institutional_zero_cost_integer_rounding(
                wp,
                wm,
                c_plus,
                c_minus,
                max_buy=prefs.max_buy_contracts,
                max_sell=prefs.max_sell_contracts,
                step=prefs.step,
                tol=prefs.zero_cost_tolerance,
            )
            spend_r = float(c_plus @ wp_r + c_minus @ wm_r)
    else:
        wp_r, wm_r = wp, wm
        spend_r = float(c_plus @ wp_r + c_minus @ wm_r)

    pnl_hedged_r = y_unh + (Xp @ wp_r) + (Xm @ wm_r)

    if verbose:
        print("\n=== Executable Portfolio (after rounding) ===")
        for lbl, b, s, n in zip(labels, wp_r, wm_r, (wp_r - wm_r)):
            print(f"{lbl:>8s}: buy={b:4.1f}, sell={s:4.1f}, net={n:5.1f}")
        print(f"Rounded initial spend: ${spend_r:.2f}")

    return {
        "lp_result": res,
        "rounded": {
            "buy": wp_r,
            "sell": wm_r,
            "net": wp_r - wm_r,
            "spend": spend_r,
        },
        "pnl": {"unhedged": y_unh, "hedged": pnl_hedged_r},
        "labels": labels,
    }


__all__ = [
    "Prefs",
    "validate_quotes_df_bidask_strict",
    "simulate_terminal",
    "simulate_terminal_from_pool",
    "optimize_hedge",
    "run_hedge_workflow",
]
