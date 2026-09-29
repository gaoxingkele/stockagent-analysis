"""S20-Pure v1: parameterised three-state labels + amplitude-band funnel.

Design record: wiki/2026-09-28_s20-pure-chain.md (R01-R10).

    label factory   first-passage panel, every barrier a parameter (U, D, M, H)
    axis 1 (floor)  stage1 daily Top-`pool_size`: removes chop / dead names
    axis 2 (cap)    drop the pool's highest-natr14 `cap` share; the share is tied
                    to the take-profit target (small U -> larger cap)
    axis 3 (dir)    empty until new, point-in-time information passes review
    day level       no timing model; equal 1/N daily sleeves, hard stop at -D

Nothing here is a probability promise. Each list is reported with the
historical win rate / expectancy of its exit rule.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .s20 import anchored_residual_probability

ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = ROOT / "output/production/s20_pure_v1"
CONTRACT_PATH = ROOT / "config/s20_pure_v1.json"
SAFE_CONTRACT_PATH = ROOT / "config/s20_pure_v1_1_safe.json"
STAGE1_SEEDS = (20260907, 20260921, 20261005)
MAX_DELTA_LOGIT = 1.0


@dataclass(frozen=True)
class ExitRule:
    name: str
    take_profit: float      # U, percent
    stop_loss: float        # D, percent
    amplitude_cap: float    # share of the pool dropped by highest natr14
    shakeout: float | None = None  # M, percent: tolerated adverse move before +U
    horizon: int = 20


@dataclass(frozen=True)
class PureConfig:
    pool_size: int = 100
    top_k: int = 20
    natr_window: int = 14
    cost_pct: float = 0.3
    daily_sleeve: float = 1 / 20
    rules: tuple[ExitRule, ...] = field(default_factory=lambda: (
        ExitRule("U15D10", 15.0, 10.0, 0.40, shakeout=5.0),
        ExitRule("U20D15", 20.0, 15.0, 0.20, shakeout=8.0),
    ))
    primary_rule: str = "U15D10"

    def rule(self, name: str | None = None) -> ExitRule:
        name = name or self.primary_rule
        for r in self.rules:
            if r.name == name:
                return r
        raise KeyError(name)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rules"] = [asdict(r) for r in self.rules]
        return d


# ---------------------------------------------------------------- labels
def first_touch(path: np.ndarray, level: float, up: bool) -> np.ndarray:
    """First 1-based session where `path` (H, n) crosses `level`; 0 = never."""
    hit = path >= level if up else path <= level
    return np.where(hit.any(0), hit.argmax(0) + 1, 0).astype(np.int16)


def three_state(up_day, dn_day, shake_day=None, horizon: int = 20) -> np.ndarray:
    """pure_up / dirty_up / pure_down / chop / ambiguous from first-touch days.

    up_day: first session of +U; dn_day: first session of -D; shake_day: first
    session of -M (M < D). All 0 when untouched.
    """
    up = np.asarray(up_day, dtype=int)
    dn = np.asarray(dn_day, dtype=int)
    up = np.where((up > 0) & (up <= horizon), up, 0)
    dn = np.where((dn > 0) & (dn <= horizon), dn, 0)
    out = np.full(len(up), "chop", dtype=object)
    up_first = (up > 0) & ((dn == 0) | (up < dn))
    out[up_first] = "pure_up"
    out[(dn > 0) & ((up == 0) | (dn < up))] = "pure_down"
    out[(up > 0) & (up == dn)] = "ambiguous"
    if shake_day is not None:
        sh = np.asarray(shake_day, dtype=int)
        out[up_first & (sh > 0) & (sh <= up)] = "dirty_up"
    return out


def exit_return(up_day, dn_day, close_ret, rule: ExitRule, cost_pct: float = 0.3) -> np.ndarray:
    """Per-trade return (%) under take-profit +U / stop -D / time exit at H.

    Same-session double touch is booked as the stop (conservative).
    """
    up = np.asarray(up_day, dtype=int)
    dn = np.asarray(dn_day, dtype=int)
    H = rule.horizon
    up = np.where((up > 0) & (up <= H), up, 0)
    dn = np.where((dn > 0) & (dn <= H), dn, 0)
    r = np.asarray(close_ret, dtype=float).copy()
    r = np.where((up > 0) & ((dn == 0) | (up < dn)), rule.take_profit, r)
    r = np.where((dn > 0) & ((up == 0) | (dn <= up)), -rule.stop_loss, r)
    return r - cost_pct


# ---------------------------------------------------------------- features
def natr(daily: pd.DataFrame, window: int = 14) -> pd.Series:
    """Normalised ATR from daily bars (ts_code, trade_date, high, low, close, pre_close)."""
    d = daily.sort_values(["ts_code", "trade_date"])
    tr = np.maximum(d["high"] - d["low"],
                    np.maximum((d["high"] - d["pre_close"]).abs(), (d["low"] - d["pre_close"]).abs()))
    atr = tr.groupby(d["ts_code"]).transform(lambda s: s.rolling(window, min_periods=max(2, window - 4)).mean())
    return (atr / d["close"]).reindex(daily.index)


# ---------------------------------------------------------------- stage1 scorer
@lru_cache(maxsize=1)
def _frozen_models():
    import lightgbm as lgb

    anchor = lgb.Booster(model_file=str(FROZEN_DIR / "anchor.txt"))
    residuals = [lgb.Booster(model_file=str(FROZEN_DIR / f"residual_seed{s}.txt")) for s in STAGE1_SEEDS]
    features = json.loads((FROZEN_DIR / "features.json").read_text(encoding="utf-8"))
    return anchor, residuals, features


def stage1_probability(frame: pd.DataFrame) -> np.ndarray:
    """Frozen stage1 score: R20 anchor probability + bounded residual logit."""
    anchor, residuals, features = _frozen_models()
    x = frame[features]
    base = anchor.predict(x)
    delta = np.mean([m.predict(x, raw_score=True) for m in residuals], axis=0)
    return anchored_residual_probability(base, delta, max_absolute_residual=MAX_DELTA_LOGIT)


# ---------------------------------------------------------------- funnel
def select(frame: pd.DataFrame, cfg: PureConfig = PureConfig(), rule: str | None = None,
           score_col: str = "stage1_probability", natr_col: str = "natr14",
           industry_cap: int | None = None, industry_mode: str = "expand") -> pd.DataFrame:
    """Daily list: stage1 Top-pool -> drop highest-natr cap share -> stage1 Top-K.

    Returns the selected rows with `pool_rank`, `natr_pct_in_pool`, `list_rank`
    and `rule`. Names without natr (too little history) stay in the pool.
    With `industry_cap`, needs an `industry` column; see `_industry_fill`.
    """
    r = cfg.rule(rule)
    f = frame[frame[score_col].notna()].copy()
    f["pool_rank"] = f.groupby("trade_date")[score_col].rank(ascending=False, method="first")
    pool = f[f["pool_rank"] <= cfg.pool_size].copy()
    pool["natr_pct_in_pool"] = pool.groupby("trade_date")[natr_col].rank(pct=True)
    keep = pool["natr_pct_in_pool"].isna() | (pool["natr_pct_in_pool"] <= 1 - r.amplitude_cap)
    out = pool[keep].copy()
    out["list_rank"] = out.groupby("trade_date")[score_col].rank(ascending=False, method="first")
    if industry_cap is not None:
        out = pd.concat([_industry_fill(g, cfg.top_k, industry_cap, industry_mode)
                         for _, g in out.groupby("trade_date")], ignore_index=True)
    else:
        out = out[out["list_rank"] <= cfg.top_k].copy()
    out["rule"] = r.name
    return out.sort_values(["trade_date", "list_rank"]).reset_index(drop=True)


@dataclass(frozen=True)
class SafeConfig:
    """v1.1 "safe and up" list (wiki/2026-09-29_s20-pure-r13-vol-band.md).

    Safety and upside share one axis, volatility. Keep the calmer part of the
    market (daily natr14 percentile <= natr_pct_max), rank it by stage1, take
    Top-K, then widen the list with other industries when one industry holds
    more than `industry_cap` names. Exit is a take-profit band.
    """
    natr_pct_max: float = 0.4
    top_k: int = 20
    industry_cap: int = 4
    industry_mode: str = "expand"
    band_low: float = 5.0      # a: sell half, move stop to entry
    band_high: float = 15.0    # b: sell the rest
    crash_line: float = 10.0   # D: full exit
    horizon: int = 20
    cost_pct: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


def select_safe(frame: pd.DataFrame, cfg: SafeConfig = SafeConfig(),
                score_col: str = "stage1_probability", natr_col: str = "natr14") -> pd.DataFrame:
    """Daily safe list. Needs trade_date, ts_code, industry, score and natr columns.

    The natr percentile is taken over the whole scored universe of the day.
    """
    f = frame[frame[score_col].notna() & frame[natr_col].notna()].copy()
    f["natr_pct"] = f.groupby("trade_date")[natr_col].rank(pct=True)
    f = f[f["natr_pct"] <= cfg.natr_pct_max].copy()
    f["list_rank"] = f.groupby("trade_date")[score_col].rank(ascending=False, method="first")
    f = f[f["list_rank"] <= max(200, 3 * cfg.top_k)]
    out = pd.concat([_industry_fill(g, cfg.top_k, cfg.industry_cap, cfg.industry_mode)
                     for _, g in f.groupby("trade_date")], ignore_index=True)
    out["rule"] = "safe_v1_1"
    return out.sort_values(["trade_date", "list_rank"]).reset_index(drop=True)


def _industry_fill(day: pd.DataFrame, top_k: int, cap: int, mode: str) -> pd.DataFrame:
    """Per-day industry handling on candidates already ordered by `list_rank`.

    mode="expand"  keep the original Top-K; for every name an industry holds above
                   `cap`, add the next-best candidate from an industry still under
                   `cap` (the list grows from K to K + excess).
    mode="replace" strict cap: walk down the ranking, skip names whose industry is
                   full, stop at K names.
    """
    day = day.sort_values("list_rank")
    ind = day["industry"].fillna("unknown").astype(str).to_numpy()
    if mode == "replace":
        seen: dict[str, int] = {}
        keep = []
        for i, x in enumerate(ind):
            if seen.get(x, 0) < cap:
                seen[x] = seen.get(x, 0) + 1
                keep.append(i)
            if len(keep) == top_k:
                break
        res = day.iloc[keep].copy()
        res["fill"] = "top"
        return res
    head = day.iloc[:top_k].copy()
    head["fill"] = "top"
    counts = head["industry"].fillna("unknown").astype(str).value_counts()
    need = int((counts - cap).clip(lower=0).sum())
    seen = counts.to_dict()
    extra = []
    for i in range(top_k, len(day)):
        if len(extra) == need:
            break
        if seen.get(ind[i], 0) < cap:
            seen[ind[i]] = seen.get(ind[i], 0) + 1
            extra.append(i)
    tail = day.iloc[extra].copy()
    tail["fill"] = "industry_substitute"
    return pd.concat([head, tail])
