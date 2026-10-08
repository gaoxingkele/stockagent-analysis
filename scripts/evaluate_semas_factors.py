#!/usr/bin/env python
"""Evaluate the 21 SEMAS 5-day factor expressions on the full market and on the S20 funnel.

Registration: wiki/2026-10-08_semas-5d-factors-on-s20.md (written before any result).
The expressions are evaluated with SEMAS's own parser/evaluator from the pinned commit b50aa062
(copied verbatim into the scratch folder given by --semas), on a panel built here with SEMAS's
conventions: unadjusted daily bars, TA-Lib rsi_14 / adx_14 / willr_14 / macd_hist, vwap = amount/volume,
financial indicators as of their announcement date, money flow with net_elg_amount = buy - sell elg.

Output: output/experiments/semas_5d_20261008/{factors.parquet, report.txt}
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import talib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_cross_filter_v2 import nw_se  # noqa: E402

OUT = ROOT / "output/experiments/semas_5d_20261008"
LIST = Path("C:/aicoding/SEMAS/FACTOR_5D_SHARPE_GT_1_5.md")
START, END, LAST_SIGNAL = "20240102", "20260805", "20260728"
SEMAS_END = "20260601"
DEV_END = "20260126"
FINA_FIELDS = "ts_code,ann_date,end_date,eps,ocfps,netprofit_yoy,roe_dt,grossprofit_margin"


def expressions() -> dict[str, str]:
    text = LIST.read_text(encoding="utf-8")
    return dict(re.findall(r"### (F\d\d)\s*\n\s*```text\n(.+?)\n```", text))


def fina() -> pd.DataFrame:
    cache = OUT / "fina_indicator.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import tushare as ts
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    parts = []
    for y in range(2023, 2027):
        for md in ("0331", "0630", "0930", "1231"):
            period = f"{y}{md}"
            if period > "20260630":
                continue
            d = pro.fina_indicator_vip(period=period, fields=FINA_FIELDS)
            if d is not None and len(d):
                parts.append(d)
    f = pd.concat(parts, ignore_index=True).dropna(subset=["ann_date"])
    f = f.sort_values(["ts_code", "ann_date", "end_date"]).drop_duplicates(["ts_code", "ann_date"], keep="last")
    f.to_parquet(cache, index=False)
    return f


def panel() -> pd.DataFrame:
    def load(folder: str, cols: list[str] | None = None) -> pd.DataFrame:
        parts = []
        for f in sorted((ROOT / "output/tushare_cache" / folder).glob("*.parquet")):
            if START <= f.stem <= END:
                parts.append(pd.read_parquet(f, columns=cols))
        x = pd.concat(parts, ignore_index=True)
        x["trade_date"] = x.trade_date.astype(str)
        return x
    d = load("daily", ["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"])
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "name"])
    st = set(basic.loc[basic.name.fillna("").str.contains("ST", regex=False), "ts_code"])
    d = d[d.ts_code.str.endswith((".SH", ".SZ")) & ~d.ts_code.isin(st)]
    d = d.merge(load("daily_basic", ["ts_code", "trade_date", "turnover_rate", "pb", "total_mv", "circ_mv"]), how="left")
    mf = load("moneyflow", ["ts_code", "trade_date", "buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
                             "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount", "net_mf_amount"])
    mf["net_elg_amount"] = mf.buy_elg_amount - mf.sell_elg_amount
    d = d.merge(mf, how="left")
    d = d.sort_values(["trade_date", "ts_code"])
    fi = fina()
    fi = fi.assign(asof=pd.to_datetime(fi.ann_date.astype(str))).drop(columns=["ann_date", "end_date"]).sort_values("asof")
    d["asof"] = pd.to_datetime(d.trade_date)
    d = pd.merge_asof(d.sort_values("asof"), fi, on="asof", by="ts_code", direction="backward").drop(columns=["asof"])
    d = d.rename(columns={"ts_code": "symbol", "vol": "volume"})
    d["date"] = pd.to_datetime(d.trade_date)
    d = d.set_index(["symbol", "date"]).sort_index()
    g = d.groupby(level="symbol")
    d["return"] = g.close.pct_change()
    d["vwap"] = d.amount / d.volume.replace(0, np.nan)
    d["hk_vol"] = np.nan

    def per_symbol(x: pd.DataFrame) -> pd.DataFrame:
        c, h, lo = x.close.to_numpy(float), x.high.to_numpy(float), x.low.to_numpy(float)
        out = pd.DataFrame(index=x.index)
        out["rsi_14"] = talib.RSI(c, 14)
        out["macd_hist"] = talib.MACD(c)[2]
        out["adx_14"] = talib.ADX(h, lo, c, 14)
        out["willr_14"] = talib.WILLR(h, lo, c, 14)
        return out
    d = d.join(g.apply(per_symbol).droplevel(0))
    return d


def tstat(s: pd.Series, lag: int) -> float:
    s = s.dropna()
    if len(s) < 20:
        return np.nan
    x = s.to_numpy()
    n = len(x)
    e = x - x.mean()
    v = e @ e / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n
    return x.mean() / np.sqrt(v / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semas", required=True, help="folder holding the pinned china_a_share_alpha package")
    a = ap.parse_args()
    sys.path.insert(0, a.semas)
    from china_a_share_alpha.factor import parse_expression  # noqa: E402
    OUT.mkdir(parents=True, exist_ok=True)
    data = panel()
    exprs = expressions()
    fac = pd.DataFrame(index=data.index)
    for k, e in exprs.items():
        try:
            fac[k] = parse_expression(e).eval(data).replace([np.inf, -np.inf], np.nan)
        except Exception as err:  # noqa: BLE001
            print(f"{k} failed: {err}", flush=True)
            fac[k] = np.nan
    g = data.groupby(level="symbol")
    entry = g.open.shift(-1)
    fac["fwd5_close"] = g.close.shift(-5) / data.close - 1
    fac["fwd5_trade"] = (g.close.shift(-5) / entry - 1) * 100 - 0.3
    fac["log_mv"] = np.log(data.total_mv)
    fac["past5"] = g.close.pct_change(5)
    fac = fac.reset_index()
    fac["trade_date"] = fac.date.dt.strftime("%Y%m%d")
    fac = fac[(fac.trade_date >= "20240601") & (fac.trade_date <= LAST_SIGNAL)]
    fac.to_parquet(OUT / "factors.parquet", index=False)
    fac["half"] = np.where(fac.trade_date >= SEMAS_END, "after SEMAS",
                           fac.trade_date.str[:4] + np.where(fac.trade_date.str[4:6] <= "06", "H1", "H2"))
    fac["big"] = fac.groupby("trade_date").log_mv.rank(ascending=False) <= 800
    fac["fwd5_ex"] = fac.fwd5_trade - fac.groupby("trade_date").fwd5_trade.transform("mean")
    names = list(exprs)
    lines = [f"panel: {fac.symbol.nunique()} stocks, {fac.trade_date.nunique()} signal days ({fac.trade_date.min()}..{fac.trade_date.max()})",
             "IC = mean daily Spearman with the 5-day close-to-close return (NW lag 4); long = top fifth, next open -> close day 5, cost 0.3%, excess over the same day's market",
             "", "== 1-3. full market / top-800 by market cap / long-only top fifth =="]
    halves = sorted(fac.half.unique(), key=lambda h: (h == "after SEMAS", h))
    head = "factor | coverage | rho logMV  rho past5 | IC all (t) | IC by half: " + " ".join(halves) + " | IC top800 (t) | long excess % (t) win% | long excess by half"
    lines.append(head)
    for k in names:
        x = fac.dropna(subset=[k, "fwd5_close"])
        if x.empty:
            lines.append(f"{k} | no values")
            continue
        ic = x.groupby("trade_date").apply(lambda d: d[k].corr(d.fwd5_close, method="spearman"))
        ic_h = x.groupby("half").apply(lambda d: d.groupby("trade_date").apply(lambda e: e[k].corr(e.fwd5_close, method="spearman")).mean())
        xb = x[x.big]
        icb = xb.groupby("trade_date").apply(lambda d: d[k].corr(d.fwd5_close, method="spearman"))
        rmv = x.groupby("trade_date").apply(lambda d: d[k].corr(d.log_mv, method="spearman")).mean()
        rp5 = x.groupby("trade_date").apply(lambda d: d[k].corr(d.past5, method="spearman")).mean()
        top = x[x.groupby("trade_date")[k].rank(pct=True) > 0.8]
        lx = top.groupby("trade_date").fwd5_ex.mean()
        lx_h = top.groupby("half").fwd5_ex.mean()
        lines.append(f"{k} | {x.symbol.nunique():4d} | {rmv:+.2f} {rp5:+.2f} | {ic.mean():+.4f} ({tstat(ic, 4):+.1f}) | "
                     + " ".join(f"{ic_h.get(h, np.nan):+.3f}" for h in halves)
                     + f" | {icb.mean():+.4f} ({tstat(icb, 4):+.1f}) | {lx.mean():+.2f} ({tstat(lx, 4):+.1f}) {100 * (top.fwd5_trade > 0).mean():.0f}% | "
                     + " ".join(f"{lx_h.get(h, np.nan):+.2f}" for h in halves))

    # ---- S20 funnel survivors
    from s20_pure_history import outcomes  # noqa: E402
    from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402
    fr = pd.read_parquet(ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet")
    fr = fr[fr.trade_date <= LAST_SIGNAL]
    surv = select(fr, PureConfig(top_k=100), "U15D10")[["ts_code", "trade_date"]].rename(columns={"ts_code": "symbol"})
    out = outcomes().rename(columns={"ts_code": "symbol"})[["symbol", "trade_date", "ret_v1", "state"]]
    s = surv.merge(fac[["symbol", "trade_date", *names, "log_mv"]], on=["symbol", "trade_date"]).merge(out, on=["symbol", "trade_date"])
    s["stop"] = (s.state == "pure_down").astype(float)
    s["seg"] = np.where(s.trade_date <= DEV_END, "dev", "conf")
    lines += ["", f"== 4. S20 funnel survivors: {len(s):,} rows, {s.trade_date.nunique()} days; high third minus low third, same day ==",
              "factor | rho logMV in pool | per trade dev (t) conf (t) all (t) | stop-first pts dev (t) conf (t) all (t) | verdict"]
    passed = []
    for k in names:
        x = s.dropna(subset=[k]).copy()
        if x.empty or x[k].nunique() < 3:
            lines.append(f"{k} | too few values")
            continue
        x["t"] = x.groupby("trade_date")[k].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=False) if len(v) >= 9 else np.nan)
        rmv = x.groupby("trade_date").apply(lambda d: d[k].corr(d.log_mv, method="spearman")).mean()
        res = {}
        for m in ("ret_v1", "stop"):
            d = x[x.t == 2].groupby("trade_date")[m].mean() - x[x.t == 0].groupby("trade_date")[m].mean()
            seg = x.groupby("trade_date").seg.first().reindex(d.index)
            res[m] = {"dev": d[seg == "dev"], "conf": d[seg == "conf"], "all": d}
        ok = {}
        for m in ("ret_v1", "stop"):
            r = res[m]
            ok[m] = (np.sign(r["dev"].mean()) == np.sign(r["conf"].mean()) and abs(tstat(r["dev"], 19)) >= 2
                     and abs(tstat(r["conf"], 19)) >= 2 and abs(tstat(r["all"], 19)) >= 3.2)
        verdict = " ".join(f"PASS {m}" for m in ok if ok[m]) or "-"
        if any(ok.values()):
            passed.append(k)
        sc = {"ret_v1": 1, "stop": 100}
        lines.append(f"{k} | {rmv:+.2f} | " + " ".join(f"{sc['ret_v1'] * res['ret_v1'][g2].mean():+.2f}({tstat(res['ret_v1'][g2], 19):+.1f})" for g2 in ("dev", "conf", "all"))
                     + " | " + " ".join(f"{sc['stop'] * res['stop'][g2].mean():+.1f}({tstat(res['stop'][g2], 19):+.1f})" for g2 in ("dev", "conf", "all"))
                     + f" | {verdict}")
    lines += ["", f"S20 passed: {passed if passed else 'none'}"]
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
