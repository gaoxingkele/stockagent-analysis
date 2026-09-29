#!/usr/bin/env python
"""Jev as a market-wide sell-off detector: leakage probe (phase 0) + retro skill (phase 1).

Question (known after the signal-day close, CCTV evening news included):
  will at least one broad sell-off session occur in the next 5 A-share sessions?
  broad sell-off = median A-share return <= -2.5% OR >= 100 limit-down closes.

Conditions on the same stratified sample of signal days:
  A real      true as_of date + market snapshot + that evening's CCTV news
  B shifted   same snapshot + news, as_of replaced by a hash-assigned other date
  C no_date   snapshot + news, no date
  D numeric   snapshot only, no date, no news
Reading: A >> B means the model recalls outcomes by date (retro use contaminated);
C vs D isolates what the news text adds; everything is compared with a free
logistic baseline on the same snapshot numbers (expanding-window, no look-ahead).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/jev_market"
CCTV = OUT / "cctv"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
N_PER_CLASS = 45
QUESTION = (
    "A-share market, China. Consider the next 5 trading sessions after the evening described in the state. "
    "A 'broad sell-off session' is a session in which the median A-share stock falls 2.5% or more, "
    "or at least 100 A-share stocks close limit-down. Judge only from the information given. "
    "Will at least one broad sell-off session occur in those next 5 sessions?"
)
CRITERIA = {"true": "at least one broad sell-off session occurs in the next 5 sessions",
            "false": "no broad sell-off session occurs in the next 5 sessions"}


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("JEV_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("JEV_API_KEY missing in .env")


def ask(state: dict, key: str, attempts: int = 5) -> dict:
    payload = json.dumps({"model": MODEL, "state": state, "questions": {
        "selloff5": {"type": "noul", "instructions": QUESTION, "criteria": CRITERIA}}},
        ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=payload,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    for i in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                b = json.loads(r.read().decode("utf-8"))
            return {"ok": True, "p": float(b["answers"]["selloff5"]["noul"]), "model": b.get("model")}
        except urllib.error.HTTPError as e:
            if e.code in (429, 529):
                time.sleep(2 ** i)
                continue
            return {"ok": False, "error": f"HTTP {e.code}", "detail": e.read().decode("utf-8")[:200]}
        except Exception as e:  # noqa: BLE001
            time.sleep(1 + i)
            last = str(e)
    return {"ok": False, "error": last}


def market_snapshot() -> pd.DataFrame:
    rows = []
    for f in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")):
        d = pd.read_parquet(f, columns=["ts_code", "pct_chg", "amount"])
        d = d[d.ts_code.str.endswith((".SH", ".SZ"))]
        board20 = d.ts_code.str[:3].isin(["300", "301", "688", "689"])
        rows.append({"date": f.stem, "median_ret": d.pct_chg.median(), "up_share": (d.pct_chg > 0).mean(),
                     "limit_up": int((d.pct_chg >= np.where(board20, 19.5, 9.5)).sum()),
                     "limit_down": int((d.pct_chg <= np.where(board20, -19.5, -9.5)).sum()),
                     "amount": d.amount.sum()})
    m = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    m["median_ret_5d"] = m.median_ret.rolling(5).sum()
    m["median_ret_20d"] = m.median_ret.rolling(20).sum()
    m["limit_down_5d"] = m.limit_down.rolling(5).sum()
    m["amount_z60"] = (m.amount - m.amount.rolling(60).mean()) / m.amount.rolling(60).std()
    m["crash"] = (m.median_ret <= -2.5) | (m.limit_down >= 100)
    fut = np.full(len(m), np.nan)
    for i in range(len(m) - 5):
        fut[i] = float(m.crash.iloc[i + 1:i + 6].any())
    m["crash_next5"] = fut
    m["label_ok"] = np.arange(len(m)) + 5 < len(m)
    return m


FEATS = ["median_ret", "up_share", "limit_up", "limit_down", "median_ret_5d", "median_ret_20d",
         "limit_down_5d", "amount_z60"]


def cctv(date: str) -> list[dict]:
    p = CCTV / f"{date}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    import akshare as ak
    df = ak.news_cctv(date=date)
    items = [{"title": str(r.title), "content": str(r.content)[:160]} for r in df.itertuples()]
    p.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


def snapshot_dict(row) -> dict:
    return {"median_stock_return_today_pct": round(row.median_ret, 2),
            "share_of_stocks_up_today": round(row.up_share, 3),
            "limit_up_count_today": int(row.limit_up), "limit_down_count_today": int(row.limit_down),
            "sum_of_daily_median_returns_last_5_sessions_pct": round(row.median_ret_5d, 2),
            "sum_of_daily_median_returns_last_20_sessions_pct": round(row.median_ret_20d, 2),
            "limit_down_count_last_5_sessions": int(row.limit_down_5d),
            "turnover_zscore_vs_60_sessions": round(row.amount_z60, 2)}


def iso(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def main() -> int:
    CCTV.mkdir(parents=True, exist_ok=True)
    m = market_snapshot()
    m = m[m.label_ok & m.amount_z60.notna() & (m.date >= "20240401")].reset_index(drop=True)

    # free baseline: expanding-window logistic regression (refit monthly, predict next month)
    m["base_p"] = np.nan
    months = sorted(m.date.str[:6].unique())
    for mo in months[6:]:
        tr = m[(m.date.str[:6] < mo)].iloc[:-5]           # drop last 5 rows: labels not yet known
        te = m.date.str[:6] == mo
        if tr.crash_next5.nunique() < 2:
            continue
        lr = LogisticRegression(max_iter=2000).fit(tr[FEATS], tr.crash_next5.astype(int))
        m.loc[te, "base_p"] = lr.predict_proba(m.loc[te, FEATS])[:, 1]

    rng = np.random.default_rng(20260929)
    pos = m[m.crash_next5.astype(bool)].date.to_numpy()
    neg = m[~m.crash_next5.astype(bool)].date.to_numpy()
    sample = sorted(list(rng.choice(pos, N_PER_CLASS, replace=False)) + list(rng.choice(neg, N_PER_CLASS, replace=False)))
    all_dates = m.date.tolist()

    def shifted(d: str) -> str:   # hash-assigned, independent of the label
        h = int(hashlib.sha256(d.encode()).hexdigest(), 16)
        cand = [x for x in all_dates if abs(int(x[:6]) - int(d[:6])) >= 3]
        return cand[h % len(cand)]

    for d in sample:
        cctv(d)
    key = api_key()
    jobs = []
    for d in sample:
        row = m[m.date == d].iloc[0]
        snap, news = snapshot_dict(row), cctv(d)
        jobs += [(d, "A_real", {"as_of": iso(d), "market_after_close": snap, "cctv_evening_news": news}),
                 (d, "B_shifted", {"as_of": iso(shifted(d)), "market_after_close": snap, "cctv_evening_news": news}),
                 (d, "C_no_date", {"market_after_close": snap, "cctv_evening_news": news}),
                 (d, "D_numeric", {"market_after_close": snap})]
    res = []
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(ask, s, key): (d, c) for d, c, s in jobs}
        for fu in cf.as_completed(futs):
            d, c = futs[fu]
            res.append({"date": d, "cond": c, **fu.result()})
    r = pd.DataFrame(res)
    r.to_csv(OUT / "probe_results.csv", index=False)
    t = r[r.ok].pivot(index="date", columns="cond", values="p").join(
        m.set_index("date")[["crash_next5", "base_p"]])
    y = t.crash_next5.astype(int)
    lines = [f"sample {len(t)} dates ({int(y.sum())} positive), calls ok {int(r.ok.sum())}/{len(r)}, model {r.model.dropna().unique()}"]
    for c in ("A_real", "B_shifted", "C_no_date", "D_numeric", "base_p"):
        ok = t[c].notna()
        lines.append(f"{c:10s} AUC {roc_auc_score(y[ok], t.loc[ok, c]):.3f}  mean p {t.loc[ok, c].mean():.3f}  (n={int(ok.sum())})")
    both = t[["A_real", "base_p"]].dropna()
    lines.append(f"corr(A_real, baseline) = {both.A_real.corr(both.base_p):.2f}")
    full = m[m.base_p.notna()]
    lines.append(f"baseline on all {len(full)} days: AUC {roc_auc_score(full.crash_next5.astype(int), full.base_p):.3f}")
    text = "\n".join(lines)
    (OUT / "probe_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
