#!/usr/bin/env python
"""Retrain R5 and S5 to veto names likely to print -5% or -7% within 5 sessions.

Does not touch production models. Feature cache and stage-1 ranks overlap only
through 2026-01-26, so the clock is split inside that window:

  train  first 70% of signal dates
  test   last 30%  (not used to choose the model or the threshold)

R5 is fit on the daily top 200 by the nested R20 score.
S5 is fit on the daily top 200 by stage-1.
Label: the low versus the next session's open hits -5% (or -7%) inside 5 sessions.

List rule, walked in rank order: skip a name when its probability is at least
0.50, keep going until 20 names or the rank cap (20 = no refill, 50 or 80).
JEV is an auxiliary vote on S5 only, and only for a capped sample of names
whose S5 probability sits near 0.50. It is not applied to the whole list.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXP = ROOT / "output/experiments/s20_pure_20260928"
OUT = ROOT / "output/experiments/r5_s5_break_20261004"
FEAT = EXP / "feat_cache_bps5000.parquet"
KEEP_FEATS = [
    "adx", "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",
    "ma5_ma20", "ma20_ma60", "macd_hist", "rsi_6", "rsi_14", "rsi_24",
    "kdj_k", "kdj_d", "boll_pct", "boll_width", "atr_pct", "bias_5",
    "bias_10", "bias_20", "roc_10", "roc_20", "cci_14", "wr_14", "mfi_14",
    "trix", "vol_ratio_5", "vol_ratio_20", "amplitude", "amount_ratio_20",
]


def auc(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_frame() -> pd.DataFrame:
    rank = pd.read_parquet(
        EXP / "r04_preds_U15_D10.parquet",
        columns=["ts_code", "trade_date", "stage1_score", "s1_rank", "dn5_day"],
    )
    r20 = pd.read_parquet(
        ROOT / "output/experiments/s20_20r_residual_portable/predictions.parquet",
        columns=["ts_code", "trade_date", "r20_nested_anchor"],
    )
    for df in (rank, r20):
        df["trade_date"] = df["trade_date"].astype(str)
        df["ts_code"] = df["ts_code"].astype(str)
    m = rank.merge(r20, on=["ts_code", "trade_date"], how="inner")
    m["s_rank"] = m["s1_rank"].astype(int)
    m["r_rank"] = m.groupby("trade_date")["r20_nested_anchor"].rank(ascending=False, method="first").astype(int)
    use = m[(m.s_rank <= 200) | (m.r_rank <= 200)].copy()
    feats = pd.read_parquet(FEAT, columns=["ts_code", "trade_date", *KEEP_FEATS])
    feats["trade_date"] = feats["trade_date"].astype(str)
    feats["ts_code"] = feats["ts_code"].astype(str)
    use = use.merge(feats, on=["ts_code", "trade_date"], how="left")
    use["y5"] = ((use.dn5_day > 0) & (use.dn5_day <= 5)).astype(np.int8)
    return use


def add_y7(frame: pd.DataFrame) -> pd.DataFrame:
    """Min low over the 5 sessions starting at the next open, versus that open."""
    daily_dir = ROOT / "output/tushare_cache/daily"
    files = sorted(p for p in daily_dir.glob("*.parquet") if p.stem >= "20250201")
    px = pd.concat(
        [pd.read_parquet(p, columns=["ts_code", "trade_date", "open", "low"]) for p in files],
        ignore_index=True,
    )
    px["trade_date"] = px["trade_date"].astype(str)
    px["ts_code"] = px["ts_code"].astype(str)
    dates = np.array(sorted(px.trade_date.unique()))
    idx = {d: i for i, d in enumerate(dates)}
    need = set(frame.ts_code)
    px = px[px.ts_code.isin(need)]
    op = px.pivot(index="trade_date", columns="ts_code", values="open").reindex(dates)
    lo = px.pivot(index="trade_date", columns="ts_code", values="low").reindex(dates)
    codes = {c: op.columns.get_loc(c) for c in frame.ts_code.unique() if c in op.columns}
    y7 = np.full(len(frame), np.nan)
    y5_check = np.full(len(frame), np.nan)
    for n, (code, day) in enumerate(zip(frame.ts_code.to_numpy(), frame.trade_date.to_numpy())):
        j = codes.get(code)
        i = idx.get(day)
        if j is None or i is None or i + 5 >= len(dates):
            continue
        entry = op.iat[i + 1, j]
        if not (entry > 0) or not np.isfinite(entry):
            continue
        window = lo.iloc[i + 1:i + 6, j].to_numpy(dtype=float)
        if np.isfinite(window).sum() < 4:
            continue
        dd = np.nanmin(window) / entry - 1.0
        y7[n] = dd <= -0.07
        y5_check[n] = dd <= -0.05
    frame = frame.copy()
    frame["y7"] = y7
    agree = np.nanmean(y5_check == frame.y5.to_numpy())
    print(f"y5 calendar check agreement {agree:.3f}  y7 rate {np.nanmean(y7):.3f}", flush=True)
    return frame


def fit_one(train: pd.DataFrame, valid: pd.DataFrame, label: str):
    import lightgbm as lgb

    cols = [c for c in KEEP_FEATS if c in train.columns]
    y = train[label].to_numpy()
    ok = np.isfinite(y)
    pos = max(float(y[ok].sum()), 1.0)
    neg = max(float(ok.sum() - y[ok].sum()), 1.0)
    model = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, scale_pos_weight=neg / pos,
        random_state=20261004, n_jobs=4, verbose=-1,
    )
    model.fit(
        train.loc[ok, cols], y[ok].astype(int),
        eval_set=[(valid[cols], valid[label].astype(float))],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )
    return model, cols


def predict(model, cols, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[cols])[:, 1]


def take(day: pd.DataFrame, pcol: str, cap: int, drop_at: float | None) -> pd.DataFrame:
    q = day[day["rank"] <= cap].sort_values("rank")
    if drop_at is None:
        return q.nsmallest(20, "rank")
    keep = q[~(q[pcol] >= drop_at)]
    return keep.head(20)


def metrics(picked: pd.DataFrame, band: pd.DataFrame) -> dict:
    if picked.empty:
        return {"n": 0}
    m = picked.merge(band, on=["ts_code", "trade_date"], how="inner")
    if m.empty:
        return {"n": 0}
    per = m.groupby("trade_date").size()
    return {
        "names": int(len(m)),
        "days": int(m.trade_date.nunique()),
        "avg_len": round(float(per.mean()), 2),
        "full20": round(float((per >= 20).mean()), 3),
        "band_ok": round(100 * m.cls_a5_d10.isin([1, 3]).mean(), 1),
        "band_bad": round(100 * (m.cls_a5_d10 == 2).mean(), 1),
        "ret_band": round(float(m.ret_a5_b15_d10.mean()), 2),
    }


def jev_key() -> str:
    for name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    for path in (ROOT / ".env", Path("C:/aicoding/jev.env.txt")):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip() in ("TYPESAFE_API_KEY", "JEV_API_KEY") and v.strip():
                return v.strip().strip("\"'")
            if path.name == "jev.env.txt" and line.strip().startswith("apikey_"):
                return line.strip()
    raise RuntimeError("no jev credential")


def jev_batch(rows: list[dict], barrier: int) -> list[float | None]:
    """One request, one noul per name. Returns probabilities of 'will break'."""
    questions = {}
    for i, row in enumerate(rows):
        questions[f"n{i}"] = {
            "type": "noul",
            "instructions": (
                f"只根据这段推荐日已经知道的摘要，判断未来 5 个交易日的最低价"
                f"会不会触及相对次日开盘 {barrier}% 。"
                f"近 5 日价格位置 bias_5={row['bias_5']:.2f}，波动 atr_pct={row['atr_pct']:.2f}，"
                f"振幅 amplitude={row['amplitude']:.2f}，RSI14={row['rsi_14']:.1f}，"
                f"在 stage1 候选池中的名次={int(row['s_rank'])}。"
                "不要使用摘要里没有的信息。"
            ),
            "criteria": {"true": "会先触及该下跌线", "false": "不会先触及该下跌线"},
        }
    payload = {
        "model": "jev-latest",
        "state": {
            "task": "5日最低价是否先触及固定下跌线",
            "known_at": "推荐日收盘",
            "entry": "次日开盘",
            "horizon_sessions": 5,
            "barrier_pct": barrier,
            "n": len(rows),
        },
        "questions": questions,
    }
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        "https://api.typesafe.ai/v1/systemone",
        data=raw,
        headers={"Authorization": "Bearer " + jev_key(), "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        body = json.load(resp)
    out = []
    for i in range(len(rows)):
        ans = body.get("answers", {}).get(f"n{i}", {})
        val = ans.get("noul")
        out.append(float(val) if isinstance(val, (int, float)) else None)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("load", flush=True)
    frame = add_y7(load_frame())
    days = np.array(sorted(frame.trade_date.unique()))
    cut = days[int(len(days) * 0.70)]
    vcut = days[int(len(days) * 0.55)]
    print(f"dates {days[0]}..{days[-1]}  n={len(days)}  train<{vcut}  valid {vcut}..{cut}  test>={cut}", flush=True)
    train_mask = frame.trade_date < vcut
    valid_mask = (frame.trade_date >= vcut) & (frame.trade_date < cut)
    test_mask = frame.trade_date >= cut

    specs = {
        "S5": ("s_rank", "y5", "y7"),
        "R5": ("r_rank", "y5", "y7"),
    }
    models = {}
    test = frame[test_mask].copy()
    for name, (rank_col, lab5, lab7) in specs.items():
        pool = frame[frame[rank_col] <= 200]
        tr, va = pool[pool.trade_date < vcut], pool[(pool.trade_date >= vcut) & (pool.trade_date < cut)]
        for lab, tag in ((lab5, "5"), (lab7, "7")):
            print(f"fit {name} barrier {tag} train {tr[lab].notna().sum()} pos {tr[lab].mean():.3f}", flush=True)
            model, cols = fit_one(tr[tr[lab].notna()], va[va[lab].notna()], lab)
            models[(name, tag)] = (model, cols, rank_col)
            te = test[test[rank_col] <= 200]
            p = predict(model, cols, te)
            test.loc[te.index, f"p_{name}_{tag}"] = p
            print(f"  test AUC {auc(te[lab], p):.3f}  base {te[lab].mean():.3f}  pred mean {p.mean():.3f}", flush=True)

    band = pd.read_parquet(
        EXP / "band_panel.parquet",
        columns=["ts_code", "trade_date", "cls_a5_d10", "ret_a5_b15_d10"],
    )
    band["trade_date"] = band.trade_date.astype(str)
    band["ts_code"] = band.ts_code.astype(str)
    band = band[band.trade_date >= cut]

    lines = [f"train dates < {vcut}  early-stop {vcut}..{cut}  test >= {cut}", ""]
    rows = []
    for name, rank_col in (("S5", "s_rank"), ("R5", "r_rank")):
        queue = test[test[rank_col] <= 80].copy()
        queue["rank"] = queue[rank_col]
        for tag in ("5", "7"):
            pcol = f"p_{name}_{tag}"
            for cap, drop, label in (
                (20, None, "top20"),
                (20, 0.50, "filter20"),
                (50, 0.50, "refill50"),
                (80, 0.50, "refill80"),
            ):
                parts = []
                for _, day in queue.groupby("trade_date"):
                    parts.append(take(day, pcol, cap, drop))
                picked = pd.concat(parts, ignore_index=True) if parts else queue.iloc[0:0]
                stat = metrics(picked, band)
                stat.update(model=name, barrier=tag, rule=label)
                rows.append(stat)
                lines.append(f"{name} -{tag}% {label:8s} {stat}")
        # secondary: drop the riskiest 30% inside that day's top 50, then refill inside 50
        pcol = f"p_{name}_5"
        parts = []
        for _, day in queue.groupby("trade_date"):
            d = day[day["rank"] <= 50].copy()
            if len(d) >= 5:
                cut_p = d[pcol].quantile(0.70)
                d = d[d[pcol] < cut_p]
            parts.append(d.sort_values("rank").head(20))
        picked = pd.concat(parts, ignore_index=True)
        stat = metrics(picked, band)
        stat.update(model=name, barrier="5", rule="drop_riskiest_30_of_50")
        rows.append(stat)
        lines.append(f"{name} -5% drop_riskiest_30_of_50 {stat}")

    # JEV auxiliary on S5, barrier -5, probability in [0.40, 0.60], capped sample
    border = test[(test.s_rank <= 50) & test.p_S5_5.between(0.40, 0.60)].copy()
    border = border.dropna(subset=["bias_5", "atr_pct", "amplitude", "rsi_14", "y5"])
    sample = border.sample(n=min(16, len(border)), random_state=20261004)
    jev_note = "JEV unavailable"
    try:
        probs = []
        recs = sample.to_dict("records")
        for start in range(0, len(recs), 8):
            chunk = recs[start:start + 8]
            probs.extend(jev_batch(chunk, -5))
            time.sleep(0.3)
        sample = sample.copy()
        sample["jev"] = probs
        both = sample.dropna(subset=["jev"])
        jev_auc = auc(both.y5, both.jev)
        s5_auc = auc(both.y5, both.p_S5_5)
        # auxiliary rule on this sample only: trust S5 outside the band;
        # inside the band, drop when JEV noul >= 0.60
        jev_hit = float(((both.jev >= 0.60) == (both.y5 == 1)).mean()) if len(both) else float("nan")
        s5_hit = float(((both.p_S5_5 >= 0.50) == (both.y5 == 1)).mean()) if len(both) else float("nan")
        jev_note = (
            f"S5 uncertain band n={len(both)}  JEV AUC {jev_auc:.3f}  S5 AUC {s5_auc:.3f}  "
            f"JEV drop-if-noul>=0.60 accuracy {jev_hit:.3f}  S5 p>=0.50 accuracy {s5_hit:.3f}"
        )
        both[["ts_code", "trade_date", "s_rank", "y5", "p_S5_5", "jev"]].to_csv(
            OUT / "jev_border_sample.csv", index=False
        )
    except Exception as exc:  # noqa: BLE001 — record the failure, keep the price-model result
        jev_note = f"JEV call failed: {type(exc).__name__}"
    lines += ["", jev_note]
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text, encoding="utf-8")
    pd.DataFrame(rows).to_csv(OUT / "grid.csv", index=False)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
