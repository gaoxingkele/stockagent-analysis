#!/usr/bin/env python
"""S20-Pure v1 shadow run on data after the freeze.

1. Load stage1 factors rebuilt for 2026-07-27..latest (build_s20_20r_confirmation_factors.py).
2. Integrity: on the 8 overlap days (07-27..08-05) compare rebuilt factors and
   frozen-model scores with the archived confirmation run.
3. Score every day with the frozen stage1, apply the frozen funnel, write the
   daily lists (including the latest, unmatured days).
4. Evaluate only signal days >= 20260806 whose 20-session horizon has matured,
   against the two pre-registered baselines. Shadow evidence, not promotion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.s20_pure import (  # noqa: E402
    CONTRACT_PATH,
    PureConfig,
    _frozen_models,
    exit_return,
    natr,
    select,
    stage1_probability,
    three_state,
)

SHADOW = ROOT / "output/experiments/s20_pure_v1_shadow"
CONFIRM = ROOT / "output/experiments/s20_20r_confirmation"
EVAL_START = "20260806"
REGIME_COLS = {"ret_5d": "mkt_ret_5d", "ret_20d": "mkt_ret_20d", "ret_60d": "mkt_ret_60d",
               "rsi14": "mkt_rsi14", "vol_ratio": "mkt_vol_ratio"}


def load_factors(factor_dir: Path, features: list[str], start: str) -> pd.DataFrame:
    raw = [f for f in features if f not in {"industry_id", "regime_id", *REGIME_COLS.values(), "regime_days_in",
                                             "regime_intensity", "hs300_ret60_z60", "cyb_rel_strength",
                                             "zz500_rel_strength"}]
    parts = [pd.read_parquet(p, columns=["ts_code", "trade_date", "industry", *raw])
             for p in sorted(factor_dir.glob("group_*.parquet"))]
    f = pd.concat(parts, ignore_index=True)
    f["ts_code"], f["trade_date"] = f.ts_code.astype(str), f.trade_date.astype(str)
    f = f[f.trade_date >= start]
    reg = pd.read_parquet(ROOT / "output/regimes/daily_regime.parquet").rename(columns=REGIME_COLS)
    ext = pd.read_parquet(ROOT / "output/regime_extra/regime_extra.parquet")
    for x in (reg, ext):
        x["trade_date"] = x["trade_date"].astype(str)
    reg = reg.merge(ext, on="trade_date", how="left")
    f = f.merge(reg[["trade_date", *[c for c in features if c in reg.columns]]], on="trade_date", how="left")
    meta = json.loads((ROOT / "output/lgbm_maxgain/feature_meta.json").read_text(encoding="utf-8"))
    imap = meta.get("industry_map", {})
    f["industry_id"] = f["industry"].fillna("unknown").astype(str).map(lambda v: imap.get(v, -1))
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet", columns=["ts_code", "name"])
    st = set(basic.loc[basic.name.fillna("").str.contains("ST", regex=False), "ts_code"].astype(str))
    f = f[~f.ts_code.isin(st)].copy()
    for c in features:
        f[c] = pd.to_numeric(f[c], errors="coerce") if c in f else np.nan
    return f.merge(basic, on="ts_code", how="left").sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def daily_natr(start: str) -> pd.DataFrame:
    files = [p for p in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")) if p.stem >= "20260501"]
    d = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "high", "low", "close", "pre_close"])
                   for p in files], ignore_index=True)
    d["trade_date"] = d.trade_date.astype(str)
    d["natr14"] = natr(d, 14)
    return d.loc[d.trade_date >= start, ["ts_code", "trade_date", "natr14"]]


def summarize(q: pd.DataFrame, rule, name: str) -> dict:
    r = rule
    up, dn = q[f"up{int(r.take_profit)}_day"], q[f"dn{int(r.stop_loss)}_day"]
    shake = q[f"dn{int(r.shakeout)}_day"] if r.shakeout else None
    st = pd.Series(three_state(up, dn, shake, r.horizon))
    ex = exit_return(up, dn, q["ret20"], r)
    return {"rule": r.name, "set": name, "days": q.trade_date.nunique(), "n": len(q),
            "win": round(100 * float((ex > 0).mean()), 1), "mean": round(float(np.nanmean(ex)), 2),
            "pure_up": round(100 * float((st == "pure_up").mean()), 1),
            "dirty_up": round(100 * float((st == "dirty_up").mean()), 1),
            "pure_down": round(100 * float((st == "pure_down").mean()), 1),
            "chop": round(100 * float((st == "chop").mean()), 1)}


def main() -> int:
    cfg = PureConfig()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["funnel"] == json.loads(json.dumps(cfg.to_dict())), "config drifted from frozen contract"
    _, _, features = _frozen_models()
    f = load_factors(SHADOW / "factor_groups", features, "20260727")
    f["stage1_probability"] = stage1_probability(f)
    f = f.merge(daily_natr("20260727"), on=["ts_code", "trade_date"], how="left")
    report = {"frozen_contract": str(CONTRACT_PATH.relative_to(ROOT)), "dates": [f.trade_date.min(), f.trade_date.max()],
              "rows": len(f)}

    # --- integrity on the overlap with the archived confirmation window
    ov = f[f.trade_date <= "20260805"]
    old = load_factors(CONFIRM / "factor_groups", features, "20260727")
    old = old[old.trade_date <= "20260805"]
    j = ov.merge(old, on=["ts_code", "trade_date"], suffixes=("", "_old"))
    num = [c for c in features if c in j and f"{c}_old" in j and c != "industry_id"]
    mism = {c: float((np.abs(j[c] - j[f"{c}_old"]) > 1e-6 * (1 + np.abs(j[f"{c}_old"]))).mean()) for c in num}
    saved = pd.read_parquet(CONFIRM / "run_v1/predictions.parquet", columns=["ts_code", "trade_date", "stage1_probability"])
    k = ov.merge(saved, on=["ts_code", "trade_date"], suffixes=("", "_saved"))
    overlap = k.groupby("trade_date").apply(lambda g: len(set(g.nlargest(20, "stage1_probability").ts_code)
                                                          & set(g.nlargest(20, "stage1_probability_saved").ts_code)) / 20)
    report["integrity_overlap_0727_0805"] = {
        "rows_matched": len(j), "features_with_any_mismatch": {c: round(v, 4) for c, v in mism.items() if v > 0},
        "stage1_max_abs_diff_vs_archive": float((k.stage1_probability - k.stage1_probability_saved).abs().max()),
        "top20_overlap_vs_archive": round(float(overlap.mean()), 3)}
    print(json.dumps(report["integrity_overlap_0727_0805"], indent=1, ensure_ascii=False), flush=True)

    # --- daily lists
    new = f[f.trade_date >= EVAL_START].copy()
    lists = pd.concat([select(new, cfg, r.name) for r in cfg.rules], ignore_index=True)
    cols = ["trade_date", "rule", "list_rank", "ts_code", "name", "industry", "stage1_probability",
            "pool_rank", "natr14", "natr_pct_in_pool"]
    lists[cols].to_csv(SHADOW / "daily_lists.csv", index=False, encoding="utf-8-sig")
    last = lists.trade_date.max()
    report["latest_list_date"] = last

    # --- matured evaluation
    panel = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/path_panel.parquet")
    panel = panel[panel.trade_date >= EVAL_START]
    matured_to = panel.trade_date.max()
    rows = []
    for r in cfg.rules:
        L = lists[(lists.rule == r.name)].merge(panel, on=["ts_code", "trade_date"])
        base = new.copy()
        base["rk"] = base.groupby("trade_date")["stage1_probability"].rank(ascending=False, method="first")
        B = base[base.rk <= cfg.top_k].merge(panel, on=["ts_code", "trade_date"])
        Uv = new[["ts_code", "trade_date"]].merge(panel, on=["ts_code", "trade_date"])
        rows += [summarize(L, r, "s20_pure_v1"), summarize(B, r, "stage1_top20_nocap"), summarize(Uv, r, "universe")]
    ev = pd.DataFrame(rows)
    report["evaluation"] = {"signal_days": [EVAL_START, matured_to], "matured_days": int(ev.days.max()),
                            "promotion_review_needs": 60, "table": ev.to_dict(orient="records")}
    (SHADOW / "shadow_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(ev.to_string(index=False))

    md = [f"# S20-Pure v1 名单 {last}\n", "冻结合约 `config/s20_pure_v1.json`；影子运行，非投资建议。\n"]
    for r in cfg.rules:
        q = lists[(lists.trade_date == last) & (lists.rule == r.name)]
        md.append(f"\n## 规则 {r.name}（止盈 +{r.take_profit:g}% / 止损 -{r.stop_loss:g}% / {r.horizon} 日，"
                  f"池内截掉 natr 最高 {int(r.amplitude_cap*100)}%）\n\n| # | 代码 | 名称 | 行业 | stage1 | 池内名次 | natr14 |\n|---|---|---|---|---|---|---|\n")
        for _, x in q.iterrows():
            md.append(f"| {int(x.list_rank)} | {x.ts_code} | {x.get('name','')} | {x.get('industry','')} | "
                      f"{x.stage1_probability:.3f} | {int(x.pool_rank)} | {x.natr14:.3f} |\n")
    (SHADOW / f"list_{last}.md").write_text("".join(md), encoding="utf-8")
    print(f"latest list {last}: {SHADOW / f'list_{last}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
