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
from stockagent_analysis.market_valve import (  # noqa: E402
    LEVEL_CN,
    ValveConfig,
    apply_actions,
    daily_breadth,
    load_contract,
    valve_levels,
)
from stockagent_analysis.s20_pure import (  # noqa: E402
    CONTRACT_PATH,
    SAFE_CONTRACT_PATH,
    PureConfig,
    SafeConfig,
    select_safe,
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


def valve_table() -> pd.DataFrame:
    """Valve level per date: Tushare daily cache, extended by the live Sina rows."""
    files = [p for p in sorted((ROOT / "output/tushare_cache/daily").glob("*.parquet")) if p.stem >= "20260401"]
    b = daily_breadth(pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "pct_chg"]) for p in files]))
    live = ROOT / "output/jev_market/live_market_days.csv"
    if live.exists():
        lv = pd.read_csv(live, dtype={"date": str})[["date", "limit_down"]].rename(columns={"date": "trade_date"})
        b = pd.concat([b, lv[~lv.trade_date.isin(b.trade_date)]], ignore_index=True)
    return valve_levels(b, ValveConfig())


def main() -> int:
    cfg = PureConfig()
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["funnel"] == json.loads(json.dumps(cfg.to_dict())), "config drifted from frozen contract"
    safe_cfg = SafeConfig()
    safe_contract = json.loads(SAFE_CONTRACT_PATH.read_text(encoding="utf-8"))
    assert safe_contract["list"] == json.loads(json.dumps(safe_cfg.to_dict())), "safe config drifted"
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
    lists = pd.concat([select(new, cfg, r.name) for r in cfg.rules] + [select_safe(new, safe_cfg)],
                      ignore_index=True)
    cols = ["trade_date", "rule", "list_rank", "ts_code", "name", "industry", "stage1_probability",
            "pool_rank", "natr14", "natr_pct_in_pool", "natr_pct", "fill"]
    cols = [c for c in cols if c in lists.columns] + ["valve_level", "served_by"]
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
    # "safe and up" view (band exit a5/b15/D10) for every list
    band = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/band_panel.parquet",
                           columns=["ts_code", "trade_date", "maxdd20", "cls_a5_d10", "ret_a5_b15_d10"])
    band = band[band.trade_date >= EVAL_START]

    def safe_metrics(q, name):
        c, r = q.cls_a5_d10, q.ret_a5_b15_d10 - safe_cfg.cost_pct
        return {"set": name, "days": q.trade_date.nunique(), "avg_len": round(len(q) / max(q.trade_date.nunique(), 1), 1),
                "success": round(100 * float(c.isin([1, 3]).mean()), 1), "bad": round(100 * float((c == 2).mean()), 1),
                "crash15": round(100 * float((q.maxdd20 <= -15).mean()), 1), "band_mean": round(float(r.mean()), 2)}
    srows = [safe_metrics(lists[lists.rule == n].merge(band, on=["ts_code", "trade_date"]), n)
             for n in ("safe_v1_1", "U15D10", "U20D15")]
    srows.append(safe_metrics(new[["ts_code", "trade_date"]].merge(band, on=["ts_code", "trade_date"]), "universe"))
    sev = pd.DataFrame(srows)
    report["evaluation_safe_band_a5_b15_d10"] = sev.to_dict(orient="records")

    # --- market valve (monitor mode) + counterfactuals of the pre-registered actions
    vc = load_contract()
    assert vc["config"] == json.loads(json.dumps(ValveConfig().to_dict())), "valve config drifted"
    vt = valve_table().set_index("date")
    # lists as served: action B (user-enabled) swaps the aggressive lists for the safe list on red days
    served = apply_actions(lists, vt.reset_index(), ValveConfig())
    served[cols].to_csv(SHADOW / "daily_lists.csv", index=False, encoding="utf-8-sig")
    lists[[c for c in cols if c in lists.columns]].to_csv(SHADOW / "daily_lists_raw.csv", index=False, encoding="utf-8-sig")
    path = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/path_panel.parquet",
                           columns=["ts_code", "trade_date", "up15_day", "dn10_day", "dn8_day", "ret20"])
    band8 = pd.read_parquet(ROOT / "output/experiments/s20_pure_20260928/band_panel.parquet",
                            columns=["ts_code", "trade_date", "ret_a5_b15_d10", "ret_a5_b15_d8"])
    r1 = cfg.rule("U15D10")
    v1m = lists[lists.rule == "U15D10"].merge(path, on=["ts_code", "trade_date"])
    v1m["ret"] = exit_return(v1m.up15_day, v1m.dn10_day, v1m.ret20, r1)
    v1m["ret_t"] = exit_return(v1m.up15_day, v1m.dn8_day, v1m.ret20, r1)
    sfm = lists[lists.rule == "safe_v1_1"].merge(band8, on=["ts_code", "trade_date"])
    sfm["ret"], sfm["ret_t"] = sfm.ret_a5_b15_d10 - 0.3, sfm.ret_a5_b15_d8 - 0.3
    day = pd.DataFrame({"v1": v1m.groupby("trade_date").ret.mean(), "v1_t": v1m.groupby("trade_date").ret_t.mean(),
                        "safe": sfm.groupby("trade_date").ret.mean(), "safe_t": sfm.groupby("trade_date").ret_t.mean()})
    day["level"] = vt.level.reindex(day.index)
    red, hot = day.level.eq("red"), day.level.isin(["orange", "red"])
    cf_rows = []
    for lst in ("v1", "safe"):
        pol = {"none": day[lst], "A_skip_red": day[lst].where(~red, 0.0),
               "C_tight_orange": day[lst].where(~hot, day[f"{lst}_t"])}
        if lst == "v1":
            pol["B_safe_red"] = day[lst].where(~red, day["safe"])
        for k, sr in pol.items():
            cf_rows.append({"list": lst, "action": k, "days": int(sr.notna().sum()),
                            "affected_days": int((red if "red" in k else hot)[sr.notna()].sum()) if k != "none" else 0,
                            "mean_sleeve": round(float(sr.mean()), 3),
                            "worst_month": round(float(sr.groupby(sr.index.str[:6]).mean().min()), 2)})
    cfd = pd.DataFrame(cf_rows)
    report["valve_counterfactuals"] = cfd.to_dict(orient="records")
    report["valve_levels_new_window"] = day.level.value_counts().to_dict()
    print(cfd.to_string(index=False))
    print(sev.to_string(index=False))
    report["evaluation"] = {"signal_days": [EVAL_START, matured_to], "matured_days": int(ev.days.max()),
                            "promotion_review_needs": 60, "table": ev.to_dict(orient="records")}
    print(ev.to_string(index=False))

    vcfg = ValveConfig()
    lv = vt.loc[last] if last in vt.index else None
    level_today = lv.level if lv is not None else "unknown"
    b_on = "B_safe_red" in vcfg.enabled_actions
    triggered = b_on and level_today == "red"
    valve_line = (f"市场预警阀门（{last}）：**{LEVEL_CN.get(level_today, '未知')}**，近 5 日跌停合计 "
                  f"{int(lv.limit_down_5d) if lv is not None and pd.notna(lv.limit_down_5d) else '—'} 家"
                  f"（黄 ≥{vcfg.yellow_at}，橙 ≥{vcfg.orange_at}，红 ≥{vcfg.red_at}）。"
                  f"动作 B（红色时进攻版改用稳健版）{'已启用' if b_on else '未启用'}，"
                  f"今日{'**已触发**：进攻版名单由稳健版替代' if triggered else '未触发'}。")
    newest = max(vt.index[vt.level != "unknown"])
    if newest > last:
        valve_line += f" 最新一晚 {newest} 的阀门为 **{LEVEL_CN.get(vt.loc[newest].level)}**（近 5 日跌停 {int(vt.loc[newest].limit_down_5d)} 家）。"
    report["latest_valve"] = {"list_date": last, "level": level_today, "action_B_triggered": bool(triggered),
                              "newest_valve": {"date": newest, "level": vt.loc[newest].level}}
    report["served_lists_red_days"] = sorted(served.loc[served.served_by.eq("safe_v1_1") & served.rule.ne("safe_v1_1"),
                                                        "trade_date"].unique().tolist())
    md = [f"# S20-Pure v1 名单 {last}\n",
          "冻结合约 `config/s20_pure_v1.json`、`config/s20_pure_valve_v1.json`；影子运行，非投资建议。\n",
          f"\n{valve_line}\n"]
    for r in cfg.rules:
        q = lists[(lists.trade_date == last) & (lists.rule == r.name)]
        md.append(f"\n## 进攻版 {r.name}（止盈 +{r.take_profit:g}% / 止损 -{r.stop_loss:g}% / {r.horizon} 日，"
                  f"池内截掉 natr 最高 {int(r.amplitude_cap*100)}%）\n\n")
        if triggered:
            md.append("> 今日红色预警，按动作 B 本规则改用下方稳健版名单；原名单仅供参考，不建议建仓。\n\n")
        md.append("| # | 代码 | 名称 | 行业 | stage1 | 池内名次 | natr14 |\n|---|---|---|---|---|---|---|\n")
        for _, x in q.iterrows():
            md.append(f"| {int(x.list_rank)} | {x.ts_code} | {x.get('name','')} | {x.get('industry','')} | "
                      f"{x.stage1_probability:.3f} | {int(x.pool_rank)} | {x.natr14:.3f} |\n")
    q = lists[(lists.trade_date == last) & (lists.rule == "safe_v1_1")]
    md.append(f"\n## 稳健版 v1.1（全市场波动最低 {int(safe_cfg.natr_pct_max*100)}% → stage1 排序 → 前 {safe_cfg.top_k}，"
              f"每行业 ≤ {safe_cfg.industry_cap}，超出部分由其他行业替补；止盈区间 +{safe_cfg.band_low:g}%～+{safe_cfg.band_high:g}%，"
              f"止损 -{safe_cfg.crash_line:g}%）\n\n| # | 代码 | 名称 | 行业 | stage1 | 波动分位 | 来源 |\n|---|---|---|---|---|---|---|\n")
    for i, (_, x) in enumerate(q.iterrows(), 1):
        md.append(f"| {i} | {x.ts_code} | {x.get('name','')} | {x.get('industry','')} | {x.stage1_probability:.3f} | "
                  f"{x.natr_pct:.2f} | {'替补' if x.fill == 'industry_substitute' else '前20'} |\n")
    (SHADOW / f"list_{last}.md").write_text("".join(md), encoding="utf-8")
    (SHADOW / "shadow_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"latest list {last}: {SHADOW / f'list_{last}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
