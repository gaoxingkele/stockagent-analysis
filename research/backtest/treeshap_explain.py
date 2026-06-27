# -*- coding: utf-8 -*-
"""DLV-001 (Roadmap A2) — TreeSHAP 可解释层: 对 PIT-clean r20 当月 picks 逐只票加性归因.

北极星 (部署/审计, 非 alpha):
  给 PIT-clean r20 (dep001 剔 holder_pct 的月度 walk-forward 模型) 的每日推荐附"为什么打高分" —
  用 LightGBM booster 内置 TreeSHAP (`predict(..., pred_contrib=True)`, 无需 shap 库) 对每只 pick 算
  逐特征加性贡献, 输出 top 正/负贡献因子表。用途 = 前向 paper-trade 解释 / 向出资人说清"赚的什么钱"
  (QuantML 论文核心用例), 提高可审计性, **不提高选股** (这是解释层, 不是新信号)。

加性性质 (TreeSHAP 保证): 对回归 booster, 每行 sum(feature_contribs) + base_value == 模型 raw 预测
  (= pred_r20 分数)。因此可把每只 pick 的分数完整拆成 "各因子推高/拉低了多少分"。

口径 (与 dep001 / WFE-001 逐字一致, apples-to-apples, R01/R05):
  - 模型: research/cache/dep001/r20_models/202604/classifier.txt (PIT-clean r20, 剔 holder_pct, embargo WF)
  - 特征: feature_meta.json 的 246 列 (含 industry_id), booster.feature_name() 顺序逐位对齐
  - 数据: load_window(DATA_START, DATA_END, with_mfk=True) 同 dep001 (ST 源头排除 R06), 取 202604 月行
  - picks: research/cache/dep001/picks_by_month/202604.parquet (PIT-clean OOS 当月入选持仓)
  - 缺失值不 fillna (同 dep001 r20 预测口径: float32 + replace inf->nan), LightGBM 原生处理 NaN

demo (最新月 202604) 落 output/:
  - output/treeshap_dep001_202604_contribs.csv  逐 (date,stock,feature) 贡献长表 (top±各因子)
  - output/treeshap_dep001_202604_demo.md       最新再平衡日 picks 的人读归因表 (top5 正/负 + base)

生产线 V12.31 只读冻结 (R05, 仅 load 缓存模型, 未碰生产文件)。大缓存 gitignored (R04)。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"
sys.path.insert(0, str(RESEARCH))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))

from train_v15_refresh import load_window
from wfe001_gen_picks import DATA_START, DATA_END

DEMO_MONTH = "202604"
TOP_K = 5  # 每只 pick 展示 top-K 正 / top-K 负贡献因子

DEP_DIR = ROOT / "research" / "cache" / "dep001"
MODEL_FILE = DEP_DIR / "r20_models" / DEMO_MONTH / "classifier.txt"
META_FILE = DEP_DIR / "r20_models" / DEMO_MONTH / "feature_meta.json"
PICKS_FILE = DEP_DIR / "picks_by_month" / f"{DEMO_MONTH}.parquet"

OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUT_DIR / f"treeshap_dep001_{DEMO_MONTH}_contribs.csv"
MD_OUT = OUT_DIR / f"treeshap_dep001_{DEMO_MONTH}_demo.md"
VERDICT = ROOT / "research" / "verdicts" / "DLV-001.json"


def stock_name_map() -> dict:
    p = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
    if not p.exists():
        return {}
    b = pd.read_parquet(p)[["ts_code", "name"]].drop_duplicates("ts_code")
    return dict(zip(b["ts_code"], b["name"]))


def main():
    t0 = time.time()
    print(f"\n=== DLV-001 TreeSHAP 可解释层: PIT-clean r20 {DEMO_MONTH} picks 逐只票加性归因 ===\n",
          flush=True)
    for f in (MODEL_FILE, META_FILE, PICKS_FILE):
        if not f.exists():
            raise FileNotFoundError(f"缺前置工件: {f}")

    booster = lgb.Booster(model_str=MODEL_FILE.read_text(encoding="utf-8"))
    feat_cols = booster.feature_name()
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    assert feat_cols == meta["feature_cols"], "booster 特征顺序与 feature_meta 不一致"
    assert "holder_pct" not in feat_cols, "PIT-clean 模型不应含 holder_pct"
    print(f"[model] PIT-clean r20 {DEMO_MONTH}: {len(feat_cols)} 特征 (剔 holder_pct), "
          f"target={meta['target']}", flush=True)

    # ── 与 dep001 逐字一致地重建 202604 特征 (faithful) ──
    print(f"[data] load_window {DATA_START}-{DATA_END} (ST 源头排除 R06, +mfk) ...", flush=True)
    daily = load_window(DATA_START, DATA_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    industries = pd.Categorical(daily["industry"].fillna("unknown"))
    daily["industry_id"] = industries.codes.astype(int)
    for c in feat_cols:
        if c not in daily.columns:
            daily[c] = np.nan  # 与 full-window dep001 一致: 缺源因子为 NaN (非 0), LGBM 原生处理
    dm = daily[daily["trade_date"].str.startswith(DEMO_MONTH)].copy()
    print(f"[data] {DEMO_MONTH} 月 {len(dm):,} 行 / {dm['ts_code'].nunique()} 股", flush=True)

    picks = pd.read_parquet(PICKS_FILE)
    picks["trade_date"] = picks["trade_date"].astype(str)
    print(f"[picks] {len(picks):,} 持仓行 / {picks['trade_date'].nunique()} 再平衡日 / "
          f"{picks['ts_code'].nunique()} 只票", flush=True)

    # 把 picks 与当月特征对齐 (ts_code, trade_date)
    keep = ["ts_code", "trade_date", "alloc_pct"]
    df = picks[keep].merge(dm[["ts_code", "trade_date"] + feat_cols],
                           on=["ts_code", "trade_date"], how="left")
    miss = df[feat_cols].isna().all(axis=1)
    if miss.any():
        print(f"[warn] {int(miss.sum())} 只 pick 在当月特征缺失 (剔除), 余 {int((~miss).sum())}", flush=True)
        df = df[~miss].reset_index(drop=True)

    # ── TreeSHAP: booster 内置 pred_contrib (无需 shap 库) ──
    X = df[feat_cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    contrib = booster.predict(X.values, pred_contrib=True)  # (n, n_feat+1), 末列 = base_value
    base_value = float(contrib[0, -1])
    feat_contrib = contrib[:, :-1]  # (n, n_feat)
    pred = booster.predict(X.values)  # raw 预测 (= pred_r20 分数)

    # 加性自检: sum(feature_contribs) + base == pred
    recon = feat_contrib.sum(axis=1) + contrib[:, -1]
    max_err = float(np.nanmax(np.abs(recon - pred)))
    print(f"[shap] base_value={base_value:+.4f}; 加性自检 max|Σcontrib+base - pred|={max_err:.2e} "
          f"({'PASS' if max_err < 1e-4 else 'FAIL'})", flush=True)
    assert max_err < 1e-4, "TreeSHAP 加性性质未满足"

    df["pred_r20"] = pred
    df["base_value"] = base_value
    name_map = stock_name_map()
    df["name"] = df["ts_code"].map(name_map).fillna("")

    # ── 逐只 pick 提取 top±贡献因子 → 长表 ──
    rows = []
    fc_arr = np.array(feat_cols)
    for i, r in df.iterrows():
        c = feat_contrib[i]
        order = np.argsort(c)  # 升序: 最负在前
        neg_idx = [j for j in order if c[j] < 0][:TOP_K]
        pos_idx = [j for j in order[::-1] if c[j] > 0][:TOP_K]
        for sign, idxs in (("+", pos_idx), ("-", neg_idx)):
            for rank, j in enumerate(idxs, 1):
                rows.append({
                    "trade_date": r["trade_date"], "ts_code": r["ts_code"], "name": r["name"],
                    "alloc_pct": r["alloc_pct"], "pred_r20": round(float(r["pred_r20"]), 4),
                    "base_value": round(base_value, 4),
                    "sign": sign, "rank": rank, "feature": fc_arr[j],
                    "feature_value": round(float(X.iloc[i, j]), 6) if pd.notna(X.iloc[i, j]) else None,
                    "shap_contrib": round(float(c[j]), 5),
                })
    long_df = pd.DataFrame(rows)
    long_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"[out] 逐 (date,stock,feature) 贡献长表 -> {CSV_OUT.relative_to(ROOT)} ({len(long_df):,} 行)",
          flush=True)

    # ── 全期 top 驱动因子 (|mean shap| 排序, 审计概览) ──
    mean_abs = np.nanmean(np.abs(feat_contrib), axis=0)
    mean_signed = np.nanmean(feat_contrib, axis=0)
    drivers = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs,
                            "mean_signed_shap": mean_signed}).sort_values(
        "mean_abs_shap", ascending=False).head(15)

    # ── 人读 demo: 最新再平衡日 picks 的归因表 ──
    last_day = df["trade_date"].max()
    day = df[df["trade_date"] == last_day].sort_values("pred_r20", ascending=False).reset_index(drop=True)
    lines = []
    lines.append(f"# DLV-001 TreeSHAP 可解释层 — PIT-clean r20 {DEMO_MONTH} picks 逐只票加性归因\n")
    lines.append(f"> 模型: `dep001/r20_models/{DEMO_MONTH}` (PIT-clean r20, 剔 holder_pct, label-embargo "
                 f"walk-forward) · TreeSHAP via LightGBM `pred_contrib=True` (无 shap 库) · "
                 f"base_value (全样本均分锚) = **{base_value:+.4f}**\n")
    lines.append(f"加性性质: 每只 pick 的 `pred_r20 = base_value + Σ(各因子 SHAP 贡献)`, "
                 f"加性自检 max 误差 {max_err:.1e} (PASS)。SHAP 单位 = r20 预测分 (≈ 20 日前向收益率%)。\n")
    lines.append(f"## 全期 (整月 {len(df)} 只 picks) top-15 驱动因子 (按 |mean SHAP| 排序)\n")
    lines.append("| # | 因子 | mean\\|SHAP\\| | mean SHAP (signed) |")
    lines.append("|---|------|------------|--------------------|")
    for k, (_, rr) in enumerate(drivers.iterrows(), 1):
        lines.append(f"| {k} | `{rr['feature']}` | {rr['mean_abs_shap']:.4f} | "
                     f"{rr['mean_signed_shap']:+.4f} |")
    lines.append(f"\n## 最新再平衡日 {last_day} 的 {len(day)} 只 picks — 逐只 top{TOP_K} 正/负贡献\n")
    for _, r in day.iterrows():
        i = df.index[(df["trade_date"] == r["trade_date"]) & (df["ts_code"] == r["ts_code"])][0]
        c = feat_contrib[i]
        order = np.argsort(c)
        neg = [(fc_arr[j], c[j]) for j in order if c[j] < 0][:TOP_K]
        pos = [(fc_arr[j], c[j]) for j in order[::-1] if c[j] > 0][:TOP_K]
        nm = f" {r['name']}" if r["name"] else ""
        lines.append(f"### {r['ts_code']}{nm}  · pred_r20 = **{r['pred_r20']:+.3f}** "
                     f"(base {base_value:+.3f} + Σcontrib {r['pred_r20']-base_value:+.3f}) "
                     f"· alloc {r['alloc_pct']*100:.1f}%\n")
        lines.append(f"- **推高 (top{TOP_K} 正)**: " +
                     ", ".join(f"`{f}` {v:+.3f}" for f, v in pos))
        lines.append(f"- **拉低 (top{TOP_K} 负)**: " +
                     ", ".join(f"`{f}` {v:+.3f}" for f, v in neg) + "\n")
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] 人读 demo (最新日 {last_day}, {len(day)} 只) -> {MD_OUT.relative_to(ROOT)}", flush=True)

    # ── verdict ──
    top3 = drivers.head(3)[["feature", "mean_abs_shap"]].to_dict("records")
    verdict = {
        "id": "DLV-001",
        "title": "A2 TreeSHAP 可解释层 (逐只票加性归因)",
        "status": "built",
        "conclusion": (
            f"TreeSHAP 解释层就绪: 用 LightGBM 内置 pred_contrib (无 shap 库) 对 PIT-clean r20 "
            f"{DEMO_MONTH} 全月 {len(df)} 只 picks 算逐只票加性归因, 加性自检通过 "
            f"(max|Σcontrib+base-pred|={max_err:.1e}<1e-4, base_value={base_value:+.4f})。"
            f"全期 top3 驱动因子 (|mean SHAP|): "
            + ", ".join(f"{d['feature']}({d['mean_abs_shap']:.3f})" for d in top3)
            + f"。每 pick 输出 top{TOP_K} 正/负贡献因子表 → 可向 paper-trade/出资人解释'为什么打高分'。"
            f"纯解释层, 不改选股 (R05 生产冻结)。"),
        "metrics": {
            "demo_month": DEMO_MONTH,
            "n_picks_explained": int(len(df)),
            "n_rebalance_days": int(df["trade_date"].nunique()),
            "n_features": len(feat_cols),
            "base_value": round(base_value, 4),
            "additivity_max_err": max_err,
            "top3_drivers": top3,
            "latest_day": last_day,
        },
        "artifacts": {
            "script": "research/backtest/treeshap_explain.py",
            "model": f"research/cache/dep001/r20_models/{DEMO_MONTH}/classifier.txt",
            "contribs_csv": f"output/treeshap_dep001_{DEMO_MONTH}_contribs.csv",
            "demo_md": f"output/treeshap_dep001_{DEMO_MONTH}_demo.md",
        },
        "discipline": ("R05 生产只读 (仅 load dep001 缓存模型); R06 ST 源头排除; R04 大缓存 gitignored; "
                       "解释层非 alpha (R03 中间指标≠ship 不适用, 无落地宣称)。"),
    }
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] -> {VERDICT.relative_to(ROOT)} (status=built)", flush=True)
    print(f"\n[done] DLV-001 TreeSHAP 解释层完成, 耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
