# -*- coding: utf-8 -*-
"""MH-002 — Phase-1 廉价筛查: regime 分层 rank-IC + 朴素消融对照.

防自欺核心 (SIGN-R12)。检验梅花特征 mh_*/mhs_* 对前向 r20 的横截面预测力,
**并扣除"公历月 + 板块cohort"朴素对照后看是否还有独立残差**。若被朴素对照吃掉
= 信号来自世俗来源被卦象重新打包 → no_residual_signal → 廉价 REJECT (MH-003/004 skip)。

口径 (纯描述, 非 gate, SIGN-R03; 是否落地只认 MH-004 walk-forward):

  目标 (label, 不是特征): r20 = close[t+20交易日]/close[t]-1, 全因果, 只用作被预测量,
    绝不写入 research/features/ (留 research/cache/, 不撞 leakage guard, SIGN-R04)。

  横截面 rank-IC: 逐交易日 Spearman(特征, r20) 再跨日平均; t = mean/std*sqrt(N_dates)。
    注意: 横截面 IC 天然差掉"当日全市场常数"项 ⇒ 纯公历月季节性无法进入 IC,
    朴素对照里"公历月"只通过 月×板块 交互起作用 (这是诚实的口径说明)。

  目标编码 (给名义卦象 id 公平机会, 无泄漏): 月度扩张窗 OOF target encoding,
    target 月 M 的某类编码 = 该类在 月 <= M-gap 的历史 r20 均值 (gap=2 月留余量,
    保证 r20 的 20 日前向窗不与 M 重叠) → 再算其横截面 rank-IC。纯噪声类 → 编码近常数 → IC≈0。

  朴素对照: 同法 OOF 编码 (公历月, 板块) cohort → naive_pred; 残差 = r20 - naive_pred。
    梅花残差 IC = Spearman(梅花 OOF 编码, 残差)。若梅花编码对残差仍有 IC ⇒ 独立信号。

  分层 (SIGN-R11): 全期 + 按 RG-001 regime(momentum/mixed/reversal) 分别报。

  mhs_* 静态卦 = 纯代码 = 卦象版股票 ID; 其任何 OOF IC 本质是"个股固定效应", 非卦象占卜,
    决策聚焦动态 mh_* 在扣除板块后是否有残差。

裁决 (Phase-1 screen 标准, 启动即固定, 非冻结 gate):
  residual_signal  当且仅当 某动态 mh_* 特征 全期残差 |IC|>=0.01 且 |t|>=2.0;
  否则 no_residual_signal (廉价收手)。

  ST 已在 MH-001 源头排除 (join 自然继承, SIGN-R06)。
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
MH_P = ROOT / "research" / "features" / "meihua_features.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
CACHE = ROOT / "research" / "cache"
PANEL_CACHE = CACHE / "mh002_panel.parquet"     # 含 r20 (forward label) → 留 cache, 不进 features
RESULTS = CACHE / "mh002_results.json"
VERDICT = ROOT / "research" / "verdicts" / "MH-002.json"

KEY = ["ts_code", "trade_date"]
HORIZON = 20            # 前向 r20
GAP_MONTHS = 2          # OOF 月度编码留 2 月 gap (保证 20d 前向窗不与 target 月重叠)
IC_FLOOR = 0.01         # Phase-1 screen 残差 |IC| 下限
T_FLOOR = 2.0           # Phase-1 screen 残差 |t| 下限

MH_DYN = [c for c in (
    "mh_base_gua", "mh_mutual_gua", "mh_changed_gua", "mh_upper", "mh_lower",
    "mh_moving", "mh_ti_elem", "mh_yong_elem", "mh_relation", "mh_yang_count")]
MH_STAT = [c.replace("mh_", "mhs_") for c in MH_DYN]
ALL_FEATS = MH_DYN + MH_STAT


# ───────────────────────── 数据装配 ─────────────────────────
def build_panel() -> pd.DataFrame:
    if PANEL_CACHE.exists():
        print(f"[panel] checkpoint 命中 {PANEL_CACHE.name}", flush=True)
        return pd.read_parquet(PANEL_CACHE)

    print("[panel] 加载梅花特征 ...", flush=True)
    mh = pd.read_parquet(MH_P)
    mh["trade_date"] = mh["trade_date"].astype(str)

    print("[panel] 加载 daily close, 算前向 r20 (label, 因果) ...", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(DAILY.glob("*.parquet"))]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = (px[px["close"] > 0].drop_duplicates(KEY, keep="last")
          .sort_values(KEY).reset_index(drop=True))
    # 前向 20 交易日收益 (per stock 时间序列 shift -HORIZON)
    fwd = px.groupby("ts_code")["close"].shift(-HORIZON)
    px["r20"] = fwd / px["close"] - 1.0
    px = px[["ts_code", "trade_date", "r20"]]

    df = mh.merge(px, on=KEY, how="inner")        # inner: ST 已在 mh 排除, 自然继承
    df = df[df["r20"].notna()].reset_index(drop=True)

    print("[panel] 合并 regime + 板块 + 公历月 ...", flush=True)
    rg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]]
    rg["trade_date"] = rg["trade_date"].astype(str)
    df = df.merge(rg, on="trade_date", how="left")
    df["regime"] = df["regime"].fillna("unknown")

    basic = pd.read_parquet(BASIC)[["ts_code", "industry"]].drop_duplicates("ts_code")
    basic["industry"] = basic["industry"].fillna("unknown")
    df = df.merge(basic, on="ts_code", how="left")
    df["industry"] = df["industry"].fillna("unknown")
    df["ym"] = df["trade_date"].str[:6]
    df["cal_month"] = df["trade_date"].str[4:6]

    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PANEL_CACHE, index=False)
    print(f"[panel] 落盘 {PANEL_CACHE.name} ({len(df):,} 行)  "
          f"日期 {df.trade_date.min()}~{df.trade_date.max()}", flush=True)
    return df


# ───────────────────── 横截面 rank-IC (向量化) ─────────────────────
def xs_ic(df: pd.DataFrame, feat: str, target: str = "r20") -> dict:
    """逐日 Spearman(feat, target) (= 日内秩的 pearson), 跨日平均 + t 值。"""
    sub = df[["trade_date", feat, target]].dropna()
    if sub.empty:
        return {"ic_mean": np.nan, "ic_t": np.nan, "n_dates": 0}
    g = sub.groupby("trade_date")
    rf = g[feat].rank()
    rt = g[target].rank()
    tmp = pd.DataFrame({"d": sub["trade_date"].values,
                        "f": rf.values, "t": rt.values})
    a = tmp.groupby("d").agg(
        n=("f", "size"), sf=("f", "sum"), st=("t", "sum"),
        sff=("f", lambda x: float(np.dot(x, x))),
        stt=("t", lambda x: float(np.dot(x, x))))
    # f*t 的 per-date 和
    tmp["ft"] = tmp["f"] * tmp["t"]
    a["sft"] = tmp.groupby("d")["ft"].sum()
    n = a["n"]
    cov = a["sft"] - a["sf"] * a["st"] / n
    vf = a["sff"] - a["sf"] ** 2 / n
    vt = a["stt"] - a["st"] ** 2 / n
    denom = np.sqrt(vf * vt)
    ic = (cov / denom).where((denom > 0) & (n >= 2))
    ic = ic.dropna()
    if len(ic) < 2:
        return {"ic_mean": np.nan, "ic_t": np.nan, "n_dates": int(len(ic))}
    m, s = float(ic.mean()), float(ic.std(ddof=1))
    t = m / s * np.sqrt(len(ic)) if s > 0 else np.nan
    return {"ic_mean": m, "ic_t": float(t), "n_dates": int(len(ic)),
            "ic_std": s, "ic_ir": (m / s if s > 0 else np.nan)}


# ───────────────────── 月度扩张窗 OOF target encoding ─────────────────────
def oof_encode(df: pd.DataFrame, cat_cols: list[str], target: str = "r20",
               gap: int = GAP_MONTHS) -> pd.Series:
    """target 月 M 的某 cohort 编码 = 该 cohort 在 月<=M-gap 历史 r20 均值。无泄漏。"""
    months = sorted(df["ym"].unique())
    key = (df[cat_cols[0]].astype(str) if len(cat_cols) == 1
           else df[cat_cols].astype(str).agg("|".join, axis=1))
    tmp = pd.DataFrame({"ym": df["ym"].values, "key": key.values,
                        "y": df[target].values}, index=df.index)
    grp = tmp.groupby(["ym", "key"])["y"].agg(["sum", "count"]).reset_index()
    piv_s = grp.pivot(index="ym", columns="key", values="sum").reindex(months).fillna(0).cumsum()
    piv_c = grp.pivot(index="ym", columns="key", values="count").reindex(months).fillna(0).cumsum()
    piv_mean = (piv_s / piv_c.replace(0, np.nan)).shift(gap)     # 滞后 gap 月 = 因果可用
    enc_long = piv_mean.stack().rename("enc").reset_index()
    out = (tmp.reset_index().merge(enc_long, on=["ym", "key"], how="left")
           .set_index("index")["enc"])
    return out.reindex(df.index)


# ───────────────────── regime 分层 IC 报表 ─────────────────────
def ic_table(df: pd.DataFrame, feat_col_or_series, label: str,
             target: str = "r20") -> dict:
    """全期 + 分 regime 的横截面 IC。feat 既可是列名也可是已挂到 df 的临时列名。"""
    res = {"feature": label}
    res["all"] = xs_ic(df, feat_col_or_series, target)
    for reg in ["momentum", "mixed", "reversal"]:
        sub = df[df["regime"] == reg]
        res[reg] = xs_ic(sub, feat_col_or_series, target) if len(sub) else {
            "ic_mean": np.nan, "ic_t": np.nan, "n_dates": 0}
    return res


def main() -> int:
    t0 = time.time()
    print("\n=== MH-002 Phase-1 廉价筛查 (regime 分层 IC + 朴素消融) ===\n", flush=True)
    df = build_panel()
    print(f"[panel] {len(df):,} 行 / {df['ts_code'].nunique()} 股 / "
          f"{df['trade_date'].nunique()} 日; regime 分布: "
          f"{df['regime'].value_counts(normalize=True).round(3).to_dict()}", flush=True)

    # ── 1) 原始 (序数码) 横截面 IC, 全特征, 全期+分 regime (描述) ──
    print("\n[1] 原始序数码 横截面 rank-IC ...", flush=True)
    raw_ic = {}
    for f in ALL_FEATS:
        raw_ic[f] = ic_table(df, f, f)
        a = raw_ic[f]["all"]
        print(f"  {f:18s} IC={a['ic_mean']:+.4f} t={a['ic_t']:+.2f}", flush=True)

    # ── 2) OOF target encoding (给名义卦象公平机会), 算其 IC ──
    print(f"\n[2] OOF 月度扩张 target encoding (gap={GAP_MONTHS}月) → IC ...", flush=True)
    oof_ic = {}
    oof_cols = {}
    for f in ALL_FEATS:
        col = f"__oof_{f}"
        df[col] = oof_encode(df, [f])
        oof_cols[f] = col
        oof_ic[f] = ic_table(df, col, f)
        a = oof_ic[f]["all"]
        print(f"  {f:18s} OOF-IC={a['ic_mean']:+.4f} t={a['ic_t']:+.2f} "
              f"(n_dates={a['n_dates']})", flush=True)

    # ── 3) 朴素对照: (公历月, 板块) cohort OOF 编码 → naive_pred → 残差 ──
    print(f"\n[3] 朴素对照 (公历月×板块) OOF 编码 + 残差 ...", flush=True)
    df["__naive"] = oof_encode(df, ["cal_month", "industry"])
    naive_ic = ic_table(df, "__naive", "naive(cal_month x industry)")
    print(f"  naive(cal_month×industry) OOF-IC={naive_ic['all']['ic_mean']:+.4f} "
          f"t={naive_ic['all']['ic_t']:+.2f}", flush=True)
    # 残差: r20 - naive_pred (naive 冷启动 NaN 的行不参与残差检验)
    df["__resid"] = df["r20"] - df["__naive"]
    n_resid = int(df["__resid"].notna().sum())
    print(f"  残差有效行: {n_resid:,} / {len(df):,}", flush=True)

    # ── 4) 梅花 OOF 编码 对残差的 IC (扣朴素对照后是否仍有独立信号) ──
    print(f"\n[4] 梅花 OOF 编码 vs 残差 (独立残差 IC) ...", flush=True)
    resid_ic = {}
    for f in ALL_FEATS:
        resid_ic[f] = ic_table(df, oof_cols[f], f, target="__resid")
        a = resid_ic[f]["all"]
        ar = oof_ic[f]["all"]
        print(f"  {f:18s} resid-IC={a['ic_mean']:+.4f} t={a['ic_t']:+.2f}  "
              f"(raw OOF-IC={ar['ic_mean']:+.4f}, 保留 "
              f"{(a['ic_mean']/ar['ic_mean']*100) if ar['ic_mean'] not in (0,None) and not np.isnan(ar['ic_mean']) else float('nan'):.0f}%)",
              flush=True)

    # ── 裁决: 动态 mh_* 残差 |IC|>=floor 且 |t|>=floor ──
    def absmax_feat(d, feats):
        best, bf = -1.0, None
        for f in feats:
            v = d[f]["all"]
            ic = v.get("ic_mean")
            if ic is None or (isinstance(ic, float) and np.isnan(ic)):
                continue
            if abs(ic) > best:
                best, bf = abs(ic), f
        return bf

    bf_dyn = absmax_feat(resid_ic, MH_DYN)
    dyn_resid = resid_ic[bf_dyn]["all"] if bf_dyn else {"ic_mean": np.nan, "ic_t": np.nan}
    passed = (bf_dyn is not None
              and not np.isnan(dyn_resid["ic_mean"])
              and abs(dyn_resid["ic_mean"]) >= IC_FLOOR
              and not np.isnan(dyn_resid["ic_t"])
              and abs(dyn_resid["ic_t"]) >= T_FLOOR)
    status = "residual_signal" if passed else "no_residual_signal"

    def clean(d):
        return {k: (None if isinstance(v, float) and np.isnan(v) else
                    (round(v, 6) if isinstance(v, float) else v))
                for k, v in d.items()}

    def pack(table):
        return {seg: clean(table[seg]) for seg in ["all", "momentum", "mixed", "reversal"]}

    results = {
        "task": "MH-002",
        "horizon_days": HORIZON,
        "oof_gap_months": GAP_MONTHS,
        "screen_thresholds": {"ic_floor": IC_FLOOR, "t_floor": T_FLOOR,
                              "applies_to": "dynamic mh_* residual IC (full sample)"},
        "n_rows": int(len(df)),
        "n_stocks": int(df["ts_code"].nunique()),
        "n_dates": int(df["trade_date"].nunique()),
        "regime_dist": df["regime"].value_counts(normalize=True).round(4).to_dict(),
        "raw_ordinal_ic": {f: pack(raw_ic[f]) for f in ALL_FEATS},
        "oof_target_ic": {f: pack(oof_ic[f]) for f in ALL_FEATS},
        "naive_control_ic": pack(naive_ic),
        "residual_ic_after_naive": {f: pack(resid_ic[f]) for f in ALL_FEATS},
        "best_dynamic_feature_by_residual": bf_dyn,
        "best_dynamic_residual_all": clean(dyn_resid),
        "decision": status,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[results] -> {RESULTS.relative_to(ROOT)}", flush=True)

    # ── verdict ──
    bdr = dyn_resid
    conclusion = (
        f"梅花 Phase-1 廉价筛查: 最强动态 mh_* 残差信号 = {bf_dyn} "
        f"(扣 公历月×板块 朴素对照后 残差 IC={bdr['ic_mean']:+.4f}, t={bdr['ic_t']:+.2f}); "
        f"朴素对照本身 OOF-IC={naive_ic['all']['ic_mean']:+.4f} (t={naive_ic['all']['ic_t']:+.2f})。"
        f"判定 {status} (阈值 |IC|>={IC_FLOOR} 且 |t|>={T_FLOOR})。"
    )
    if status == "no_residual_signal":
        conclusion += (" 动态卦象残差不足以独立于世俗(板块×月历)来源 ⇒ 廉价收手, "
                       "MH-003/004 置 skip, MH-005 skipped (SIGN-R12 消融未存活)。")
    else:
        conclusion += " 存在独立残差 ⇒ 进 MH-003/004 walk-forward (最终裁决只认 walk-forward α, SIGN-R03)。"

    verdict = {
        "id": "MH-002",
        "status": status,
        "conclusion": conclusion,
        "metrics": {
            "horizon_days": HORIZON,
            "best_dynamic_feature": bf_dyn,
            "best_dynamic_residual_ic": clean(bdr).get("ic_mean"),
            "best_dynamic_residual_t": clean(bdr).get("ic_t"),
            "naive_control_oof_ic": clean(naive_ic["all"]).get("ic_mean"),
            "naive_control_oof_t": clean(naive_ic["all"]).get("ic_t"),
            "screen_ic_floor": IC_FLOOR,
            "screen_t_floor": T_FLOOR,
            "n_dates": int(df["trade_date"].nunique()),
            "regime_dist": df["regime"].value_counts(normalize=True).round(4).to_dict(),
        },
        "ablation": {
            "naive_control": "OOF (cal_month x industry) target encoding",
            "note": ("横截面 IC 天然差掉当日全市场常数项, 纯公历月季节性不进 IC; "
                     "朴素对照通过 月×板块 交互起作用。mhs_* 静态卦=卦象版股票ID, "
                     "其 OOF IC 是个股固定效应非占卜。"),
            "best_dynamic_raw_oof_ic": clean(oof_ic[bf_dyn]["all"]).get("ic_mean") if bf_dyn else None,
        },
        "artifacts": ["research/cache/mh002_results.json",
                      "research/cache/mh002_panel.parquet"],
        "guardrails": ["SIGN-R03 纯描述非gate", "SIGN-R04 r20 留cache不进features",
                       "SIGN-R06 ST源头排除继承", "SIGN-R11 regime分层", "SIGN-R12 朴素消融对照"],
    }
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] status={status} -> {VERDICT.relative_to(ROOT)}", flush=True)
    print(f"\n[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
