# -*- coding: utf-8 -*-
"""MT-004 — 三组对比 (B/A/C) + 决策 gate: 升级 apples-to-apples 消融.

北极星问题: **梅花对走势的数论分桶能否超过/补充标准技术指标对同一走势的编码?**

Phase-1 三组裁决 (复用 MT-001/002/003 verdict):
  A (数字起卦)   = no_residual   (扣线性动量后无残差, 廉价 REJECT)
  B (体用趋势)   = residual_signal (mhB_upper resid2-IC=+0.0214 t=10.32) — 诚实预警: 体卦=多尺度MA堆叠换皮
  C (累积卦序列) = residual_signal (mhC_move_std resid2-IC=+0.0463 t=10.87) — 诚实预警: move_std≈已实现波动率换皮

但 Phase-1 的 resid2 **只扣了线性动量/趋势** (mom_5/mom_20/ma_pos/rsi_14)。
MT-001/MT-003 都把硬约束甩给 MT-004: apples-to-apples baseline **必须额外含**
  (1) 已实现波动率 std(ret_20)/ATR  (MT-003 要求, 针对 move_std)
  (2) 多尺度 MA 堆叠/非线性趋势      (MT-001 要求, 针对 mhB_upper)
否则会把'波动率/非线性趋势'这些**标准 TA 对同一走势的编码**误归功于卦象 = 自欺 (SIGN-R12)。

本脚本完成 Phase-1 故意留白的那一步 = 升级消融 resid3:
  resid1 = r20 − naive(公历月×板块 OOF)
  resid2 = resid1 逐日横截面扣 线性动量 (mom_5,mom_20,ma_pos,rsi_14)        ← 重现 Phase-1
  resid3 = resid1 逐日横截面扣 完整标准TA (线性动量 + 已实现波动率 + 多尺度MA堆叠)  ← apples-to-apples
然后看最强候选 (B=mhB_upper / C=mhC_move_std) 对 resid3 是否仍 |IC|>=0.01 且 |t|>=3。

决策 (SIGN-R12 + phase2_gate):
  · 若最强候选对 resid3 仍存活 → 有独立于标准TA编码的残差 → 跑 19月 walk-forward (PASS/REJECT 由 α 定)。
  · 若 resid3 上全部坍塌 → 残差=标准TA对同一走势的编码换皮, 无独立信号 → REJECT (廉价, 不进 walk-forward)。
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
CACHE = ROOT / "research" / "cache"
MT001_PANEL = CACHE / "mt001_panel.parquet"      # mhB_* + mom + r20 + 分层
MT003_PANEL = CACHE / "mt003_panel.parquet"      # mhC_* + mom + r20 + 分层
CTRL_CACHE = CACHE / "mt004_controls.parquet"    # 升级 TA 控制集 (波动率+MA堆叠)
WORK_CACHE = CACHE / "mt004_panel.parquet"       # 合并工作面板 (含 resid)
RESULTS = CACHE / "mt004_results.json"
VERDICT = ROOT / "research" / "verdicts" / "MT-004.json"
MT001_V = ROOT / "research" / "verdicts" / "MT-001.json"
MT002_V = ROOT / "research" / "verdicts" / "MT-002.json"
MT003_V = ROOT / "research" / "verdicts" / "MT-003.json"

KEY = ["ts_code", "trade_date"]
HORIZON = 20
GAP_MONTHS = 2
IC_FLOOR = 0.01
T_FLOOR = 3.0

# 候选 (Phase-1 两个 residual_signal 的最强动态特征)
CAND_B = "mhB_upper"        # 名义卦象 → OOF target encoding
CAND_C = "mhC_move_std"     # 连续序列标量 → 原始 rank-IC

LINEAR_MOM = ["mom_5", "mom_20", "ma_pos", "rsi_14"]          # Phase-1 线性动量 (resid2)
# 升级控制集 = 线性动量 + 已实现波动率 + 多尺度MA堆叠/非线性趋势 (resid3, apples-to-apples)
VOL_CTRL = ["vol_20", "atr_20"]
MASTACK_CTRL = ["mom_10", "mom_60", "r_c_ma5", "r_ma5_ma10", "r_ma10_ma20", "r_ma20_ma60"]
FULL_TA = LINEAR_MOM + VOL_CTRL + MASTACK_CTRL


# ───────────────────────── 升级 TA 控制集 (从 raw daily, 因果) ─────────────────────────
def build_controls() -> pd.DataFrame:
    if CTRL_CACHE.exists():
        print(f"[ctrl] checkpoint 命中 {CTRL_CACHE.name}", flush=True)
        return pd.read_parquet(CTRL_CACHE)
    print("[ctrl] 加载 daily OHLC 计算 波动率/多尺度MA堆叠 (因果) ...", flush=True)
    cols = ["ts_code", "trade_date", "high", "low", "close", "pre_close"]
    parts = [pd.read_parquet(f, columns=cols) for f in sorted(DAILY.glob("*.parquet"))]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = (px[(px["close"] > 0) & (px["pre_close"] > 0)]
          .drop_duplicates(KEY, keep="last").sort_values(KEY).reset_index(drop=True))
    g = px.groupby("ts_code", sort=False)["close"]
    ret = g.transform(lambda s: s.pct_change())
    px["ret"] = ret
    # 已实现波动率 (MT-003 要求): 20日日收益 std + ATR代理
    px["vol_20"] = px.groupby("ts_code", sort=False)["ret"].transform(
        lambda s: s.rolling(20, min_periods=20).std())
    tr = (px["high"] - px["low"]) / px["pre_close"]
    px["atr_20"] = tr.groupby(px["ts_code"]).transform(
        lambda s: s.rolling(20, min_periods=20).mean())
    # 额外动量尺度
    px["mom_10"] = g.transform(lambda s: s / s.shift(10) - 1.0)
    px["mom_60"] = g.transform(lambda s: s / s.shift(60) - 1.0)
    # 多尺度 MA 堆叠/非线性趋势 (MT-001 要求): 各级 MA 之间的连续斜率
    ma5 = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma10 = g.transform(lambda s: s.rolling(10, min_periods=10).mean())
    ma20 = g.transform(lambda s: s.rolling(20, min_periods=20).mean())
    ma60 = g.transform(lambda s: s.rolling(60, min_periods=60).mean())
    px["r_c_ma5"] = px["close"] / ma5 - 1.0
    px["r_ma5_ma10"] = ma5 / ma10 - 1.0
    px["r_ma10_ma20"] = ma10 / ma20 - 1.0
    px["r_ma20_ma60"] = ma20 / ma60 - 1.0
    out = px[KEY + VOL_CTRL + MASTACK_CTRL].copy()
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CTRL_CACHE, index=False)
    print(f"[ctrl] -> {CTRL_CACHE.relative_to(ROOT)} ({len(out):,} 行)", flush=True)
    return out


def build_panel() -> pd.DataFrame:
    if WORK_CACHE.exists():
        print(f"[panel] checkpoint 命中 {WORK_CACHE.name}", flush=True)
        return pd.read_parquet(WORK_CACHE)
    print("[panel] 合并 mt003(C) spine + mt001(B) mhB_upper + 升级控制集 ...", flush=True)
    c = pd.read_parquet(MT003_PANEL, columns=(
        KEY + [CAND_C] + LINEAR_MOM + ["r20", "regime", "industry", "ym", "cal_month"]))
    b = pd.read_parquet(MT001_PANEL, columns=(KEY + [CAND_B]))
    ctrl = build_controls()
    df = c.merge(b, on=KEY, how="inner").merge(ctrl, on=KEY, how="inner")
    df = df[df["r20"].notna()].reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WORK_CACHE, index=False)
    print(f"[panel] -> {WORK_CACHE.relative_to(ROOT)} ({len(df):,} 行)", flush=True)
    return df


# ───────────────────── 横截面 rank-IC (向量化, 同 MT-003) ─────────────────────
def xs_ic(df: pd.DataFrame, feat: str, target: str) -> dict:
    sub = df[["trade_date", feat, target]].dropna()
    if sub.empty:
        return {"ic_mean": np.nan, "ic_t": np.nan, "n_dates": 0}
    rf = sub.groupby("trade_date")[feat].rank()
    rt = sub.groupby("trade_date")[target].rank()
    tmp = pd.DataFrame({"d": sub["trade_date"].values, "f": rf.values, "t": rt.values})
    tmp["ft"] = tmp["f"] * tmp["t"]
    a = tmp.groupby("d").agg(n=("f", "size"), sf=("f", "sum"), st=("t", "sum"),
                             sff=("f", lambda x: float(np.dot(x, x))),
                             stt=("t", lambda x: float(np.dot(x, x))),
                             sft=("ft", "sum"))
    n = a["n"]
    cov = a["sft"] - a["sf"] * a["st"] / n
    vf = a["sff"] - a["sf"] ** 2 / n
    vt = a["stt"] - a["st"] ** 2 / n
    denom = np.sqrt(vf * vt)
    ic = (cov / denom).where((denom > 0) & (n >= 2)).dropna()
    if len(ic) < 2:
        return {"ic_mean": np.nan, "ic_t": np.nan, "n_dates": int(len(ic))}
    mn, s = float(ic.mean()), float(ic.std(ddof=1))
    t = mn / s * np.sqrt(len(ic)) if s > 0 else np.nan
    return {"ic_mean": mn, "ic_t": float(t), "n_dates": int(len(ic)),
            "ic_std": s, "ic_ir": (mn / s if s > 0 else np.nan)}


def oof_encode(df: pd.DataFrame, cat_cols: list[str], target: str = "r20",
               gap: int = GAP_MONTHS) -> pd.Series:
    months = sorted(df["ym"].unique())
    key = (df[cat_cols[0]].astype(str) if len(cat_cols) == 1
           else df[cat_cols].astype(str).agg("|".join, axis=1))
    tmp = pd.DataFrame({"ym": df["ym"].values, "key": key.values,
                        "y": df[target].values}, index=df.index)
    grp = tmp.groupby(["ym", "key"])["y"].agg(["sum", "count"]).reset_index()
    piv_s = grp.pivot(index="ym", columns="key", values="sum").reindex(months).fillna(0).cumsum()
    piv_c = grp.pivot(index="ym", columns="key", values="count").reindex(months).fillna(0).cumsum()
    piv_mean = (piv_s / piv_c.replace(0, np.nan)).shift(gap)
    enc_long = piv_mean.stack().rename("enc").reset_index()
    out = (tmp.reset_index().merge(enc_long, on=["ym", "key"], how="left")
           .set_index("index")["enc"])
    return out.reindex(df.index)


def residualize(df: pd.DataFrame, ycol: str, factor_cols: list[str]) -> pd.Series:
    out = np.full(len(df), np.nan)
    pos = {ix: i for i, ix in enumerate(df.index)}
    for _, sub in df.groupby("trade_date"):
        y = sub[ycol].values
        X = sub[factor_cols].values
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if mask.sum() < len(factor_cols) + 5:
            continue
        Xm, ym = X[mask], y[mask]
        sd = Xm.std(0)
        sd[sd == 0] = 1.0
        Xs = (Xm - Xm.mean(0)) / sd
        A = np.column_stack([np.ones(mask.sum()), Xs])
        beta, *_ = np.linalg.lstsq(A, ym, rcond=None)
        r = ym - A @ beta
        idxs = [pos[ix] for ix in sub.index[mask]]
        out[idxs] = r
    return pd.Series(out, index=df.index)


def ic_table(df: pd.DataFrame, feat: str, target: str) -> dict:
    res = {"all": xs_ic(df, feat, target)}
    for reg in ["momentum", "mixed", "reversal"]:
        sub = df[df["regime"] == reg]
        res[reg] = (xs_ic(sub, feat, target) if len(sub)
                    else {"ic_mean": np.nan, "ic_t": np.nan, "n_dates": 0})
    return res


def clean(d):
    return {k: (None if isinstance(v, float) and np.isnan(v) else
                (round(v, 6) if isinstance(v, float) else v)) for k, v in d.items()}


def pack(table):
    return {seg: clean(table[seg]) for seg in ["all", "momentum", "mixed", "reversal"]}


def survives(tbl_all: dict) -> bool:
    ic, t = tbl_all.get("ic_mean"), tbl_all.get("ic_t")
    return (ic is not None and not (isinstance(ic, float) and np.isnan(ic))
            and abs(ic) >= IC_FLOOR and t is not None
            and not (isinstance(t, float) and np.isnan(t)) and abs(t) >= T_FLOOR)


def main() -> int:
    t0 = time.time()
    print("\n=== MT-004 三组对比 + 决策 gate (升级 apples-to-apples 消融) ===\n", flush=True)

    # ── Phase-1 三组裁决摘要 (复用 verdict) ──
    v1 = json.loads(MT001_V.read_text(encoding="utf-8"))
    v2 = json.loads(MT002_V.read_text(encoding="utf-8"))
    v3 = json.loads(MT003_V.read_text(encoding="utf-8"))
    phase1 = {
        "A": {"scheme": "数字起卦", "status": v2["status"],
              "best_feature": v2["metrics"]["best_dynamic_feature"],
              "resid2_ic": v2["metrics"]["best_dynamic_resid2_ic"],
              "resid2_t": v2["metrics"]["best_dynamic_resid2_t"]},
        "B": {"scheme": "体用=趋势vs当日", "status": v1["status"],
              "best_feature": v1["metrics"]["best_dynamic_feature"],
              "resid2_ic": v1["metrics"]["best_dynamic_resid2_ic"],
              "resid2_t": v1["metrics"]["best_dynamic_resid2_t"]},
        "C": {"scheme": "累积卦+序列特征", "status": v3["status"],
              "best_feature": v3["metrics"]["best_dynamic_feature"],
              "resid2_ic": v3["metrics"]["best_dynamic_resid2_ic"],
              "resid2_t": v3["metrics"]["best_dynamic_resid2_t"]},
    }
    print("[Phase-1 三组] (resid2 = 仅扣 月×板块 + 线性动量)")
    for k in ["A", "B", "C"]:
        p = phase1[k]
        print(f"  {k} {p['scheme']:14s} {p['status']:15s} "
              f"best={p['best_feature']} resid2-IC={p['resid2_ic']:+.4f} t={p['resid2_t']:+.2f}",
              flush=True)

    df = build_panel()
    print(f"\n[panel] {len(df):,} 行 / {df['ts_code'].nunique()} 股 / "
          f"{df['trade_date'].nunique()} 日; regime "
          f"{df['regime'].value_counts(normalize=True).round(3).to_dict()}", flush=True)

    # ── naive 月×板块 OOF → resid1 ──
    print("\n[resid1] 朴素对照 cal_month×industry OOF ...", flush=True)
    df["__naive"] = oof_encode(df, ["cal_month", "industry"], "r20")
    df["resid1"] = df["r20"] - df["__naive"]

    # ── resid2 (线性动量, 重现 Phase-1) vs resid3 (完整标准TA) ──
    print(f"[resid2] 逐日横截面扣 线性动量 {LINEAR_MOM} ...", flush=True)
    df["resid2"] = residualize(df, "resid1", LINEAR_MOM)
    print(f"[resid3] 逐日横截面扣 完整标准TA {FULL_TA} (apples-to-apples) ...", flush=True)
    df["resid3"] = residualize(df, "resid1", FULL_TA)
    print(f"  resid2 有效 {int(df['resid2'].notna().sum()):,} / "
          f"resid3 有效 {int(df['resid3'].notna().sum()):,}", flush=True)

    # ── 候选编码: B=OOF / C=原始 ──
    df["__oof_B"] = oof_encode(df, [CAND_B], "r20")

    # ── 控制集自身 IC (描述: 显示波动率/MA堆叠对 r20 的标准TA信号强度) ──
    print("\n[ctrl-IC] 升级控制集自身 vs r20 ...", flush=True)
    ctrl_ic = {}
    for f in VOL_CTRL + MASTACK_CTRL:
        ctrl_ic[f] = clean(xs_ic(df, f, "r20"))
        print(f"  {f:14s} IC(r20)={ctrl_ic[f]['ic_mean']:+.4f} t={ctrl_ic[f]['ic_t']:+.2f}",
              flush=True)

    # ── 候选 IC: raw r20 / resid2 / resid3, 全期+分regime ──
    print("\n[候选] B=mhB_upper(OOF) / C=mhC_move_std(原始) 对 r20/resid2/resid3 ...", flush=True)
    cand = {
        "B": {"feature": CAND_B, "src": "__oof_B", "encoding": "OOF"},
        "C": {"feature": CAND_C, "src": CAND_C, "encoding": "raw_rankIC"},
    }
    cand_ic = {}
    for k, c in cand.items():
        src = c["src"]
        cand_ic[k] = {
            "feature": c["feature"], "encoding": c["encoding"],
            "raw_r20": pack(ic_table(df, src, "r20")),
            "resid2": pack(ic_table(df, src, "resid2")),
            "resid3": pack(ic_table(df, src, "resid3")),
        }
        r2 = cand_ic[k]["resid2"]["all"]
        r3 = cand_ic[k]["resid3"]["all"]
        print(f"  {k} {c['feature']:16s}  resid2-IC={r2['ic_mean']:+.4f} t={r2['ic_t']:+.2f}  "
              f"→ resid3-IC={r3['ic_mean']:+.4f} t={r3['ic_t']:+.2f}  "
              f"{'存活' if survives(r3) else '坍塌'}", flush=True)

    # ── 决策: 任一候选对 resid3 仍存活 → 需 walk-forward; 否则 REJECT ──
    surv_B = survives(cand_ic["B"]["resid3"]["all"])
    surv_C = survives(cand_ic["C"]["resid3"]["all"])
    any_survive = surv_B or surv_C
    # 选最强 (按 resid3 |IC|)
    def r3abs(k):
        ic = cand_ic[k]["resid3"]["all"].get("ic_mean")
        return abs(ic) if (ic is not None and not (isinstance(ic, float) and np.isnan(ic))) else -1
    strongest = "C" if r3abs("C") >= r3abs("B") else "B"

    if any_survive:
        status = "PASS_PHASE1_NEEDS_WF"   # 占位; 若进到这里需补 walk-forward 才能 PASS
    else:
        status = "REJECT"

    results = {
        "task": "MT-004", "kind": "gate (B/A/C 对比 + 升级 apples-to-apples 消融)",
        "window_days": 20, "horizon_days": HORIZON,
        "screen_thresholds": {"ic_floor": IC_FLOOR, "t_floor": T_FLOOR},
        "phase1_summary": phase1,
        "upgraded_controls": {
            "linear_momentum_resid2": LINEAR_MOM,
            "full_TA_resid3": FULL_TA,
            "added_vs_phase1": VOL_CTRL + MASTACK_CTRL,
            "rationale": ("MT-003 要求加 已实现波动率(vol_20/atr_20) 针对 move_std; "
                          "MT-001 要求加 多尺度MA堆叠/非线性趋势 针对 mhB_upper。"
                          "resid3 = 标准TA对同一走势的完整编码, 用于 apples-to-apples。"),
        },
        "control_self_ic_r20": ctrl_ic,
        "candidates": cand_ic,
        "survives_resid3": {"B": surv_B, "C": surv_C},
        "strongest_candidate": strongest,
        "n_rows": int(len(df)), "n_dates": int(df["trade_date"].nunique()),
        "regime_dist": df["regime"].value_counts(normalize=True).round(4).to_dict(),
        "decision_prelim": status,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[results] -> {RESULTS.relative_to(ROOT)}  (任一存活={any_survive})", flush=True)
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0 if not any_survive else 2   # 2 = 需补 walk-forward (本轮再处理)


if __name__ == "__main__":
    sys.exit(main())
