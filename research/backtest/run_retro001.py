# -*- coding: utf-8 -*-
"""RETRO-001 — 0501+ 真实 picks 表现回顾 (vs 全市场 / 概念均值). 纯只读, 描述性.

北极星 (RETRO 阶段): 回顾 2025-05-01 起 V12.31 每日评分股票 (picks) 的真实表现 ——
真实涨没涨? 相对全市场跑赢没? 相对其所属概念 (merged 概念库) 跑赢没?

picks_source (prd.preRegisteredGate): 复用 BT-001 已建的 V12.31 baseline 每日持仓
  research/cache/bt001/picks_v1231_daily.parquet (= V7c dual-track 池, 池内按 ratio_s5 排序),
  截 trade_date >= 0501。**model-vintage caveat**: 生产 r20 排序模型 (r20_v16_long_nost) 训练窗
  < 20250930, 故 0501+ 的早 ~5 个月 (202505~202509) 对 r20 有 in-sample 暴露 → 这些月的表现
  带乐观偏差, 须标注 (与 WF-001/WFE-001 的 de-lookahead 诊断一致)。本 task 描述性, 不做 gate。

实现收益 (close[t] 为基, 日线到最新): ret_h = close[t+h]/close[t]-1 (h=5/10/20);
  max_gain_20 = max(high[t+1..t+20])/close[t]-1 (= winner_def 口径)。
  前向 horizon 不满的 pick (临近数据末端) 标 partial 并从满 20d 统计中剔除。

对照:
  - 全市场均值: 同 pick_date 全市场 (ST 源头排除) 均值收益 → 超额 = pick - 市场。
  - 概念均值: 每 pick 映射 merged 概念库其所属概念, 取 (pick_date, concept) 概念成员均值,
    再对该股所属多个概念求均值 = 概念 benchmark → 概念超额 = pick - 概念均值。

R 纪律: R01 不动门柱 (描述性非 gate); R02 描述性=合法完成; R03 中间指标≠ship;
  R04 大缓存 research/cache/retro001/ gitignored 无 features 写入; R05 生产只读 (复用缓存 picks);
  R06 ST 源头排除 (load_prices); R11 regime 分层必带。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))
from run_bt002 import load_prices

PICKS = ROOT / "research" / "cache" / "bt001" / "picks_v1231_daily.parquet"
CONCEPT = ROOT / "output" / "concept_local" / "concept_merged.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
CACHE = ROOT / "research" / "cache" / "retro001"
CACHE.mkdir(parents=True, exist_ok=True)
VERDICT = ROOT / "research" / "verdicts" / "RETRO-001.json"

START = "20250501"            # 0501 起 (前向真实 OOS)
HORIZONS = [5, 10, 20]
MAXGAIN_H = 20
# r20 生产模型训练窗 < 20250930 → 早期月 in-sample 暴露 (vintage caveat)
VINTAGE_INSAMPLE_BEFORE = "202510"


def _fwd_close_ret(close_piv: np.ndarray, h: int) -> np.ndarray:
    """ret_h[t] = close[t+h]/close[t]-1, 末端不足 h 日 -> nan."""
    T = close_piv.shape[0]
    out = np.full_like(close_piv, np.nan)
    if T > h:
        out[:T - h] = close_piv[h:] / close_piv[:T - h] - 1.0
    return out


def _fwd_max_high(high_piv: np.ndarray, close_piv: np.ndarray, h: int):
    """max_gain[t] = max(high[t+1..t+h])/close[t]-1; n_fwd[t] = 可用前向天数 (<h 则 partial)."""
    T = high_piv.shape[0]
    M = np.full_like(high_piv, -np.inf)
    nfwd = np.zeros(high_piv.shape, dtype=np.int16)
    for k in range(1, h + 1):
        if T - k <= 0:
            break
        M[:T - k] = np.fmax(M[:T - k], high_piv[k:])
        nfwd[:T - k] += 1
    M[~np.isfinite(M)] = np.nan
    mg = M / close_piv - 1.0
    return mg, nfwd


def main():
    t0 = time.time()
    print("\n=== RETRO-001: 0501+ V12.31 picks 真实表现回顾 (vs 全市场/概念均值, 只读) ===\n", flush=True)

    # ── picks (0501+) ──
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    picks = picks[picks["trade_date"] >= START].copy()
    pick_dates = sorted(picks["trade_date"].unique())
    print(f"[picks] {len(picks):,} (date,stock) 观测 / {len(pick_dates)} 交易日 "
          f"/ {picks['ts_code'].nunique()} 股  [{pick_dates[0]}~{pick_dates[-1]}]", flush=True)

    # ── 价表 (全市场, ST 源头排除 R06); 透视 close/high ──
    print(f"[prices] load >= {START} (ST 源头排除) ...", flush=True)
    px = load_prices(START)
    cal = sorted(px["trade_date"].unique())
    print(f"[prices] {len(px):,} 行 / {len(cal)} 交易日 [{cal[0]}~{cal[-1]}]", flush=True)
    close = px.pivot_table(index="trade_date", columns="ts_code", values="close")
    high = px.pivot_table(index="trade_date", columns="ts_code", values="high")
    close = close.reindex(cal); high = high.reindex(high.index.union(cal)).reindex(cal)
    codes = list(close.columns)
    date_idx = {d: i for i, d in enumerate(cal)}
    cp = close.to_numpy(dtype="float64"); hp = high.to_numpy(dtype="float64")

    # ── 前向收益矩阵 (全市场, 描述性) ──
    ret_mats = {h: _fwd_close_ret(cp, h) for h in HORIZONS}
    mg_mat, nfwd_mat = _fwd_max_high(hp, cp, MAXGAIN_H)

    def _mat_to_long(mat, name):
        df = pd.DataFrame(mat, index=cal, columns=codes)
        df = df.loc[pick_dates]
        s = df.stack(future_stack=True)
        s.index.names = ["trade_date", "ts_code"]
        return s.rename(name)

    # 全市场每日均值 (同 pick_date 基准) — 全样本均值, 非仅 picks
    mkt_mean = {}
    for h in HORIZONS:
        dfm = pd.DataFrame(ret_mats[h], index=cal, columns=codes).loc[pick_dates]
        mkt_mean[f"ret_{h}"] = dfm.mean(axis=1)
    dfg = pd.DataFrame(mg_mat, index=cal, columns=codes).loc[pick_dates]
    mkt_mean["max_gain_20"] = dfg.mean(axis=1)
    mkt_mean = pd.DataFrame(mkt_mean)                      # index=pick_date

    # ── picks 前向收益 ──
    cols = {}
    for h in HORIZONS:
        cols[f"ret_{h}"] = _mat_to_long(ret_mats[h], f"ret_{h}")
    cols["max_gain_20"] = _mat_to_long(mg_mat, "max_gain_20")
    cols["nfwd_20"] = _mat_to_long(nfwd_mat.astype("float64"), "nfwd_20")
    longp = pd.concat(cols.values(), axis=1)
    pk = picks.set_index(["trade_date", "ts_code"]).join(longp, how="left").reset_index()
    pk["full_20d"] = pk["nfwd_20"] >= MAXGAIN_H

    # 全市场超额 (同 pick_date)
    for h in HORIZONS:
        pk[f"exc_mkt_{h}"] = pk[f"ret_{h}"] - pk["trade_date"].map(mkt_mean[f"ret_{h}"])

    # ── 概念均值 (merged 概念库) ──
    print("[concept] 映射 merged 概念 + 算 (pick_date, concept) 成员均值 ...", flush=True)
    mem = pd.read_parquet(CONCEPT)[["stock_code", "concept_ts_code"]].dropna().drop_duplicates()
    mem = mem.rename(columns={"stock_code": "ts_code", "concept_ts_code": "concept"})
    pick_codes = set(picks["ts_code"].unique())
    pick_concepts = set(mem[mem["ts_code"].isin(pick_codes)]["concept"].unique())
    memf = mem[mem["concept"].isin(pick_concepts)].copy()          # 仅 picks 所属概念
    print(f"[concept] picks 覆盖 {len(pick_concepts)} 概念; 这些概念成员对 {len(memf):,} 行", flush=True)

    # 全市场前向收益 long (仅 pick 日 × 这些概念的成员股) → 概念成员均值
    mem_codes = [c for c in codes if c in set(memf["ts_code"])]
    cpos = [codes.index(c) for c in mem_codes]
    cmean_parts = []
    for h in HORIZONS + ["mg"]:
        mat = mg_mat if h == "mg" else ret_mats[h]
        name = "max_gain_20" if h == "mg" else f"ret_{h}"
        sub = pd.DataFrame(mat[:, cpos], index=cal, columns=mem_codes).loc[pick_dates]
        s = sub.stack(future_stack=True); s.index.names = ["trade_date", "ts_code"]
        cmean_parts.append(s.rename(name))
    retmem = pd.concat(cmean_parts, axis=1).reset_index()
    retmem = retmem.merge(memf, on="ts_code", how="inner")        # (date,stock,concept,ret)
    concept_mean = retmem.groupby(["trade_date", "concept"]).mean(numeric_only=True)
    concept_mean = concept_mean.rename(columns=lambda c: f"cmean_{c}")

    # 每 pick: 其所属概念的概念均值, 对多概念求均值
    pk_mem = pk[["trade_date", "ts_code"]].merge(memf, on="ts_code", how="left")
    pk_mem = pk_mem.merge(concept_mean.reset_index(), on=["trade_date", "concept"], how="left")
    cb = pk_mem.groupby(["trade_date", "ts_code"]).mean(numeric_only=True)   # 概念 benchmark/股
    pk = pk.merge(cb.reset_index(), on=["trade_date", "ts_code"], how="left")
    n_concept_cov = pk["cmean_ret_20"].notna().sum()
    for h in HORIZONS:
        pk[f"exc_con_{h}"] = pk[f"ret_{h}"] - pk[f"cmean_ret_{h}"]
    pk["exc_con_mg"] = pk["max_gain_20"] - pk["cmean_max_gain_20"]

    # ── regime (R11) ──
    reg = pd.read_parquet(REGIME)[["trade_date", "regime"]]
    reg["trade_date"] = reg["trade_date"].astype(str)
    pk = pk.merge(reg, on="trade_date", how="left")
    pk["ym"] = pk["trade_date"].str[:6]
    pk["vintage"] = np.where(pk["ym"] < VINTAGE_INSAMPLE_BEFORE, "in_sample_r20", "true_oos")

    pk.to_parquet(CACHE / "retro001_picks_perf.parquet", index=False)

    # ── 汇总 ──
    full = pk[pk["full_20d"]]
    n_total, n_full = len(pk), len(full)

    def _stats(df):
        d = {}
        for h in HORIZONS:
            d[f"ret_{h}_mean"] = float(df[f"ret_{h}"].mean())
            d[f"ret_{h}_median"] = float(df[f"ret_{h}"].median())
        d["max_gain_20_mean"] = float(df["max_gain_20"].mean())
        d["win_ret_20"] = float((df["ret_20"] > 0).mean())
        d["win_vs_mkt_20"] = float((df["exc_mkt_20"] > 0).mean())
        d["exc_mkt_5_mean"] = float(df["exc_mkt_5"].mean())
        d["exc_mkt_10_mean"] = float(df["exc_mkt_10"].mean())
        d["exc_mkt_20_mean"] = float(df["exc_mkt_20"].mean())
        cc = df["exc_con_20"].dropna()
        d["exc_con_5_mean"] = float(df["exc_con_5"].mean())
        d["exc_con_10_mean"] = float(df["exc_con_10"].mean())
        d["exc_con_20_mean"] = float(df["exc_con_20"].mean())
        d["win_vs_con_20"] = float((cc > 0).mean()) if len(cc) else None
        d["n"] = int(len(df))
        return d

    overall = _stats(full)
    by_month = {m: _stats(g) for m, g in full.groupby("ym")}
    by_regime = {r: _stats(g) for r, g in full.groupby("regime") if pd.notna(r)}
    by_vintage = {v: _stats(g) for v, g in full.groupby("vintage")}

    # 批次表 (月) 落盘
    bm = pd.DataFrame(by_month).T
    bm.to_parquet(CACHE / "retro001_by_month.parquet")

    print("\n── overall (full-20d picks) ──", flush=True)
    print(f"  n={overall['n']}  ret20 mean={overall['ret_20_mean']:+.4f} median={overall['ret_20_median']:+.4f} "
          f"win={overall['win_ret_20']:.2%}  max_gain20 mean={overall['max_gain_20_mean']:+.4f}", flush=True)
    print(f"  vs市场: exc20 mean={overall['exc_mkt_20_mean']:+.4f} win={overall['win_vs_mkt_20']:.2%}", flush=True)
    print(f"  vs概念: exc20 mean={overall['exc_con_20_mean']:+.4f} win={overall['win_vs_con_20']}", flush=True)
    print("\n── by vintage ──", flush=True)
    for v, s in by_vintage.items():
        print(f"  {v:14s} n={s['n']:5d} ret20={s['ret_20_mean']:+.4f} exc_mkt20={s['exc_mkt_20_mean']:+.4f} "
              f"exc_con20={s['exc_con_20_mean']:+.4f} win={s['win_ret_20']:.2%}", flush=True)
    print("\n── by regime (R11) ──", flush=True)
    for r, s in by_regime.items():
        print(f"  {r:10s} n={s['n']:5d} ret20={s['ret_20_mean']:+.4f} exc_mkt20={s['exc_mkt_20_mean']:+.4f} "
              f"exc_con20={s['exc_con_20_mean']:+.4f}", flush=True)

    # ── 裁决 ──
    pos_mkt = overall["exc_mkt_20_mean"] > 0
    pos_con = (overall["exc_con_20_mean"] or 0) > 0
    if pos_mkt and pos_con:
        concl = (f"0501+ V12.31 picks 真实表现: 20d 均值收益 {overall['ret_20_mean']:+.2%} (胜率 {overall['win_ret_20']:.0%}), "
                 f"相对全市场超额 {overall['exc_mkt_20_mean']:+.2%} (胜率 {overall['win_vs_mkt_20']:.0%}) 且相对所属概念超额 "
                 f"{overall['exc_con_20_mean']:+.2%} → picks 真实涨且双双跑赢市场与概念基准 (但早 5 月含 r20 in-sample vintage 偏差)")
    elif pos_mkt and not pos_con:
        concl = (f"0501+ picks 20d 收益 {overall['ret_20_mean']:+.2%}, 相对市场超额 {overall['exc_mkt_20_mean']:+.2%} 为正, "
                 f"但相对所属概念超额 {overall['exc_con_20_mean']:+.2%} ≈0/为负 → picks 的相对收益主要是'选对概念(板块beta)', "
                 f"概念内未系统跑赢 (为 RETRO-002/003 的桶内对比埋下伏笔)")
    else:
        concl = (f"0501+ picks 20d 收益 {overall['ret_20_mean']:+.2%}, 相对市场超额 {overall['exc_mkt_20_mean']:+.2%} / "
                 f"相对概念超额 {overall['exc_con_20_mean']:+.2%} → 真实表现未跑赢基准, 须 RETRO-002/003 深究")

    verdict = {
        "task": "RETRO-001",
        "status": "built",
        "conclusion": concl,
        "metrics": {
            "coverage": {
                "n_obs_total": int(n_total),
                "n_obs_full20d": int(n_full),
                "n_pick_dates": len(pick_dates),
                "n_stocks": int(picks["ts_code"].nunique()),
                "date_range": [pick_dates[0], pick_dates[-1]],
                "n_partial_horizon": int(n_total - n_full),
                "concept_coverage_frac": float(n_concept_cov / max(n_total, 1)),
            },
            "overall_full20d": overall,
            "by_vintage": by_vintage,
            "by_month": by_month,
            "by_regime": by_regime,
        },
        "caveats": [
            "model-vintage: 生产 r20 (r20_v16_long_nost) 训练窗 < 20250930, 0501+ 早 5 月 (202505~202509) "
            "对 r20 有 in-sample 暴露 → by_vintage['in_sample_r20'] 带乐观偏差 (见 WF-001/WFE-001 de-lookahead)",
            "收益用 close[t]->close[t+h] (生产实际 next_open 进场); 概念均值含 pick 自身, 为概念 beta 基准",
            "(date,stock) 观测含 carryover 持仓 (同股跨日重复), 非独立样本; 描述性回顾非 gate (R03)",
        ],
        "artifacts": [
            "research/backtest/run_retro001.py",
            "research/cache/retro001/retro001_picks_perf.parquet",
            "research/cache/retro001/retro001_by_month.parquet",
        ],
        "discipline": "R01 不动门柱(描述性); R02 描述性=合法完成; R03 中间指标≠ship; "
                      "R04 大缓存 gitignored 无 features 写入; R05 复用缓存 picks 未碰生产; "
                      "R06 ST 源头排除; R11 regime+vintage 分层必带",
    }
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (CACHE / "retro001_results.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
