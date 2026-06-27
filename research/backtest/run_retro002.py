# -*- coding: utf-8 -*-
"""RETRO-002 — 概念内"漏掉的赢家"画像 (ex-ante 特征 vs hindsight 收益). 纯只读, 描述性.

北极星 (RETRO 阶段): 把 0501+ V12.31 picks 和它们**所属概念里后来涨得好但没被选中**的股票
("漏掉的赢家" = missed_winner) 做对比, 看差异到底在**下单前可观测的 ex-ante 特征** (可能可学)
还是只在**事后实现收益** (hindsight, 不可交易, = SLV-002 幸存者偏差陷阱)。

winner_def (prd.preRegisteredGate, 冻结 R01):
  主桶 = merged 概念库 (output/concept_local/concept_merged.parquet) 映射每股概念;
  每 (pick_date, concept) 内**全成员**按后续 20 日 max_gain (=max(high[t+1..t+20])/close-1)
  排序, **TOP 20% = winner**; missed_winner = winner 且当日**未在 V12.31 池**。
  仅在含 ≥1 个 pick 的概念 (pick_concepts) 内打标 (= 有 pick 可对比的桶, 同 RETRO-001)。
  仅对 full-20d (前向满 20 交易日) 的 (date) 打标, 末端不满 20d 的日剔除 (公平排序)。

ex-ante 特征 (prd.preRegisteredGate.exante_features, 全部下单前 = D 收盘可知):
  模型分 (= V12.31 实际决策变量, 来自 BT-003 scored_daily 全宇宙评分):
    pred_r20 (V7c 池排序分) / ratio_s5 (pump_up) / pump_down_s5 / pyr_velocity_20_60
  价量派生 (本脚本从日线 backward 计算): past_r5 / mom_20 / mom_60 / rsi_14 / turn_amt_20
  ADX (BT-004 px_feats 全宇宙) / total_mv 市值 (factor_lab, 覆盖到 ~20260126, 部分)
realized (hindsight, **不可交易**): max_gain_20 / ret_20 — 严格只作收益差分, 不混入 ex-ante。

R 纪律: R01 winner_def/特征集冻结不在结果上调; R02 描述性=合法完成; R03 中间指标≠ship;
  R04 大缓存 research/cache/retro002/ gitignored 无 features 写入; R05 复用缓存评分未碰生产;
  R06 ST 源头排除 (load_prices); R11 regime 分层必带; hindsight 陷阱严格 ex-ante vs realized 分解。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))
from run_bt002 import load_prices, DAILY_DIR, BASIC

PICKS = ROOT / "research" / "cache" / "bt001" / "picks_v1231_daily.parquet"
CONCEPT = ROOT / "output" / "concept_local" / "concept_merged.parquet"
SCORED = ROOT / "research" / "cache" / "bt003" / "scored_daily.parquet"
PXFEATS = ROOT / "research" / "cache" / "bt004" / "px_feats.parquet"   # ts_code,trade_date,ctx,ma20_slope,adx
FLAB = ROOT / "output" / "factor_lab_3y" / "factor_groups" / "group_001.parquet"  # total_mv
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
CACHE = ROOT / "research" / "cache" / "retro002"
CACHE.mkdir(parents=True, exist_ok=True)
VERDICT = ROOT / "research" / "verdicts" / "RETRO-002.json"

START = "20250501"            # 0501 起 (前向真实 OOS, picks 区间)
PX_START = "20250101"         # 多留 lookback 供 mom_60 / rsi (backward 窗)
MAXGAIN_H = 20
WINNER_Q = 0.80              # 概念内 TOP 20% = winner (冻结 R01)
VINTAGE_INSAMPLE_BEFORE = "202510"   # r20 训练窗 < 20250930 -> 早月 in-sample (vintage caveat)

# 模型分 (scored_daily 列名)
MODEL_FEATS = ["pred_r20", "ratio_s5", "pump_down_s5", "pyr_velocity_20_60"]
# 价量派生 (本脚本算) + adx + total_mv
PX_FEATS = ["past_r5", "mom_20", "mom_60", "rsi_14", "turn_amt_20"]
EXANTE_FEATS = MODEL_FEATS + PX_FEATS + ["adx", "total_mv"]
REALIZED_FEATS = ["max_gain_20", "ret_20"]   # hindsight, 不可交易


def _fwd_close_ret(close_piv, h):
    T = close_piv.shape[0]
    out = np.full_like(close_piv, np.nan)
    if T > h:
        out[:T - h] = close_piv[h:] / close_piv[:T - h] - 1.0
    return out


def _fwd_max_high(high_piv, close_piv, h):
    T = high_piv.shape[0]
    M = np.full_like(high_piv, -np.inf)
    nfwd = np.zeros(high_piv.shape, dtype=np.int16)
    for k in range(1, h + 1):
        if T - k <= 0:
            break
        M[:T - k] = np.fmax(M[:T - k], high_piv[k:])
        nfwd[:T - k] += 1
    M[~np.isfinite(M)] = np.nan
    return M / close_piv - 1.0, nfwd


def _bwd_ret(close_piv, lag):
    """past return over `lag` days (backward, ex-ante): close[t]/close[t-lag]-1."""
    T = close_piv.shape[0]
    out = np.full_like(close_piv, np.nan)
    if T > lag:
        out[lag:] = close_piv[lag:] / close_piv[:T - lag] - 1.0
    return out


def _rsi_wilder(close_piv, n=14):
    """Wilder RSI-14 (backward, ex-ante), per column."""
    d = np.diff(close_piv, axis=0)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    T, C = close_piv.shape
    rsi = np.full((T, C), np.nan)
    au = np.full(C, np.nan); ad = np.full(C, np.nan)
    for t in range(1, T):
        u = up[t - 1]; v = dn[t - 1]
        if t == n:
            au = np.nanmean(up[:n], axis=0); ad = np.nanmean(dn[:n], axis=0)
        elif t > n:
            au = (au * (n - 1) + u) / n
            ad = (ad * (n - 1) + v) / n
        else:
            continue
        rs = au / (ad + 1e-12)
        rsi[t] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _turn_amt(amount_piv, win=20):
    """relative turnover/liquidity proxy (backward): amount[t] / mean(amount[t-win..t-1])."""
    df = pd.DataFrame(amount_piv)
    ma = df.rolling(win, min_periods=max(5, win // 2)).mean().shift(1)
    return (df / (ma + 1e-12)).to_numpy()


def _smd(a, b):
    """standardized mean difference (pick - missed) / pooled std (effect size)."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt((va + vb) / 2.0)
    if sp <= 0:
        return None
    return float((np.mean(a) - np.mean(b)) / sp)


def main():
    t0 = time.time()
    print("\n=== RETRO-002: 概念内漏赢家画像 (ex-ante 特征 vs hindsight 收益, 只读) ===\n", flush=True)

    # ── picks (0501+) ──
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    picks = picks[picks["trade_date"] >= START].copy()
    pick_dates = sorted(picks["trade_date"].unique())
    pick_key = set(zip(picks["trade_date"], picks["ts_code"]))
    print(f"[picks] {len(picks):,} (date,stock) / {len(pick_dates)} 日 / {picks['ts_code'].nunique()} 股", flush=True)

    # ── 价表 (全市场, ST 源头排除 R06) ──
    print(f"[prices] load >= {PX_START} (ST 源头排除) ...", flush=True)
    px = load_prices(PX_START)
    cal = sorted(px["trade_date"].unique())
    close = px.pivot_table(index="trade_date", columns="ts_code", values="close").reindex(cal)
    high = px.pivot_table(index="trade_date", columns="ts_code", values="high").reindex(cal)
    # amount 单独从日线缓存读 (load_prices 不含 amount), ST 源头排除
    aparts = []
    for f in sorted(DAILY_DIR.glob("*.parquet")):
        d = pd.read_parquet(f, columns=["ts_code", "trade_date", "amount"])
        d["trade_date"] = d["trade_date"].astype(str)
        d = d[d["trade_date"] >= PX_START]
        if not d.empty:
            aparts.append(d)
    amt = pd.concat(aparts, ignore_index=True)
    amount = amt.pivot_table(index="trade_date", columns="ts_code", values="amount").reindex(cal)
    amount = amount.reindex(columns=close.columns)
    codes = list(close.columns)
    cp = close.to_numpy("float64"); hp = high.to_numpy("float64"); ap = amount.to_numpy("float64")
    print(f"[prices] {len(px):,} 行 / {len(codes)} 股 / {len(cal)} 日 [{cal[0]}~{cal[-1]}]", flush=True)

    # ── 前向 (realized, hindsight) + backward (ex-ante 价量) ──
    mg_mat, nfwd = _fwd_max_high(hp, cp, MAXGAIN_H)
    ret20_mat = _fwd_close_ret(cp, MAXGAIN_H)
    bwd = {"past_r5": _bwd_ret(cp, 5), "mom_20": _bwd_ret(cp, 20), "mom_60": _bwd_ret(cp, 60),
           "rsi_14": _rsi_wilder(cp, 14), "turn_amt_20": _turn_amt(ap, 20)}

    def _mat_long(mat, name, dates):
        df = pd.DataFrame(mat, index=cal, columns=codes).loc[dates]
        s = df.stack(future_stack=True); s.index.names = ["trade_date", "ts_code"]
        return s.rename(name)

    # ── 概念成员 (仅 pick_concepts) ──
    mem = pd.read_parquet(CONCEPT)[["stock_code", "concept_ts_code"]].dropna().drop_duplicates()
    mem = mem.rename(columns={"stock_code": "ts_code", "concept_ts_code": "concept"})
    pick_concepts = set(mem[mem["ts_code"].isin(set(picks["ts_code"]))]["concept"].unique())
    memf = mem[mem["concept"].isin(pick_concepts)].copy()
    print(f"[concept] pick_concepts={len(pick_concepts)}  成员对={len(memf):,}", flush=True)

    # ── winner 打标: 仅 full-20d 的 pick_date (公平排序; 末端不满 20d 的日剔除) ──
    full_dates = [d for d in pick_dates if cal.index(d) + MAXGAIN_H < len(cal)]
    print(f"[winner] full-20d 可打标 pick_dates: {len(full_dates)}/{len(pick_dates)} "
          f"[{full_dates[0]}~{full_dates[-1]}]  (末端不满 20d 的日剔除)", flush=True)

    mg_long = _mat_long(mg_mat, "max_gain_20", full_dates).reset_index().dropna(subset=["max_gain_20"])
    # 成员级: (date, stock, concept, max_gain) -> 概念内 TOP20% = winner_in_concept
    dsc = mg_long.merge(memf, on="ts_code", how="inner")
    dsc["rk"] = dsc.groupby(["trade_date", "concept"])["max_gain_20"].rank(pct=True, method="average")
    dsc["win_in_con"] = dsc["rk"] >= WINNER_Q
    # 聚合到 (date, stock): winner = 在其任一概念内 TOP20%
    winflag = dsc.groupby(["trade_date", "ts_code"])["win_in_con"].max().rename("is_winner").reset_index()

    # ── 对比宇宙 = pick_concepts 全成员 (full-20d 日), 标 is_pick / is_winner / missed_winner ──
    uni = winflag.copy()
    uni["is_pick"] = [ (d, c) in pick_key for d, c in zip(uni["trade_date"], uni["ts_code"]) ]
    uni["missed_winner"] = uni["is_winner"] & (~uni["is_pick"])
    # realized 收益 (hindsight)
    uni = uni.merge(mg_long, on=["trade_date", "ts_code"], how="left")
    uni = uni.merge(_mat_long(ret20_mat, "ret_20", full_dates).reset_index(), on=["trade_date", "ts_code"], how="left")

    n_pick = int(uni["is_pick"].sum())
    n_win = int(uni["is_winner"].sum())
    n_missed = int(uni["missed_winner"].sum())
    pick_winhit = float(uni.loc[uni["is_pick"], "is_winner"].mean())   # picks 自身的 winner 命中率
    print(f"[universe] {len(uni):,} (date,stock); picks={n_pick} winner={n_win} missed_winner={n_missed}; "
          f"picks 自身 winner 命中率={pick_winhit:.1%}", flush=True)

    # ── ex-ante 特征合入 ──
    # 模型分 (scored_daily 全宇宙)
    sc = pd.read_parquet(SCORED, columns=["ts_code", "trade_date"] + MODEL_FEATS)
    sc["trade_date"] = sc["trade_date"].astype(str)
    uni = uni.merge(sc, on=["trade_date", "ts_code"], how="left")
    # adx (bt004 px_feats 全宇宙)
    pf = pd.read_parquet(PXFEATS, columns=["ts_code", "trade_date", "adx"])
    pf["trade_date"] = pf["trade_date"].astype(str)
    uni = uni.merge(pf, on=["trade_date", "ts_code"], how="left")
    # 价量派生 (本脚本算) — 只取 full_dates 行
    for name, mat in bwd.items():
        uni = uni.merge(_mat_long(mat, name, full_dates).reset_index(), on=["trade_date", "ts_code"], how="left")
    # total_mv (factor_lab, 部分覆盖)
    fl = pd.read_parquet(FLAB, columns=["ts_code", "trade_date", "total_mv"])
    fl["trade_date"] = fl["trade_date"].astype(str)
    uni = uni.merge(fl, on=["trade_date", "ts_code"], how="left")
    mv_cov = float(uni["total_mv"].notna().mean())

    # ── regime / vintage (R11) ──
    reg = pd.read_parquet(REGIME)[["trade_date", "regime"]]
    reg["trade_date"] = reg["trade_date"].astype(str)
    uni = uni.merge(reg, on="trade_date", how="left")
    uni["ym"] = uni["trade_date"].str[:6]
    uni["vintage"] = np.where(uni["ym"] < VINTAGE_INSAMPLE_BEFORE, "in_sample_r20", "true_oos")
    uni.to_parquet(CACHE / "retro002_universe.parquet", index=False)

    # ── 对比表: picks vs missed_winners ──
    A = uni[uni["is_pick"]]                 # picks
    B = uni[uni["missed_winner"]]           # 漏掉的赢家

    def _cmp(feats, A, B):
        tab = {}
        for f in feats:
            a = A[f].to_numpy("float64"); b = B[f].to_numpy("float64")
            tab[f] = {
                "pick_mean": float(np.nanmean(a)) if np.isfinite(a).any() else None,
                "pick_median": float(np.nanmedian(a)) if np.isfinite(a).any() else None,
                "missed_mean": float(np.nanmean(b)) if np.isfinite(b).any() else None,
                "missed_median": float(np.nanmedian(b)) if np.isfinite(b).any() else None,
                "diff_mean": (float(np.nanmean(a) - np.nanmean(b))
                              if np.isfinite(a).any() and np.isfinite(b).any() else None),
                "smd": _smd(a, b),
                "pick_cov": float(np.isfinite(a).mean()), "missed_cov": float(np.isfinite(b).mean()),
            }
        return tab

    exante_tab = _cmp(EXANTE_FEATS, A, B)
    realized_tab = _cmp(REALIZED_FEATS, A, B)

    # 按 |smd| 排 ex-ante 差异
    ranked = sorted([(f, v["smd"]) for f, v in exante_tab.items() if v["smd"] is not None],
                    key=lambda kv: -abs(kv[1]))

    print("\n── ex-ante 特征对比 (pick vs missed_winner, 按 |SMD| 排) ──", flush=True)
    for f, smd in ranked:
        v = exante_tab[f]
        print(f"  {f:18s} pick {v['pick_mean']:+.4g} vs missed {v['missed_mean']:+.4g}  "
              f"diff {v['diff_mean']:+.4g}  SMD {smd:+.3f}", flush=True)
    print("\n── realized (hindsight, 不可交易) ──", flush=True)
    for f, v in realized_tab.items():
        print(f"  {f:18s} pick {v['pick_mean']:+.4g} vs missed {v['missed_mean']:+.4g}  "
              f"diff {v['diff_mean']:+.4g}  SMD {v['smd']:+.3f}", flush=True)

    # regime 分层: realized max_gain gap + 头号 ex-ante 特征 SMD
    by_regime = {}
    for rg, g in uni.groupby("regime"):
        if pd.isna(rg):
            continue
        ga, gb = g[g["is_pick"]], g[g["missed_winner"]]
        if len(ga) < 5 or len(gb) < 5:
            continue
        by_regime[rg] = {
            "n_pick": int(len(ga)), "n_missed": int(len(gb)),
            "max_gain_gap": float(np.nanmean(ga["max_gain_20"]) - np.nanmean(gb["max_gain_20"])),
            "ret20_gap": float(np.nanmean(ga["ret_20"]) - np.nanmean(gb["ret_20"])),
            "top_exante_smd": {f: exante_tab[f]["smd"] for f, _ in ranked[:3]},
        }

    # ── 裁决 (描述性) ──
    # ex-ante 差异主要在哪? realized gap 多大?
    top3 = [f for f, _ in ranked[:3]]
    mg_gap = realized_tab["max_gain_20"]["diff_mean"]
    # 模型分主导 = picks 被自己的模型分选出 (预期高); 非模型 ex-ante (mom/size/turn/rsi) 差异是潜在可学信号
    nonmodel = [f for f in PX_FEATS + ["adx", "total_mv"]]
    nm_ranked = sorted([(f, exante_tab[f]["smd"]) for f in nonmodel if exante_tab[f]["smd"] is not None],
                       key=lambda kv: -abs(kv[1]))
    nm_top = nm_ranked[:3]

    conclusion = (
        f"0501+ pick_concepts 内: winner(概念20d max_gain TOP20%)={n_win}, 其中 missed_winner(未入池)={n_missed}; "
        f"picks 自身概念 winner 命中率仅 {pick_winhit:.0%} (= 多数 picks 不是其概念里涨最猛的 20%)。 "
        f"realized(hindsight)差异: missed_winner 的 20d max_gain 比 picks 高 {-mg_gap:+.1%} (天然: winner 按 max_gain 选的, 不可交易)。 "
        f"ex-ante 可观测差异按 |SMD| 排序前三 = {top3}; 其中**非模型分** ex-ante 特征 (动量/市值/换手/RSI/ADX) 最大差异 = "
        + ", ".join(f"{f}(SMD{smd:+.2f})" for f, smd in nm_top) + "。 "
        f"模型分 (pred_r20/ratio_s5) picks 远高于 missed_winner = 预期 (picks 正是被这些分选出的); "
        f"真问题 (missed_winner 是否有可学的 ex-ante 残差结构) 留 RETRO-003 扣桶 beta + 扣现有因子后判 (此处仅描述)。"
    )

    verdict = {
        "task": "RETRO-002", "status": "built", "conclusion": conclusion,
        "metrics": {
            "coverage": {
                "n_pick_dates": len(pick_dates), "n_full20d_dates": len(full_dates),
                "n_universe_obs": int(len(uni)), "n_pick": n_pick,
                "n_winner": n_win, "n_missed_winner": n_missed,
                "pick_winner_hit_rate": pick_winhit,
                "missed_winner_base_rate": float(n_missed / max(len(uni), 1)),
                "total_mv_coverage": mv_cov, "winner_q": WINNER_Q,
            },
            "exante_compare": exante_tab,
            "realized_compare_hindsight": realized_tab,
            "exante_smd_ranked": ranked,
            "nonmodel_exante_smd_ranked": nm_ranked,
            "by_regime": by_regime,
        },
        "ex_ante_vs_hindsight": {
            "ex_ante_observable": EXANTE_FEATS,
            "realized_hindsight_not_tradeable": REALIZED_FEATS,
            "note": "ex-ante = D 收盘可知 (模型分/动量/市值/换手/RSI/ADX); realized max_gain/ret_20 = "
                    "事后实现, winner 正是按 max_gain 选的 → realized 差异是定义性 hindsight, 不可交易 (SLV-002 陷阱)。"
                    "模型分差异是套套逻辑 (picks 被模型分选出); 真正可能可学的是**非模型 ex-ante 特征**残差 → RETRO-003 gate。",
        },
        "caveats": [
            "winner/missed_winner 是 forward-conditioned (按后续 20d max_gain 定义), '差异'的 realized 部分纯 hindsight 不可交易;",
            "model-vintage: r20 训练窗 < 20250930, 早月 (202505~09) picks 的 pred_r20 带 in-sample 暴露 (vintage 标注);",
            "total_mv 来自 factor_lab 覆盖到 ~20260126 ({:.0%} 行有值); 其余日市值缺失, 仅作部分参考;".format(mv_cov),
            "(date,stock) 含 carryover/多概念聚合, 非独立样本; 描述性回顾非 gate (R03)。",
        ],
        "artifacts": [
            "research/backtest/run_retro002.py",
            "research/cache/retro002/retro002_universe.parquet",
        ],
        "discipline": "R01 winner_def/特征集冻结(描述性); R02 描述性=合法完成; R03 中间指标≠ship; "
                      "R04 大缓存 gitignored 无 features 写入; R05 复用缓存评分未碰生产; R06 ST 源头排除; "
                      "R11 regime 分层必带; ex-ante vs hindsight 严格分解 (hindsight 陷阱)。",
    }
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (CACHE / "retro002_results.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
