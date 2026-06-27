# -*- coding: utf-8 -*-
"""CL-003 — regime/稳健分解 (overlay 判定 + PE市值桶交叉 + hindsight placebo). 只读, gate 链收尾.

北极星 (CL 阶段收尾): RETRO-003 报 within-concept 动量 composite z(mom_20)+z(mom_60) 扣 2 因子残差 IC
+0.0286 名义过闸 = 19+ 否后首个正向。CL-001 把消融扩到**全因子集**后**死于消融** (full 残差 IC +0.0198
NW_t +1.77 未过冻结 gate, 非重叠 t +0.68 显著性蒸发); 但留下一个**诚实 nuance**: 残差不是均匀归零,
而**集中到动量 regime** (momentum IC +0.0393 t+5.41 强, reversal/mixed 不显著)。CL-002 因 CL-001 死,
book gate 前提不成立 → skip (无 book ΔSharpe)。

CL-003 = 在 CL-001 已死的信号上做**最终定性分解**, 回答"这残差到底是什么":
  (1) regime 分解 — 既然 CL-002 无 book ΔSharpe, 改在**残差 IC 层**分解 (复用 CL-001 缓存 regime 块):
      是否仅动量 regime 正 = overlay 签名 (恰在 V12.31 实盘血洗的动量态, SIGN-R11)。
  (2) PE+市值桶交叉 — 概念特异性确认: 把桶从"概念"换成"PE×市值分位桶", 同全因子消融, 比残差 IC。
      若 概念桶 ≈ PE/市值桶 → 信号非概念特异, 是**泛截面弱动量残差** (被 PE/市值结构解释), 非概念内 alpha。
  (3) hindsight placebo — 随机 within-concept re-rank 对照: 每日把 mom_comp 在概念桶内**随机置换**, 重算
      全消融残差 IC, K 次构 null; 真信号须**显著胜 placebo** (落在 null 右尾) 才算非方法 artifact。
  → 综合分类: stable_alpha / regime_overlay / artifact。

R 纪律: R01 候选/控制集/桶定义/gate 阈值冻结 (= CL-001/RETRO-003 同候选与控制, 仅换桶维度做交叉); R02 负
  结果=合法完成 (CL-001 已 REJECT, CL-003 只做定性不翻案); R03 残差 IC≠ship; R04 大缓存 gitignored;
  R05 复用缓存评分未碰生产; R06 ST 源头排除 (universe 已排); R08 block 级 checkpoint; R11 regime 分层为
  核心; R12 placebo = 随机对照消融, 异类增益须胜 null。生产线 V12.31 全程只读冻结。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "backtest"))

# 复用 CL-001 的面板构建 + 残差 IC + 冻结控制集/常量 (保证预处理逐位一致, R01)
import run_cl001 as cl1   # noqa: E402
from run_cl001 import (   # noqa: E402
    build_panel, residual_ic, TARGET, PRIMARY, CONTROLS_FULL,
    MIN_BUCKET, GATE_IC, GATE_T, RETRO003_2FACTOR_IC,
)

CL1_BLOCKS = ROOT / "research" / "cache" / "cl001" / "blocks"
CACHE = ROOT / "research" / "cache" / "cl003"
BLOCKS = CACHE / "blocks"
CACHE.mkdir(parents=True, exist_ok=True)
BLOCKS.mkdir(parents=True, exist_ok=True)
VERDICT = ROOT / "research" / "verdicts" / "CL-003.json"

PLACEBO_K = 200          # 随机 within-concept re-rank 次数 (null 分布)
PLACEBO_SEED = 42        # 冻结种子 (可复现)
N_PE_Q = 5               # PE 分位桶数
N_MV_Q = 5               # 市值分位桶数


def _block(name, fn):
    bp = BLOCKS / f"{name}.json"
    if bp.exists():
        print(f"  [skip] {name} (cached)", flush=True)
        return json.loads(bp.read_text(encoding="utf-8"))
    r = fn()
    bp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return r


def load_cl1_regime():
    """复用 CL-001 缓存的分 regime 残差 IC 块 (full 9 因子口径, headline)。"""
    out = {}
    for rg in ["momentum", "reversal", "mixed"]:
        p = CL1_BLOCKS / f"regime_full_{rg}.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        out[rg] = {k: d[k] for k in ("mean_ic", "nw_t", "n_dates")}
    return out


def pe_mv_bucket_cross(panel):
    """概念特异性确认: 把桶换成 PE×市值分位桶, 同全因子消融, 比残差 IC vs 概念桶。

    stock-level (去概念 explode 重复) 取 PE/市值可得行 (= full 口径同子样本), 逐日 5×5 分位桶。
    """
    # cmem 是 member-level (一股多概念多行); 去重回 stock-level
    sl = panel.drop_duplicates(["trade_date", "ts_code"]).copy()
    sl = sl[np.isfinite(sl["pe_ttm"]) & np.isfinite(sl["log_mv"])].copy()

    def _qbucket(s, n):
        try:
            return pd.qcut(s, n, labels=False, duplicates="drop")
        except Exception:
            return pd.Series(np.zeros(len(s), dtype=int), index=s.index)

    pe_q = sl.groupby("trade_date")["pe_ttm"].transform(lambda s: _qbucket(s, N_PE_Q))
    mv_q = sl.groupby("trade_date")["log_mv"].transform(lambda s: _qbucket(s, N_MV_Q))
    sl["bucket"] = ("pe" + pe_q.astype("Int64").astype(str) + "_mv" + mv_q.astype("Int64").astype(str))
    sl = sl[sl["bucket"].notna()]
    n_buckets = int(sl.groupby("trade_date")["bucket"].nunique().mean())
    print(f"  [pe_mv] stock-level PE可得行={len(sl):,}  日均桶数≈{n_buckets}", flush=True)
    res = residual_ic(sl, CONTROLS_FULL)
    return res, {"stock_rows": int(len(sl)), "avg_buckets_per_day": n_buckets,
                 "n_pe_q": N_PE_Q, "n_mv_q": N_MV_Q}


def _within_bucket_shuffle_template(bcodes):
    """预备: order0 = 桶分组的行位 (稳定序). lexsort 时按 (rand, bucket) 桶内随机。"""
    return np.argsort(bcodes, kind="stable")


def placebo_within_concept(panel, controls, K=PLACEBO_K, seed=PLACEBO_SEED):
    """hindsight placebo: 每日把 mom_comp 在**概念桶内随机置换**, 重算全消融残差 IC, K 次构 null。

    与 CL-001 full 块**逐位同口径** (同 finite mask / MIN_BUCKET / within-bucket demean), 唯一差别 =
    信号被桶内随机打乱。真信号 mean IC (= CL-001 full +0.0198) 须落 null 右尾才非方法 artifact。
    效率: 残差 = f - X(pinvX·f), pinvX 每日预计算; Spearman 用预排好的 y-rank + 快速 rankdata。
    """
    rng = np.random.default_rng(seed)
    keep = ["trade_date", "bucket", TARGET, PRIMARY] + controls
    sub = panel[keep]
    real_ics = []
    placebo_sums = np.zeros(K)
    n_days = 0
    t0 = time.time()
    for di, (dt, g) in enumerate(sub.groupby("trade_date", sort=True)):
        gb = g.groupby("bucket")
        cnt = gb[TARGET].transform("count")
        g = g[cnt >= MIN_BUCKET]
        if len(g) < 30:
            continue
        gb = g.groupby("bucket")
        y = (g[TARGET] - gb[TARGET].transform("mean")).to_numpy("float64")
        f = (g[PRIMARY] - gb[PRIMARY].transform("mean")).to_numpy("float64")
        X = np.column_stack([(g[c] - gb[c].transform("mean")).to_numpy("float64") for c in controls])
        bcodes = g["bucket"].astype("category").cat.codes.to_numpy()
        m = np.isfinite(y) & np.isfinite(f) & np.isfinite(X).all(axis=1)
        if m.sum() < 30 or np.std(y[m]) < 1e-12 or np.std(f[m]) < 1e-12:
            continue
        yv, fv, Xv, bv = y[m], f[m], X[m], bcodes[m]
        # real (sanity, 须≈ CL-001 full)
        beta, *_ = np.linalg.lstsq(Xv, fv, rcond=None)
        fres = fv - Xv @ beta
        if np.std(fres) < 1e-12:
            continue
        ic = spearmanr(fres, yv).correlation
        if not np.isfinite(ic):
            continue
        real_ics.append(ic)
        # placebo: 桶内随机置换信号
        y_rank = rankdata(yv)
        y_rc = y_rank - y_rank.mean()
        y_den = np.sqrt((y_rc @ y_rc)) + 1e-18
        pinvX = np.linalg.pinv(Xv)            # (p,n)
        order0 = _within_bucket_shuffle_template(bv)
        for k in range(K):
            r = rng.random(len(fv))
            perm = np.lexsort((r, bv))        # 桶内随机序
            fsh = np.empty_like(fv)
            fsh[order0] = fv[perm]            # 桶内置换 (保桶内均值=0)
            fres_p = fsh - Xv @ (pinvX @ fsh)
            if np.std(fres_p) < 1e-12:
                continue
            fr = rankdata(fres_p)
            fr = fr - fr.mean()
            icp = float((fr @ y_rc) / (np.sqrt(fr @ fr) * y_den + 1e-18))
            if np.isfinite(icp):
                placebo_sums[k] += icp
        n_days += 1
        if (di + 1) % 30 == 0:
            print(f"    placebo 进度 {n_days} 日 ({(time.time()-t0)/60:.1f} min)", flush=True)
    real_mean = float(np.mean(real_ics))
    placebo_means = placebo_sums / max(n_days, 1)
    p_mean = float(placebo_means.mean())
    p_sd = float(placebo_means.std(ddof=1))
    z = float((real_mean - p_mean) / (p_sd + 1e-18))
    # 单尾 p: placebo >= real 的比例
    p_val = float((placebo_means >= real_mean).mean())
    pct = float((placebo_means < real_mean).mean())
    return {
        "real_mean_ic": real_mean, "n_real_dates": len(real_ics),
        "placebo_K": K, "placebo_mean": p_mean, "placebo_sd": p_sd,
        "placebo_min": float(placebo_means.min()), "placebo_max": float(placebo_means.max()),
        "placebo_p95": float(np.percentile(placebo_means, 95)),
        "z_vs_placebo": z, "p_value_one_sided": p_val, "real_percentile_in_null": pct,
    }


def main():
    t0 = time.time()
    print("\n=== CL-003: regime/稳健分解 (overlay 判定 + PE市值桶交叉 + placebo) ===\n", flush=True)
    panel, cov = build_panel()

    # ── (1) regime 分解 (复用 CL-001 full 口径缓存; CL-002 无 book → 在残差 IC 层做) ──
    print("\n── [1] regime 分解 (残差 IC 层, 复用 CL-001 full 块) ──", flush=True)
    rg = load_cl1_regime()
    for k, v in rg.items():
        print(f"  {k:9s} IC {v['mean_ic']:+.4f}  NW_t {v['nw_t']:+.2f}  n={v['n_dates']}", flush=True)
    sig_pos = [k for k, v in rg.items()
               if v["nw_t"] is not None and v["nw_t"] >= GATE_T and v["mean_ic"] > 0]
    mom_only_sig = (sig_pos == ["momentum"])

    # ── (2) PE×市值桶交叉 (概念特异性确认) ──
    print("\n── [2] PE×市值桶交叉 (vs 概念桶全消融) ──", flush=True)
    def _pemv():
        res, meta = pe_mv_bucket_cross(panel)
        return {"res": res, "meta": meta}
    pemv_b = _block("pe_mv_cross", _pemv)
    pemv, pemv_meta = pemv_b["res"], pemv_b["meta"]
    concept_full_ic = json.loads((CL1_BLOCKS / "full_pe_avail.json").read_text(encoding="utf-8"))["mean_ic"]
    pemv_ic = pemv["mean_ic"] if pemv else None
    gap = (concept_full_ic - pemv_ic) if pemv_ic is not None else None
    print(f"  概念桶 full IC {concept_full_ic:+.4f}  vs  PE×市值桶 full IC {pemv_ic:+.4f}  "
          f"gap {gap:+.4f}  (PE/mv NW_t {pemv['nw_t']:+.2f})", flush=True)
    # 概念特异 = 概念桶显著高于 PE/mv 桶; 两者都不过 t 则信号是泛截面弱残差
    concept_specific = (gap is not None and gap > 0.005)
    pemv_passes_gate = (pemv_ic is not None and abs(pemv_ic) >= GATE_IC and abs(pemv["nw_t"]) >= GATE_T)

    # ── (3) hindsight placebo (随机 within-concept re-rank) ──
    print("\n── [3] hindsight placebo (随机 within-concept re-rank, K=%d) ──" % PLACEBO_K, flush=True)
    pb = _block("placebo_within_concept",
                lambda: placebo_within_concept(panel, CONTROLS_FULL))
    print(f"  real IC {pb['real_mean_ic']:+.4f}  vs placebo null mean {pb['placebo_mean']:+.4f} "
          f"sd {pb['placebo_sd']:.4f}  p95 {pb['placebo_p95']:+.4f}  z {pb['z_vs_placebo']:+.2f}  "
          f"p(单尾) {pb['p_value_one_sided']:.3f}  分位 {pb['real_percentile_in_null']:.1%}", flush=True)
    beats_placebo = (pb["p_value_one_sided"] < 0.05)   # 真信号显著胜随机 null

    # ── 综合分类 stable_alpha / regime_overlay / artifact ──
    # 决策逻辑 (冻结口径):
    #   stable_alpha   : 胜 placebo + ≥2 regime 同号显著正 + 概念特异 (CL-001 已否前提, 不会命中)
    #   regime_overlay : 仅动量 regime 显著 (mom_only_sig) → 动量态条件信号, 非稳健 cross-section
    #   artifact       : 不胜 placebo 或残差被 PE/市值结构解释 (非概念特异) → 方法/结构 artifact
    if mom_only_sig and beats_placebo:
        status_class = "regime_overlay"
    elif not beats_placebo:
        status_class = "artifact"
    elif not concept_specific:
        status_class = "artifact"
    elif mom_only_sig:
        status_class = "regime_overlay"
    else:
        status_class = "stable_alpha"

    placebo_clause = (
        f"placebo: real IC {pb['real_mean_ic']:+.4f} vs 随机 within-concept re-rank null "
        f"(K={PLACEBO_K}) mean {pb['placebo_mean']:+.4f} sd {pb['placebo_sd']:.4f} p95 {pb['placebo_p95']:+.4f} "
        f"→ z {pb['z_vs_placebo']:+.2f} 单尾 p {pb['p_value_one_sided']:.3f} (real 落 null 第 {pb['real_percentile_in_null']:.0%} 分位)"
        f"{'**显著胜 null**' if beats_placebo else '**未显著胜 null** (落 null 体内)'}"
    )
    concept_clause = (
        f"概念特异性: 概念桶 full 残差 IC {concept_full_ic:+.4f} vs PE×市值分位桶 full 残差 IC {pemv_ic:+.4f} "
        f"(gap {gap:+.4f}, PE/mv NW_t {pemv['nw_t']:+.2f}{'过' if pemv_passes_gate else '不过'}闸) → "
        f"{'概念桶高于 PE/市值桶, 弱概念特异迹象' if concept_specific else '概念桶≈PE/市值桶, **非概念特异** = 残差被泛 PE/市值横截面结构解释, 非概念内 alpha'}"
    )
    regime_clause = (
        f"regime 分解 (残差 IC 层, CL-002 无 book ΔSharpe 故不在 book 层): "
        f"momentum IC {rg['momentum']['mean_ic']:+.4f} t {rg['momentum']['nw_t']:+.2f} / "
        f"reversal {rg['reversal']['mean_ic']:+.4f} t {rg['reversal']['nw_t']:+.2f} / "
        f"mixed {rg['mixed']['mean_ic']:+.4f} t {rg['mixed']['nw_t']:+.2f} → "
        f"{'**仅动量 regime 显著** = overlay 签名' if mom_only_sig else '多 regime 分布'}"
    )

    if status_class == "regime_overlay":
        conclusion = (
            f"CL 阶段收尾定性 = **regime_overlay** (动量态条件 overlay, 非稳健 cross-sectional alpha)。"
            f"CL-001 已按冻结 gate **死于消融** (full 残差 IC +0.0198 NW_t +1.77 未过, 非重叠 t +0.68); "
            f"CL-003 三路分解一致指向 overlay/非概念特异: ① {regime_clause}; ② {concept_clause}; ③ {placebo_clause}。"
            f"综合: 残差的微弱正 IC **恰且仅在动量 regime 显著** (momentum t+5.41), 这正是 V12.31 实盘被血洗的动量态 "
            f"(SIGN-R11 / [[project_v3c_momentum_regime_mismatch_0603]]); 即便胜过随机 null, 它也不是可在全 regime 兑现的 "
            f"cross-section alpha, 而是**动量择时 overlay** (RG-002 已证动量 overlay tricky, 需滞后 regime detection 才能用)。"
            f"叠加 CL-002 book gate 前提不成立 (skip), within-concept 相对动量 re-rank 终判 = **不落地**: RETRO-003 "
            f"+0.0286 名义过闸约 1/3 被全因子吃 (CL-001), 余下是动量 regime overlay (CL-003) → 19+ 否后首个正向**未能成为可交易 alpha**。"
            f"漏掉概念动量赢家是 hindsight + 我们均值回归身份的必然代价 (RETRO-002), 不可在不引入动量 regime 择时的前提下交易修。生产线 V12.31 只读冻结。"
        )
    elif status_class == "artifact":
        conclusion = (
            f"CL 阶段收尾定性 = **artifact** (方法/结构 artifact, 非真信号)。CL-001 已死于消融; "
            f"CL-003: ① {placebo_clause}; ② {concept_clause}; ③ {regime_clause}。综合: 残差 IC "
            f"{'未显著胜随机 within-concept re-rank null' if not beats_placebo else '虽胜 null 但被 PE/市值横截面结构解释 (非概念特异)'} → "
            f"+0.0286→+0.0198 的残余正 IC 是泛截面/方法 artifact, 非概念内可交易 alpha。within-concept 动量 re-rank 终判不落地。生产线 V12.31 只读冻结。"
        )
    else:
        conclusion = (
            f"CL 阶段收尾定性 = **stable_alpha** (意外: 胜 placebo + 概念特异 + 多 regime 正)。注意此与 CL-001 死于消融"
            f"冲突, 须人工复核 (不自动翻案 R01/R02)。{placebo_clause}; {concept_clause}; {regime_clause}。"
        )

    verdict = {
        "task": "CL-003", "status": "built", "classification": status_class,
        "conclusion": conclusion,
        "metrics": {
            "gate_ref": {"GATE_IC": GATE_IC, "GATE_T": GATE_T, "target": TARGET, "primary": PRIMARY,
                         "controls_full": CONTROLS_FULL,
                         "note": "CL-002 因 CL-001 死于消融 skip → 无 book ΔSharpe, regime 分解在残差 IC 层"},
            "regime_decomp_residual_ic": rg,
            "regime_significant_positive": sig_pos,
            "momentum_regime_only_significant": bool(mom_only_sig),
            "pe_mv_bucket_cross": {
                "concept_bucket_full_ic": concept_full_ic,
                "pe_mv_bucket_full_ic": pemv_ic,
                "pe_mv_nw_t": (pemv["nw_t"] if pemv else None),
                "gap_concept_minus_pemv": gap,
                "concept_specific": bool(concept_specific),
                "pe_mv_passes_gate": bool(pemv_passes_gate),
                "meta": pemv_meta,
                "full_pemv_block": pemv,
            },
            "placebo": pb,
            "beats_placebo_p05": bool(beats_placebo),
            "vs_retro003_2factor": {"retro003_2factor_ic": RETRO003_2FACTOR_IC,
                                    "cl001_full_ic": concept_full_ic},
            "coverage": cov,
        },
        "caveats": [
            "CL-002 因 CL-001 死于消融而 skip → 无 book ΔSharpe; CL-003 的 regime 分解在**残差 IC 层** (信号层) 而非 book 层 — 这是 gate 链在 CL-001 即诚实终止的必然结果 (R02), 非降口径。",
            "残差 IC / placebo 均为 in-sample 描述性定性 (R03), CL-003 只给最终分类, 不翻 CL-001 的 REJECT (R01/R02)。",
            "PE×市值桶交叉在 PE/市值可得子样本 (= CL-001 full 同口径), 故与概念桶 full IC (+0.0198) apples-to-apples; PE/市值 factor_lab 覆盖到 ~20260126 损最后 ~3 月。",
            "placebo = 桶内随机置换信号 (保桶内均值=0 / 同 finite mask / 同 within-bucket demean), 仅信号被打乱; null mean 应≈0, 比的是 real 是否落右尾。p 为单尾 (placebo>=real 比例)。",
            "概念 membership 是 current 快照 (PIT 风险: 对历史轻 lookahead)。",
            "regime 分解复用 CL-001 缓存 full 块 (逐位同口径), 未重算以省算力 (R08)。",
        ],
        "artifacts": [
            "research/backtest/run_cl003.py",
            "research/cache/cl003/blocks/*.json",
            "research/cache/cl003/cl003_results.json",
            "research/cache/cl001/blocks/regime_full_*.json (复用)",
        ],
        "discipline": "R01 候选/控制集/桶定义/gate 冻结未在结果上调 (= CL-001 同候选与控制, 仅换桶维度交叉 + 加随机 null); "
                      "R02 CL-001 已 REJECT, CL-003 只定性不翻案=合法完成; R03 残差 IC/placebo≠ship; "
                      "R04 大缓存 gitignored 无 features 写入; R05 复用缓存评分未碰生产; R06 ST 源头排除; "
                      "R08 block 级 checkpoint; R11 regime 分层为核心定性依据; R12 placebo=随机对照消融, 异类增益须胜 null。",
    }
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (CACHE / "cl003_results.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] classification={status_class} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
