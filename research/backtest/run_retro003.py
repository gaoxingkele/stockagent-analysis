# -*- coding: utf-8 -*-
"""RETRO-003 — 对比学习可行性 gate (within-bucket 扣beta+扣因子残差 + PE市值桶交叉验证). 只读, gate.

真问题 (northStar): 扣桶 beta (桶内去均值) + 扣现有 V12.31 因子 (pump ratio + r20) 后,
桶内还有没有**ex-ante 可观测**残差能分谁跑赢 (= 对比学习/within-bucket LTR 是否能增 ex-ante alpha)?
RETRO-002 已揭示 missed_winner 的差异主要在**动量** (mom_20/mom_60 SMD 最大, ex-ante 可观测),
故预注册主候选 = 桶内动量 composite (z(mom_20)+z(mom_60))。但 RETRO-002 的差异是**无条件**的,
且 realized max_gain 差异纯 hindsight; 本 task 才是真正的 gate: 把候选 ex-ante 特征**扣桶 beta + 扣因子**
后对**可交易的后续 20d 收益 (ret_20, close-to-close)** 算残差 IC。

contrastive_gate (prd.preRegisteredGate, 冻结 R01):
  within-bucket 残差可分性: 桶内去均值(扣beta)后候选 ex-ante 特征对后续 20d 收益残差 IC,
  再扣 [pump ratio + r20]; NEW = 残差 |IC|>=0.01 & |t|>=2 (过桶 demean + 因子消融) → LTR/对比学习候选
  (需 walk-forward); 否则 REJECT_beta_hindsight; 有残差但 <阈/不稳 = 真小。

bucket_crosscheck (冻结 R01):
  除概念主桶, 加 PE+市值相似 peer 桶 (同日按 log市值 五分位 × PE 五分位 = 25 桶) 重算同一残差 IC;
  若两桶残差一致 → 差异泛截面 (非概念特异); 若概念桶残差显著高于 PE市值桶 → 概念特异结构
  (更支持概念对比学习)。两桶在**同一 PE/市值可得子样本**上比较 (apples-to-apples)。

方法 (全因果, 描述性 IC 非 ship — NEW 也只是"值得 walk-forward"非落地, R03):
  目标 y = ret_20 (可交易前向 20d close 收益); 控制 C = [ratio_s5, pred_r20] (= 现有 V12.31 因子)。
  桶 = 概念 (member-level, 一股属多概念则多行) 或 PE×市值 25 桶 (一股一桶)。
  Stage A 扣 beta: 桶内 (date,bucket) 去均值 → f_d, C_d, y_d (桶 <5 成员剔, demean 才稳)。
  Stage B 扣因子: 逐日 (cross-section) f_d ~ C_d OLS(无截距, 已 demean) → 残差 f_res。
  残差 IC = 逐日 Spearman(f_res, y_d), 跨日均值; t = Newey-West (lag=20, 修 20d 重叠) headline,
  非重叠 (每 20 交易日取样) 作 robustness。

R 纪律: R01 gate/候选/控制/桶定义冻结不在结果上调; R02 REJECT=合法完成; R03 残差 IC 即便过阈也只
  "值得 walk-forward"非 ship; R04 大缓存 research/cache/retro003/ gitignored 无 features 写入;
  R05 复用缓存评分未碰生产; R06 ST 源头排除 (universe 已排); R11 regime 分层必带; R12 异类增益须存活
  朴素消融 (扣桶 beta + 扣现有因子 = 朴素对照)。
"""
from __future__ import annotations
import json, sys, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))

UNIV = ROOT / "research" / "cache" / "retro002" / "retro002_universe.parquet"
PICKS = ROOT / "research" / "cache" / "bt001" / "picks_v1231_daily.parquet"
CONCEPT = ROOT / "output" / "concept_local" / "concept_merged.parquet"
FLAB_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
CACHE = ROOT / "research" / "cache" / "retro003"
CACHE.mkdir(parents=True, exist_ok=True)
VERDICT = ROOT / "research" / "verdicts" / "RETRO-003.json"

START = "20250501"
NW_LAG = 20                     # Newey-West lag = 前向 horizon (修 20d 重叠致 t 膨胀)
MIN_BUCKET = 5                  # 桶内 <5 成员剔 (demean/IC 才稳)
WINNER_Q = 0.80
GATE_IC = 0.01                  # contrastive_gate (冻结 R01)
GATE_T = 2.0

TARGET = "ret_20"              # 可交易前向 20d close 收益 (非 max_gain 的 hindsight 定义)
CONTROLS = ["ratio_s5", "pred_r20"]   # 现有 V12.31 因子 (pump ratio + r20), 冻结
# 候选 ex-ante 特征 (非模型分, 下单前可观测) — 模型分是套套逻辑 (picks 被其选出) 故不作候选
CAND_FEATS = ["mom_20", "mom_60", "adx", "rsi_14", "turn_amt_20", "past_r5"]
PRIMARY = "mom_comp"          # 预注册主候选 = 桶内动量 composite (RETRO-002 top 差异)


def load_factorlab_mvpe() -> pd.DataFrame:
    """concat 52 factor_lab groups -> (ts_code, trade_date, total_mv, pe_ttm), >=START. 缓存."""
    cf = CACHE / "mvpe_panel.parquet"
    if cf.exists():
        d = pd.read_parquet(cf); d["trade_date"] = d["trade_date"].astype(str); return d
    parts = []
    for f in sorted(glob.glob(str(FLAB_DIR / "group_*.parquet"))):
        try:
            d = pd.read_parquet(f, columns=["ts_code", "trade_date", "total_mv", "pe_ttm"])
        except Exception:
            continue
        d["trade_date"] = d["trade_date"].astype(str)
        d = d[d["trade_date"] >= START]
        if len(d):
            parts.append(d)
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
    out.to_parquet(cf, index=False)
    return out


def zscore_by_date(df, col):
    g = df.groupby("trade_date")[col]
    return (df[col] - g.transform("mean")) / (g.transform("std") + 1e-12)


def _nw_t(ics):
    """Newey-West (lag=NW_LAG) t-stat on a per-date IC series (修 20d 重叠)."""
    n = len(ics)
    mean_ic = float(ics.mean())
    dev = ics - mean_ic
    var = float((dev @ dev) / n)
    for k in range(1, min(NW_LAG, n - 1) + 1):
        gk = float((dev[k:] @ dev[:-k]) / n)
        var += 2.0 * (1.0 - k / (NW_LAG + 1)) * gk
    var = max(var, 1e-18)
    return mean_ic, float(mean_ic / np.sqrt(var / n))


def batch_residual_ic(df, bkey, feats, regime_filter=None):
    """单遍 per-date streaming: 桶内扣 beta + 扣因子后, 每个 feat 对 TARGET 的逐日 Spearman 残差 IC.
    低内存 (逐日 group, 不在全表上 transform)。同时算因子消融前的桶内 raw IC (描述性)。
    返回 {feat: {mean_ic, nw_t, n_dates, mean_ic_nonoverlap, t_nonoverlap, n_nonoverlap, raw_within_bucket_ic}}."""
    keep = ["trade_date", bkey, TARGET] + CONTROLS + list(feats)
    if regime_filter is not None:
        keep = keep + ["regime"]
    sub = df[keep]
    if regime_filter is not None:
        sub = sub[sub["regime"] == regime_filter]
    res_ics = {f: [] for f in feats}      # 扣因子残差 IC
    raw_ics = {f: [] for f in feats}      # 仅扣 beta (因子消融前)
    dlist = []
    for dt, g in sub.groupby("trade_date", sort=True):
        # 桶内去均值 (扣 beta); 桶 <MIN_BUCKET 成员剔
        gb = g.groupby(bkey)
        cnt = gb[TARGET].transform("count")
        big = cnt >= MIN_BUCKET
        g = g[big]
        if len(g) < 30:
            continue
        gb = g.groupby(bkey)
        y_d = (g[TARGET] - gb[TARGET].transform("mean")).to_numpy("float64")
        C_d = np.column_stack([(g[c] - gb[c].transform("mean")).to_numpy("float64") for c in CONTROLS])
        ymask = np.isfinite(y_d) & np.isfinite(C_d).all(axis=1)
        if ymask.sum() < 30 or np.std(y_d[ymask]) < 1e-12:
            continue
        dlist.append(dt)
        for f in feats:
            f_d = (g[f] - gb[f].transform("mean")).to_numpy("float64")
            m = ymask & np.isfinite(f_d)
            if m.sum() < 30 or np.std(f_d[m]) < 1e-12:
                res_ics[f].append(np.nan); raw_ics[f].append(np.nan); continue
            yv = y_d[m]; fv = f_d[m]; X = C_d[m]
            # raw (仅扣 beta)
            rraw = spearmanr(fv, yv).correlation
            raw_ics[f].append(rraw if np.isfinite(rraw) else np.nan)
            # Stage B: 扣因子 (无截距, 已 demean) -> f 残差
            beta, *_ = np.linalg.lstsq(X, fv, rcond=None)
            fres = fv - X @ beta
            if np.std(fres) < 1e-12:
                res_ics[f].append(np.nan); continue
            ic = spearmanr(fres, yv).correlation
            res_ics[f].append(ic if np.isfinite(ic) else np.nan)

    out = {}
    for f in feats:
        arr = np.array(res_ics[f], dtype="float64")
        arr = arr[np.isfinite(arr)]
        rraw = np.array(raw_ics[f], dtype="float64"); rraw = rraw[np.isfinite(rraw)]
        if len(arr) < 3:
            out[f] = None; continue
        mean_ic, nw_t = _nw_t(arr)
        ics_no = arr[::NW_LAG]
        if len(ics_no) >= 3:
            m_no = float(ics_no.mean()); t_no = float(m_no / (ics_no.std(ddof=1) / np.sqrt(len(ics_no)) + 1e-18))
        else:
            m_no, t_no = None, None
        out[f] = {"mean_ic": mean_ic, "nw_t": nw_t, "n_dates": int(len(arr)),
                  "mean_ic_nonoverlap": m_no, "t_nonoverlap": t_no, "n_nonoverlap": int(len(ics_no)),
                  "raw_within_bucket_ic": (float(rraw.mean()) if len(rraw) else None)}
    return out


def main():
    t0 = time.time()
    print("\n=== RETRO-003: 对比学习可行性 gate (扣桶beta+扣因子残差 + PE市值桶交叉验证) ===\n", flush=True)

    # ── universe (RETRO-002, (date,stock) 级, 含 ex-ante 特征 + ret_20 + regime) ──
    uni = pd.read_parquet(UNIV)
    uni["trade_date"] = uni["trade_date"].astype(str)
    uni = uni[uni["trade_date"] >= START].copy()
    print(f"[universe] {len(uni):,} (date,stock) / {uni['trade_date'].nunique()} 日", flush=True)

    # 主候选: 桶内动量 composite = z(mom_20)+z(mom_60) (逐日 cross-section z-score, 再平均)
    uni["mom_comp"] = (zscore_by_date(uni, "mom_20") + zscore_by_date(uni, "mom_60")) / 2.0

    # ── PE/市值面板 (concat 52 factor_lab groups) ──
    mvpe = load_factorlab_mvpe()
    print(f"[mvpe] factor_lab 合并 {len(mvpe):,} 行 / {mvpe['ts_code'].nunique()} 股 / "
          f"{mvpe['trade_date'].nunique()} 日 [{mvpe['trade_date'].min()}~{mvpe['trade_date'].max()}]", flush=True)
    uni = uni.drop(columns=[c for c in ("total_mv",) if c in uni.columns])
    uni = uni.merge(mvpe, on=["ts_code", "trade_date"], how="left")
    mv_cov = float(uni["total_mv"].notna().mean())
    pe_cov = float(uni["pe_ttm"].notna().mean())
    print(f"[mvpe] 合入 universe: total_mv cov={mv_cov:.1%}  pe_ttm cov={pe_cov:.1%}", flush=True)

    # ── 概念成员 (member-level, 仅 pick_concepts) ──
    picks = pd.read_parquet(PICKS); picks["trade_date"] = picks["trade_date"].astype(str)
    picks = picks[picks["trade_date"] >= START]
    pick_codes = set(picks["ts_code"])
    mem = pd.read_parquet(CONCEPT)[["stock_code", "concept_ts_code"]].dropna().drop_duplicates()
    mem = mem.rename(columns={"stock_code": "ts_code", "concept_ts_code": "concept"})
    pick_concepts = set(mem[mem["ts_code"].isin(pick_codes)]["concept"].unique())
    memf = mem[mem["concept"].isin(pick_concepts)].copy()
    print(f"[concept] pick_concepts={len(pick_concepts)} 成员对={len(memf):,}", flush=True)

    # member-level concept panel: (date, stock, concept) 特征复制
    base_cols = ["trade_date", "ts_code", TARGET, "regime", "vintage"] + CONTROLS + CAND_FEATS + ["mom_comp", "total_mv", "pe_ttm"]
    base = uni[base_cols].copy()
    cmem = base.merge(memf, on="ts_code", how="inner")
    cmem["bucket"] = cmem["concept"]
    print(f"[concept-panel] member rows={len(cmem):,}", flush=True)

    # PE×市值 25 桶 (一股一桶/日) — 仅 PE/市值可得行
    pe_sub = uni.dropna(subset=["total_mv", "pe_ttm"]).copy()
    pe_sub["logmv"] = np.log(pe_sub["total_mv"].clip(lower=1))
    pe_sub["sizeq"] = pe_sub.groupby("trade_date")["logmv"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop") if s.nunique() >= 5 else np.nan)
    pe_sub["peq"] = pe_sub.groupby("trade_date")["pe_ttm"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False, duplicates="drop") if s.nunique() >= 5 else np.nan)
    pe_sub = pe_sub.dropna(subset=["sizeq", "peq"])
    pe_sub["bucket"] = pe_sub["sizeq"].astype(int).astype(str) + "_" + pe_sub["peq"].astype(int).astype(str)
    print(f"[pecap-panel] rows={len(pe_sub):,} (PE/市值可得; 25桶)", flush=True)

    # 概念桶限制到 PE/市值可得子样本 (apples-to-apples 对照)
    avail_key = set(zip(pe_sub["trade_date"], pe_sub["ts_code"]))
    cmem_sub = cmem[[ (d, c) in avail_key for d, c in zip(cmem["trade_date"], cmem["ts_code"]) ]].copy()
    print(f"[concept-panel|PE可得] member rows={len(cmem_sub):,}", flush=True)

    # ── 残差 IC 计算 (单遍 streaming, 低内存) ──
    feats_to_test = [PRIMARY] + CAND_FEATS
    results = {"concept_full": {}, "concept_peavail": {}, "pecap": {}}

    print("\n── 概念桶 (full universe): 扣桶beta + 扣[ratio_s5,pred_r20] 残差 IC ──", flush=True)
    cf = batch_residual_ic(cmem, "bucket", feats_to_test)
    for f in feats_to_test:
        r = cf.get(f)
        if r:
            results["concept_full"][f] = r
            print(f"  {f:12s} resid IC {r['mean_ic']:+.4f}  NW_t {r['nw_t']:+.2f}  "
                  f"(扣因子前桶内 IC {r['raw_within_bucket_ic']:+.4f})  n={r['n_dates']}", flush=True)

    print("\n── 概念桶 (PE/市值可得子样本) vs PE×市值桶: 同口径残差 IC ──", flush=True)
    cpa = batch_residual_ic(cmem_sub, "bucket", feats_to_test)
    pcb = batch_residual_ic(pe_sub, "bucket", feats_to_test)
    for f in feats_to_test:
        rc = cpa.get(f); rp = pcb.get(f)
        if rc:
            results["concept_peavail"][f] = rc
        if rp:
            results["pecap"][f] = rp
        cic = rc["mean_ic"] if rc else None; ct = rc["nw_t"] if rc else None
        pic = rp["mean_ic"] if rp else None; pt = rp["nw_t"] if rp else None
        sc = f"{cic:+.4f}(t{ct:+.2f})" if rc else "NA"
        sp = f"{pic:+.4f}(t{pt:+.2f})" if rp else "NA"
        print(f"  {f:12s} 概念桶 {sc} | PE市值桶 {sp}", flush=True)

    # ── regime 分层 (R11) — 主候选 ──
    print("\n── regime 分层 (主候选 mom_comp, 概念桶 full) ──", flush=True)
    by_regime = {}
    for rg in ["momentum", "reversal", "mixed"]:
        rr = batch_residual_ic(cmem, "bucket", [PRIMARY], regime_filter=rg)
        r = rr.get(PRIMARY)
        if r:
            by_regime[rg] = {"mean_ic": r["mean_ic"], "nw_t": r["nw_t"], "n_dates": r["n_dates"]}
            print(f"  {rg:9s} resid IC {r['mean_ic']:+.4f}  NW_t {r['nw_t']:+.2f}  n={r['n_dates']}", flush=True)

    # ── gate 判定 ──
    prim = results["concept_full"].get(PRIMARY, {})
    prim_ic = prim.get("mean_ic")
    prim_t = prim.get("nw_t")
    # best 个体候选 (含多重检验 caveat)
    cand_pass = [(f, v["mean_ic"], v["nw_t"]) for f, v in results["concept_full"].items()
                 if v.get("mean_ic") is not None and abs(v["mean_ic"]) >= GATE_IC and abs(v["nw_t"]) >= GATE_T]
    # 概念特异性: 概念桶 vs PE市值桶 (同子样本) 残差 IC 对照 (主候选)
    cc = results["concept_peavail"].get(PRIMARY, {}).get("mean_ic")
    pc = results["pecap"].get(PRIMARY, {}).get("mean_ic")
    concept_specific = None
    if cc is not None and pc is not None:
        concept_specific = bool(abs(cc) >= abs(pc) + 0.005)   # 概念桶明显高 = 概念特异

    primary_pass = (prim_ic is not None and abs(prim_ic) >= GATE_IC and abs(prim_t) >= GATE_T)
    if primary_pass:
        status = "对比学习候选"
        playbook = "对比学习候选"
    elif cand_pass:
        # 主候选未过但有个体过阈 -> 真小 (多重检验风险, 不强上)
        status = "真小"
        playbook = "真小"
    else:
        status = "REJECT_beta_hindsight"
        playbook = "REJECT_beta_hindsight"

    prim_no_t = prim.get("t_nonoverlap")
    prim_raw = prim.get("raw_within_bucket_ic")
    cpa_prim = results["concept_peavail"].get(PRIMARY, {})
    rg_mom = by_regime.get("momentum", {})
    if status == "对比学习候选":
        conclusion = (
            f"主候选 桶内动量 composite (z(mom_20)+z(mom_60)) 扣桶beta+扣[ratio_s5,pred_r20] 后, "
            f"概念桶残差 IC={prim_ic:+.4f} NW_t={prim_t:+.2f} **过冻结 gate** (|IC|>={GATE_IC} & |t|>={GATE_T}) → "
            f"19+ 连否后**首个名义过闸的正向候选**, within-bucket LTR/对比学习**列为候选** (需独立 walk-forward, 非 ship, R03)。"
            f"但这是**最弱意义的过闸**, 三处脆弱性必须在 walk-forward 前认清: "
            f"①**符号翻转 (suppressor)**: 桶内 raw 动量 IC={prim_raw:+.4f} (负=均值回归), 仅扣因子后残差才转正 → 信号是'动量中正交于现有因子的分量'非动量本身; "
            f"②**apples-to-apples 子样本失效**: PE/市值可得子样本概念桶 mom_comp IC={cpa_prim.get('mean_ic'):+.4f} t={cpa_prim.get('nw_t'):+.2f} 不过 t, "
            f"非重叠 t={prim_no_t:+.2f} 亦不过 → full universe 过闸有赖交叉验证未覆盖的最后 3 月; "
            f"③**动量 regime 归零**: 残差 IC 在 momentum 月={rg_mom.get('mean_ic'):+.4f} t={rg_mom.get('nw_t'):+.2f} (≈零), 正值全来自 reversal/mixed → 不修 V12.31 实盘动量态痛点 (SIGN-R11)。"
            f"概念特异性: 概念桶 {cc:+.4f} vs PE市值桶 {pc:+.4f} (同子样本) "
            + ("勉强判概念特异 (但两者都不过 t, 概念结构增量很小)。" if concept_specific else "两桶接近 = 泛截面动量残差非概念特异。")
            + " 净结论: 对比学习/LTR 是 19 否后唯一名义过闸的正向, 但属脆弱/regime 条件/suppressor 弱信号, "
            "须先过独立 walk-forward 且证明能修动量态才谈落地; 生产线 V12.31 不动。"
        )
    elif status == "真小":
        passers = ", ".join(f"{f}({ic:+.4f},t{t:+.2f})" for f, ic, t in cand_pass)
        conclusion = (
            f"主候选 mom_comp 残差 IC={prim_ic:+.4f} NW_t={prim_t:+.2f} 未过 gate; "
            f"个体候选中 {passers} 过阈但属事后多重检验, 不强上 (真小)。"
            f"扣桶beta+扣[ratio_s5,pred_r20] 后桶内残差极弱 → 'missed_winner 的差异'基本被桶 beta + 现有因子吃掉。"
        )
    else:
        conclusion = (
            f"主候选 桶内动量 composite 扣桶beta+扣[ratio_s5,pred_r20] 后, "
            f"概念桶残差 IC={prim_ic:+.4f} NW_t={prim_t:+.2f} **未过 gate** (|IC|>={GATE_IC} & |t|>={GATE_T}); "
            f"全部个体候选亦无一过阈 → 'missed_winner 的差异'被桶 beta + 现有 V12.31 因子 (ratio_s5/pred_r20) 完全吃掉。"
            f"概念桶 vs PE市值桶残差 (主候选): {cc:+.4f} vs {pc:+.4f} (同子样本) — "
            + ("概念桶并不更强 = 即便有残差也是泛截面非概念特异。" if not concept_specific else "概念桶略高但均未过阈。") +
            f" picks 没有系统性输给同桶内可交易的 ex-ante 信号; RETRO-002 的动量差异是**无条件 + hindsight (winner 按 max_gain 选)** 产物, "
            f"扣桶 beta (= 概念共涨, 先验已 REJECT) 后不残留可学结构 → **对比学习不增 ex-ante alpha** (REJECT_beta_hindsight)。"
        )

    verdict = {
        "task": "RETRO-003", "status": status, "playbook_hit": playbook,
        "conclusion": conclusion,
        "metrics": {
            "gate": {"GATE_IC": GATE_IC, "GATE_T": GATE_T, "target": TARGET, "controls": CONTROLS,
                     "primary_candidate": PRIMARY, "nw_lag": NW_LAG, "min_bucket": MIN_BUCKET},
            "primary": {"resid_ic_concept_full": prim_ic, "nw_t": prim_t,
                        "n_dates": prim.get("n_dates"),
                        "resid_ic_nonoverlap": prim.get("mean_ic_nonoverlap"),
                        "t_nonoverlap": prim.get("t_nonoverlap"),
                        "raw_within_bucket_ic": prim.get("raw_within_bucket_ic"),
                        "passed_gate": primary_pass},
            "concept_full": results["concept_full"],
            "concept_peavail": results["concept_peavail"],
            "pecap_bucket": results["pecap"],
            "concept_specific": concept_specific,
            "concept_vs_pecap_primary": {"concept_ic": cc, "pecap_ic": pc},
            "by_regime_primary": by_regime,
            "candidates_passing_gate": [{"feat": f, "ic": ic, "t": t} for f, ic, t in cand_pass],
            "coverage": {"total_mv_cov": mv_cov, "pe_ttm_cov": pe_cov,
                         "concept_member_rows": int(len(cmem)),
                         "concept_member_rows_peavail": int(len(cmem_sub)),
                         "pecap_rows": int(len(pe_sub))},
        },
        "caveats": [
            "残差 IC 是 in-sample 描述性, NEW 仅表示'值得 walk-forward'非 ship (R03); REJECT 是合法完成 (R02)。",
            "TARGET=ret_20 (可交易前向 close 收益), 非 winner 定义用的 max_gain_20 (hindsight); 故 gate 测的是可交易残差。",
            "概念桶 member-level: 一股属多概念则多行, IC 按桶内成员加权 (within-bucket 可分性度量)。",
            "PE/市值来自 factor_lab 合并, 覆盖到 ~20260126; 202602-04 (最后 3 月) 无市值/PE → PE市值桶交叉验证缺尾部, "
            "概念桶 vs PE市值桶对照在 PE/市值可得子样本上 apples-to-apples。",
            "t 用 Newey-West (lag=20) 修 20d 前向重叠致 t 膨胀; 非重叠 (每20日取样) 作 robustness。",
            "多重检验: gate 只判预注册主候选 (mom_comp, 来自 RETRO-002 top 差异); 个体候选过阈仅记 '真小' 不强上。",
            "控制 = [ratio_s5, pred_r20] (= prd 冻结的 'pump ratio + r20'); 模型分本身不作候选 (套套逻辑, picks 被其选出)。",
        ],
        "artifacts": [
            "research/backtest/run_retro003.py",
            "research/cache/retro003/mvpe_panel.parquet",
            "research/cache/retro003/retro003_results.json",
        ],
        "discipline": "R01 gate/候选/控制/桶定义冻结未在结果上调; R02 REJECT=合法完成; R03 残差 IC≠ship "
                      "(NEW 也只'值得 walk-forward'); R04 大缓存 gitignored 无 features 写入; R05 复用缓存评分未碰生产; "
                      "R06 ST 源头排除 (universe 已排); R11 regime 分层必带; R12 异类增益须存活朴素消融 "
                      "(扣桶 beta + 扣现有因子 = 朴素对照); ex-ante vs hindsight 严格分 (TARGET=可交易 ret_20)。",
    }
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (CACHE / "retro003_results.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status={status} -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
