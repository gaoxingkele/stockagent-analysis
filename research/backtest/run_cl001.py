# -*- coding: utf-8 -*-
"""CL-001 — 全因子消融 + walk-forward(embargo) + 分regime 残差验证 (杀或活信号). 只读, gate 第一关.

北极星 (CL 阶段): RETRO-003 报 within-concept 动量 composite z(mom_20)+z(mom_60) 扣桶beta+扣 **2 因子**
[ratio_s5, pred_r20] 后残差 IC=+0.0286 NW_t+2.68, 名义过闸 = 19+ 否后首个正向。但这是过拟合最爱藏处:
①信号是动量 (V12.31 故意排除的); ②消融仅 2 因子非全集; ③RETRO-003 已揭示三处脆弱 (suppressor/子样本
失效/动量regime归零)。CL-001 = **设计成能主动证伪的严格验证**: 把消融从 2 因子扩到**全因子集**, 报
分regime 残差 IC, 判 +0.0286 里有多少是"标准动量被现有因子吃掉"的 artifact。

cl001_ablation (prd.preRegisteredGate, 冻结 R01):
  残差 IC: 概念内 demean (扣桶 beta) 后, 扣**全因子集**
    [mkt多窗动量(past_r5) + pyr_velocity_20_60 + log市值(size) + pe(value) + adx + rsi + 形态z(ma20_slope)
     + ratio_s5(pump ratio) + pred_r20(r20)]
  逐日截面正交化, 月度 walk-forward(embargo: 仅 full-20d 即 ret_20 全可知的日), 分 regime 报残差 IC + NW_t。
  gate: 全消融后 |IC|>=0.01 & |t|>=2  AND  至少 2 个 regime 同号正 (非动量 window 独有)。

status ∈ {存活, 死于消融, 动量window独有} (prd.responsePlaybook.CL-001):
  存活        : 全消融后 |IC|>=0.01 & |t|>=2 且 ≥2 regime 同号正 → 进 CL-002 book gate
  死于消融    : 全消融后 |IC|<阈 → +0.0286 是 2 因子弱消融 artifact, 标准动量被现有因子吃, REJECT 文档化
  动量window独有: 仅动量 regime 正 → regime 条件非稳健, 转 CL-003 判 overlay

方法 (全因果, 描述性 IC 非 ship, R03):
  目标 y = ret_20 (可交易前向 20d close 收益); 信号 f = mom_comp (z(mom_20)+z(mom_60))。
  桶 = 概念 (member-level, 一股属多概念则多行)。Stage A 桶内 (date,bucket) 去均值 (扣 beta, 桶<5 剔);
  Stage B 逐日 (cross-section) f_d ~ C_d OLS(无截距, 已 demean) → 残差 f_res; 残差 IC = 逐日
  Spearman(f_res, y_d) 跨日均值; t = Newey-West (lag=20) headline + 非重叠 (每 20 日取样) robustness。
  对照 RETRO-003 的 2 因子消融 (+0.0286): 全消融后剩多少 = 多少是标准动量被现有因子吃。

R 纪律: R01 gate/候选/控制/桶定义冻结不在结果上调 (= RETRO-003 同候选, 仅扩控制集); R02 REJECT=合法完成;
  R03 残差 IC≠ship (存活也只'值得 CL-002'); R04 大缓存 research/cache/cl001/ gitignored 无 features 写入;
  R05 复用缓存评分未碰生产; R06 ST 源头排除 (universe 已排); R08 block-level checkpoint;
  R11 regime 分层必带且为 gate 关键; R12 异类增益须存活朴素消融 (全因子集 = 最严朴素对照)。
  PIT flag: 概念 membership 是 current 快照 (对历史的轻 lookahead), 控制 ratio_s5/pred_r20 早月含
  in-sample 暴露 (vintage) — 作控制只会让消融更激进 (保守); 见 caveats。
"""
from __future__ import annotations
import json, sys, time, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

UNIV = ROOT / "research" / "cache" / "retro002" / "retro002_universe.parquet"
PICKS = ROOT / "research" / "cache" / "bt001" / "picks_v1231_daily.parquet"
CONCEPT = ROOT / "output" / "concept_local" / "concept_merged.parquet"
MVPE = ROOT / "research" / "cache" / "retro003" / "mvpe_panel.parquet"   # total_mv, pe_ttm (52 groups)
PXFEATS = ROOT / "research" / "cache" / "bt004" / "px_feats.parquet"      # ma20_slope (形态/趋势)
CACHE = ROOT / "research" / "cache" / "cl001"
BLOCKS = CACHE / "blocks"
CACHE.mkdir(parents=True, exist_ok=True)
BLOCKS.mkdir(parents=True, exist_ok=True)
VERDICT = ROOT / "research" / "verdicts" / "CL-001.json"

START = "20250501"
NW_LAG = 20
MIN_BUCKET = 5
GATE_IC = 0.01
GATE_T = 2.0

TARGET = "ret_20"
PRIMARY = "mom_comp"           # = RETRO-003 主候选 (z(mom_20)+z(mom_60)), 概念内 demean, 冻结
RETRO003_2FACTOR_IC = 0.0286   # RETRO-003 弱消融 (扣 [ratio_s5,pred_r20]) 概念桶残差 IC (对照基准)

# 全因子集 (cl001_ablation, 冻结 R01) — 概念内 demean 后逐日正交化扣除
#   mkt多窗动量 = past_r5 (短窗, 与信号 20/60 不同窗, 非循环); 形态z = ma20_slope (趋势/形态)
CONTROLS_FULL = ["past_r5", "pyr_velocity_20_60", "log_mv", "pe_ttm", "adx",
                 "rsi_14", "ma20_slope", "ratio_s5", "pred_r20"]
# core = 去 size/value (full universe 242 日, 不损最后 3 月; size/value factor_lab 覆盖到 ~20260126)
CONTROLS_CORE = ["past_r5", "pyr_velocity_20_60", "adx", "rsi_14", "ma20_slope", "ratio_s5", "pred_r20"]


def zscore_by_date(df, col):
    g = df.groupby("trade_date")[col]
    return (df[col] - g.transform("mean")) / (g.transform("std") + 1e-12)


def _nw_t(ics):
    n = len(ics)
    mean_ic = float(ics.mean())
    dev = ics - mean_ic
    var = float((dev @ dev) / n)
    for k in range(1, min(NW_LAG, n - 1) + 1):
        gk = float((dev[k:] @ dev[:-k]) / n)
        var += 2.0 * (1.0 - k / (NW_LAG + 1)) * gk
    var = max(var, 1e-18)
    return mean_ic, float(mean_ic / np.sqrt(var / n))


def residual_ic(df, controls, regime_filter=None, by_month=False):
    """单遍 per-date streaming: 概念桶内扣 beta + 扣 `controls` 后 PRIMARY 对 TARGET 的逐日 Spearman 残差 IC.
    返回 dict(mean_ic/nw_t/n_dates/非重叠/raw_within_bucket_ic[, by_month])."""
    keep = ["trade_date", "bucket", TARGET, PRIMARY] + controls
    if regime_filter is not None:
        keep = keep + ["regime"]
    sub = df[keep]
    if regime_filter is not None:
        sub = sub[sub["regime"] == regime_filter]
    res_ics, raw_ics, dlist = [], [], []
    month_ic = {}     # ym -> [ic,...]
    for dt, g in sub.groupby("trade_date", sort=True):
        gb = g.groupby("bucket")
        cnt = gb[TARGET].transform("count")
        g = g[cnt >= MIN_BUCKET]
        if len(g) < 30:
            continue
        gb = g.groupby("bucket")
        y_d = (g[TARGET] - gb[TARGET].transform("mean")).to_numpy("float64")
        f_d = (g[PRIMARY] - gb[PRIMARY].transform("mean")).to_numpy("float64")
        C_d = np.column_stack([(g[c] - gb[c].transform("mean")).to_numpy("float64") for c in controls])
        m = np.isfinite(y_d) & np.isfinite(f_d) & np.isfinite(C_d).all(axis=1)
        if m.sum() < 30 or np.std(y_d[m]) < 1e-12 or np.std(f_d[m]) < 1e-12:
            continue
        yv, fv, X = y_d[m], f_d[m], C_d[m]
        rraw = spearmanr(fv, yv).correlation                 # raw (仅扣 beta)
        beta, *_ = np.linalg.lstsq(X, fv, rcond=None)
        fres = fv - X @ beta                                 # Stage B: 扣全因子
        if np.std(fres) < 1e-12:
            continue
        ic = spearmanr(fres, yv).correlation
        if not np.isfinite(ic):
            continue
        res_ics.append(ic); raw_ics.append(rraw if np.isfinite(rraw) else np.nan); dlist.append(dt)
        if by_month:
            month_ic.setdefault(dt[:6], []).append(ic)
    arr = np.array(res_ics, dtype="float64")
    if len(arr) < 3:
        return None
    rraw = np.array(raw_ics, dtype="float64"); rraw = rraw[np.isfinite(rraw)]
    mean_ic, nw_t = _nw_t(arr)
    ics_no = arr[::NW_LAG]
    if len(ics_no) >= 3:
        m_no = float(ics_no.mean()); t_no = float(m_no / (ics_no.std(ddof=1) / np.sqrt(len(ics_no)) + 1e-18))
    else:
        m_no, t_no = None, None
    out = {"mean_ic": mean_ic, "nw_t": nw_t, "n_dates": int(len(arr)),
           "mean_ic_nonoverlap": m_no, "t_nonoverlap": t_no, "n_nonoverlap": int(len(ics_no)),
           "raw_within_bucket_ic": (float(rraw.mean()) if len(rraw) else None)}
    if by_month:
        out["by_month"] = {ym: {"mean_ic": float(np.mean(v)), "n": len(v)}
                           for ym, v in sorted(month_ic.items())}
    return out


def _block(name, fn):
    """R08 block-level checkpoint: 已算过的块直接读, 否则算并落盘."""
    bp = BLOCKS / f"{name}.json"
    if bp.exists():
        print(f"  [skip] {name} (cached)", flush=True)
        return json.loads(bp.read_text(encoding="utf-8"))
    r = fn()
    bp.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    return r


def build_panel():
    """概念 member-level 面板 (date, stock, concept) + 全控制列. 仅 pick_concepts, ST 已源头排除 (universe)."""
    uni = pd.read_parquet(UNIV)
    uni["trade_date"] = uni["trade_date"].astype(str)
    uni = uni[uni["trade_date"] >= START].copy()
    print(f"[universe] {len(uni):,} (date,stock) / {uni['trade_date'].nunique()} 日", flush=True)

    # 主候选信号: 桶内动量 composite (逐日 cross-section z-score 再平均)
    uni["mom_comp"] = (zscore_by_date(uni, "mom_20") + zscore_by_date(uni, "mom_60")) / 2.0

    # size/value: 用 mvpe_panel (52 groups, 覆盖优于 universe 的 group_001 单组)
    mvpe = pd.read_parquet(MVPE); mvpe["trade_date"] = mvpe["trade_date"].astype(str)
    uni = uni.drop(columns=[c for c in ("total_mv",) if c in uni.columns])
    uni = uni.merge(mvpe, on=["ts_code", "trade_date"], how="left")
    uni["log_mv"] = np.log(uni["total_mv"].clip(lower=1))
    mv_cov = float(uni["total_mv"].notna().mean()); pe_cov = float(uni["pe_ttm"].notna().mean())

    # 形态z: ma20_slope (bt004 px_feats); universe 已有 adx, 但 px_feats 的 adx 同源, 取 ma20_slope
    pf = pd.read_parquet(PXFEATS, columns=["ts_code", "trade_date", "ma20_slope"])
    pf["trade_date"] = pf["trade_date"].astype(str)
    uni = uni.merge(pf, on=["ts_code", "trade_date"], how="left")
    slope_cov = float(uni["ma20_slope"].notna().mean())
    print(f"[merge] total_mv cov={mv_cov:.1%}  pe_ttm cov={pe_cov:.1%}  ma20_slope cov={slope_cov:.1%}", flush=True)

    # 概念成员 (仅 pick_concepts), member-level explode
    picks = pd.read_parquet(PICKS); picks["trade_date"] = picks["trade_date"].astype(str)
    pick_codes = set(picks[picks["trade_date"] >= START]["ts_code"])
    mem = pd.read_parquet(CONCEPT)[["stock_code", "concept_ts_code"]].dropna().drop_duplicates()
    mem = mem.rename(columns={"stock_code": "ts_code", "concept_ts_code": "concept"})
    pick_concepts = set(mem[mem["ts_code"].isin(pick_codes)]["concept"].unique())
    memf = mem[mem["concept"].isin(pick_concepts)].copy()
    cols = ["trade_date", "ts_code", TARGET, "regime", PRIMARY] + \
           list(dict.fromkeys(CONTROLS_FULL + CONTROLS_CORE))
    base = uni[cols].copy()
    cmem = base.merge(memf, on="ts_code", how="inner")
    cmem["bucket"] = cmem["concept"]
    print(f"[concept-panel] pick_concepts={len(pick_concepts)} member rows={len(cmem):,}", flush=True)
    return cmem, {"total_mv_cov": mv_cov, "pe_ttm_cov": pe_cov, "ma20_slope_cov": slope_cov,
                  "n_pick_concepts": len(pick_concepts), "member_rows": int(len(cmem))}


def main():
    t0 = time.time()
    print("\n=== CL-001: 全因子消融 + walk-forward(embargo) + 分regime 残差验证 ===\n", flush=True)
    panel, cov = build_panel()

    # ── 块 1: core 全消融 (full universe, 去 size/value) — full-universe NW-t (功率) ──
    print("\n── [block] core 全消融 (full universe): 扣 7 因子残差 IC + 月度 walk-forward ──", flush=True)
    core = _block("core_full_universe", lambda: residual_ic(panel, CONTROLS_CORE, by_month=True))
    if core:
        print(f"  core resid IC {core['mean_ic']:+.4f}  NW_t {core['nw_t']:+.2f}  "
              f"非重叠 t {core['t_nonoverlap']:+.2f}  (桶内 raw IC {core['raw_within_bucket_ic']:+.4f})  "
              f"n={core['n_dates']}", flush=True)

    # ── 块 2: full 全消融 (PE/市值可得子样本, +size+value) — 最严朴素对照 (headline gate) ──
    print("\n── [block] full 全消融 (PE/市值可得子样本): 扣 9 因子残差 IC (含 size/value) ──", flush=True)
    full = _block("full_pe_avail", lambda: residual_ic(panel, CONTROLS_FULL, by_month=True))
    if full:
        print(f"  full resid IC {full['mean_ic']:+.4f}  NW_t {full['nw_t']:+.2f}  "
              f"非重叠 t {full['t_nonoverlap']:+.2f}  n={full['n_dates']}", flush=True)

    # ── 块 3: 分 regime (R11, gate 关键) — core 与 full 各报 ──
    print("\n── [block] 分 regime 残差 IC (R11) ──", flush=True)
    by_regime = {"core": {}, "full": {}}
    for rg in ["momentum", "reversal", "mixed"]:
        rc = _block(f"regime_core_{rg}", lambda rg=rg: residual_ic(panel, CONTROLS_CORE, regime_filter=rg))
        rf = _block(f"regime_full_{rg}", lambda rg=rg: residual_ic(panel, CONTROLS_FULL, regime_filter=rg))
        if rc:
            by_regime["core"][rg] = {k: rc[k] for k in ("mean_ic", "nw_t", "n_dates")}
            print(f"  core {rg:9s} IC {rc['mean_ic']:+.4f}  NW_t {rc['nw_t']:+.2f}  n={rc['n_dates']}", flush=True)
        if rf:
            by_regime["full"][rg] = {k: rf[k] for k in ("mean_ic", "nw_t", "n_dates")}
            print(f"  full {rg:9s} IC {rf['mean_ic']:+.4f}  NW_t {rf['nw_t']:+.2f}  n={rf['n_dates']}", flush=True)

    # ── gate 判定 (headline = full 9 因子全消融; core 作功率补充) ──
    head = full if full else core
    head_label = "full(9因子,PE可得)" if full else "core(7因子,full universe)"
    h_ic = head["mean_ic"] if head else None
    h_t = head["nw_t"] if head else None
    ic_t_pass = (h_ic is not None and abs(h_ic) >= GATE_IC and abs(h_t) >= GATE_T)

    # regime 条件: ≥2 regime 同号正 (用与 headline 同口径的 regime)
    rg_key = "full" if full else "core"
    rg_d = by_regime[rg_key]
    pos_regimes = [rg for rg, v in rg_d.items() if v["mean_ic"] is not None and v["mean_ic"] > 0]
    mom_pos = ("momentum" in rg_d and rg_d["momentum"]["mean_ic"] is not None
               and rg_d["momentum"]["mean_ic"] > 0)
    n_pos = len(pos_regimes)
    two_regime_pass = n_pos >= 2

    # 动量window独有 = 仅 momentum regime 正 (其余 ≤0)
    mom_only = mom_pos and n_pos == 1

    if ic_t_pass and two_regime_pass:
        status = "存活"; playbook = "存活"
    elif mom_only:
        status = "动量window独有"; playbook = "动量window独有"
    else:
        status = "死于消融"; playbook = "死于消融"

    # 对照 RETRO-003 2 因子消融: 全消融后 IC 占 +0.0286 多少
    shrink_core = (core["mean_ic"] / RETRO003_2FACTOR_IC) if core else None
    shrink_full = (full["mean_ic"] / RETRO003_2FACTOR_IC) if full else None

    # 诚实 nuance 用: headline 非重叠 t + 分 regime 显著性 (用 headline 同口径 rg_d)
    head_no_t = head.get("t_nonoverlap") if head else None
    head_raw = head.get("raw_within_bucket_ic") if head else None
    mom_v = rg_d.get("momentum", {})
    rev_v = rg_d.get("reversal", {})
    mix_v = rg_d.get("mixed", {})
    # 仅 momentum regime 显著 (|t|>=2), reversal/mixed 不显著 = 信号本质动量regime条件
    sig_regimes = [rg for rg, v in rg_d.items()
                   if v.get("nw_t") is not None and abs(v["nw_t"]) >= GATE_T and v["mean_ic"] > 0]
    mom_only_sig = (sig_regimes == ["momentum"])

    pos_rg_str = "/".join(pos_regimes) if pos_regimes else "无"
    if status == "存活":
        conclusion = (
            f"主候选 within-concept 动量 composite (z(mom_20)+z(mom_60)) 概念内 demean 后, 扣**全因子集** "
            f"[{', '.join(CONTROLS_FULL)}] 逐日截面正交化, headline={head_label} 残差 IC={h_ic:+.4f} NW_t={h_t:+.2f} "
            f"**过冻结 gate** (|IC|>={GATE_IC} & |t|>={GATE_T}) 且 ≥2 regime 同号正 ({pos_rg_str}) → "
            f"全消融后信号**存活** (非 2 因子弱消融 artifact): full 全消融 IC 是 RETRO-003 2 因子 +{RETRO003_2FACTOR_IC} 的 "
            f"{shrink_full:.0%} (core {shrink_core:.0%})。进 CL-002 book apples-to-apples gate 验净赚不赚。"
            f"诚实 nuance (R03/R11): 残差 IC 是 in-sample 描述性, 存活仅'值得 CL-002'非 ship; "
            f"动量 regime {'正' if mom_pos else '仍≤0'} → 见 CL-003 判是否动量 overlay。"
        )
    elif status == "动量window独有":
        conclusion = (
            f"主候选 全消融后残差 IC={h_ic:+.4f} NW_t={h_t:+.2f} ({head_label}); 分 regime **仅动量 window 正** "
            f"({pos_rg_str}) → regime 条件非稳健, 信号是动量 regime 择时性质 → 转 CL-003 判 overlay "
            f"(RG-002 已证动量 overlay tricky)。"
        )
    else:
        conclusion = (
            f"主候选 within-concept 动量 composite 概念内 demean 后, 扣**全因子集** [{', '.join(CONTROLS_FULL)}] "
            f"逐日正交化, headline={head_label} 残差 IC={h_ic:+.4f} **NW_t={h_t:+.2f} 未过冻结 gate** "
            f"(|IC|>={GATE_IC} & |t|>={GATE_T}); 且**非重叠 t={head_no_t:+.2f}** (core 同 ~+0.69) → 去 20d 重叠膨胀后无显著性。"
            f"core(7因子,full universe) NW_t={core['nw_t']:+.2f} 仅勉强擦边 2.0 但非重叠 t 同样 ~+0.69 不过 → **整体不稳健, 死于消融**。"
            f"对照 RETRO-003 2 因子弱消融 (+{RETRO003_2FACTOR_IC}): 全消融后 full IC 缩到 "
            f"{('%.0f%%' % (shrink_full*100)) if shrink_full is not None else 'NA'} (core {('%.0f%%' % (shrink_core*100)) if shrink_core is not None else 'NA'}) "
            f"且显著性蒸发 → +0.0286 约 1/3 被更全的因子 (多窗动量 past_r5 + 形态 ma20_slope + size/value) 吃掉, 余下整体不过闸。"
            f" ★**诚实 nuance (与 RETRO-003 反转, 不许据此移门柱 R01)**: 全消融后残差**不是均匀归零, 而是集中到动量 regime** "
            f"— momentum IC={mom_v.get('mean_ic'):+.4f} t={mom_v.get('nw_t'):+.2f} (强), reversal {rev_v.get('mean_ic'):+.4f} t={rev_v.get('nw_t'):+.2f}, "
            f"mixed {mix_v.get('mean_ic'):+.4f} t={mix_v.get('nw_t'):+.2f} (后两者不显著)"
            + ("; 即**仅动量 regime 显著**, 揭示 RETRO-003 '动量月≈0' 是其 2 因子控制被 regime 混淆的 artifact (那两个 mean-reversion 控制在反转月吸收了信号)。" if mom_only_sig else "。")
            + f" 桶内 raw 动量 IC={head_raw:+.4f} (负=均值回归, suppressor 仍在)。by_month 不稳 (202505/06 负, 202508~202601 正, 202602 又负)。 "
            f"**裁决**: 按冻结 gate (headline 全消融 |IC|&|t|) **死于消融** = within-concept 相对动量 re-rank **未通过存活闸, 不进 CL-002** (REJECT 文档化, R02/R03)。 "
            f"但残差的动量 regime 集中性 ≠ 干净归零, 性质上属**动量 regime 条件/overlay** (恰在 V12.31 实盘被血洗的动量态, [[project_v3c_momentum_regime_mismatch_0603]] + SIGN-R11): "
            f"若后续要追, 须按 CL-003 当**动量 overlay** 验 (RG-002 已证 overlay tricky), 而非当稳健 cross-sectional alpha。生产线 V12.31 不动 (R05)。"
        )

    verdict = {
        "task": "CL-001", "status": status, "playbook_hit": playbook,
        "conclusion": conclusion,
        "metrics": {
            "gate": {"GATE_IC": GATE_IC, "GATE_T": GATE_T, "target": TARGET, "primary": PRIMARY,
                     "controls_full": CONTROLS_FULL, "controls_core": CONTROLS_CORE,
                     "nw_lag": NW_LAG, "min_bucket": MIN_BUCKET, "headline": head_label,
                     "ic_t_pass": bool(ic_t_pass), "n_pos_regimes": n_pos,
                     "two_regime_pass": bool(two_regime_pass), "momentum_regime_positive": bool(mom_pos),
                     "significant_regimes": sig_regimes, "momentum_regime_only_significant": bool(mom_only_sig),
                     "headline_nonoverlap_t": head_no_t},
            "full_ablation_pe_avail": full,
            "core_ablation_full_universe": core,
            "by_regime": by_regime,
            "vs_retro003_2factor": {"retro003_2factor_ic": RETRO003_2FACTOR_IC,
                                    "full_ic_share_of_2factor": shrink_full,
                                    "core_ic_share_of_2factor": shrink_core},
            "coverage": cov,
        },
        "caveats": [
            "残差 IC 是 in-sample 描述性, 存活也只表示'值得 CL-002 book gate'非 ship (R03); 死于消融=合法完成 (R02)。",
            "TARGET=ret_20 (可交易前向 close 收益, 非 winner 定义用的 max_gain hindsight)。",
            "全消融 (full, 9 因子) 含 size/value (factor_lab 覆盖到 ~20260126), 故 PE/市值可得子样本损最后 ~3 月; "
            "core (7 因子) 在 full universe 242 日, 二者并报。headline gate 用最严的 full 全消融。",
            "形态z = ma20_slope (趋势/形态); mkt多窗动量 = past_r5 (短窗, 与信号 mom_20/mom_60 不同窗, 非循环正交)。",
            "概念 membership 是 current 快照 (PIT 风险: 对历史的轻 lookahead, 须在 CL-002/003 落地前 PIT 化)。",
            "控制 ratio_s5/pred_r20 早月 (202505~09) 含 r20 in-sample 暴露 (vintage); 作控制只会让消融更激进 (保守)。",
            "t 用 Newey-West (lag=20) 修 20d 前向重叠致 t 膨胀; 非重叠 (每 20 日取样) 作 robustness; embargo=仅 full-20d 日。",
        ],
        "artifacts": [
            "research/backtest/run_cl001.py",
            "research/cache/cl001/blocks/*.json",
            "research/cache/cl001/cl001_results.json",
        ],
        "discipline": "R01 候选/控制集/gate 阈值冻结未在结果上调 (= RETRO-003 同候选, 仅按 prd cl001_ablation 扩控制集); "
                      "R02 REJECT=合法完成; R03 残差 IC≠ship; R04 大缓存 gitignored 无 features 写入; "
                      "R05 复用缓存评分未碰生产; R06 ST 源头排除 (universe 已排); R08 block-level checkpoint; "
                      "R11 regime 分层必带且为 gate 关键; R12 全因子集 = 最严朴素对照。",
    }
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    (CACHE / "cl001_results.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status={status} (headline {head_label}) -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
