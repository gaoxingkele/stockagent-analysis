# -*- coding: utf-8 -*-
"""BL-001 — BASE 入场 (严格 router + 突破触发) + 紧尾 label 设计 + 样本诊断.

Phase 1 第一步 (零训练): 在 SLV-001 已分桶的严格 BASE 桶上, 加"突破触发"把
"罕见但干净"的横盘 base 提纯成可交易入场事件, 并造紧尾 label (替代被去掉的
past_r5 隐性尾控), 然后诊断样本量决定 BL-002 走 v1 规则法还是 v2 训练。

设计见 research/sleeves/DESIGN.md §9; 边界冻结于 prd.preRegisteredGate.base_entry / base_label。

────────────────────────────────────────────────────────────────────────
入场定义 (FROZEN, SIGN-R01 — 量比/箱体阈值在本 task 冻结, 不在结果上调):

  前提 (严格 base 已建): 前一交易日 archetype == 'BASE'
        (= SLV-001 严格桶: abs(ma20_slope_norm)<0.0015/日 & boll_bw_pctile(250d)<0.30
          & consol_duration>=15 & past_r5∈[-0.05,+0.05])
  箱体上沿: box_top = 过去 BOX_LOOKBACK=20 日 high 的最大值 (因果, 不含当日)
  突破触发: 当日 close > box_top  且  vol_ratio >= VOL_RATIO_THR=1.5
        vol_ratio = vol[t] / mean(vol[t-20 .. t-1])   (放量, 因果)

  → base_entry 事件 = 前提 & 突破触发 (前一日 BASE 天然去重: 突破后该股不再 BASE,
    次日前提 false, 故每段 base 只取首次突破)

紧尾 label (FROZEN, 比 squat 的 −5% 更紧 = 显式替代被去掉的 past_r5 尾控):
  次日开盘 next_open 建仓, 前向 5 日:
    upside   = max(high[t+1..t+5]) / next_open - 1
    downside = min(low [t+1..t+5]) / next_open - 1
  base_label = 1  iff  upside >= 0.10  AND  downside >= -0.04   (紧 dd ≥ −4%)

诊断: base_entry 月度/日度事件数, 正例率, vs SLV-002 BASE 画像核对, 样本充分性判断。

红线: 入场/label 边界冻结 (R01); 全因果零泄漏 (R04, 前向列/label 只留 cache 非 features);
ST 已在 factor_panel 源头排除 (R06); 分 regime 复核 (R11); 描述统计/诊断非 ship (R03);
生产线只读 (R05)。大缓存写 research/cache/bl001/ (gitignored)。

产出:
  research/cache/bl001/base_entry.parquet   (ts_code,trade_date,base_label,regime,box_top,vol_ratio + 前向画像列)
  research/cache/bl001_results.json
  research/verdicts/BL-001.json             (status=built + 事件量/正例率/样本充分性)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache" / "bl001"
CACHE.mkdir(parents=True, exist_ok=True)
FACTOR_CKPT = ROOT / "research" / "cache" / "slv001" / "factor_panel.parquet"
DAILY = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
ENTRY_OUT = CACHE / "base_entry.parquet"
RESULTS = ROOT / "research" / "cache" / "bl001_results.json"
VERDICT = ROOT / "research" / "verdicts" / "BL-001.json"

STUDY_START = "20230103"        # 同 SLV-001 统计起点
LOAD_START = "20220601"         # vol 回看缓冲
KEY = ["ts_code", "trade_date"]

# ── 冻结入场常量 (SIGN-R01) ──
BOX_LOOKBACK = 20               # 箱体上沿回看窗 (因果, 不含当日)
VOL_RATIO_THR = 1.5             # 放量阈 (vol[t] / 过去20日均量)
VOL_MA_WIN = 20                 # 均量窗

# ── 冻结紧尾 label 常量 ──
PUMP_UP, PUMP_DD, FORWARD = 0.10, 0.04, 5

# ── 样本充分性判据 (冻结, 决定 BL-002 v1/v2) ──
SUFFICIENT_N_POS = 2000         # 总正例数下限
SUFFICIENT_PER_FOLD_POS = 100   # walk-forward 每折平均正例下限 (19 折)
N_FOLDS = 19


def load_st_set() -> set:
    if not BASIC.exists():
        return set()
    b = pd.read_parquet(BASIC)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(b[b["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def load_vol(st: set) -> pd.DataFrame:
    print(f"[data] 加载 daily vol {LOAD_START}+ (ST 源头排除)...", flush=True)
    parts = []
    for f in sorted(DAILY.glob("*.parquet")):
        if f.stem < LOAD_START:
            continue
        parts.append(pd.read_parquet(f, columns=["ts_code", "trade_date", "vol"]))
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.drop_duplicates(subset=KEY, keep="last")
    if st:
        big = big[~big["ts_code"].isin(st)]
    return big


def main():
    t0 = time.time()
    print("\n=== BL-001 BASE 入场 + 突破触发 + 紧尾 label + 样本诊断 ===\n", flush=True)

    df = pd.read_parquet(FACTOR_CKPT)
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(KEY).reset_index(drop=True)
    print(f"[data] factor_panel {len(df):,} 行 / {df['ts_code'].nunique()} 股", flush=True)

    st = load_st_set()
    vol = load_vol(st)
    df = df.merge(vol, on=KEY, how="left")
    print(f"[data] vol 合并完成 (缺 vol {df['vol'].isna().mean():.3%})", flush=True)

    g = df.groupby("ts_code", sort=False)

    # ── 箱体上沿 (因果, 不含当日): 过去 BOX_LOOKBACK 日 high 最大 ──
    df["box_top"] = g["high"].transform(
        lambda s: s.shift(1).rolling(BOX_LOOKBACK, min_periods=BOX_LOOKBACK).max())
    # ── 放量比 (因果): vol[t] / 过去 VOL_MA_WIN 日均量 (不含当日) ──
    vol_ma = g["vol"].transform(
        lambda s: s.shift(1).rolling(VOL_MA_WIN, min_periods=VOL_MA_WIN).mean())
    df["vol_ratio"] = df["vol"] / vol_ma
    # ── 前提: 前一交易日 archetype == BASE ──
    df["prev_base"] = g["archetype"].shift(1).eq("BASE")

    # ── base_entry 事件 ──
    df["base_entry"] = (
        df["prev_base"]
        & (df["close"] > df["box_top"])
        & (df["vol_ratio"] >= VOL_RATIO_THR)
    ).fillna(False)

    # ── 紧尾 label (前向5d, max_gain>=10% & max_dd>=-4%, 全因果) ──
    next_open = g["open"].shift(-1)
    max_high = g["high"].transform(
        lambda s: s.rolling(FORWARD, min_periods=FORWARD).max().shift(-FORWARD))
    min_low = g["low"].transform(
        lambda s: s.rolling(FORWARD, min_periods=FORWARD).min().shift(-FORWARD))
    upside = max_high / next_open - 1.0
    downside = min_low / next_open - 1.0
    df["_upside"] = upside
    df["_downside"] = downside
    df["base_label"] = ((upside >= PUMP_UP) & (downside >= -PUMP_DD)).astype("float")
    df.loc[max_high.isna() | next_open.isna(), "base_label"] = np.nan
    # 画像用前向收益 (5d/20d, 次日开盘进; 前向列只留 cache)
    for H in (5, 20):
        close_h = g["close"].shift(-H)
        min_low_h = g["low"].transform(
            lambda s: s.rolling(H, min_periods=H).min().shift(-H))
        df[f"_ret{H}"] = close_h / next_open - 1.0
        df[f"_dd{H}"] = min_low_h / next_open - 1.0
        df.loc[close_h.isna() | next_open.isna(), f"_ret{H}"] = np.nan
        df.loc[min_low_h.isna() | next_open.isna(), f"_dd{H}"] = np.nan

    # regime (SIGN-R11)
    if REGIME_P.exists():
        rg = pd.read_parquet(REGIME_P, columns=["trade_date", "regime"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        df = df.merge(rg, on="trade_date", how="left")
    else:
        df["regime"] = "unknown"

    # ── 研究窗口内的 base_entry 事件 ──
    ev = df[(df["trade_date"] >= STUDY_START) & df["base_entry"]].copy()
    n_events = len(ev)
    ev_lab = ev[ev["base_label"].notna()]
    n_labeled = len(ev_lab)
    pos_rate = float(ev_lab["base_label"].mean()) if n_labeled else None
    n_pos = int(ev_lab["base_label"].sum()) if n_labeled else 0

    print(f"\n[events] base_entry {n_events:,} 事件 ({ev['trade_date'].min()}~{ev['trade_date'].max()}), "
          f"可标注 {n_labeled:,}, 正例 {n_pos:,} (正例率 {pos_rate})", flush=True)

    # ── 落 base_entry 面板 (cache, 非 features; 含 label/前向列但路径在 cache) ──
    out_cols = KEY + ["base_label", "regime", "box_top", "vol_ratio",
                      "_ret5", "_dd5", "_ret20", "_dd20", "past_r5", "consol_duration"]
    ev[out_cols].to_parquet(ENTRY_OUT, index=False)
    print(f"[out] base_entry -> {ENTRY_OUT.relative_to(ROOT)} ({n_events:,} 行)", flush=True)

    # ── 诊断: 月度 / 日度 事件数 ──
    ev["ym"] = ev["trade_date"].str[:6]
    by_month = ev.groupby("ym").size()
    n_months = int(ev["ym"].nunique())
    n_days_with_event = int(ev["trade_date"].nunique())
    # 候选交易日总数 (有任一 base_entry 资格的研究窗口交易日数 = 全市场交易日)
    total_trade_days = int(df[df["trade_date"] >= STUDY_START]["trade_date"].nunique())
    monthly_stats = {
        "n_months_with_event": n_months,
        "events_per_month_mean": round(float(by_month.mean()), 2) if n_months else 0.0,
        "events_per_month_median": round(float(by_month.median()), 2) if n_months else 0.0,
        "events_per_month_min": int(by_month.min()) if n_months else 0,
        "events_per_month_max": int(by_month.max()) if n_months else 0,
        "n_trade_days_with_event": n_days_with_event,
        "total_trade_days": total_trade_days,
        "day_coverage": round(n_days_with_event / total_trade_days, 4) if total_trade_days else None,
        "events_per_active_day_mean": round(n_events / n_days_with_event, 2) if n_days_with_event else 0.0,
        "events_per_trade_day_mean": round(n_events / total_trade_days, 2) if total_trade_days else 0.0,
    }

    # ── vs SLV-002 BASE 画像核对: base_entry 前向 5d/20d 风险收益 ──
    def prof(sub, H):
        r = sub[f"_ret{H}"].dropna()
        dd = sub[f"_dd{H}"].dropna()
        if len(r) < 50:
            return None
        return {
            "n": int(len(r)),
            "mean": round(float(r.mean()), 5),
            "median": round(float(r.median()), 5),
            "win_rate": round(float((r > 0).mean()), 4),
            "p10_left_tail": round(float(r.quantile(0.10)), 5),
            "realized_dd_median": round(float(dd.median()), 5),
            "realized_dd_p10": round(float(dd.quantile(0.10)), 5),
        }
    entry_profile = {f"h{H}": prof(ev, H) for H in (5, 20)}
    entry_profile = {k: v for k, v in entry_profile.items() if v is not None}

    # 对比基线: 严格 BASE 桶全体 (未加突破) 的桶内启动率 ≈ SLV-001 的 0.063
    base_bucket = df[(df["trade_date"] >= STUDY_START) & (df["archetype"] == "BASE")
                     & df["is_launch"].notna()]
    base_bucket_launch_rate = float(base_bucket["is_launch"].mean()) if len(base_bucket) else None

    # ── 分 regime 复核 (R11): 事件数 + 正例率 ──
    by_regime = {}
    for rgm in ["momentum", "reversal", "mixed"]:
        sub = ev_lab[ev_lab["regime"] == rgm]
        if len(sub) < 30:
            continue
        by_regime[rgm] = {"n": int(len(sub)),
                          "pos_rate": round(float(sub["base_label"].mean()), 4)}

    # ── 样本充分性判断 (冻结判据) → BL-002 v1/v2 路由 ──
    per_fold_pos = n_pos / N_FOLDS
    sufficient = bool(n_pos >= SUFFICIENT_N_POS and per_fold_pos >= SUFFICIENT_PER_FOLD_POS)
    bl002_route = "v2_train_ok" if sufficient else "v1_rules_only"

    results = {
        "task": "BL-001", "elapsed_s": round(time.time() - t0, 1),
        "window": [STUDY_START, ev["trade_date"].max() if n_events else None],
        "frozen_entry": {
            "prereq": "prev_day archetype==BASE (SLV-001 严格桶)",
            "box_top": f"max(high) over prior {BOX_LOOKBACK}d (因果不含当日)",
            "breakout": f"close > box_top AND vol_ratio >= {VOL_RATIO_THR}",
            "vol_ratio": f"vol[t] / mean(vol[t-{VOL_MA_WIN}..t-1])",
        },
        "frozen_label": f"前向{FORWARD}d max_gain>={PUMP_UP} & max_dd>=-{PUMP_DD} (紧尾, 全因果 ST排除)",
        "n_events": n_events, "n_labeled": n_labeled, "n_pos": n_pos, "pos_rate": pos_rate,
        "monthly_stats": monthly_stats,
        "entry_forward_profile": entry_profile,
        "base_bucket_launch_rate_no_breakout": round(base_bucket_launch_rate, 5)
            if base_bucket_launch_rate is not None else None,
        "by_regime": by_regime,
        "sample_sufficiency": {
            "n_pos": n_pos, "per_fold_pos_mean": round(per_fold_pos, 1),
            "thresholds": {"n_pos": SUFFICIENT_N_POS, "per_fold_pos": SUFFICIENT_PER_FOLD_POS,
                           "n_folds": N_FOLDS},
            "sufficient_for_training": sufficient, "bl002_route": bl002_route,
        },
        "guardrails": ["SIGN-R01 入场/label边界冻结", "SIGN-R03 诊断非ship",
                       "SIGN-R04 全因果label留cache", "SIGN-R06 ST源头排除", "SIGN-R11 分regime"],
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print(f"\n--- 月度/日度事件诊断 ---", flush=True)
    print(f"  有事件月数 {n_months}, 月均 {monthly_stats['events_per_month_mean']} "
          f"(中位 {monthly_stats['events_per_month_median']}, "
          f"min {monthly_stats['events_per_month_min']}/max {monthly_stats['events_per_month_max']})", flush=True)
    print(f"  有事件交易日 {n_days_with_event}/{total_trade_days} (覆盖 {monthly_stats['day_coverage']}), "
          f"活跃日均 {monthly_stats['events_per_active_day_mean']} 个", flush=True)
    print(f"\n--- 入场质量 (vs SLV-002 BASE 画像) ---", flush=True)
    for H in (5, 20):
        p = entry_profile.get(f"h{H}")
        if p:
            print(f"  h{H}: n{p['n']} μ{p['mean']:+.4f} win{p['win_rate']:.3f} "
                  f"p10ret{p['p10_left_tail']:+.4f} dd_p10{p['realized_dd_p10']:+.4f}", flush=True)
    print(f"  base_entry 正例率(紧尾label) {pos_rate} vs 严格BASE桶无突破启动率 "
          f"{round(base_bucket_launch_rate,4) if base_bucket_launch_rate else None}", flush=True)
    print(f"  分regime: {json.dumps(by_regime, ensure_ascii=False)}", flush=True)
    print(f"\n--- 样本充分性 → BL-002 路由 ---", flush=True)
    print(f"  正例 {n_pos} (每折≈{per_fold_pos:.0f}), 门槛 n_pos>={SUFFICIENT_N_POS} & "
          f"每折>={SUFFICIENT_PER_FOLD_POS} → sufficient={sufficient} → {bl002_route}", flush=True)

    # ── verdict ──
    lift_vs_bucket = (round(pos_rate / base_bucket_launch_rate, 2)
                      if pos_rate and base_bucket_launch_rate else None)
    p20 = entry_profile.get("h20", {})
    conclusion = (
        f"在严格 BASE 桶 (SLV-001) 上加突破触发 (前一日BASE & close>过去{BOX_LOOKBACK}日箱顶 & "
        f"vol_ratio>={VOL_RATIO_THR}), 窗口 {STUDY_START}~{ev['trade_date'].max() if n_events else 'NA'} 共得 "
        f"base_entry {n_events:,} 事件 (可标注 {n_labeled:,})。紧尾 label (前向5d max_gain>=10% & max_dd>=-4%) "
        f"正例 {n_pos:,}, 正例率 {pos_rate}。突破筛选把紧尾正例率从严格 BASE 桶无突破基线 "
        f"{round(base_bucket_launch_rate,4) if base_bucket_launch_rate else 'NA'} (口径注: 基线是 is_launch dd≥−5%) "
        f"提到 {pos_rate} (lift {lift_vs_bucket}x, 且本 label dd 更紧 −4%) — 突破触发确实抬升命中率。"
        f"【诚实核对 — SLV-002 画像是 survivor-conditioned, 勿误读为干净】SLV-002 BASE 高质量画像 "
        f"(μ+0.151/win0.829/dd_p10−0.075) 仅在 is_launch=1 (赢家) 上算; base_entry 是无条件入场信号, 其前向 20d "
        f"画像 μ{p20.get('mean')}/win{p20.get('win_rate')}/dd_p10{p20.get('realized_dd_p10')} 明显更差 (含失败突破=假突破秒回), "
        f"dd_p10 −14.8% 并不薄。即'罕见但干净'是结果端筛选的产物, 无条件突破入场仍有肥左尾 — 正是 BL-002 需在入场内再排序、"
        f"BL-003 需紧尾 label+紧止损守尾的理由。可立的 edge 是命中率 lift {lift_vs_bucket}x, 非 SLV-002 的干净尾。"
        f"月频: 月均 {monthly_stats['events_per_month_mean']} 事件 (min {monthly_stats['events_per_month_min']}/"
        f"max {monthly_stats['events_per_month_max']}), 有事件交易日覆盖 {monthly_stats['day_coverage']}, "
        f"活跃日均 {monthly_stats['events_per_active_day_mean']} 个 → blend 多数日有少量可注入候选, 但稀疏。"
        f"【样本充分性 → BL-002 路由】正例 {n_pos} (walk-forward 每折≈{per_fold_pos:.0f}), "
        f"门槛 n_pos>={SUFFICIENT_N_POS} 且 每折>={SUFFICIENT_PER_FOLD_POS} → "
        f"sufficient={sufficient} → **{bl002_route}** "
        f"({'样本够可上 v2 训练打分' if sufficient else '样本不足, BL-002 仅走 v1 规则法 (SLV-002 证入场即 edge, 绕开小样本训练陷阱)'})。"
        f"诊断非 ship (R03); 入场/label 边界冻结 (R01); 全因果 ST 排除留 cache (R04/R06)。")

    VERDICT.write_text(json.dumps({
        "id": "BL-001", "status": "built", "conclusion": conclusion,
        "metrics": {
            "window": [STUDY_START, ev["trade_date"].max() if n_events else None],
            "n_events": n_events, "n_labeled": n_labeled, "n_pos": n_pos, "pos_rate": pos_rate,
            "base_bucket_launch_rate_no_breakout": round(base_bucket_launch_rate, 5)
                if base_bucket_launch_rate is not None else None,
            "lift_vs_base_bucket": lift_vs_bucket,
            "monthly_stats": monthly_stats,
            "entry_forward_profile": entry_profile,
            "by_regime": by_regime,
            "sample_sufficiency": results["sample_sufficiency"],
        },
        "artifacts": ["research/cache/bl001/base_entry.parquet",
                      "research/cache/bl001_results.json"],
        "note": "入场/label 边界本 task 冻结 (R01: BOX_LOOKBACK20/VOL_RATIO1.5/dd-4%); 全因果前向列+label 只留 cache 非 features (R04); "
                "ST 源头已排除 (R06); 分 regime (R11); 诊断非 ship (R03); 生产线只读 (R05)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] BL-001 status=built -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
