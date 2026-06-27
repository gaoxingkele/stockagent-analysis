# -*- coding: utf-8 -*-
"""DK-002 — 底部确认翻多窗口定义 + 前向画像 (大涨/横盘/失败).

承 DK-001 (duokongK 状态机面板 research/cache/dk001/duokongk_panel.parquet)。
按 NOTES.md §2 预注册窗口定义标 dk_window, 算窗口后 20/40d 前向收益分布 + 三态占比
(大涨≥+15% / 横盘|ret|<5% / 失败<−5%) + realized max_dd, 分 regime; 事件量 + 月度频次;
与用户"窗口含一个或多个启动子"核对 (窗口内 is_launch 命中率)。

窗口定义 (FROZEN, NOTES.md §2):
  - pivot_low: dk_dir=-1 段内, low[t] = min(low[t-5..t+5]) (中心 11 根, 右 5 根确认 → 标注合法但实时迟到)。
  - 空翻多窗口: pivot_low 之后 dk_dir 由 -1 翻 +1 且持续 ≥3 根 +1 (flip 当根 + 后 2 根均 +1)。
  - 窗口区域 dk_window = [pivot_low_idx .. flip+3]; 取该 -1 段内"段底"qualifying pivot (最低 low)。
  - 前向锚点 = flip+3 (窗口末根, 最早可实时决策点), 从 close[flip+3] 量 20/40d 前向收益。

红线纪律:
  - 窗口定义冻结不在结果上调 (SIGN-R01); 前向/label 列只留 cache 不写 features (SIGN-R04);
    ST 源头排除复用 SLV-001 宇宙 (SIGN-R06); 分 regime 评估 (SIGN-R11);
    描述统计非 ship (SIGN-R03); 生产线只读 (SIGN-R05); checkpoint (SIGN-R08)。

产出:
  research/cache/dk002/dk_windows.parquet  (ts_code, anchor_date, pivot_date, flip_date,
                                            seg_len, has_launch, fwd_r20, fwd_r40, maxdd40, regime)
  research/cache/dk002_results.json
  research/verdicts/DK-002.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache" / "dk002"
CACHE.mkdir(parents=True, exist_ok=True)
DK_PANEL = ROOT / "research" / "cache" / "dk001" / "duokongk_panel.parquet"
SRC_PANEL = ROOT / "research" / "cache" / "slv001" / "factor_panel.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
WIN_OUT = CACHE / "dk_windows.parquet"
RESULTS = ROOT / "research" / "cache" / "dk002_results.json"
VERDICT = ROOT / "research" / "verdicts" / "DK-002.json"

STUDY_START = "20230103"
KEY = ["ts_code", "trade_date"]
PIVOT_HALF = 5          # 中心 11 根 (左5右5) 局部低点确认
MIN_PERSIST = 3         # 空翻多持续 ≥3 根 +1
WIN_END_OFFSET = 3      # 窗口末根 = flip + 3
H20, H40 = 20, 40       # 前向 horizon


def build_windows(df: pd.DataFrame) -> pd.DataFrame:
    """逐股扫描, 标空翻多底部确认窗口, 算前向画像。df 已按 (ts_code, trade_date) 升序。"""
    out = []
    for code, sub in df.groupby("ts_code", sort=False):
        n = len(sub)
        if n < PIVOT_HALF * 2 + MIN_PERSIST + WIN_END_OFFSET + 1:
            continue
        low = sub["low"].to_numpy()
        close = sub["close"].to_numpy()
        dkd = sub["dk_dir"].to_numpy()
        launch = sub["is_launch"].to_numpy()
        dates = sub["trade_date"].to_numpy()
        # 中心 11 根局部低点 (右 5 根确认) + 在 -1 段
        # 滚动中心 min: pivot 候选 = low[t] == min(low[t-5..t+5]) 且 dk_dir[t]==-1
        is_pivot = np.zeros(n, dtype=bool)
        for t in range(PIVOT_HALF, n - PIVOT_HALF):
            if dkd[t] == -1 and low[t] == low[t - PIVOT_HALF:t + PIVOT_HALF + 1].min():
                is_pivot[t] = True

        # 找 -1 → +1 翻转点
        for f in range(1, n):
            if not (dkd[f] == 1 and dkd[f - 1] == -1):
                continue
            # 持续 ≥3 根 +1
            if f + (MIN_PERSIST - 1) >= n:
                continue
            if not np.all(dkd[f:f + MIN_PERSIST] == 1):
                continue
            # 回溯本 -1 段起点 (dk_dir==-1 连续段, 终于 f-1)
            seg_start = f - 1
            while seg_start > 0 and dkd[seg_start - 1] == -1:
                seg_start -= 1
            # 段内 qualifying pivot, 取段底 (最低 low)
            piv_idx = [p for p in range(seg_start, f) if is_pivot[p]]
            if not piv_idx:
                continue  # 无底部确认 → 不符窗口定义, skip
            p = min(piv_idx, key=lambda i: low[i])
            anchor = f + WIN_END_OFFSET   # 窗口末根 = 前向锚点
            if anchor >= n:
                continue
            # 前向收益 (锚点 close 起)
            c0 = close[anchor]
            r20 = close[anchor + H20] / c0 - 1.0 if anchor + H20 < n else np.nan
            r40 = close[anchor + H40] / c0 - 1.0 if anchor + H40 < n else np.nan
            # realized max_dd over 40d 前向路径 (锚点起)
            end40 = min(anchor + H40, n - 1)
            path = close[anchor:end40 + 1]
            if len(path) > 1:
                run_max = np.maximum.accumulate(path)
                maxdd = float((path / run_max - 1.0).min())
            else:
                maxdd = np.nan
            # 窗口内是否含启动子 (is_launch==1, 区间 [p..anchor])
            has_launch = int(np.nanmax(launch[p:anchor + 1]) == 1) if anchor + 1 > p else 0
            out.append({
                "ts_code": code,
                "anchor_date": dates[anchor],
                "pivot_date": dates[p],
                "flip_date": dates[f],
                "seg_len": int(f - seg_start),
                "win_len": int(anchor - p + 1),
                "has_launch": has_launch,
                "fwd_r20": r20,
                "fwd_r40": r40,
                "maxdd40": maxdd,
            })
    return pd.DataFrame(out)


def classify(ret: pd.Series) -> dict:
    """三态 + mild 占比 (NOTES: 大涨≥+15% / 横盘|ret|<5% / 失败<−5%)。"""
    r = ret.dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}
    surge = float((r >= 0.15).mean())
    fail = float((r <= -0.05).mean())
    flat = float((r.abs() < 0.05).mean())
    mild = float(((r >= 0.05) & (r < 0.15)).mean())
    return {
        "n": int(n),
        "mean": round(float(r.mean()), 4),
        "median": round(float(r.median()), 4),
        "surge_ge15pct": round(surge, 4),
        "mild_5to15pct": round(mild, 4),
        "flat_lt5pct": round(flat, 4),
        "fail_le_neg5pct": round(fail, 4),
        "win_rate_pos": round(float((r > 0).mean()), 4),
    }


def main():
    t0 = time.time()
    print("\n=== DK-002 底部确认翻多窗口 + 前向画像 ===\n", flush=True)

    if WIN_OUT.exists():
        print(f"[ckpt] 命中 {WIN_OUT.name}, 跳过窗口重算", flush=True)
        win = pd.read_parquet(WIN_OUT)
    else:
        print("[data] 加载 dk001 面板 + SLV-001 OHLC/is_launch", flush=True)
        dk = pd.read_parquet(DK_PANEL, columns=KEY + ["dk_dir"])
        dk["trade_date"] = dk["trade_date"].astype(str)
        px = pd.read_parquet(SRC_PANEL, columns=KEY + ["low", "close", "is_launch"])
        px["trade_date"] = px["trade_date"].astype(str)
        df = px.merge(dk, on=KEY, how="inner").sort_values(KEY).reset_index(drop=True)
        print(f"[data] merged {len(df):,} 行 / {df['ts_code'].nunique()} 股", flush=True)
        print("[calc] 扫描底部确认翻多窗口...", flush=True)
        win = build_windows(df)
        # regime 标注 (按 anchor_date)
        rg = pd.read_parquet(REGIME, columns=["trade_date", "regime"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        win = win.merge(rg.rename(columns={"trade_date": "anchor_date"}),
                        on="anchor_date", how="left")
        win.to_parquet(WIN_OUT, index=False)
        print(f"[ckpt] dk_windows 落盘 {WIN_OUT.name} ({len(win):,} 窗口)", flush=True)

    # ── 研究窗口过滤 (anchor >= STUDY_START) ──
    win = win[win["anchor_date"] >= STUDY_START].copy()
    n_win = len(win)
    n_stk = win["ts_code"].nunique()
    print(f"\n--- dk_window 事件 (anchor>={STUDY_START}) ---", flush=True)
    print(f"  窗口数 {n_win:,} / 覆盖 {n_stk} 股", flush=True)

    # 月度频次
    win["month"] = win["anchor_date"].str[:6]
    monthly = win.groupby("month").size()
    monthly_stats = {
        "n_months": int(monthly.shape[0]),
        "mean_per_month": round(float(monthly.mean()), 1),
        "median_per_month": float(monthly.median()),
        "min": int(monthly.min()), "max": int(monthly.max()),
    }

    # 启动子命中率
    launch_hit = round(float(win["has_launch"].mean()), 4)

    # 前向画像 (全期 + 分 regime), 20d / 40d
    prof_all_20 = classify(win["fwd_r20"])
    prof_all_40 = classify(win["fwd_r40"])
    maxdd_med = round(float(win["maxdd40"].median()), 4)
    maxdd_mean = round(float(win["maxdd40"].mean()), 4)

    prof_by_regime_40 = {}
    prof_by_regime_20 = {}
    for reg, g in win.groupby("regime"):
        prof_by_regime_40[str(reg)] = classify(g["fwd_r40"])
        prof_by_regime_20[str(reg)] = classify(g["fwd_r20"])

    # 段长/窗口长
    seg_stats = {
        "seg_len_median": float(win["seg_len"].median()),
        "seg_len_mean": round(float(win["seg_len"].mean()), 1),
        "win_len_median": float(win["win_len"].median()),
        "win_len_mean": round(float(win["win_len"].mean()), 1),
    }

    print(f"  启动子命中率 (窗口内 is_launch): {launch_hit:.1%}", flush=True)
    print(f"  前向 40d: {prof_all_40}", flush=True)
    print(f"  前向 20d: {prof_all_20}", flush=True)
    print(f"  max_dd40: median {maxdd_med:.1%} / mean {maxdd_mean:.1%}", flush=True)
    for reg in prof_by_regime_40:
        print(f"  [{reg}] 40d: {prof_by_regime_40[reg]}", flush=True)

    results = {
        "task": "DK-002", "elapsed_s": round(time.time() - t0, 1),
        "window_def": {"pivot_half": PIVOT_HALF, "min_persist": MIN_PERSIST,
                       "win_end_offset": WIN_END_OFFSET, "anchor": "flip+3 (窗口末根)"},
        "n_windows": n_win, "n_stocks": n_stk,
        "monthly": monthly_stats,
        "launch_hit_rate": launch_hit,
        "seg_stats": seg_stats,
        "maxdd40_median": maxdd_med, "maxdd40_mean": maxdd_mean,
        "profile_all_20d": prof_all_20,
        "profile_all_40d": prof_all_40,
        "profile_by_regime_20d": prof_by_regime_20,
        "profile_by_regime_40d": prof_by_regime_40,
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── verdict ──
    s40 = prof_all_40
    surge40 = s40.get("surge_ge15pct", 0)
    flat40 = s40.get("flat_lt5pct", 0)
    fail40 = s40.get("fail_le_neg5pct", 0)
    # base rate 参考: 该宇宙任意点 40d 大涨基率 (粗算 vs 窗口) — 用窗口 mean 与三态描述定性
    corresponds_surge = surge40 >= 0.30 and surge40 > fail40
    conclusion = (
        f"按 NOTES.md §2 预注册定义 (pivot_low 右5根确认 + 空翻多持续≥{MIN_PERSIST}根, "
        f"段底qualifying pivot, 锚点=flip+{WIN_END_OFFSET}) 全量标出 {n_win:,} 个底部确认翻多窗口 "
        f"(覆盖 {n_stk} 股, {monthly_stats['n_months']} 月, 月均 {monthly_stats['mean_per_month']} 个)。"
        f"窗口内启动子(is_launch)命中率 {launch_hit:.1%}, 印证'窗口含一个或多个启动子'但非满命中。"
        f"前向 40d 三态: 大涨(≥+15%) {surge40:.1%} / 横盘(|ret|<5%) {flat40:.1%} / 失败(<−5%) {fail40:.1%} "
        f"(均值 {s40.get('mean')}, 中位 {s40.get('median')}, 正收益率 {s40.get('win_rate_pos')}); "
        f"前向 20d 大涨 {prof_all_20.get('surge_ge15pct'):.1%} / 横盘 {prof_all_20.get('flat_lt5pct'):.1%} / "
        f"失败 {prof_all_20.get('fail_le_neg5pct'):.1%}; realized max_dd40 中位 {maxdd_med:.1%}。"
        f"分 regime (40d 大涨占比): " +
        " / ".join(f"{r} {prof_by_regime_40[r].get('surge_ge15pct',0):.1%}"
                   for r in sorted(prof_by_regime_40)) + "。"
        f"{'窗口大体对应后续大涨倾向' if corresponds_surge else '窗口大涨占比不高, 大量横盘/失败 — 反转确认≠预测大涨'}。"
        f"描述统计非 ship (SIGN-R03); 下一步 DK-003 测与 SQUAT 重叠率 + 残差增量消融 (换皮判定 gate)。")

    VERDICT.write_text(json.dumps({
        "id": "DK-002", "status": "built", "conclusion": conclusion,
        "metrics": {
            "n_windows": n_win, "n_stocks": n_stk,
            "launch_hit_rate": launch_hit,
            "monthly": monthly_stats, "seg_stats": seg_stats,
            "maxdd40_median": maxdd_med, "maxdd40_mean": maxdd_mean,
            "profile_all_20d": prof_all_20, "profile_all_40d": prof_all_40,
            "profile_by_regime_40d": prof_by_regime_40,
            "corresponds_to_surge": bool(corresponds_surge),
        },
        "artifacts": ["research/cache/dk002/dk_windows.parquet",
                      "research/cache/dk002_results.json"],
        "note": "窗口定义冻结不在结果上调 (R01); 前向/label 列只留 cache 非 features (R04); "
                "ST 源头排除复用 SLV-001 宇宙 (R06); 分 regime (R11); checkpoint (R08); "
                "描述统计非 ship (R03); 生产线只读 (R05)。锚点=flip+3 (窗口末根, 最早可实时决策点); "
                "pivot 右5根确认=标注合法但实时迟到~5+3根 (NOTES §2 已明示)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] DK-002 status=built -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
