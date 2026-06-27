# -*- coding: utf-8 -*-
"""DK-001 — duokongK 多空K线状态机全因果移植 + 全量面板序列生成.

起因 (2026-06-13): 用户提供 TradingView Pine `多空K线` 脚本, 提议把 duokongK 状态机
当独立标注维度。本 task 是 faithful port: 把状态机逐字移植 (klen=30, 单根内 a/b/c 顺序,
hhk/llk 用 [1] 全因果), 在全量 OHLC 面板 (复用 SLV-001 同宇宙, ST 已源头排除) 逐股跑出
dk_dir/dk_flip/bars_since_flip/dist_to_slLow/slHigh 序列, 落 cache 供 DK-002/003 用。

状态机 (FROZEN, NOTES.md §1 逐字, 全因果):
  hhk = highest(high,30)[1]  # 前30根(不含当根)最高
  llk = lowest(low,30)[1]    # 前30根(不含当根)最低
  单根内按顺序:
    (a) 首次突破:  close>hhk 且 dk_dir!=+1 → dk_dir=+1; slLow=low;  slHigh=0
                   close<llk 且 dk_dir!=-1 → dk_dir=-1; slHigh=high; slLow=0
    (b) 失败翻转 (用上一根遗留的 slLow/slHigh, 先判后更新):
                   dk_dir==+1 且 slLow>0 且 close<slLow  → dk_dir=-1; slHigh=high; slLow=0
                   dk_dir==-1 且 slHigh>0 且 close>slHigh → dk_dir=+1; slLow=low;  slHigh=0
    (c) 移动止损 (跟极值):
                   slHigh>0 且 high>slHigh → slHigh=high
                   slLow>0  且 low<slLow   → slLow=low

红线纪律:
  - 状态机冻结 (SIGN-R01, 移植规范不在结果上调); ST 源头排除 (SIGN-R06, 复用 SLV-001 宇宙);
    全因果零泄漏 (SIGN-R04, hhk/llk 用 [1], 状态只依赖历史; 输出列名不撞 forward 黑名单)。
  - 大缓存写 research/cache/dk001/ (gitignored), 不写 research/features/。
  - 长任务 checkpoint (SIGN-R08, parquet 落盘命中即跳过)。
  - 描述统计非 ship (SIGN-R03); 生产线只读 (SIGN-R05)。

产出:
  research/cache/dk001/duokongk_panel.parquet  (ts_code,trade_date,dk_dir,dk_flip,
                                                bars_since_flip,sl_low,sl_high,
                                                dist_to_sllow,dist_to_slhigh)
  research/cache/dk001_results.json
  research/verdicts/DK-001.json                (status=built + 覆盖/事件/段长)
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache" / "dk001"
CACHE.mkdir(parents=True, exist_ok=True)
SRC_PANEL = ROOT / "research" / "cache" / "slv001" / "factor_panel.parquet"  # 同宇宙, ST已排除
PANEL_OUT = CACHE / "duokongk_panel.parquet"
RESULTS = ROOT / "research" / "cache" / "dk001_results.json"
VERDICT = ROOT / "research" / "verdicts" / "DK-001.json"

STUDY_START = "20230103"   # 与 SLV-001 一致 (前面是 hhk/llk 30 根 warmup 缓冲)
KLEN = 30
KEY = ["ts_code", "trade_date"]


def run_state_machine(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      hhk: np.ndarray, llk: np.ndarray):
    """单股全因果扫描, 返回 (dk_dir, sl_low, sl_high)。NOTES.md §1 逐字顺序 a/b/c。

    hhk/llk 已是 highest(high,30)[1] / lowest(low,30)[1] (含当根之前30根, NaN=未足窗)。
    """
    n = len(close)
    dk_dir = np.zeros(n, dtype=np.int8)
    sl_low = np.zeros(n, dtype=np.float64)
    sl_high = np.zeros(n, dtype=np.float64)
    d = 0          # 持久 dk_dir
    sH = 0.0       # slHigh (空腿移动止损上沿)
    sL = 0.0       # slLow  (多腿移动止损下沿)
    for t in range(n):
        c = close[t]; h = high[t]; lo = low[t]
        up = hhk[t]; dn = llk[t]
        # (a) 首次突破平台 (hhk/llk NaN 时比较为 False, 自然跳过 warmup)
        if up == up and c > up and d != 1:
            d = 1; sL = lo; sH = 0.0
        if dn == dn and c < dn and d != -1:
            d = -1; sH = h; sL = 0.0
        # (b) 失败翻转 (用上一根遗留的 slLow/slHigh, 先判后更新)
        if d == 1 and sL > 0 and c < sL:
            d = -1; sH = h; sL = 0.0
        if d == -1 and sH > 0 and c > sH:
            d = 1; sL = lo; sH = 0.0
        # (c) 移动结构止损 (跟极值)
        if sH > 0 and h > sH:
            sH = h
        if sL > 0 and lo < sL:
            sL = lo
        dk_dir[t] = d
        sl_low[t] = sL
        sl_high[t] = sH
    return dk_dir, sl_low, sl_high


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """逐股算 hhk/llk (因果) + 跑状态机 + 派生 dk_flip/bars_since_flip/dist。"""
    g = df.groupby("ts_code", sort=False)
    # hhk = highest(high,30)[1]; llk = lowest(low,30)[1]  (因果: shift(1) 排除当根)
    df["hhk"] = g["high"].transform(
        lambda s: s.rolling(KLEN, min_periods=KLEN).max().shift(1))
    df["llk"] = g["low"].transform(
        lambda s: s.rolling(KLEN, min_periods=KLEN).min().shift(1))

    dk_dir = np.zeros(len(df), dtype=np.int8)
    sl_low = np.zeros(len(df), dtype=np.float64)
    sl_high = np.zeros(len(df), dtype=np.float64)
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    hhk = df["hhk"].to_numpy(); llk = df["llk"].to_numpy()

    nstk = 0
    for _, idx in g.indices.items():
        idx = np.sort(idx)  # groupby.indices 已按出现序, df 已排序故升序
        d, sL, sH = run_state_machine(hi[idx], lo[idx], cl[idx], hhk[idx], llk[idx])
        dk_dir[idx] = d
        sl_low[idx] = sL
        sl_high[idx] = sH
        nstk += 1
        if nstk % 1000 == 0:
            print(f"  ...状态机 {nstk} 股", flush=True)

    df["dk_dir"] = dk_dir
    df["sl_low"] = sl_low
    df["sl_high"] = sl_high
    # dk_flip = 方向相对上一根变化 (首根不算翻转)
    prev_dir = g["close"].cumcount()  # 占位, 实际用 shift
    df["dk_flip"] = (df["dk_dir"] != df.groupby("ts_code", sort=False)["dk_dir"].shift(1)) & \
                    df.groupby("ts_code", sort=False)["dk_dir"].shift(1).notna()
    df["dk_flip"] = df["dk_flip"].astype("int8")
    # bars_since_flip: 自上次 flip 起的连续根数 (含当根=0 在 flip 当根)
    flip = df["dk_flip"].to_numpy()
    grp_id = df.groupby("ts_code", sort=False).ngroup().to_numpy()
    bars = np.zeros(len(df), dtype=np.int32)
    cnt = 0; last_grp = -1
    for i in range(len(df)):
        if grp_id[i] != last_grp:
            cnt = 0; last_grp = grp_id[i]
        elif flip[i] == 1:
            cnt = 0
        else:
            cnt += 1
        bars[i] = cnt
    df["bars_since_flip"] = bars
    # 因果距离 (用当前 close 与当前状态止损; slX=0 表示该方向无止损 → NaN)
    sl_low_arr = df["sl_low"].to_numpy()
    sl_high_arr = df["sl_high"].to_numpy()
    df["dist_to_sllow"] = np.where(sl_low_arr > 0,
                                   cl / np.where(sl_low_arr > 0, sl_low_arr, 1.0) - 1.0, np.nan)
    df["dist_to_slhigh"] = np.where(sl_high_arr > 0,
                                    cl / np.where(sl_high_arr > 0, sl_high_arr, 1.0) - 1.0, np.nan)
    return df


def causal_self_check(df: pd.DataFrame, sample_codes: list[str]) -> list[dict]:
    """抽样核对: 每个 flip 当根必满足 a/b 触发条件之一; hhk/llk 不含当根 (无未来)。"""
    checks = []
    for code in sample_codes:
        sub = df[df["ts_code"] == code].sort_values("trade_date").reset_index(drop=True)
        if len(sub) < KLEN + 10:
            continue
        flips = sub[sub["dk_flip"] == 1]
        ok_trigger = 0; bad_trigger = 0
        for _, row in flips.iterrows():
            c, up, dn = row["close"], row["hhk"], row["llk"]
            i = row.name
            prev = sub.iloc[i - 1]
            d_new = row["dk_dir"]
            # 翻多: close>hhk(首破) 或 close>上一根slHigh(失败翻转); 翻空对称
            trig = False
            if d_new == 1:
                trig = (pd.notna(up) and c > up) or (prev["sl_high"] > 0 and c > prev["sl_high"])
            elif d_new == -1:
                trig = (pd.notna(dn) and c < dn) or (prev["sl_low"] > 0 and c < prev["sl_low"])
            ok_trigger += int(trig); bad_trigger += int(not trig)
        # 因果性: hhk[t] 应 == max(high[t-30..t-1]) (不含当根)
        leak = 0
        for i in range(KLEN, min(len(sub), KLEN + 50)):
            ref = sub["high"].iloc[i - KLEN:i].max()
            if pd.notna(sub["hhk"].iloc[i]) and abs(sub["hhk"].iloc[i] - ref) > 1e-9:
                leak += 1
        checks.append({"ts_code": code, "n_flips": int(len(flips)),
                       "trigger_ok": int(ok_trigger), "trigger_bad": int(bad_trigger),
                       "hhk_leak_count": int(leak)})
    return checks


def main():
    t0 = time.time()
    print("\n=== DK-001 duokongK 状态机全因果移植 ===\n", flush=True)

    if PANEL_OUT.exists():
        print(f"[ckpt] 命中 {PANEL_OUT.name}, 跳过状态机重算", flush=True)
        df = pd.read_parquet(PANEL_OUT)
    else:
        print(f"[data] 加载 SLV-001 同宇宙 OHLC (ST 已源头排除) {SRC_PANEL.relative_to(ROOT)}", flush=True)
        df = pd.read_parquet(SRC_PANEL, columns=KEY + ["open", "high", "low", "close"])
        df["trade_date"] = df["trade_date"].astype(str)
        df = df.sort_values(KEY).reset_index(drop=True)
        print(f"[data] {len(df):,} 行 / {df['ts_code'].nunique()} 股 / "
              f"{df['trade_date'].nunique()} 日", flush=True)
        print("[calc] 算 hhk/llk + 跑全因果状态机 (逐股)...", flush=True)
        df = build_panel(df)
        out_cols = KEY + ["dk_dir", "dk_flip", "bars_since_flip",
                          "sl_low", "sl_high", "dist_to_sllow", "dist_to_slhigh"]
        df[out_cols].to_parquet(PANEL_OUT, index=False)
        print(f"[ckpt] duokongk_panel 落盘 {PANEL_OUT.name} ({len(df):,} 行)", flush=True)
        df = df[out_cols]

    # ── 泄漏自查: 输出列名不撞 forward 黑名单 ──
    fwd_exact = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                {f"dd{n}" for n in (5, 10, 20, 30, 40)} | {"max_gain", "fwd_ret", "label", "target"}
    bad = [c for c in df.columns if c.lower() in fwd_exact
           or c.lower().startswith(("fwd_", "future_", "next_")) or c.lower().endswith("_forward")]
    assert not bad, f"duokongk_panel 列名撞 forward 黑名单: {bad}"

    # ── 因果自检 (抽样核对 flip 触发逻辑 + hhk 无未来) ──
    # 重建带 hhk/llk/sl 的子集供自检 (仅抽样股, 省内存)
    full = pd.read_parquet(SRC_PANEL, columns=KEY + ["high", "low", "close"])
    full["trade_date"] = full["trade_date"].astype(str)
    full = full.sort_values(KEY).reset_index(drop=True)
    g = full.groupby("ts_code", sort=False)
    full["hhk"] = g["high"].transform(lambda s: s.rolling(KLEN, min_periods=KLEN).max().shift(1))
    full["llk"] = g["low"].transform(lambda s: s.rolling(KLEN, min_periods=KLEN).min().shift(1))
    chk_df = full.merge(df[KEY + ["dk_dir", "dk_flip", "sl_low", "sl_high"]], on=KEY, how="left")
    codes = sorted(df["ts_code"].unique())
    sample = [codes[i] for i in (0, len(codes) // 4, len(codes) // 2,
                                 3 * len(codes) // 4, len(codes) - 1)]
    self_check = causal_self_check(chk_df, sample)
    total_bad = sum(c["trigger_bad"] for c in self_check)
    total_leak = sum(c["hhk_leak_count"] for c in self_check)
    print(f"[check] 抽样 {len(self_check)} 股: trigger_bad={total_bad}, hhk_leak={total_leak}", flush=True)

    # ── 研究窗口统计 ──
    panel = df[df["trade_date"] >= STUDY_START].copy()
    n_rows = len(panel)
    n_stk = panel["ts_code"].nunique()
    n_days = panel["trade_date"].nunique()

    # 方向分布
    dir_share = panel["dk_dir"].value_counts(normalize=True).round(4).to_dict()
    dir_share = {str(int(k)): float(v) for k, v in dir_share.items()}

    # 翻多/翻空事件频次 (flip 且 dk_dir 切到 +1 / -1)
    flips = panel[panel["dk_flip"] == 1]
    flip_to_long = int((flips["dk_dir"] == 1).sum())
    flip_to_short = int((flips["dk_dir"] == -1).sum())
    n_flip = int(len(flips))

    # 段长分布: 每股连续同向段长度 (基于全窗 panel, 用 bars_since_flip 段重建)
    # 段终点 = 下一根 flip 或 股票末根; 段长 = bars_since_flip 在段末值 + 1
    pp = panel.sort_values(KEY).reset_index(drop=True)
    nxt_flip = pp.groupby("ts_code", sort=False)["dk_flip"].shift(-1).fillna(1)
    seg_end = (nxt_flip == 1)
    seg_len = (pp.loc[seg_end, "bars_since_flip"] + 1)
    seg_stats = {
        "n_segments": int(seg_end.sum()),
        "mean": round(float(seg_len.mean()), 2),
        "median": float(seg_len.median()),
        "p25": float(seg_len.quantile(0.25)),
        "p75": float(seg_len.quantile(0.75)),
        "p95": float(seg_len.quantile(0.95)),
        "max": int(seg_len.max()),
    }
    # 分方向段长
    seg_dir = pp.loc[seg_end, "dk_dir"]
    seg_len_long = float((seg_len[seg_dir == 1]).mean()) if (seg_dir == 1).any() else None
    seg_len_short = float((seg_len[seg_dir == -1]).mean()) if (seg_dir == -1).any() else None

    results = {
        "task": "DK-001", "elapsed_s": round(time.time() - t0, 1),
        "window": [STUDY_START, panel["trade_date"].max()],
        "klen": KLEN,
        "n_rows": n_rows, "n_stocks": n_stk, "n_days": n_days,
        "dir_share": dir_share,
        "n_flips": n_flip, "flip_to_long": flip_to_long, "flip_to_short": flip_to_short,
        "seg_len_stats": seg_stats,
        "seg_len_mean_long": round(seg_len_long, 2) if seg_len_long else None,
        "seg_len_mean_short": round(seg_len_short, 2) if seg_len_short else None,
        "self_check": self_check,
        "self_check_trigger_bad_total": total_bad,
        "self_check_hhk_leak_total": total_leak,
        "guardrails": ["SIGN-R01 状态机冻结逐字移植", "SIGN-R03 描述统计非ship",
                       "SIGN-R04 全因果 hhk/llk[1] 列名不撞黑名单", "SIGN-R06 ST源头排除(复用SLV-001宇宙)",
                       "SIGN-R08 checkpoint"],
    }
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print(f"\n--- 研究窗口 {STUDY_START}~{panel['trade_date'].max()} ---", flush=True)
    print(f"  覆盖: {n_rows:,} 行 / {n_stk} 股 / {n_days} 日", flush=True)
    print(f"  dk_dir 分布: {dir_share}", flush=True)
    print(f"  翻转事件: 总 {n_flip:,} (翻多 {flip_to_long:,} / 翻空 {flip_to_short:,})", flush=True)
    print(f"  段长 (根): mean {seg_stats['mean']} / median {seg_stats['median']} / "
          f"p95 {seg_stats['p95']} / max {seg_stats['max']}; "
          f"多腿均 {seg_len_long:.1f} / 空腿均 {seg_len_short:.1f}", flush=True)

    # ── verdict ──
    causal_ok = (total_bad == 0 and total_leak == 0)
    conclusion = (
        f"duokongK 状态机按 NOTES.md §1 faithful 移植 (klen=30, 单根内 a/b/c 顺序, "
        f"hhk/llk 用 [1] 全因果), 在全量 OHLC 面板 (复用 SLV-001 同宇宙, ST 源头排除) "
        f"逐股跑出 dk_dir/dk_flip/bars_since_flip/dist 序列。研究窗口 {STUDY_START}~"
        f"{panel['trade_date'].max()} 覆盖 {n_rows:,} 行 / {n_stk} 股 / {n_days} 日。"
        f"dk_dir 分布 {dir_share} (多腿/空腿/中性占比); 翻转事件总 {n_flip:,} "
        f"(翻多 {flip_to_long:,} / 翻空 {flip_to_short:,}); 段长中位 {seg_stats['median']} 根、"
        f"均值 {seg_stats['mean']} 根 (多腿均 {seg_len_long:.1f} / 空腿均 {seg_len_short:.1f})。"
        f"因果自检抽样 {len(self_check)} 股: flip 触发逻辑不符 {total_bad} 例、"
        f"hhk 未来泄漏 {total_leak} 例 → {'全因果无泄漏' if causal_ok else '⚠存在异常需排查'}。"
        f"描述统计非 ship; 下一步 DK-002 标底部确认翻多窗口 + 前向画像 (SIGN-R03)。")

    VERDICT.write_text(json.dumps({
        "id": "DK-001", "status": "built", "conclusion": conclusion,
        "metrics": {
            "window": [STUDY_START, panel["trade_date"].max()],
            "klen": KLEN, "n_rows": n_rows, "n_stocks": n_stk, "n_days": n_days,
            "dir_share": dir_share,
            "n_flips": n_flip, "flip_to_long": flip_to_long, "flip_to_short": flip_to_short,
            "seg_len_stats": seg_stats,
            "seg_len_mean_long": round(seg_len_long, 2) if seg_len_long else None,
            "seg_len_mean_short": round(seg_len_short, 2) if seg_len_short else None,
            "causal_self_check_ok": bool(causal_ok),
            "self_check_trigger_bad_total": total_bad,
            "self_check_hhk_leak_total": total_leak,
        },
        "artifacts": ["research/cache/dk001/duokongk_panel.parquet",
                      "research/cache/dk001_results.json"],
        "note": "状态机冻结逐字移植 (SIGN-R01); 全因果 hhk/llk[1], 输出列名不撞 forward 黑名单 (R04); "
                "ST 源头排除复用 SLV-001 宇宙 (R06); 大缓存写 research/cache/ 非 features; "
                "checkpoint (R08); 描述统计非 ship (R03); 生产线只读 (R05)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] DK-001 status=built -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
