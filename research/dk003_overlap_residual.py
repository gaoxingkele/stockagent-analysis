# -*- coding: utf-8 -*-
"""DK-003 — 与 SQUAT 重叠率 + 残差增量消融 (换皮判定 gate).

承 DK-001 (状态机面板) + DK-002 (底部确认翻多窗口)。按 NOTES.md §3 + prd.preRegisteredGate
做三问的第②③问 + gate:
  ① (DK-002 已做) 窗口前向画像。
  ② 重叠: dk_window onset / 窗口跨度 vs SLV-001 SQUAT 桶 launch 的重叠率 (Jaccard / 命中率)。
  ③ 残差增量: 构因果 dk 特征 (dk_state / 翻多recency / 反转确认强度) 前向 RankIC,
     扣 [past_r5 + pyr_velocity多窗 + ADX + RSI] 后残差 RankIC + t 值 (逐日截面正交化);
     SQUAT 外残差窗口前向是否显著兑现。

预注册 gate (FROZEN, prd.preRegisteredGate.gate, 禁在结果上调 SIGN-R01):
  NEW_INFO            = 残差 |IC| ≥ 0.01 且 |t| ≥ 2 且 SQUAT 外残差窗口前向显著兑现
  REJECT_reskin       = 残差被标准动量吃掉 (|IC| < 0.01)
  REJECT_no_residual_pay = 残差存在但前向不兑现 / 全在横盘

红线纪律:
  - gate 冻结 (R01); 负结果=合法完成 (R02); 中间 IC≠ship (R03);
    前向/label/IC 用列只留 cache 非 features (R04); ST 源头排除复用 SLV-001 宇宙 (R06);
    分 regime (R11); checkpoint (R08); 生产线只读 (R05)。

产出:
  research/cache/dk003/feat_panel.parquet   (因果特征 + 控制量 + forward 列, gitignored cache)
  research/cache/dk003_results.json
  research/verdicts/DK-003.json
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "research" / "cache" / "dk003"
CACHE.mkdir(parents=True, exist_ok=True)
DK_PANEL = ROOT / "research" / "cache" / "dk001" / "duokongk_panel.parquet"
SRC_PANEL = ROOT / "research" / "cache" / "slv001" / "factor_panel.parquet"
ROUTER = ROOT / "research" / "cache" / "slv001" / "router_panel.parquet"
WINDOWS = ROOT / "research" / "cache" / "dk002" / "dk_windows.parquet"
FEAT_OUT = CACHE / "feat_panel.parquet"
RESULTS = ROOT / "research" / "cache" / "dk003_results.json"
VERDICT = ROOT / "research" / "verdicts" / "DK-003.json"

STUDY_START = "20230103"
KEY = ["ts_code", "trade_date"]
H20, H40 = 20, 40
# dk 因果信号 + 控制量
DK_FEATS = ["dk_long_fresh", "dk_state", "rev_strength"]
CONTROLS = ["past_r5", "pyr_velocity_20_60", "rsi14", "adx14"]
GATE_FEAT = "dk_long_fresh"   # gate 主信号: "刚由空腿翻多" 的新鲜度 = 底部确认 onset
IC_MIN_OBS = 50               # 截面 IC 最小样本


# ────────────────────────── 因果指标 (RSI/ADX, Wilder ewm) ──────────────────────────
def add_rsi_adx(df: pd.DataFrame) -> pd.DataFrame:
    """逐股因果算 RSI14 / ADX14 (Wilder smoothing ≈ ewm alpha=1/14)。只用当根及历史。"""
    g = df.groupby("ts_code", sort=False)
    a = 1.0 / 14.0
    # RSI
    delta = g["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.groupby(df["ts_code"], sort=False).transform(
        lambda s: s.ewm(alpha=a, adjust=False).mean())
    avg_loss = loss.groupby(df["ts_code"], sort=False).transform(
        lambda s: s.ewm(alpha=a, adjust=False).mean())
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    df["rsi14"] = 100.0 - 100.0 / (1.0 + rs)
    # ADX
    up_move = g["high"].diff()
    dn_move = -g["low"].diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    prev_close = g["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["_plus_dm"] = plus_dm
    df["_minus_dm"] = minus_dm
    df["_tr"] = tr
    gg = df.groupby("ts_code", sort=False)
    atr = gg["_tr"].transform(lambda s: s.ewm(alpha=a, adjust=False).mean())
    plus_di = 100.0 * gg["_plus_dm"].transform(lambda s: s.ewm(alpha=a, adjust=False).mean()) / atr
    minus_di = 100.0 * gg["_minus_dm"].transform(lambda s: s.ewm(alpha=a, adjust=False).mean()) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    df["adx14"] = dx.groupby(df["ts_code"], sort=False).transform(
        lambda s: s.ewm(alpha=a, adjust=False).mean())
    df.drop(columns=["_plus_dm", "_minus_dm", "_tr"], inplace=True)
    return df


def build_feat_panel() -> pd.DataFrame:
    print("[data] 加载 factor_panel (OHLC + past_r5 + pyr_velocity) + dk001 面板", flush=True)
    px = pd.read_parquet(SRC_PANEL, columns=KEY + ["high", "low", "close",
                                                   "past_r5", "pyr_velocity_20_60"])
    px["trade_date"] = px["trade_date"].astype(str)
    dk = pd.read_parquet(DK_PANEL, columns=KEY + ["dk_dir", "bars_since_flip",
                                                  "dist_to_sllow", "dist_to_slhigh"])
    dk["trade_date"] = dk["trade_date"].astype(str)
    df = px.merge(dk, on=KEY, how="inner").sort_values(KEY).reset_index(drop=True)
    print(f"[data] merged {len(df):,} 行 / {df['ts_code'].nunique()} 股", flush=True)

    print("[calc] 因果 RSI14 / ADX14 ...", flush=True)
    df = add_rsi_adx(df)

    print("[calc] dk 因果信号 + forward 收益 (label 仅写 cache) ...", flush=True)
    # dk 因果信号
    is_long = (df["dk_dir"] == 1)
    df["dk_state"] = df["dk_dir"].astype(np.float64)
    # 翻多 recency: 刚由空腿翻多 = 新鲜度高, 随持有衰减 (仅多腿有效, 空腿=0)
    df["dk_long_fresh"] = np.where(is_long, 1.0 / (1.0 + df["bars_since_flip"].astype(float)), 0.0)
    # 反转确认强度: 多腿时 close 相对结构止损下沿的距离 (确认越强越高), 空腿=0
    df["rev_strength"] = np.where(is_long, df["dist_to_sllow"].fillna(0.0), 0.0)

    # forward 收益 (label/forward 列, 只留 cache; verify.py 黑名单仅扫 research/features/)
    g = df.groupby("ts_code", sort=False)
    df["fwd_r20"] = g["close"].shift(-H20) / df["close"] - 1.0
    df["fwd_r40"] = g["close"].shift(-H40) / df["close"] - 1.0
    return df


# ────────────────────────── 逐日截面 RankIC + 残差正交化 ──────────────────────────
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _orth_residual(feat: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    """feat 对 ctrl(含截距) 截面 OLS, 返回残差。"""
    X = np.column_stack([np.ones(len(feat)), ctrl])
    beta, *_ = np.linalg.lstsq(X, feat, rcond=None)
    return feat - X @ beta


def daily_ic(df: pd.DataFrame, feat: str, fwd: str, controls: list[str] | None = None):
    """逐日截面 RankIC (raw); controls 给定则先正交化 feat 再 IC (residual)。
    返回 (mean_ic, t_stat, n_days, ic_by_regime)。"""
    ics, regs = [], []
    cols = [feat, fwd] + (controls or [])
    for d, gd in df.groupby("trade_date", sort=False):
        sub = gd[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < IC_MIN_OBS:
            continue
        f = sub[feat].to_numpy(float)
        y = sub[fwd].to_numpy(float)
        if controls:
            C = sub[controls].to_numpy(float)
            if np.linalg.matrix_rank(np.column_stack([np.ones(len(C)), C])) < C.shape[1] + 1:
                continue
            f = _orth_residual(f, C)
        ic = _spearman(f, y)
        if ic == ic:
            ics.append(ic)
            regs.append(gd["regime"].iloc[0] if "regime" in gd else np.nan)
    ics = np.array(ics, float)
    if len(ics) == 0:
        return np.nan, np.nan, 0, {}
    mean = float(ics.mean())
    t = float(mean / (ics.std(ddof=1) / np.sqrt(len(ics)))) if ics.std(ddof=1) > 0 else np.nan
    by_reg = {}
    regs = np.array(regs, dtype=object)
    for r in set(regs[regs == regs]):
        m = ics[regs == r]
        by_reg[str(r)] = {"mean_ic": round(float(m.mean()), 5), "n_days": int(len(m))}
    return round(mean, 5), round(t, 2), len(ics), by_reg


def classify(ret: pd.Series) -> dict:
    r = ret.dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}
    return {
        "n": int(n),
        "mean": round(float(r.mean()), 4),
        "median": round(float(r.median()), 4),
        "surge_ge15pct": round(float((r >= 0.15).mean()), 4),
        "flat_lt5pct": round(float((r.abs() < 0.05).mean()), 4),
        "fail_le_neg5pct": round(float((r <= -0.05).mean()), 4),
        "win_rate_pos": round(float((r > 0).mean()), 4),
    }


# ────────────────────────── ② 重叠率 ──────────────────────────
def compute_overlap():
    """dk_window 跨度 / onset 与 SQUAT 桶 launch 的重叠 (命中率 + Jaccard)。"""
    win = pd.read_parquet(WINDOWS)
    win = win[win["anchor_date"] >= STUDY_START].copy()
    win["pivot_i"] = win["pivot_date"].astype(int)
    win["anchor_i"] = win["anchor_date"].astype(int)
    win["flip_i"] = win["flip_date"].astype(int)

    r = pd.read_parquet(ROUTER, columns=["ts_code", "trade_date", "archetype", "is_launch"])
    r["trade_date"] = r["trade_date"].astype(str)
    sq = r[(r["archetype"] == "SQUAT") & (r["is_launch"] == 1) &
           (r["trade_date"] >= STUDY_START)].copy()
    sq["d_i"] = sq["trade_date"].astype(int)
    sq_by = {c: np.sort(s["d_i"].to_numpy()) for c, s in sq.groupby("ts_code")}

    # 窗口跨度 [pivot..anchor] 内是否含 ≥1 SQUAT launch (同股)
    win_contains = np.zeros(len(win), dtype=bool)
    # onset (flip 当根) ±3 交易日匹配 (用排序日数组近似; ±3 自然日近似交易日)
    onset_match = np.zeros(len(win), dtype=bool)
    codes = win["ts_code"].to_numpy()
    piv = win["pivot_i"].to_numpy(); anc = win["anchor_i"].to_numpy(); flp = win["flip_i"].to_numpy()
    for i in range(len(win)):
        arr = sq_by.get(codes[i])
        if arr is None:
            continue
        lo = np.searchsorted(arr, piv[i], "left")
        hi = np.searchsorted(arr, anc[i], "right")
        if hi > lo:
            win_contains[i] = True
        # onset: flip 当根附近有 SQUAT launch (用 idx 距离≤3, 因 SQUAT launch 数组是交易日)
        j = np.searchsorted(arr, flp[i])
        for k in (j - 1, j):
            if 0 <= k < len(arr) and abs(int(arr[k]) - int(flp[i])) <= 5:
                onset_match[i] = True
                break
    win["contains_squat"] = win_contains
    win["onset_match_squat"] = onset_match

    # 反向: SQUAT launch 落入任一 dk_window 跨度的比例
    win_by = {c: s[["pivot_i", "anchor_i"]].to_numpy() for c, s in win.groupby("ts_code")}
    sq_covered = 0
    sq_total = 0
    for c, arr in sq_by.items():
        spans = win_by.get(c)
        sq_total += len(arr)
        if spans is None:
            continue
        for d in arr:
            if np.any((spans[:, 0] <= d) & (d <= spans[:, 1])):
                sq_covered += 1

    n_win = len(win)
    contain_rate = float(win_contains.mean())
    onset_rate = float(onset_match.mean())
    sq_cov_rate = sq_covered / sq_total if sq_total else np.nan
    # Jaccard (跨度级): |win∩SQUAT| / |win∪SQUAT|, 近似用 contain & coverage
    inter = int(win_contains.sum())
    union = n_win + sq_total - sq_covered  # 含 SQUAT 的窗算交, 余下并
    jaccard = inter / union if union else np.nan
    return win, {
        "n_windows": n_win, "n_squat_launch": int(sq_total),
        "window_contains_squat_rate": round(contain_rate, 4),
        "onset_pm5d_match_squat_rate": round(onset_rate, 4),
        "squat_covered_by_window_rate": round(sq_cov_rate, 4),
        "jaccard_span": round(float(jaccard), 4),
    }


def main():
    t0 = time.time()
    print("\n=== DK-003 重叠率 + 残差增量消融 (换皮判定 gate) ===\n", flush=True)

    # ── 特征面板 (checkpoint) ──
    if FEAT_OUT.exists():
        print(f"[ckpt] 命中 {FEAT_OUT.name}, 跳过特征重算", flush=True)
        df = pd.read_parquet(FEAT_OUT)
        df["trade_date"] = df["trade_date"].astype(str)
    else:
        df = build_feat_panel()
        keep = KEY + DK_FEATS + CONTROLS + ["fwd_r20", "fwd_r40"]
        df[keep].to_parquet(FEAT_OUT, index=False)
        print(f"[ckpt] feat_panel 落盘 {FEAT_OUT.name} ({len(df):,} 行)", flush=True)
        df = df[keep]

    # regime 标注 (逐日)
    rg = pd.read_parquet(ROOT / "research" / "features" / "regime_timeline.parquet",
                         columns=["trade_date", "regime"])
    rg["trade_date"] = rg["trade_date"].astype(str)
    df = df.merge(rg, on="trade_date", how="left")
    df = df[df["trade_date"] >= STUDY_START].copy()
    print(f"[data] IC 评估面板 {len(df):,} 行 / {df['trade_date'].nunique()} 日", flush=True)

    # ── ② 重叠率 ──
    print("\n--- ② 重叠率 (dk_window vs SQUAT 桶 launch) ---", flush=True)
    win, overlap = compute_overlap()
    for k, v in overlap.items():
        print(f"  {k}: {v}", flush=True)

    # ── ③ 残差增量消融 ──
    print("\n--- ③ 残差增量消融 (raw IC vs 扣[past_r5+pyr_velocity+ADX+RSI] 残差 IC) ---", flush=True)
    ic_table = {}
    for feat in DK_FEATS:
        for fwd in ("fwd_r20", "fwd_r40"):
            raw_m, raw_t, nd, raw_reg = daily_ic(df, feat, fwd, None)
            res_m, res_t, nd2, res_reg = daily_ic(df, feat, fwd, CONTROLS)
            ic_table[f"{feat}|{fwd}"] = {
                "raw_ic": raw_m, "raw_t": raw_t,
                "resid_ic": res_m, "resid_t": res_t,
                "n_days": nd, "raw_ic_by_regime": raw_reg, "resid_ic_by_regime": res_reg,
            }
            print(f"  {feat:14s} {fwd}: raw IC {raw_m} (t {raw_t}) -> "
                  f"resid IC {res_m} (t {res_t}), n_days {nd}", flush=True)

    # gate 主信号残差 (用 r40)
    gate_key = f"{GATE_FEAT}|fwd_r40"
    g_resid_ic = ic_table[gate_key]["resid_ic"]
    g_resid_t = ic_table[gate_key]["resid_t"]
    g_raw_ic = ic_table[gate_key]["raw_ic"]

    # ── SQUAT 外残差窗口前向是否兑现 ──
    print("\n--- SQUAT 外残差窗口前向兑现 (vs SQUAT 内) ---", flush=True)
    win_out = win[~win["contains_squat"]]
    win_in = win[win["contains_squat"]]
    prof_out40 = classify(win_out["fwd_r40"])
    prof_in40 = classify(win_in["fwd_r40"])
    prof_out_by_reg = {str(r): classify(g["fwd_r40"]) for r, g in win_out.groupby("regime")}
    print(f"  SQUAT 外窗口 ({len(win_out):,}) 40d: {prof_out40}", flush=True)
    print(f"  SQUAT 内窗口 ({len(win_in):,}) 40d: {prof_in40}", flush=True)

    # 兑现判定: SQUAT 外窗口前向是否"显著兑现" (大涨占比高 & 均值显著正 & 非全横盘)
    out_realizes = (prof_out40.get("surge_ge15pct", 0) >= 0.30 and
                    prof_out40.get("mean", 0) >= 0.03 and
                    prof_out40.get("surge_ge15pct", 0) > prof_out40.get("fail_le_neg5pct", 1))

    # ── 预注册 gate 判定 (FROZEN, 禁在结果上调 R01) ──
    abs_ic = abs(g_resid_ic) if g_resid_ic == g_resid_ic else 0.0
    abs_t = abs(g_resid_t) if g_resid_t == g_resid_t else 0.0
    if abs_ic < 0.01:
        status = "REJECT_reskin"
        playbook = "残差被 past_r5/pyr_velocity/ADX/RSI 吃掉 → 换皮, 文档化 (同 CB-003)"
    elif abs_ic >= 0.01 and abs_t >= 2 and out_realizes:
        status = "NEW_INFO"
        playbook = "结构反转确认在 SQUAT 之上有真残差且兑现 → 排 Phase 2 walk-forward blend"
    else:
        status = "REJECT_no_residual_pay"
        playbook = "有残差但前向不兑现/全横盘 → 窗口识别反转区但预测不了大涨vs横盘, 文档化"

    print(f"\n[GATE] resid IC({GATE_FEAT}|r40)={g_resid_ic} t={g_resid_t}  "
          f"out_realizes={out_realizes}  -> {status}", flush=True)

    metrics = {
        "overlap": overlap,
        "ic_table": ic_table,
        "gate_signal": GATE_FEAT,
        "gate_raw_ic_r40": g_raw_ic,
        "gate_resid_ic_r40": g_resid_ic,
        "gate_resid_t_r40": g_resid_t,
        "squat_out_window_profile_40d": prof_out40,
        "squat_in_window_profile_40d": prof_in40,
        "squat_out_profile_by_regime_40d": prof_out_by_reg,
        "out_realizes": bool(out_realizes),
        "controls": CONTROLS,
    }
    results = {"task": "DK-003", "elapsed_s": round(time.time() - t0, 1),
               "status": status, "playbook": playbook, "metrics": metrics}
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    conclusion = (
        f"重叠: dk_window 跨度含 SQUAT 桶 launch 命中率 {overlap['window_contains_squat_rate']:.1%}, "
        f"onset(±5交易日)匹配 SQUAT launch {overlap['onset_pm5d_match_squat_rate']:.1%}, "
        f"反向 SQUAT launch 被 dk_window 覆盖 {overlap['squat_covered_by_window_rate']:.1%} "
        f"(Jaccard {overlap['jaccard_span']:.2f})。"
        f"残差: 主信号 {GATE_FEAT} 对 r40 raw RankIC {g_raw_ic} → 扣 "
        f"[past_r5+pyr_velocity_20_60+ADX+RSI] 逐日截面正交化后残差 RankIC {g_resid_ic} (t {g_resid_t}); "
        f"dk_state/rev_strength 残差 IC 见 metrics.ic_table。"
        f"SQUAT 外窗口 ({prof_out40.get('n')}) 前向 40d: 大涨 {prof_out40.get('surge_ge15pct',0):.1%} / "
        f"横盘 {prof_out40.get('flat_lt5pct',0):.1%} / 失败 {prof_out40.get('fail_le_neg5pct',0):.1%} "
        f"(均值 {prof_out40.get('mean')}) → {'显著兑现' if out_realizes else '不兑现 (横盘/失败为主)'}。"
        f"按 prd.preRegisteredGate.gate 判定 **{status}**: {playbook}。"
        f"(gate 冻结禁在结果上调 R01; 残差被标准动量吃掉=换皮; 负结果=合法完成 R02; 中间 IC≠ship R03。)")

    VERDICT.write_text(json.dumps({
        "id": "DK-003", "status": status, "conclusion": conclusion,
        "metrics": metrics,
        "playbook_hit": playbook,
        "artifacts": ["research/cache/dk003/feat_panel.parquet",
                      "research/cache/dk003_results.json"],
        "note": "逐日截面 RankIC + 正交化残差 (扣 past_r5+pyr_velocity_20_60+ADX+RSI); "
                "RSI14/ADX14 Wilder ewm 因果自算; forward 列只留 cache 非 features (R04); "
                "ST 源头排除复用 SLV-001 宇宙 (R06); 分 regime (R11); checkpoint (R08); "
                "gate 冻结禁在结果上调 (R01); 负结果=合法完成 (R02); 描述/IC 非 ship (R03); "
                "生产线只读 (R05)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] DK-003 status={status} -> {VERDICT.relative_to(ROOT)}  "
          f"耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
