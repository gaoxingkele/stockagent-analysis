# -*- coding: utf-8 -*-
"""CB-002 — 盲点 & 僵尸误杀 量化 (描述, 直答北极星前两问).

承接 CB-001 (research/features/consolidation.parquet: base_type + cs_zombie 分解)。
本任务纯描述, 直接回答:
  ① 盲点: V12.31 偏好"下蹲后起跳"(past_r5<0), 真启动里 past_r5≥0(横盘/微涨基底) 占比多少?
     系统(V12.31 v7c 池) 抓到 / 漏掉多少?
  ② 僵尸误杀: 第 6 铁律 "NOT is_zombie" 剔掉的票里, 后续真启动的 '蓄势基底'(coiling)
     假阴性率多少 (是否误杀蓄势横盘)?

真启动 (outcome 测量, **仅 forward, 绝不入特征 / 不落 features 目录**, 零泄漏):
  与生产 pump-up 同口径 (t004 build_label, H=5): 决策点 t 之后 5 个交易日内
  max_gain = max(high[t+1..t+5])/close[t]-1 >= +10%  且  max_dd = min(low)/close[t]-1 >= -5%。
  → is_launch ∈ {True, False}; 前向不足 5 日的尾部 = NaN (剔除)。

V12.31 与本问的关系 (诚实标注硬/软):
  - **硬过滤**: 第 6 铁律 NOT is_zombie (cs_zombie==True 直接被剔出池) — 可逐位复算, 计 "硬漏"。
  - **软偏好**: pump v3c 偏好下蹲 (past_r5<0) 是模型学到的排序偏好, 非硬门 — 计 "软降权"。
  本任务量化两者各漏掉多少真启动; gate/落地由 CB-004 walk-forward 定 (SIGN-R03)。

分 regime (SIGN-R11): 盲点占比 / 僵尸启动率均按 regime_timeline (momentum/reversal) 分层,
全期平均只作参考。

纪律: ST 源头排除 (R06); forward 仅 outcome 不入 features (R04); 确定性; checkpoint 落
research/cache/cb002/ (R08)。

输出:
  research/cache/cb002/launch_panel.parquet  (key + is_launch + base_type + zombie 标签, **非 features 目录**)
  research/verdicts/CB-002.json              (命中 responsePlaybook 盲点/僵尸 分支 + 量化表)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
CONSOL_P = ROOT / "research" / "features" / "consolidation.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
CACHE_DIR = ROOT / "research" / "cache" / "cb002"
LAUNCH_CACHE = CACHE_DIR / "launch_panel.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "CB-002.json"

KEY = ["ts_code", "trade_date"]

# 真启动口径 (= 生产 pump-up s5, t004 build_label)
H = 5
GAIN_THR = 0.10
DD_CAP = 0.05

# 描述性解释阈 (非 gate; gate 由 CB-004 定。仅用于命中 playbook 分支)
BLINDSPOT_SHARE_HI = 0.30   # 真启动里 past_r5≥0 占比 >= 30% 视为"盲点占比高"
COILING_LIFT_HI = 1.20      # coiling 僵尸启动率 / dead 僵尸启动率 >= 1.2 视为"误杀(蓄势更易启动)"


def load_st_set() -> set:
    if not BASIC_P.exists():
        return set()
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def load_daily(st: set) -> pd.DataFrame:
    cols = ["ts_code", "trade_date", "close", "high", "low"]
    parts = [pd.read_parquet(f, columns=cols) for f in sorted(DAILY_DIR.glob("*.parquet"))]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.drop_duplicates(subset=KEY, keep="last")
    big = big[big["close"].notna() & (big["close"] > 0)]
    if st:
        big = big[~big["ts_code"].isin(st)]
    return big.sort_values(KEY).reset_index(drop=True)


def compute_launch(daily: pd.DataFrame) -> pd.DataFrame:
    """前向 H 日 pump-up outcome (与 t004 build_label up 分支同口径)。"""
    gh = daily.groupby("ts_code", sort=False)["high"]
    gl = daily.groupby("ts_code", sort=False)["low"]
    fwd_hi = pd.concat([gh.shift(-k) for k in range(1, H + 1)], axis=1).max(axis=1)
    fwd_lo = pd.concat([gl.shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1)
    gain = fwd_hi / daily["close"] - 1.0
    dd = fwd_lo / daily["close"] - 1.0
    valid = gain.notna() & dd.notna()
    is_launch = pd.Series(pd.NA, index=daily.index, dtype="boolean")
    is_launch[valid] = ((gain >= GAIN_THR) & (dd >= -DD_CAP))[valid]
    out = daily[KEY].copy()
    out["is_launch"] = is_launch
    return out


def rate(mask_launch: pd.Series) -> float:
    """启动率 = P(launch | 该子集); 子集内 is_launch 非空的均值。"""
    s = mask_launch.dropna()
    return float(s.mean()) if len(s) else float("nan")


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    st = load_st_set()
    print(f"[st] ST 源头排除 {len(st)} 只", flush=True)

    # ── 真启动 outcome (checkpoint) ──
    if LAUNCH_CACHE.exists():
        print(f"[launch] checkpoint 命中 {LAUNCH_CACHE.name}", flush=True)
        panel = pd.read_parquet(LAUNCH_CACHE)
    else:
        daily = load_daily(st)
        print(f"[daily] {len(daily):,} 行 / {daily['ts_code'].nunique()} 股 "
              f"({daily['trade_date'].min()}~{daily['trade_date'].max()})", flush=True)
        lp = compute_launch(daily)
        print(f"[launch] 真启动 outcome 算完 (H={H}, gain>={GAIN_THR:.0%}, dd>=-{DD_CAP:.0%})", flush=True)

        consol = pd.read_parquet(CONSOL_P, columns=KEY + [
            "base_type", "cs_past_r5", "cs_is_consolidation",
            "cs_zombie", "cs_zombie_coiling", "cs_zombie_dead"])
        consol["trade_date"] = consol["trade_date"].astype(str)
        panel = consol.merge(lp, on=KEY, how="left")

        regime = pd.read_parquet(REGIME_P, columns=["trade_date", "regime"])
        regime["trade_date"] = regime["trade_date"].astype(str)
        panel = panel.merge(regime, on="trade_date", how="left")
        panel.to_parquet(LAUNCH_CACHE, index=False)
        print(f"[launch] checkpoint 落盘 {LAUNCH_CACHE.name} ({len(panel):,} 行)", flush=True)

    # ── 确定性自检: 抽样重算 forward 标签逐位比对 ──
    daily_s = load_daily(st)
    samp = sorted(daily_s["ts_code"].unique())[::max(1, daily_s["ts_code"].nunique() // 25)][:25]
    rl = compute_launch(daily_s[daily_s["ts_code"].isin(samp)].copy())
    ra = rl.sort_values(KEY).reset_index(drop=True)
    rb = (panel[panel["ts_code"].isin(samp)][KEY + ["is_launch"]]
          .sort_values(KEY).reset_index(drop=True))
    det = bool((ra["is_launch"].astype("Int8").fillna(-1).astype("int8").values
                == rb["is_launch"].astype("Int8").fillna(-1).astype("int8").values).all())
    assert det, "确定性自检失败 (forward 标签重算不一致)"

    # 评估宇宙: 决策点须有 base_type + is_launch (前向足) 才计入
    df = panel[panel["base_type"].notna() & (panel["base_type"] != "na")
               & panel["is_launch"].notna()].copy()
    df["is_launch"] = df["is_launch"].astype(bool)
    df["cs_zombie"] = df["cs_zombie"].fillna(False).astype(bool)
    n_all = len(df)
    base_launch_rate = rate(df["is_launch"])
    launches = df[df["is_launch"]]
    n_launch = len(launches)
    print(f"[univ] 评估决策点 {n_all:,}; 真启动 {n_launch:,} (启动率 {base_launch_rate:.4%})", flush=True)

    # ───────────────────────── ① 盲点 ─────────────────────────
    # base_type 在 (全样本 / 启动样本) 的分布 + 各 base_type 启动率(lift)
    bt_all = df["base_type"].value_counts(normalize=True).round(4).to_dict()
    bt_launch = launches["base_type"].value_counts(normalize=True).round(4).to_dict()
    bt_rate = {bt: round(rate(df.loc[df["base_type"] == bt, "is_launch"]), 5)
               for bt in df["base_type"].unique()}

    # past_r5≥0 (横盘/微涨基底 = V12.31 软偏好之外) 的占比与启动率
    pos = df["cs_past_r5"] >= 0
    share_pos_in_launch = float((launches["cs_past_r5"] >= 0).mean())   # 启动里 past_r5≥0 占比
    rate_pos = rate(df.loc[pos, "is_launch"])
    rate_neg = rate(df.loc[~pos, "is_launch"])
    # 非下蹲基底 (flat_base + up_drift) 占启动比
    nonsquat = launches["base_type"].isin(["flat_base", "up_drift"])
    share_nonsquat_in_launch = float(nonsquat.mean())

    # V12.31 硬过滤(第6铁律)对真启动的硬漏: cs_zombie==True 的启动被直接剔出池
    hard_miss = float(launches["cs_zombie"].mean())              # 启动里被 6 铁律硬剔比例
    # 软降权: past_r5≥0 的启动 (下蹲偏好会把它们排在后面)
    soft_deprior = share_pos_in_launch

    # 分 regime 的盲点占比 (R11)
    reg_blind = {}
    for rg, gdf in df[df["regime"].notna()].groupby("regime"):
        lg = gdf[gdf["is_launch"]]
        if len(lg) == 0:
            continue
        reg_blind[str(rg)] = {
            "n_launch": int(len(lg)),
            "share_past_r5_ge0_in_launch": round(float((lg["cs_past_r5"] >= 0).mean()), 4),
            "share_nonsquat_in_launch": round(float(lg["base_type"].isin(
                ["flat_base", "up_drift"]).mean()), 4),
            "launch_rate": round(rate(gdf["is_launch"]), 5),
        }

    # ───────────────────────── ② 僵尸误杀 ─────────────────────────
    z = df[df["cs_zombie"]].copy()
    nz = df[~df["cs_zombie"]]
    z_rate = rate(z["is_launch"])
    nz_rate = rate(nz["is_launch"])
    coiling = z[z["cs_zombie_coiling"].fillna(False).astype(bool)]
    dead = z[z["cs_zombie_dead"].fillna(False).astype(bool)]
    coil_rate = rate(coiling["is_launch"])
    dead_rate = rate(dead["is_launch"])
    coil_lift = (coil_rate / dead_rate) if (dead_rate and dead_rate == dead_rate) else float("nan")
    # 假阴性率: 被第6铁律剔的票里, 后续真启动且基底是 coiling(蓄势) 的, 视为误杀真启动
    n_z_launch = int(z["is_launch"].sum())
    n_coil_launch = int(coiling["is_launch"].sum())
    fn_share_coiling = (n_coil_launch / n_z_launch) if n_z_launch else float("nan")  # 僵尸启动里蓄势占比
    coil_launch_rate = coil_rate

    # 分 regime 的僵尸启动率 (R11)
    reg_zombie = {}
    for rg, gdf in df[df["regime"].notna()].groupby("regime"):
        zc = gdf[gdf["cs_zombie"] & gdf["cs_zombie_coiling"].fillna(False).astype(bool)]
        zd = gdf[gdf["cs_zombie"] & gdf["cs_zombie_dead"].fillna(False).astype(bool)]
        reg_zombie[str(rg)] = {
            "coiling_launch_rate": round(rate(zc["is_launch"]), 5) if len(zc) else None,
            "dead_launch_rate": round(rate(zd["is_launch"]), 5) if len(zd) else None,
        }

    # ───────────────────────── playbook 分支判定 ─────────────────────────
    # 盲点存在 = past_r5≥0 占启动比高 且 past_r5≥0 启动率不低于 past_r5<0 (系统软偏好确实漏)
    blindspot_exists = (share_pos_in_launch >= BLINDSPOT_SHARE_HI) and (rate_pos >= rate_neg * 0.9)
    blind_branch = "存在" if blindspot_exists else "不存在"
    # 误杀 = coiling 僵尸启动率显著高于 dead, 且不低于全样本基准 (蓄势僵尸是好基底被错杀)
    misskill = (coil_lift >= COILING_LIFT_HI) and (coil_rate >= base_launch_rate * 0.9)
    zombie_branch = "误杀" if misskill else "无误杀"

    # ───────────────────────── verdict ─────────────────────────
    conclusion = (
        f"① 盲点[{blind_branch}]: 真启动里 past_r5≥0 占 {share_pos_in_launch:.1%} "
        f"(非下蹲基底 flat+drift 占启动 {share_nonsquat_in_launch:.1%}); "
        f"past_r5≥0 启动率 {rate_pos:.3%} vs past_r5<0 {rate_neg:.3%} (基准 {base_launch_rate:.3%}); "
        f"V12.31 第6铁律硬漏启动 {hard_miss:.1%}, 下蹲软偏好降权启动 {soft_deprior:.1%}。 "
        f"② 僵尸误杀[{zombie_branch}]: 僵尸启动率 {z_rate:.3%} vs 非僵尸 {nz_rate:.3%}; "
        f"蓄势横盘(coiling) 启动率 {coil_rate:.3%} vs 死横盘(dead) {dead_rate:.3%} "
        f"(lift {coil_lift:.2f}); 僵尸里真启动 {n_z_launch:,} 个, 蓄势基底占 {fn_share_coiling:.1%}。"
    )
    if blindspot_exists:
        blind_note = ("past_r5≥0 横盘/微涨启动占比高且启动率不输下蹲 → 验证盲点, CB-003/004 继续。")
    else:
        blind_note = ("启动几乎都来自下蹲 → 用户盲点假设证伪, 文档化; CB-003/004 仍可测但先验降低。")
    if misskill:
        zombie_note = ("被剔僵尸里蓄势基底(coiling)启动率显著高于死横盘 → 第6铁律误杀蓄势基底, 需精炼。")
    else:
        zombie_note = ("僵尸过滤剔的确实更接近死横盘(coiling 不比 dead 明显更易启动) → 6铁律不动它。")

    verdict = {
        "id": "CB-002",
        "status": "measured",
        "conclusion": conclusion,
        "playbook": {
            "CB-002_盲点": blind_branch, "盲点_note": blind_note,
            "CB-002_僵尸": zombie_branch, "僵尸_note": zombie_note,
        },
        "launch_def": {"horizon_days": H, "gain_threshold": GAIN_THR, "max_dd_cap": DD_CAP,
                       "note": "= 生产 pump-up s5 口径 (t004 build_label up 分支); forward-only outcome, 不入 features"},
        "interpret_thresholds": {"blindspot_share_hi": BLINDSPOT_SHARE_HI,
                                 "coiling_lift_hi": COILING_LIFT_HI,
                                 "note": "描述性解释阈, 非 gate; ship 由 CB-004 walk-forward 定 (SIGN-R03)"},
        "metrics": {
            "universe": {"n_decision_points": int(n_all), "n_launch": int(n_launch),
                         "base_launch_rate": round(base_launch_rate, 5),
                         "date_range": [df["trade_date"].min(), df["trade_date"].max()]},
            "blindspot": {
                "base_type_frac_all": bt_all,
                "base_type_frac_in_launch": bt_launch,
                "launch_rate_by_base_type": bt_rate,
                "share_past_r5_ge0_in_launch": round(share_pos_in_launch, 4),
                "share_nonsquat_in_launch": round(share_nonsquat_in_launch, 4),
                "launch_rate_past_r5_ge0": round(rate_pos, 5),
                "launch_rate_past_r5_lt0": round(rate_neg, 5),
                "v1231_hard_miss_zombie_share_in_launch": round(hard_miss, 4),
                "v1231_soft_deprior_share_in_launch": round(soft_deprior, 4),
                "by_regime": reg_blind,
            },
            "zombie_misskill": {
                "zombie_launch_rate": round(z_rate, 5),
                "nonzombie_launch_rate": round(nz_rate, 5),
                "coiling_launch_rate": round(coil_rate, 5),
                "dead_launch_rate": round(dead_rate, 5),
                "coiling_vs_dead_lift": round(coil_lift, 3),
                "n_zombie_launch": n_z_launch,
                "n_coiling_launch": n_coil_launch,
                "fn_share_coiling_in_zombie_launch": round(fn_share_coiling, 4),
                "by_regime": reg_zombie,
            },
            "deterministic": det,
            "leakage_free": True,
            "st_excluded": int(len(st)),
        },
        "artifact": str(LAUNCH_CACHE.relative_to(ROOT)),
        "guardrails": ["SIGN-R04 forward 仅 outcome 不入 features (零泄漏)",
                       "SIGN-R06 ST 源头排除", "SIGN-R08 launch_panel checkpoint",
                       "SIGN-R11 盲点/僵尸启动率分 regime", "确定性逐位比对"],
    }
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印 ──
    print("\n=== CB-002 量化 ===", flush=True)
    print(f"  评估决策点 {n_all:,} / 真启动 {n_launch:,} (启动率 {base_launch_rate:.4%})", flush=True)
    print(f"  base_type 启动率: {bt_rate}", flush=True)
    print(f"  [①盲点 {blind_branch}] past_r5≥0 占启动 {share_pos_in_launch:.1%} | "
          f"非下蹲(flat+drift) 占启动 {share_nonsquat_in_launch:.1%}", flush=True)
    print(f"     past_r5≥0 启动率 {rate_pos:.3%} vs past_r5<0 {rate_neg:.3%}", flush=True)
    print(f"     V12.31 硬漏(6铁律) {hard_miss:.1%} | 软降权(下蹲偏好) {soft_deprior:.1%}", flush=True)
    print(f"  [②僵尸 {zombie_branch}] coiling 启动率 {coil_rate:.3%} vs dead {dead_rate:.3%} "
          f"(lift {coil_lift:.2f})", flush=True)
    print(f"     僵尸启动 {n_z_launch:,} 蓄势占 {fn_share_coiling:.1%}", flush=True)
    print(f"  regime 盲点: {reg_blind}", flush=True)
    print(f"  确定性={det}", flush=True)
    print(f"[verdict] -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
