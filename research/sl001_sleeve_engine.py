# -*- coding: utf-8 -*-
"""SL-001 — r5 龙虎榜席位事件 sleeve 回测引擎 (正确载体).

承接 [[project_seat_footprint_real_signal_wrong_vehicle_0606]]: 龙虎榜席位印记 = 整轮首个
扛消融的真正交信号 (SEAT-002 残差 IC +0.0547 t8.58 全 regime), 但接 V12.31 r20 池
walk-forward REJECT = **载体错** (稀疏 3.7% + 短线 vs r20 horizon 错配, GATE-001)。
本任务造**正确载体**: 独立短线 r5 事件 sleeve — 仅龙虎榜上榜股 D+1 建仓、持有 H=5 交易日、
按席位历史战绩排序跟高战绩席位。**成败核心 = 真实成本 + 容量 + 净 α** (短线高换手成本会吃掉大半)。

事件驱动设计 (SIGN-R04 零泄漏, 复用 SEAT-001 因果防泄漏 seat_footprint):
  - 龙虎榜在交易日 D 收盘后公告 → **D+1 开盘**建仓, 持有 H 交易日, **D+1+H 开盘**平仓。
  - 候选股 = 当日 D 龙虎榜机构净买入 (∑net_buy>0) 的股 (Arm_M 全集)。
  - 席位战绩 sf_edge_r5/sf_winrate_r5 来自 SEAT-001 (席位历史 expanding 战绩, avail<=D, 信号赋 D+1),
    严格因果, D+1 入场时已知。
  - 动量对照 mom20 = close[D]/close[D-20]-1 (D 收盘已知, D+1 开盘前可得), 纯过去, 无泄漏。

三臂 apples-to-apples (隔离"席位技能" vs "龙虎榜效应"):
  - Arm_M (基线): 当日**全部**龙虎榜净买入股等权 → "无脑买龙虎榜"。
  - Arm_A (席位战绩): 在**有可信席位战绩**的候选内, 按 sf_edge_r5 降序 top-N 等权 → 跟高战绩席位。
  - Arm_B (动量对照): 在**同一**席位可信候选内, 按 mom20 降序 top-N 等权 → 控制"在龙虎榜内挑动量"。
    Arm_A vs Arm_B 唯一差异 = 排序键 (席位战绩 vs 动量) → 净增量 = 纯席位技能 (SL-003 gate)。

真实成本: round-trip 30bps (佣金双边 + 印花税 0.05% 卖 + 滑点); net = gross − cost。SL-002 跑敏感性。

纪律:
  - ST 源头排除 (SIGN-R06): 候选股 + 价格两侧。
  - 确定性 (SIGN): 全 vectorized merge, 无随机。
  - checkpoint (SIGN-R08): 价格/候选/picks 缓存 research/cache/sl001/, 已算跳过。
  - 中间指标 ≠ 落地 (SIGN-R03): 本 task 仅 status=built (引擎 + 描述统计); 裁决在 SL-003 walk-forward。

输出:
  research/cache/sl001_daily_pnl.parquet  (entry_date × arm: gross/net/n_stocks — AC 要求)
  research/cache/sl001/picks.parquet      (stock 级: 复用给 SL-002 容量 / SL-003 gate)
  research/verdicts/SL-001.json           (status=built + 三臂毛/净收益/换手/平均持仓数)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
TOPINST_P = ROOT / "output" / "tushare_cache" / "top_inst.parquet"
SF_P = ROOT / "research" / "features" / "seat_footprint.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"

CACHE = ROOT / "research" / "cache"
SL_CACHE = CACHE / "sl001"
PX_CACHE = SL_CACHE / "px.parquet"
CAND_CACHE = SL_CACHE / "candidates.parquet"
PICKS_CACHE = SL_CACHE / "picks.parquet"
DAILY_PNL = CACHE / "sl001_daily_pnl.parquet"
VERDICT = ROOT / "research" / "verdicts" / "SL-001.json"

# ── 事前注册设计常数 (冻结 SIGN-R01; 主测口径) ──
H_HOLD = 5                  # 持有交易日 (r5 短线 sleeve)
TOP_N = 10                  # 每日每臂取 top-N
COST_RT = 30.0 / 10000      # round-trip 30bps (佣金+印花税0.05%卖+滑点), 主测
MOM_WIN = 20                # 动量对照窗口 (过去交易日, 纯过去无泄漏)
SEAT_SORT = "sf_edge_r5"    # Arm_A 排序键 (期望 r5 收益, 与持有期 horizon-matched)
SEAT_TIE = "sf_winrate_r5"  # tiebreak
RET_CLIP = 0.5              # 单笔收益截断 (±50%, 防异常)


def load_st_set() -> set:
    if not BASIC_P.exists():
        return set()
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def load_prices(st: set) -> pd.DataFrame:
    """全历史 open/close (ST 源头排除); 含 amount (容量) + mom20 (动量对照)。"""
    if PX_CACHE.exists():
        print(f"[px] checkpoint 命中 {PX_CACHE.name}", flush=True)
        return pd.read_parquet(PX_CACHE)
    print(f"[px] 加载全历史 ({DAILY_DIR}) ...", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "close", "amount"])
             for f in sorted(DAILY_DIR.glob("*.parquet"))]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = px.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    px = px[px["open"].notna() & (px["open"] > 0) & px["close"].notna() & (px["close"] > 0)]
    if st:
        before = len(px)
        px = px[~px["ts_code"].isin(st)]
        print(f"  ST 源头排除: {before - len(px):,} 行 ({len(st)} 只 ST)", flush=True)
    px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    # 动量 = 过去 MOM_WIN 日收益 (该行 trade_date 收盘已知, 纯过去)
    px["mom20"] = px.groupby("ts_code")["close"].transform(
        lambda s: s / s.shift(MOM_WIN) - 1.0)
    SL_CACHE.mkdir(parents=True, exist_ok=True)
    px.to_parquet(PX_CACHE, index=False)
    print(f"[px] 落盘 {PX_CACHE.name} ({len(px):,} 行)", flush=True)
    return px


def load_candidates(st: set, valid_codes: set) -> pd.DataFrame:
    """龙虎榜机构净买入股: 每 (D, ts_code) 聚合 ∑net_buy, 取 >0 (全集 = Arm_M)。"""
    if CAND_CACHE.exists():
        print(f"[cand] checkpoint 命中 {CAND_CACHE.name}", flush=True)
        return pd.read_parquet(CAND_CACHE)
    ti = pd.read_parquet(TOPINST_P)[["trade_date", "ts_code", "net_buy"]].copy()
    ti["trade_date"] = ti["trade_date"].astype(str)
    ti = ti.dropna(subset=["net_buy"])
    if st:
        ti = ti[~ti["ts_code"].isin(st)]
    ti = ti[ti["ts_code"].isin(valid_codes)]
    cand = ti.groupby(["trade_date", "ts_code"], as_index=False)["net_buy"].sum()
    cand = cand[cand["net_buy"] > 0].rename(columns={"trade_date": "event_date"})
    cand = cand.reset_index(drop=True)
    SL_CACHE.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(CAND_CACHE, index=False)
    print(f"[cand] {len(cand):,} 候选 (D, 股) / {cand['ts_code'].nunique()} 股 / "
          f"{cand['event_date'].nunique()} 日 "
          f"({cand['event_date'].min()}~{cand['event_date'].max()})", flush=True)
    return cand


def build_picks(px: pd.DataFrame, cand: pd.DataFrame, sf: pd.DataFrame,
                cal: list[str], H: int) -> pd.DataFrame:
    """每候选: event_date D → entry=next(D) 开盘, exit=entry+H 开盘; 计 ret/动量/席位战绩/amount。

    返回 stock 级 picks (含全集 + 席位可信标记), 供三臂在 run_arms 内按排序/N 切分。
    """
    cal_idx = {d: i for i, d in enumerate(cal)}
    next_map = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    cand = cand.copy()
    cand["entry_date"] = cand["event_date"].map(next_map)
    cand = cand.dropna(subset=["entry_date"])
    cand["entry_idx"] = cand["entry_date"].map(cal_idx)
    cand = cand.dropna(subset=["entry_idx"])
    cand["entry_idx"] = cand["entry_idx"].astype(int)
    cand["exit_idx"] = cand["entry_idx"] + H
    cand = cand[cand["exit_idx"] < len(cal)].copy()
    cand["exit_date"] = cand["exit_idx"].map(lambda i: cal[i])

    op = px[["ts_code", "trade_date", "open"]]
    cand = cand.merge(op.rename(columns={"trade_date": "entry_date", "open": "entry_open"}),
                      on=["ts_code", "entry_date"], how="left")
    cand = cand.merge(op.rename(columns={"trade_date": "exit_date", "open": "exit_open"}),
                      on=["ts_code", "exit_date"], how="left")
    cand = cand.dropna(subset=["entry_open", "exit_open"])
    cand = cand[(cand["entry_open"] > 0) & (cand["exit_open"] > 0)]
    cand["ret"] = (cand["exit_open"] / cand["entry_open"] - 1.0).clip(-RET_CLIP, RET_CLIP)

    # 动量对照: event_date D 当日的 mom20 (D 收盘已知)
    mom = px[["ts_code", "trade_date", "mom20", "amount"]].rename(
        columns={"trade_date": "event_date", "amount": "entry_amount"})
    cand = cand.merge(mom, on=["ts_code", "event_date"], how="left")

    # 席位战绩: SEAT-001 seat_footprint (trade_date=信号生效日=D+1=entry_date)
    sfm = sf[["ts_code", "trade_date", SEAT_SORT, SEAT_TIE]].rename(
        columns={"trade_date": "entry_date"})
    cand = cand.merge(sfm, on=["ts_code", "entry_date"], how="left")
    cand["has_seat"] = cand[SEAT_SORT].notna()

    keep = ["event_date", "entry_date", "exit_date", "ts_code", "net_buy", "ret",
            "mom20", "entry_amount", SEAT_SORT, SEAT_TIE, "has_seat"]
    cand = cand[keep].sort_values(["entry_date", "ts_code"]).reset_index(drop=True)
    return cand


def run_arms(picks: pd.DataFrame, top_n: int, cost_rt: float) -> pd.DataFrame:
    """三臂逐 entry_date 等权篮子收益。

    Arm_M: 全候选等权。  Arm_A: 席位可信内按 sf_edge_r5 top-N。  Arm_B: 同集按 mom20 top-N。
    """
    rows = []
    seat_pool = picks[picks["has_seat"]].copy()
    for ed, g in picks.groupby("entry_date"):
        # Arm_M — 全部龙虎榜净买入股等权
        rows.append(_arm_row(ed, "M", g))
        gs = seat_pool[seat_pool["entry_date"] == ed]
        if len(gs) == 0:
            continue
        # Arm_A — 席位战绩排序 top-N
        a = gs.sort_values([SEAT_SORT, SEAT_TIE], ascending=False).head(top_n)
        rows.append(_arm_row(ed, "A", a))
        # Arm_B — 同集动量排序 top-N (对照)
        b = gs.dropna(subset=["mom20"]).sort_values("mom20", ascending=False).head(top_n)
        if len(b):
            rows.append(_arm_row(ed, "B", b))
    out = pd.DataFrame(rows)
    out["net"] = out["gross"] - cost_rt
    out["month"] = out["entry_date"].str[:6]
    return out.sort_values(["entry_date", "arm"]).reset_index(drop=True)


def _arm_row(ed: str, arm: str, g: pd.DataFrame) -> dict:
    return {"entry_date": ed, "arm": arm,
            "gross": float(g["ret"].mean()), "n_stocks": int(len(g)),
            "win": float((g["ret"] > 0).mean())}


def describe_arm(pnl: pd.DataFrame, arm: str) -> dict:
    a = pnl[pnl["arm"] == arm]
    if a.empty:
        return {}
    g = a["gross"].to_numpy()
    n = a["net"].to_numpy()
    n_per_yr = 252.0 / H_HOLD                          # 满仓连续换手的年化轮次 (容量/成本量级)
    return {
        "n_event_days": int(len(a)),
        "avg_n_stocks": round(float(a["n_stocks"].mean()), 2),
        "gross_mean_per_trade_pct": round(float(g.mean()) * 100, 4),
        "net_mean_per_trade_pct": round(float(n.mean()) * 100, 4),
        "gross_median_pct": round(float(np.median(g)) * 100, 4),
        "win_rate_basket": round(float((g > 0).mean()), 4),
        "win_rate_stock": round(float(a["win"].mean()), 4),
        "gross_std_per_trade_pct": round(float(g.std()) * 100, 4),
        "ann_roundtrips_full_invest": round(n_per_yr, 1),
        "ann_cost_drag_pct": round(n_per_yr * COST_RT * 100, 2),
    }


def main() -> int:
    t0 = time.time()
    SL_CACHE.mkdir(parents=True, exist_ok=True)
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    print("\n=== SL-001 r5 龙虎榜席位事件 sleeve 回测引擎 ===\n", flush=True)

    st = load_st_set()
    print(f"[st] ST 源头排除 {len(st)} 只", flush=True)
    px = load_prices(st)
    cal = sorted(px["trade_date"].unique())
    print(f"[cal] {len(cal)} 交易日 ({cal[0]}~{cal[-1]})", flush=True)

    cand = load_candidates(st, set(px["ts_code"].unique()))
    sf = pd.read_parquet(SF_P)
    sf["trade_date"] = sf["trade_date"].astype(str)

    if PICKS_CACHE.exists():
        print(f"[picks] checkpoint 命中 {PICKS_CACHE.name}", flush=True)
        picks = pd.read_parquet(PICKS_CACHE)
    else:
        picks = build_picks(px, cand, sf, cal, H_HOLD)
        picks.to_parquet(PICKS_CACHE, index=False)
        print(f"[picks] 落盘 {PICKS_CACHE.name} ({len(picks):,} 行, "
              f"席位可信 {int(picks['has_seat'].sum()):,})", flush=True)

    print(f"[picks] {len(picks):,} 候选 / {picks['ts_code'].nunique()} 股 / "
          f"{picks['entry_date'].nunique()} entry 日 "
          f"({picks['entry_date'].min()}~{picks['entry_date'].max()}); "
          f"席位可信覆盖 {picks['has_seat'].mean():.2%}", flush=True)

    pnl = run_arms(picks, TOP_N, COST_RT)
    pnl.to_parquet(DAILY_PNL, index=False)
    print(f"[pnl] -> {DAILY_PNL.relative_to(ROOT)} ({len(pnl):,} 行 entry×arm)", flush=True)

    arms = {"M": "等权全龙虎榜(基线)", "A": "席位战绩排序(top-N)", "B": "动量对照(top-N)"}
    desc = {a: describe_arm(pnl, a) for a in arms}

    print("\n=== 三臂描述统计 (主测 H=5, N=10, cost=30bps round-trip) ===", flush=True)
    print(f"  {'臂':24s} {'毛收益/笔%':>11s} {'净收益/笔%':>11s} "
          f"{'篮子胜率':>9s} {'股胜率':>8s} {'平均持仓':>9s}", flush=True)
    for a, name in arms.items():
        d = desc[a]
        if not d:
            continue
        print(f"  {a}:{name:21s} {d['gross_mean_per_trade_pct']:>+11.4f} "
              f"{d['net_mean_per_trade_pct']:>+11.4f} {d['win_rate_basket']:>9.1%} "
              f"{d['win_rate_stock']:>8.1%} {d['avg_n_stocks']:>9.1f}", flush=True)
    da = desc["A"]["net_mean_per_trade_pct"] - desc["B"]["net_mean_per_trade_pct"]
    print(f"\n  Arm_A − Arm_B 净收益/笔 = {da:+.4f}pp (席位技能净增量初窥, gate 在 SL-003)", flush=True)
    print(f"  年化换手 ≈ {desc['A']['ann_roundtrips_full_invest']} round-trips → "
          f"成本拖累 ≈ {desc['A']['ann_cost_drag_pct']}%/yr (短线 sleeve 核心约束, SL-002 详查)", flush=True)

    verdict = {
        "id": "SL-001", "status": "built",
        "conclusion": (
            f"r5 龙虎榜席位事件 sleeve 回测引擎已建 (D 龙虎榜→D+1 开盘建仓→D+1+{H_HOLD} 开盘平仓, "
            f"三臂 apples: M=等权全龙虎榜 / A=席位战绩(sf_edge_r5)top{TOP_N} / B=同集动量top{TOP_N}; "
            f"主测 cost=30bps round-trip)。{picks['entry_date'].nunique()} entry 日 "
            f"({picks['entry_date'].min()}~{picks['entry_date'].max()}), "
            f"全龙虎榜候选 {len(picks):,} (席位可信 {picks['has_seat'].mean():.1%}); "
            f"毛收益/笔 M={desc['M']['gross_mean_per_trade_pct']:+.3f}% "
            f"A={desc['A']['gross_mean_per_trade_pct']:+.3f}% B={desc['B']['gross_mean_per_trade_pct']:+.3f}%, "
            f"净收益/笔 A={desc['A']['net_mean_per_trade_pct']:+.3f}% (cost 吃 {COST_RT*100:.2f}pp), "
            f"Arm_A−Arm_B 净/笔 {da:+.3f}pp; 年化换手≈{desc['A']['ann_roundtrips_full_invest']}轮→"
            f"成本拖累≈{desc['A']['ann_cost_drag_pct']}%/yr。中间描述统计 ≠ 落地 (SIGN-R03), "
            f"净 α walk-forward 裁决在 SL-003。"),
        "artifact": str(DAILY_PNL.relative_to(ROOT)),
        "design": {
            "event_driven": f"D 龙虎榜机构净买入(∑net_buy>0)→entry=next(D) 开盘→exit=entry+{H_HOLD} 开盘",
            "arms": {
                "Arm_M": "当日全部龙虎榜净买入股等权 (基线='无脑买龙虎榜')",
                "Arm_A": f"席位可信候选内按 {SEAT_SORT} 降序 ({SEAT_TIE} tiebreak) top-{TOP_N} 等权",
                "Arm_B": f"同一席位可信候选内按 mom{MOM_WIN} 降序 top-{TOP_N} 等权 (动量对照)",
                "isolation": "Arm_A vs Arm_B 唯一差异=排序键(席位战绩 vs 动量)→净增量=纯席位技能 (SL-003 gate)",
            },
            "cost_main": f"round-trip {COST_RT*100:.2f}% (佣金+印花税0.05%卖+滑点); net=gross−cost",
            "causal": "席位战绩 avail<=D 信号 D+1 生效 (SEAT-001); 动量纯过去; 价格 D+1 开盘建/平仓; ST 源头排除",
            "frozen_constants": {"H_HOLD": H_HOLD, "TOP_N": TOP_N, "COST_RT_bps": int(COST_RT * 10000),
                                 "MOM_WIN": MOM_WIN, "SEAT_SORT": SEAT_SORT},
        },
        "metrics": {
            "n_entry_days": int(picks["entry_date"].nunique()),
            "date_range": [picks["entry_date"].min(), picks["entry_date"].max()],
            "n_candidates_full": int(len(picks)),
            "seat_trusted_coverage": round(float(picks["has_seat"].mean()), 4),
            "arms": desc,
            "arm_A_minus_B_net_per_trade_pp": round(da, 4),
        },
        "artifacts": [str(DAILY_PNL.relative_to(ROOT)), str(PICKS_CACHE.relative_to(ROOT))],
        "guardrails": ["SIGN-R04 因果(席位D+1/动量纯过去/价格D+1建平仓)", "SIGN-R06 ST 源头排除",
                       "SIGN-R08 px/cand/picks checkpoint", "SIGN-R03 描述统计≠落地, 裁决在 SL-003 walk-forward"],
    }
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] status=built -> {VERDICT.relative_to(ROOT)}", flush=True)
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
