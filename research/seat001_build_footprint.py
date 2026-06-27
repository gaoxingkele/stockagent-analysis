# -*- coding: utf-8 -*-
"""SEAT-001 — 龙虎榜席位印记特征 (top_inst exalter 历史 expanding 胜率 → 聪明席位跟随).

行为 alpha: 某些游资/机构席位 (exalter) 的"买入"长期跑赢。对每只当日上榜股,
按当日**买方席位**(net_buy>0) 的**历史战绩**(过去 expanding 净买入后股票表现) 加权,
得到"聪明席位跟随"信号 → 预测该股 D+1 起的表现。

因果设计 (SIGN-R04 零泄漏, playbook 数据::龙虎榜滞后+席位胜率):
  - 龙虎榜在交易日 d **收盘后**公告, 故信号在**下一交易日** t=next(d) 才生效 (feature@t 只含 ≤d 信息)。
  - 席位战绩只用**过去已兑现**的事件: 席位在历史某日 e 净买入某股, 其 fwd_r{H} 在 e+H 收盘后才可知,
    该事件的战绩仅当 avail(e)=idx(e)+H <= idx(d) 时才计入 (严格已实现, 防同期/未来泄漏)。
  - 当前事件自身的 outcome avail=idx(d)+H > idx(d), 天然被排除 (时间维 leave-one-out)。
  - 席位战绩跨股累计 (席位技能是跨标的的), 但当前股当前事件的 outcome 不入。

纪律:
  - ST 源头排除 (SIGN-R06): 上榜股 + 席位历史标的两侧都剔 ST。
  - fwd_r5/r20 仅作席位**历史** outcome, 落 research/cache, **绝不**写入 features parquet
    (verify.py forward 黑名单)。输出仅 sf_* (backward 战绩聚合)。
  - 长任务 checkpoint (SIGN-R08): fwd 收益 + 事件边际 缓存 research/cache/seat001/。
  - 确定性 (SIGN): 抽样席位重算逐位比对。

输出:
  research/features/seat_footprint.parquet (key: ts_code, trade_date; 列 sf_*)
  research/verdicts/SEAT-001.json (status=built + 席位数/覆盖)
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
TOPINST_P = ROOT / "output" / "tushare_cache" / "top_inst.parquet"
CACHE_DIR = ROOT / "research" / "cache" / "seat001"
FWD_CACHE = CACHE_DIR / "fwd_returns.parquet"      # 仅 cache, 含 forward, 不进 features
EVENT_CACHE = CACHE_DIR / "seat_events_edge.parquet"
OUT_DIR = ROOT / "research" / "features"
OUT_P = OUT_DIR / "seat_footprint.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "SEAT-001.json"

KEY = ["ts_code", "trade_date"]
HORIZONS = [5, 20]      # 席位历史 outcome 窗口 (短线为主看 r5, 兼看 r20)
MIN_HIST = 5            # 席位被信任所需的已兑现历史事件数下限

# forward 黑名单 (与 verify.py 一致), 落盘前自查 (输出列绝不命中)
FORWARD_TOKENS = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                 {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                 {"max_gain", "maxgain", "future_ret", "fwd_ret"}


def load_st_set() -> set:
    if not BASIC_P.exists():
        print("  ⚠️ stock_basic.parquet 不存在, ST 排除跳过", flush=True)
        return set()
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def load_daily_close(st: set) -> pd.DataFrame:
    print(f"[daily] 加载全历史 close ({DAILY_DIR}) ...", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(DAILY_DIR.glob("*.parquet"))]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.drop_duplicates(subset=KEY, keep="last")
    big = big[big["close"].notna() & (big["close"] > 0)]
    if st:
        before = len(big)
        big = big[~big["ts_code"].isin(st)]
        print(f"  ST 源头排除: {before - len(big):,} 行 ({len(st)} 只 ST)", flush=True)
    return big.sort_values(KEY).reset_index(drop=True)


def build_fwd_returns(close: pd.DataFrame) -> pd.DataFrame:
    """每股 forward r{H} = close[t+H]/close[t]-1 (因果 shift(-H)); 仅 cache 不进 features。"""
    if FWD_CACHE.exists():
        print(f"[fwd] checkpoint 命中 {FWD_CACHE.name}", flush=True)
        return pd.read_parquet(FWD_CACHE)
    df = close.sort_values(KEY).reset_index(drop=True)
    g = df.groupby("ts_code")["close"]
    for h in HORIZONS:
        df[f"fr{h}"] = g.transform(lambda s: s.shift(-h) / s - 1.0)
    out = df[["ts_code", "trade_date"] + [f"fr{h}" for h in HORIZONS]]
    FWD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FWD_CACHE, index=False)
    print(f"[fwd] 落盘 {FWD_CACHE.name} ({len(out):,} 行)", flush=True)
    return out


def load_events(st: set, valid_codes: set) -> pd.DataFrame:
    """top_inst → 每 (date, ts_code, exalter) 一行 net_buy (跨 side 去重)。"""
    ti = pd.read_parquet(TOPINST_P)[["trade_date", "ts_code", "exalter", "net_buy"]]
    ti["trade_date"] = ti["trade_date"].astype(str)
    ti = ti.dropna(subset=["exalter", "net_buy"])
    # 同一 (date,code,exalter) 在 side 0/1 重复出现 (net_buy 一致) → 去重
    ti = ti.drop_duplicates(subset=["trade_date", "ts_code", "exalter"], keep="first")
    if st:
        ti = ti[~ti["ts_code"].isin(st)]
    ti = ti[ti["ts_code"].isin(valid_codes)].reset_index(drop=True)
    print(f"[events] {len(ti):,} 席位-股-日事件 / {ti['exalter'].nunique()} 席位 / "
          f"{ti['ts_code'].nunique()} 股 / {ti['trade_date'].nunique()} 日 "
          f"({ti['trade_date'].min()}~{ti['trade_date'].max()})", flush=True)
    return ti


def compute_seat_edge(ev: pd.DataFrame, fwd: pd.DataFrame,
                      cal_idx: dict) -> pd.DataFrame:
    """对每个**买方席位事件** (net_buy>0) 计算该席位在事件日 d 的历史战绩 (expanding, 因果)。

    席位 s 的战绩仅累计 avail(e)=idx(e)+H <= idx(d) 的已兑现历史 outcome。
    返回每个买方席位事件一行: date, date_idx, ts_code, exalter, net_buy, sf_edge_*, sf_win5, n_hist*。
    """
    if EVENT_CACHE.exists():
        print(f"[edge] checkpoint 命中 {EVENT_CACHE.name}", flush=True)
        return pd.read_parquet(EVENT_CACHE)

    # 只对买方席位 (net_buy>0) 建信号 (聪明席位"跟买")
    ev = ev[ev["net_buy"] > 0].copy()
    ev["date_idx"] = ev["trade_date"].map(cal_idx)
    ev = ev.dropna(subset=["date_idx"])
    ev["date_idx"] = ev["date_idx"].astype(int)

    # 事件 outcome: 该股从事件日 d 的 fwd_r{H} (席位"买对没"); NaN=尚未兑现, 不入历史
    ev = ev.merge(fwd, on=KEY, how="left")

    t0 = time.time()
    cols = {"sf_edge_r5": [], "sf_edge_r20": [], "sf_win5": [],
            "n_hist5": [], "n_hist20": []}
    # 输出顺序与 ev 行对齐
    out_idx, e_edge5, e_edge20, e_win5, e_nh5, e_nh20 = [], [], [], [], [], []

    for seat, g in ev.groupby("exalter", sort=False):
        di = g["date_idx"].to_numpy()
        o5 = g["fr5"].to_numpy(float)
        o20 = g["fr20"].to_numpy(float)
        gi = g.index.to_numpy()

        # r5 战绩 (avail = date_idx + 5)
        m5 = ~np.isnan(o5)
        if m5.any():
            av5 = di[m5] + 5
            ov5 = o5[m5]
            wv5 = (ov5 > 0).astype(float)
            order = np.argsort(av5, kind="stable")
            av5s, ov5s, wv5s = av5[order], ov5[order], wv5[order]
            cum_o5 = np.concatenate([[0.0], np.cumsum(ov5s)])
            cum_w5 = np.concatenate([[0.0], np.cumsum(wv5s)])
            k5 = np.searchsorted(av5s, di, side="right")
            edge5 = np.where(k5 >= MIN_HIST, cum_o5[k5] / np.maximum(k5, 1), np.nan)
            win5 = np.where(k5 >= MIN_HIST, cum_w5[k5] / np.maximum(k5, 1), np.nan)
        else:
            k5 = np.zeros(len(di), dtype=int)
            edge5 = np.full(len(di), np.nan)
            win5 = np.full(len(di), np.nan)

        # r20 战绩 (avail = date_idx + 20)
        m20 = ~np.isnan(o20)
        if m20.any():
            av20 = di[m20] + 20
            ov20 = o20[m20]
            order = np.argsort(av20, kind="stable")
            av20s, ov20s = av20[order], ov20[order]
            cum_o20 = np.concatenate([[0.0], np.cumsum(ov20s)])
            k20 = np.searchsorted(av20s, di, side="right")
            edge20 = np.where(k20 >= MIN_HIST, cum_o20[k20] / np.maximum(k20, 1), np.nan)
        else:
            k20 = np.zeros(len(di), dtype=int)
            edge20 = np.full(len(di), np.nan)

        out_idx.append(gi)
        e_edge5.append(edge5); e_edge20.append(edge20); e_win5.append(win5)
        e_nh5.append(k5); e_nh20.append(k20)

    res = pd.DataFrame({
        "sf_edge_r5": np.concatenate(e_edge5),
        "sf_edge_r20": np.concatenate(e_edge20),
        "sf_win5": np.concatenate(e_win5),
        "n_hist5": np.concatenate(e_nh5),
        "n_hist20": np.concatenate(e_nh20),
    }, index=np.concatenate(out_idx)).sort_index()
    ev = ev.join(res)
    print(f"[edge] {len(ev):,} 买方席位事件战绩算完 ({time.time()-t0:.0f}s)", flush=True)
    keep = ["trade_date", "date_idx", "ts_code", "exalter", "net_buy",
            "sf_edge_r5", "sf_edge_r20", "sf_win5", "n_hist5", "n_hist20"]
    ev = ev[keep]
    EVENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(EVENT_CACHE, index=False)
    return ev


def aggregate_stock_day(ev: pd.DataFrame, next_map: dict) -> pd.DataFrame:
    """每 (d, ts_code): 买方席位历史战绩 net_buy 加权 → 个股聪明席位跟随信号, 赋 next(d)。"""
    e5 = ev[ev["sf_edge_r5"].notna()].copy()
    e5["w"] = e5["net_buy"].clip(lower=0)
    e5["we5"] = e5["w"] * e5["sf_edge_r5"]
    e5["ww5"] = e5["w"] * e5["sf_win5"]
    g5 = e5.groupby(["trade_date", "ts_code"]).agg(
        wsum=("w", "sum"), we5=("we5", "sum"), ww5=("ww5", "sum"),
        n_seats=("exalter", "nunique"), seat_nhist=("n_hist5", "mean")).reset_index()
    g5["sf_edge_r5"] = g5["we5"] / g5["wsum"]
    g5["sf_winrate_r5"] = g5["ww5"] / g5["wsum"]
    g5["sf_n_seats"] = g5["n_seats"].astype(int)
    g5["sf_seat_nhist"] = g5["seat_nhist"]

    e20 = ev[ev["sf_edge_r20"].notna()].copy()
    e20["w"] = e20["net_buy"].clip(lower=0)
    e20["we20"] = e20["w"] * e20["sf_edge_r20"]
    g20 = e20.groupby(["trade_date", "ts_code"]).agg(
        wsum=("w", "sum"), we20=("we20", "sum")).reset_index()
    g20["sf_edge_r20"] = g20["we20"] / g20["wsum"]

    out = g5.merge(g20[["trade_date", "ts_code", "sf_edge_r20"]],
                   on=["trade_date", "ts_code"], how="outer")
    # 事件日 d → 信号生效日 next(d) (D+1 防泄漏)
    out["signal_date"] = out["trade_date"].map(next_map)
    out = out.dropna(subset=["signal_date"])
    out = out.rename(columns={"signal_date": "_t"})
    out["trade_date"] = out["_t"]
    feat = ["sf_edge_r5", "sf_edge_r20", "sf_winrate_r5", "sf_n_seats", "sf_seat_nhist"]
    out = out[KEY + feat].sort_values(KEY).reset_index(drop=True)
    return out


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    st = load_st_set()
    print(f"[st] ST 源头排除集合 {len(st)} 只", flush=True)
    close = load_daily_close(st)
    cal = sorted(close["trade_date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}
    next_map = {cal[i]: cal[i + 1] for i in range(len(cal) - 1)}
    print(f"[cal] {len(cal)} 交易日 ({cal[0]}~{cal[-1]})", flush=True)

    fwd = build_fwd_returns(close)
    ev = load_events(st, set(close["ts_code"].unique()))
    edge = compute_seat_edge(ev, fwd, cal_idx)
    out = aggregate_stock_day(edge, next_map)

    feat_cols = [c for c in out.columns if c.startswith("sf_")]

    # ── 泄漏自查 (SIGN-R04): 输出列绝不命中 forward 黑名单 ──
    bad = [c for c in out.columns if c.lower() in FORWARD_TOKENS
           or any(__import__("re").search(p, c.lower()) for p in
                  [r"^r\d+_next", r"^fwd_", r"^future_", r"^next_", r"_forward$",
                   r"^label$", r"^target$"])]
    assert not bad, f"输出含 forward 字段 (泄漏): {bad}"
    assert all(c.startswith("sf_") for c in feat_cols), "存在非 sf_ 特征列"

    # ── 确定性自检: 随机抽一席位重算 r5 战绩逐位比对 ──
    seat_sample = edge["exalter"].dropna().iloc[len(edge) // 2]
    sub = ev[(ev["net_buy"] > 0) & (ev["exalter"] == seat_sample)].copy()
    sub["date_idx"] = sub["trade_date"].map(cal_idx)
    sub = sub.dropna(subset=["date_idx"]).merge(fwd, on=KEY, how="left")
    di = sub["date_idx"].astype(int).to_numpy()
    o5 = sub["fr5"].to_numpy(float)
    m5 = ~np.isnan(o5)
    deterministic = True
    if m5.any():
        av5 = di[m5] + 5
        ov5 = o5[m5]
        order = np.argsort(av5, kind="stable")
        cum = np.concatenate([[0.0], np.cumsum(ov5[order])])
        k = np.searchsorted(av5[order], di, side="right")
        re5 = np.where(k >= MIN_HIST, cum[k] / np.maximum(k, 1), np.nan)
        ref = edge[edge["exalter"] == seat_sample].sort_values("date_idx")["sf_edge_r5"].to_numpy()
        cur = pd.Series(re5, index=sub["date_idx"].astype(int)).sort_index().to_numpy()
        deterministic = bool(np.allclose(np.nan_to_num(ref, nan=-9),
                                         np.nan_to_num(cur, nan=-9), atol=1e-9))

    out.to_parquet(OUT_P, index=False)
    print(f"[out] -> {OUT_P.relative_to(ROOT)}  "
          f"({len(out):,} 行 × {len(out.columns)} 列)", flush=True)

    # ── 覆盖/分布 ──
    cov_stocks = int(out["ts_code"].nunique())
    cov_dates = int(out["trade_date"].nunique())
    n_seats_total = int(edge["exalter"].nunique())
    n_seats_trusted = int(edge[edge["n_hist5"] >= MIN_HIST]["exalter"].nunique())
    nonnull = {c: round(float(out[c].notna().mean()), 4) for c in feat_cols}
    desc = {c: {"mean": round(float(out[c].mean()), 6),
                "std": round(float(out[c].std()), 6),
                "p5": round(float(out[c].quantile(0.05)), 6),
                "p95": round(float(out[c].quantile(0.95)), 6)}
            for c in ["sf_edge_r5", "sf_edge_r20", "sf_winrate_r5"]}

    verdict = {
        "task": "SEAT-001",
        "status": "built",
        "conclusion": (
            f"龙虎榜席位印记特征落盘 {len(out):,} 行 × {len(feat_cols)} 特征 "
            f"({out['trade_date'].min()}~{out['trade_date'].max()}); "
            f"覆盖 {cov_stocks} 股/{cov_dates} 日; {n_seats_total} 买方席位 "
            f"({n_seats_trusted} 有 ≥{MIN_HIST} 兑现历史可信); "
            f"因果: 席位战绩仅用 avail<=d 已兑现 outcome (跨股 expanding), "
            f"信号赋 next(d)=D+1, 零泄漏, 确定性={deterministic}; ST 源头排除。"),
        "artifact": str(OUT_P.relative_to(ROOT)),
        "design": {
            "seat_track_record": (
                "每买方席位 (net_buy>0) 的历史 expanding 战绩 = 过去净买入事件的 fwd_r5/r20 均值 "
                "(+r5 胜率), 仅计 avail(e)=idx(e)+H<=idx(d) 已兑现的, 跨股累计, 当前事件自身排除。"),
            "stock_day_signal": (
                "每上榜股当日买方席位战绩按 net_buy 加权 → sf_edge_r5/r20 + sf_winrate_r5; "
                "信号赋下一交易日 (龙虎榜收盘后公告, D+1 生效)。"),
            "min_hist": MIN_HIST, "horizons": HORIZONS,
        },
        "metrics": {
            "rows": int(len(out)),
            "n_stocks": cov_stocks,
            "n_dates": cov_dates,
            "date_range": [out["trade_date"].min(), out["trade_date"].max()],
            "feature_cols": feat_cols,
            "n_seats_total": n_seats_total,
            "n_seats_trusted": n_seats_trusted,
            "nonnull_frac": nonnull,
            "dist": desc,
            "deterministic": deterministic,
            "leakage_free": True,
        },
        "guardrails": ["SIGN-R04 泄漏自查通过 + 时间维 leave-one-out", "SIGN-R06 ST 源头排除",
                       "SIGN-R08 fwd/edge checkpoint", "因果: 席位战绩 avail<=d + 信号 D+1 生效"],
    }
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"  覆盖 {cov_stocks} 股 / {cov_dates} 日, 席位 {n_seats_total} (可信 {n_seats_trusted})")
    print(f"  非空率 {nonnull}")
    print(f"  分布 {desc}")
    print(f"[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
