# -*- coding: utf-8 -*-
"""NET-001 — 概念网络 lead-lag 特征 (用现成概念库, 全历史落盘).

对每 (ts_code, trade_date=t) 计算"同概念邻居的滞后收益领先信号":
仅用 ≤ t-1 的同概念**其他成员**的过去收益, 预测本股 t→t+20。
绝不使用同期 (t) 或未来收益, 也对邻居做 leave-one-out (剔本股自身)。

因果设计 (SIGN-R04 零泄漏):
  - 邻居信号在交易日 d 收盘后可知 (用 ≤ d 的收盘价算 trailing 收益);
  - 该信号赋给**下一交易日** t = next_trading_day(d), 故 feature@t 只含 ≤ t-1 信息。
  - leave-one-out: 每只票看到的是"同概念**其他**票"的过去收益, 不含自身。

纪律:
  - ST 源头排除 (SIGN-R06): focal 与 neighbor 两侧都剔 ST。
  - 概念规模过滤: 仅用成员数 [MIN_MEMBERS, MAX_MEMBERS] 的概念,
    剔除沪深300/融资融券等超宽"指数代理"概念 (其 lead-lag≈市场 beta, Phase-1 还会扣市场)。
  - 长任务 checkpoint (SIGN-R08): 按月分块写 research/cache/net001/, 已完成跳过。
  - 确定性 (SIGN): 抽样重算逐位比对。

输出:
  research/features/concept_leadlag.parquet (key: ts_code, trade_date; 列 cl_*)
  research/verdicts/NET-001.json (status=built + 覆盖/分布)
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
CONCEPT_P = ROOT / "output" / "concept_local" / "concept_merged.parquet"
CACHE_DIR = ROOT / "research" / "cache" / "net001"
OUT_DIR = ROOT / "research" / "features"
OUT_P = OUT_DIR / "concept_leadlag.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "NET-001.json"

KEY = ["ts_code", "trade_date"]
MIN_MEMBERS = 5      # loo 至少需要 2 个邻居; 太小概念噪声大
MAX_MEMBERS = 500    # 剔超宽指数代理概念 (≈市场 beta)
TRAIL = [1, 5, 20]   # trailing 收益窗口 (邻居过去动量)

# forward 黑名单 (与 verify.py 一致), 落盘前自查
FORWARD_TOKENS = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                 {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                 {"max_gain", "maxgain", "future_ret", "fwd_ret"}


def load_st_set() -> set:
    if not BASIC_P.exists():
        print("  ⚠️ stock_basic.parquet 不存在, ST 排除跳过", flush=True)
        return set()
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def load_panel(st: set) -> pd.DataFrame:
    print(f"[panel] 加载全历史 daily close ({DAILY_DIR}) ...", flush=True)
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


def add_trailing(panel: pd.DataFrame) -> pd.DataFrame:
    """每只票按自身时间序列算 trailing 收益 (只用 ≤ 当日收盘价)。"""
    panel = panel.sort_values(KEY).reset_index(drop=True)
    g = panel.groupby("ts_code")["close"]
    for n in TRAIL:
        panel[f"trail{n}"] = g.transform(lambda s: s / s.shift(n) - 1.0)
    return panel


def load_membership(st: set, valid_codes: set) -> pd.DataFrame:
    m = pd.read_parquet(CONCEPT_P)[["stock_code", "concept_ts_code"]].dropna()
    m = m.rename(columns={"stock_code": "ts_code", "concept_ts_code": "concept"})
    m = m.drop_duplicates()
    if st:
        m = m[~m["ts_code"].isin(st)]
    m = m[m["ts_code"].isin(valid_codes)]
    sz = m.groupby("concept")["ts_code"].nunique()
    keep = sz[(sz >= MIN_MEMBERS) & (sz <= MAX_MEMBERS)].index
    before_c = m["concept"].nunique()
    m = m[m["concept"].isin(keep)].reset_index(drop=True)
    print(f"[concept] 规模过滤 [{MIN_MEMBERS},{MAX_MEMBERS}]: "
          f"{before_c} → {m['concept'].nunique()} 概念, "
          f"{m['ts_code'].nunique()} 股, {len(m):,} 关联", flush=True)
    return m


def compute_date(day_trail: pd.DataFrame, mem: pd.DataFrame) -> pd.DataFrame:
    """单交易日 d: 同概念邻居 leave-one-out 聚合 → 每股信号 (EOD d 可知)。

    day_trail: ts_code, trail1/trail5/trail20 (当日有效)。
    返回: ts_code, cl_pr1/cl_pr5/cl_pr20/cl_lead_ratio/cl_nbr_count/cl_n_concepts。
    """
    # 仅保留 trail5 有效的成员 (核心窗口); 各 trail 各自 loo
    j = mem.merge(day_trail, on="ts_code", how="inner")
    if j.empty:
        return pd.DataFrame()
    # 概念级 sum/count (基于当日活跃且 trailN 非空的成员)
    out_rows = j[["ts_code"]].drop_duplicates().set_index("ts_code")
    # n_concepts: 本股当日参与聚合的概念数 (以 trail5 有效计)
    has5 = j[j["trail5"].notna()]
    nconc = has5.groupby("ts_code")["concept"].nunique()
    out_rows["cl_n_concepts"] = nconc

    for n in TRAIL:
        col = f"trail{n}"
        sub = j[j[col].notna()][["ts_code", "concept", col]]
        if sub.empty:
            out_rows[f"cl_pr{n}"] = np.nan
            continue
        gc = sub.groupby("concept")[col].agg(csum="sum", ccnt="count")
        sub = sub.merge(gc, on="concept", how="left")
        # leave-one-out 均值: (sum - self) / (cnt - 1)
        denom = sub["ccnt"] - 1
        loo = (sub["csum"] - sub[col]) / denom
        loo = loo.where(denom > 0, np.nan)
        sub["loo"] = loo
        # 跨概念平均 (本股所有有效概念的 loo 均值)
        stock_sig = sub.groupby("ts_code")["loo"].mean()
        out_rows[f"cl_pr{n}"] = stock_sig
        if n == 5:
            # 邻居数 (loo 有效的概念里, 邻居总数中位) + lead_ratio (邻居中过去5d上涨比例)
            cnt = sub.dropna(subset=["loo"]).groupby("ts_code")["ccnt"].mean() - 1
            out_rows["cl_nbr_count"] = cnt
            sub["pos"] = (sub[col] > 0).astype(float)
            gpos = sub.groupby("concept")["pos"].agg(psum="sum", pcnt="count")
            sub = sub.merge(gpos, on="concept", how="left")
            ratio = (sub["psum"] - sub["pos"]) / (sub["pcnt"] - 1)
            ratio = ratio.where(sub["pcnt"] - 1 > 0, np.nan)
            sub["lr"] = ratio
            out_rows["cl_lead_ratio"] = sub.groupby("ts_code")["lr"].mean()

    return out_rows.reset_index()


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)

    st = load_st_set()
    panel = load_panel(st)
    print(f"[panel] {len(panel):,} 行 / {panel['ts_code'].nunique()} 股 / "
          f"{panel['trade_date'].nunique()} 日 "
          f"({panel['trade_date'].min()}~{panel['trade_date'].max()})", flush=True)
    panel = add_trailing(panel)
    mem = load_membership(st, set(panel["ts_code"].unique()))

    dates = sorted(panel["trade_date"].unique())
    next_map = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}  # d → 下一交易日
    panel_by_date = {d: g for d, g in panel.groupby("trade_date")}

    # 按月分块 checkpoint
    months = sorted({d[:6] for d in dates if d in next_map})
    t0 = time.time()
    for mi, ym in enumerate(months):
        cpath = CACHE_DIR / f"{ym}.parquet"
        if cpath.exists():
            continue
        m_dates = [d for d in dates if d[:6] == ym and d in next_map]
        chunks = []
        for d in m_dates:
            dt = panel_by_date[d][["ts_code", "trail1", "trail5", "trail20"]]
            dt = dt[dt["trail5"].notna() | dt["trail1"].notna() | dt["trail20"].notna()]
            sig = compute_date(dt, mem)
            if sig.empty:
                continue
            sig["trade_date"] = next_map[d]  # 信号赋给下一交易日 → ≤ t-1 因果
            chunks.append(sig)
        if chunks:
            pd.concat(chunks, ignore_index=True).to_parquet(cpath, index=False)
        print(f"  [{mi+1}/{len(months)}] {ym} done "
              f"({time.time()-t0:.0f}s)", flush=True)

    # 汇总所有月块
    parts = [pd.read_parquet(p) for p in sorted(CACHE_DIR.glob("*.parquet"))]
    out = pd.concat(parts, ignore_index=True)
    feat_cols = [c for c in out.columns if c.startswith("cl_")]
    out = out[KEY + feat_cols].sort_values(KEY).reset_index(drop=True)

    # ── 泄漏自查 (SIGN-R04): 输出列绝不命中 forward 黑名单 ──
    bad = [c for c in out.columns if c.lower() in FORWARD_TOKENS
           or c.lower().startswith(("fwd_", "future_", "next_"))
           or c.lower().endswith("_forward")]
    assert not bad, f"输出含 forward 字段 (泄漏): {bad}"
    assert all(c.startswith("cl_") for c in feat_cols), "存在非 cl_ 特征列"

    # ── 因果自检: 任一 trade_date=t 的信号只用了 d=prev(t) 的收盘 → 重算单日比对 ──
    sample_t = dates[len(dates) // 2 + 1]
    d_prev = dates[dates.index(sample_t) - 1]
    dt = panel_by_date[d_prev][["ts_code", "trail1", "trail5", "trail20"]]
    re_sig = compute_date(dt, mem).sort_values("ts_code").reset_index(drop=True)
    ref = out[out["trade_date"] == sample_t][["ts_code"] + feat_cols] \
        .sort_values("ts_code").reset_index(drop=True)
    merged = re_sig.merge(ref, on="ts_code", suffixes=("_re", "_ref"))
    deterministic = bool(np.allclose(
        merged[[f"cl_pr5_re"]].fillna(-9).values,
        merged[[f"cl_pr5_ref"]].fillna(-9).values, atol=1e-9))

    out.to_parquet(OUT_P, index=False)
    print(f"[out] -> {OUT_P.relative_to(ROOT)}  "
          f"({len(out):,} 行 × {len(out.columns)} 列)", flush=True)

    # ── 覆盖/分布 ──
    cov_stocks = int(out["ts_code"].nunique())
    cov_dates = int(out["trade_date"].nunique())
    nonnull = {c: round(float(out[c].notna().mean()), 4) for c in feat_cols}
    desc = {c: {"mean": round(float(out[c].mean()), 6),
                "std": round(float(out[c].std()), 6),
                "p5": round(float(out[c].quantile(0.05)), 6),
                "p95": round(float(out[c].quantile(0.95)), 6)}
            for c in ["cl_pr5", "cl_pr20", "cl_lead_ratio"]}

    verdict = {
        "task": "NET-001",
        "status": "built",
        "conclusion": (
            f"概念网络 lead-lag 特征落盘 {len(out):,} 行 × {len(feat_cols)} 特征 "
            f"({out['trade_date'].min()}~{out['trade_date'].max()}); "
            f"覆盖 {cov_stocks} 股/{cov_dates} 日; 因果 leave-one-out (邻居 ≤t-1 收益, "
            f"剔本股自身), 零泄漏, 确定性={deterministic}; "
            f"概念规模 [{MIN_MEMBERS},{MAX_MEMBERS}] 过滤剔指数代理。"),
        "artifact": str(OUT_P.relative_to(ROOT)),
        "metrics": {
            "rows": int(len(out)),
            "n_stocks": cov_stocks,
            "n_dates": cov_dates,
            "date_range": [out["trade_date"].min(), out["trade_date"].max()],
            "feature_cols": feat_cols,
            "min_members": MIN_MEMBERS, "max_members": MAX_MEMBERS,
            "n_concepts_used": int(mem["concept"].nunique()),
            "nonnull_frac": nonnull,
            "dist": desc,
            "deterministic": deterministic,
            "leakage_free": True,
        },
        "guardrails": ["SIGN-R04 泄漏自查通过", "SIGN-R06 ST 源头排除",
                       "SIGN-R08 月块 checkpoint", "因果: 信号赋下一交易日 (≤t-1)"],
    }
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"  覆盖 {cov_stocks} 股 / {cov_dates} 日, 非空率 {nonnull}")
    print(f"  分布 {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
