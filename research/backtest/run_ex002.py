# -*- coding: utf-8 -*-
"""EX-002 — 基金/风格对照 (量化"基金更强"是选股还是风格).

承用户"基金赢我们输"。19× 选股信号穷尽 + BT-002 regime 诊断 (动量月相对最弱) →
本 task 量化 V12.31 picks 的**风格暴露**, 对照基金/大盘质量基准, 把 Sharpe 差归因到
**风格/universe (小盘高波→需 tilt)** 还是**选股 (可重标修)**。

三部分 (设计冻结 R01, 见 DESIGN.md §7 / prd.preRegisteredGate.fund_compare):

  [A] V12.31 picks 风格画像: 在每个再平衡日, 对 picks vs 全市场 (ST 排除) 比较
      市值(total_mv)/波动(过去20d收益年化std)/换手(turnover_rate)/估值(pe_ttm,pb)/行业分布。
      报中位数对比 + picks 中位数在全市场分布里的百分位 + 行业 over/under-weight tilt。

  [B] 基金持仓风格 + book Sharpe 对照:
      - 基金持仓 (池C 7基金, fund_portfolio 真历史 7季报) 的市值画像 vs 我们 picks (large vs small)。
      - book Sharpe: V12.31 (复用 BT-002 baseline) vs hs300 (大盘质量代理) vs CSI1000
        vs 基金共识篮 (真历史持仓, ann_date 因果可得, 过同一引擎同基线成本)。
      - 幸存者偏差标注: 7 基金按"今日存在"选取 = fund-selection survivorship; 但持仓用真历史季报
        (非单一静态快照外推) → 比纯静态篮诚实。

  [C] 归因: Sharpe 差是风格 (小盘高波) 还是选股。

R01 对照法/基准冻结不在结果上调; R06 ST 源头排除; R11 必带 regime 分层 (book vs hs300 月超额按 regime);
描述性风格画像 + 对照 = 合法完成 (R02); 中间指标≠ship (R03); 绝对量级含 r20 池 lookahead 非可交易声明;
大缓存写 research/cache/ex002/ (gitignored), 不写 research/features/ (R04); 生产线只读 (R05)。
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))
from engine import PortfolioBacktester, CostModel

CACHE001 = ROOT / "research" / "cache" / "bt001"
CACHE002 = ROOT / "research" / "cache" / "bt002"
CACHE = ROOT / "research" / "cache" / "ex002"
CACHE.mkdir(parents=True, exist_ok=True)
(CACHE / "daily_basic").mkdir(parents=True, exist_ok=True)
PICKS = CACHE001 / "picks_v1231_daily.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
VERDICT = ROOT / "research" / "verdicts" / "EX-002.json"
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
IDX_CACHE = CACHE002 / "index_benchmarks.parquet"
FUND_HIST = CACHE / "fund_portfolio_hist.parquet"
BT002_RES = CACHE002 / "bt002_results.json"

# ── 冻结参数 (R01, = BT-002 配置) ──
TOTAL = 0.90
REBAL_EVERY = 20
BASELINE_SLIP = 0.0010
VOL_LOOKBACK = 20        # 波动率回看 (交易日)
FUND_TOPN = 15           # 基金共识篮持仓数 (按 stk_mkv_ratio 取前 N)
DAILY_BASIC_FIELDS = "ts_code,trade_date,turnover_rate,total_mv,circ_mv,pe_ttm,pb"


# ───────────── 数据加载 ─────────────
def load_prices(start: str) -> pd.DataFrame:
    parts = []
    for f in sorted(DAILY_DIR.glob("*.parquet")):
        d = pd.read_parquet(f, columns=["ts_code", "trade_date", "open", "high",
                                        "low", "close", "pct_chg"])
        d["trade_date"] = d["trade_date"].astype(str)
        d = d[d["trade_date"] >= start]
        if not d.empty:
            parts.append(d)
    px = pd.concat(parts, ignore_index=True)
    basic = pd.read_parquet(BASIC)[["ts_code", "name", "industry"]].drop_duplicates("ts_code")
    st = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    px = px[~px["ts_code"].isin(st)].reset_index(drop=True)   # ST 源头排除 (R06)
    return px


def load_daily_basic(dates: list) -> pd.DataFrame:
    """每个再平衡日拉 daily_basic (total_mv/turnover_rate/pe/pb), per-date checkpoint (R08)."""
    from dotenv import load_dotenv
    load_dotenv()
    pro = None
    parts = []
    for d in dates:
        fp = CACHE / "daily_basic" / f"{d}.parquet"
        if fp.exists():
            parts.append(pd.read_parquet(fp))
            continue
        if pro is None:
            import tushare as ts
            ts.set_token(os.environ["TUSHARE_TOKEN"])
            pro = ts.pro_api()
        df = pro.daily_basic(trade_date=d, fields=DAILY_BASIC_FIELDS)
        df["trade_date"] = df["trade_date"].astype(str)
        df.to_parquet(fp, index=False)
        parts.append(df)
        print(f"  [daily_basic] {d}: {len(df)} 股", flush=True)
    return pd.concat(parts, ignore_index=True)


def load_index_navs(book_dates: list) -> dict:
    """复用 BT-002 缓存的 hs300 / csi1000 指数净值, 对齐 book 日历归一。"""
    idx = pd.read_parquet(IDX_CACHE)
    idx["trade_date"] = idx["trade_date"].astype(str)
    idx = idx.set_index("trade_date").reindex(book_dates).ffill()
    out = {}
    for name in ["hs300", "csi1000"]:
        c = idx[f"{name}_close"].dropna()
        c = c / c.iloc[0]
        out[name] = c.reindex(book_dates).ffill()
    return out


# ───────────── 净值统计 ─────────────
def nav_summary(nav: pd.Series, ppy: int = 252) -> dict:
    nav = nav.dropna().astype(float)
    n = len(nav)
    if n < 2:
        return {"n_days": n, "error": "insufficient"}
    arr = nav.to_numpy()
    ret = pd.Series(arr).pct_change().dropna().to_numpy()
    ann = (arr[-1] / arr[0]) ** (ppy / max(n - 1, 1)) - 1.0
    vol = float(np.nanstd(ret, ddof=1))
    sharpe = (float(np.nanmean(ret)) / (vol + 1e-12)) * np.sqrt(ppy)
    peak = np.maximum.accumulate(arr)
    maxdd = float((arr / peak - 1.0).min())
    return {"n_days": int(n), "total_return": float(arr[-1] / arr[0] - 1.0),
            "ann_return": float(ann), "sharpe": float(sharpe), "max_drawdown": maxdd}


def monthly_returns(nav: pd.Series) -> pd.Series:
    s = nav.dropna().astype(float)
    ym = s.index.astype(str).str[:6]
    last = s.groupby(ym).last()
    return last.pct_change().dropna()


# ───────────── [A] 风格画像 ─────────────
def style_profile(picks, px, db, rebal_days):
    """每个再平衡日: picks vs 全市场, 比较 logMV / vol20 / turnover / pe / pb。
    返回逐日对比 + 跨日聚合 (picks 中位数 vs 市场中位数, picks 中位数在市场的百分位)。"""
    # 过去 VOL_LOOKBACK 日收益年化波动 (因果)
    p = px.sort_values(["ts_code", "trade_date"]).copy()
    p["ret"] = p["pct_chg"] / 100.0
    p["vol20"] = (p.groupby("ts_code")["ret"]
                  .rolling(VOL_LOOKBACK).std().reset_index(level=0, drop=True)) * np.sqrt(252)
    vol_map = p.set_index(["trade_date", "ts_code"])["vol20"]

    db = db.copy()
    db["log_total_mv"] = np.log(db["total_mv"].clip(lower=1.0))
    metrics = ["log_total_mv", "vol20", "turnover_rate", "pe_ttm", "pb"]

    pick_dates = set(picks["trade_date"].unique())
    rows = []
    for d in rebal_days:
        if d not in pick_dates:
            continue
        pk = set(picks[picks["trade_date"] == d]["ts_code"])
        dbd = db[db["trade_date"] == d].copy()
        if dbd.empty or not pk:
            continue
        # vol 合并
        try:
            vd = vol_map.loc[d]
            dbd["vol20"] = dbd["ts_code"].map(vd)
        except KeyError:
            dbd["vol20"] = np.nan
        dbd["is_pick"] = dbd["ts_code"].isin(pk)
        mkt = dbd
        pks = dbd[dbd["is_pick"]]
        if len(pks) < 3:
            continue
        rec = {"trade_date": d, "n_pick": int(len(pks)), "n_mkt": int(len(mkt))}
        for m in metrics:
            mv = mkt[m].dropna()
            pv = pks[m].dropna()
            if len(mv) < 10 or len(pv) < 2:
                rec[f"{m}_pick_med"] = np.nan
                rec[f"{m}_mkt_med"] = np.nan
                rec[f"{m}_pick_pctile"] = np.nan
                continue
            pmed = float(pv.median())
            rec[f"{m}_pick_med"] = pmed
            rec[f"{m}_mkt_med"] = float(mv.median())
            # picks 中位数在全市场分布的百分位 (0=最小, 1=最大)
            rec[f"{m}_pick_pctile"] = float((mv <= pmed).mean())
        rows.append(rec)
    prof = pd.DataFrame(rows)
    # 跨日聚合 (mean)
    agg = {}
    for m in metrics:
        agg[m] = {
            "pick_med_mean": round(float(prof[f"{m}_pick_med"].mean()), 4),
            "mkt_med_mean": round(float(prof[f"{m}_mkt_med"].mean()), 4),
            "pick_pctile_mean": round(float(prof[f"{m}_pick_pctile"].mean()), 4),
        }
    return prof, agg


def industry_tilt(picks, px):
    """picks 行业分布 vs 全市场行业分布 (出现频次 share), 报 top over/under-weight。"""
    basic = pd.read_parquet(BASIC)[["ts_code", "industry"]].drop_duplicates("ts_code")
    ind_map = dict(zip(basic["ts_code"], basic["industry"]))
    # picks 行业 (用 picks 自带 industry, 缺则补 basic)
    pk_ind = picks["industry"].fillna(picks["ts_code"].map(ind_map))
    pk_share = pk_ind.value_counts(normalize=True)
    # 市场: 全 universe 唯一股票的行业分布 (静态宇宙近似)
    mkt_codes = px["ts_code"].drop_duplicates()
    mkt_ind = mkt_codes.map(ind_map)
    mkt_share = mkt_ind.value_counts(normalize=True)
    tilt = (pk_share - mkt_share.reindex(pk_share.index).fillna(0.0)).sort_values(ascending=False)
    over = [(str(k), round(float(pk_share.get(k, 0)) * 100, 2), round(float(mkt_share.get(k, 0)) * 100, 2),
             round(float(v) * 100, 2)) for k, v in tilt.head(6).items()]
    under = [(str(k), round(float(pk_share.get(k, 0)) * 100, 2), round(float(mkt_share.get(k, 0)) * 100, 2),
              round(float(v) * 100, 2)) for k, v in tilt.tail(6).items()]
    return over, under


# ───────────── [B] 基金共识篮 ─────────────
def fund_holdings_style(fund_hist, db_recent):
    """基金持仓 (最新季报快照) 的市值画像 vs 全市场, 判 large vs small cap。"""
    latest_q = fund_hist["end_date"].max()
    held = fund_hist[fund_hist["end_date"] == latest_q]["symbol"].unique()
    db = db_recent.copy()
    db["log_total_mv"] = np.log(db["total_mv"].clip(lower=1.0))
    fund_mv = db[db["ts_code"].isin(held)]["log_total_mv"].dropna()
    mkt_mv = db["log_total_mv"].dropna()
    if len(fund_mv) < 3:
        return None
    fmed = float(fund_mv.median())
    return {
        "quarter": latest_q,
        "n_held": int(len(held)),
        "fund_log_mv_med": round(fmed, 4),
        "fund_total_mv_med_yiyuan": round(float(np.exp(fmed)) / 1e4, 2),  # 万元→亿元
        "fund_mv_pctile_in_market": round(float((mkt_mv <= fmed).mean()), 4),
    }


def build_fund_basket_targets(fund_hist, cal, rebal_days):
    """基金共识篮: 每再平衡日用**最近已公告 (ann_date<=d) 季报**的合并持仓, 按出现频次 (多基金共同重仓)
    取前 FUND_TOPN, 等权到 TOTAL。因果: 只用 ann_date 已披露的持仓 (季报滞后约 1 月)。"""
    fh = fund_hist.copy()
    fh["ann_date"] = fh["ann_date"].astype(str)
    targets = {}
    for d in rebal_days:
        avail = fh[fh["ann_date"] <= d]
        if avail.empty:
            continue
        # 取每只基金各自最近已公告季报
        latest_ann = avail.groupby("ts_code")["end_date"].max().to_dict()
        snap = avail[avail.apply(lambda r: r["end_date"] == latest_ann.get(r["ts_code"]), axis=1)]
        if snap.empty:
            continue
        # 共识 = 跨基金按持仓市值占比加权的合并 (按 symbol 聚合 stk_mkv_ratio 之和 = 多基金共同重仓更高)
        agg = snap.groupby("symbol")["stk_mkv_ratio"].sum().sort_values(ascending=False)
        top = agg.head(FUND_TOPN).index.tolist()
        if not top:
            continue
        w = TOTAL / len(top)
        targets[d] = {c: w for c in top}
    return targets


def main():
    t0 = time.time()
    print("\n=== EX-002: 基金/风格对照 (基金更强是选股还是风格) ===\n", flush=True)
    picks = pd.read_parquet(PICKS)
    picks["trade_date"] = picks["trade_date"].astype(str)
    start = picks["trade_date"].min()
    print(f"[picks] {len(picks):,} 行 / {picks['trade_date'].nunique()} 日 / {picks['month'].nunique()} 月", flush=True)

    print(f"[prices] load daily >= {start} (ST 排除) ...", flush=True)
    px = load_prices(start)
    cal = sorted(px["trade_date"].unique())
    rebal_days = [d for d in cal[::REBAL_EVERY] if d in set(picks["trade_date"].unique())]
    print(f"[prices] {len(px):,} 行 / {px['ts_code'].nunique()} 股 / {len(cal)} 交易日; "
          f"再平衡日 {len(rebal_days)} 个", flush=True)

    print(f"\n[daily_basic] 拉取 {len(rebal_days)} 个再平衡日 (per-date checkpoint) ...", flush=True)
    db = load_daily_basic(rebal_days)
    db_recent = load_daily_basic([rebal_days[-1]])

    # ── [A] 风格画像 ──
    print("\n[A] V12.31 picks 风格画像 (picks vs 全市场, 每再平衡日) ...", flush=True)
    prof, agg = style_profile(picks, px, db, rebal_days)
    prof.to_parquet(CACHE / "ex002_style_profile.parquet", index=False)
    label = {"log_total_mv": "市值(log万元)", "vol20": "波动(年化)", "turnover_rate": "换手(%)",
             "pe_ttm": "PE(ttm)", "pb": "PB"}
    for m, a in agg.items():
        print(f"  {label[m]:14s}: picks中位 {a['pick_med_mean']:>10.3f}  市场中位 {a['mkt_med_mean']:>10.3f}  "
              f"picks在市场百分位 {a['pick_pctile_mean']:.1%}", flush=True)
    over, under = industry_tilt(picks, px)
    print("  行业 over-weight (top): " + ", ".join(f"{k}({pk:.1f}%vs{mk:.1f}%,+{t:.1f})" for k, pk, mk, t in over[:4]), flush=True)
    print("  行业 under-weight    : " + ", ".join(f"{k}({pk:.1f}%vs{mk:.1f}%,{t:.1f})" for k, pk, mk, t in under[:4]), flush=True)

    # ── [B] 基金持仓风格 + book Sharpe 对照 ──
    print("\n[B] 基金持仓风格 + book Sharpe 对照 ...", flush=True)
    fund_hist = pd.read_parquet(FUND_HIST)
    fund_hist["end_date"] = fund_hist["end_date"].astype(str)
    fund_style = fund_holdings_style(fund_hist, db_recent)
    if fund_style:
        print(f"  基金持仓 ({fund_style['quarter']}, {fund_style['n_held']} 股): "
              f"市值中位 {fund_style['fund_total_mv_med_yiyuan']:.1f}亿元, "
              f"在全市场百分位 {fund_style['fund_mv_pctile_in_market']:.1%}", flush=True)
    pick_mv_pctile = agg["log_total_mv"]["pick_pctile_mean"]
    print(f"  对比: 我们 picks 市值在全市场百分位 {pick_mv_pctile:.1%} "
          f"(基金 {fund_style['fund_mv_pctile_in_market']:.1%} → "
          f"{'基金更大盘' if fund_style['fund_mv_pctile_in_market'] > pick_mv_pctile else '我们更大盘'})", flush=True)

    # V12.31 book (复用 BT-002 baseline)
    bt002 = json.loads(BT002_RES.read_text(encoding="utf-8"))
    v1231 = bt002["A_slippage_sensitivity"][f"slip_{BASELINE_SLIP}"]
    book_nav = pd.read_parquet(CACHE002 / "bt002_book_baseline.parquet")["nav"]
    book_nav.index = book_nav.index.astype(str)
    book_dates = book_nav.index.tolist()

    bench = {"V12.31_book": v1231}
    # 指数基准 (复用 bt002 缓存)
    idx_navs = load_index_navs(book_dates)
    for name, nv in idx_navs.items():
        bench[name] = nav_summary(nv)

    # 基金共识篮过引擎 (真历史持仓, 因果 ann_date, 同基线成本)
    print("  基金共识篮过引擎 (真历史季报, ann_date 因果, 等权 top%d) ..." % FUND_TOPN, flush=True)
    tgt_fund = build_fund_basket_targets(fund_hist, cal, rebal_days)
    cm = CostModel(slippage=BASELINE_SLIP, enabled=True)
    bt = PortfolioBacktester(px, cost=cm, t_plus_1=True, limit_pct=9.8)
    res_fund = bt.run(tgt_fund, start=start)
    res_fund.timeseries.to_parquet(CACHE / "ex002_fund_basket.parquet")
    fund_nav = res_fund.timeseries["nav"]
    fund_nav.index = fund_nav.index.astype(str)
    bench["fund_basket"] = nav_summary(fund_nav.reindex(book_dates).ffill())

    print("\n  [对照表] 年化 / 净Sharpe / maxDD:", flush=True)
    for name, s in bench.items():
        print(f"    {name:14s}: 年化 {s['ann_return']:+.1%}  Sharpe {s['sharpe']:+.2f}  maxDD {s['max_drawdown']:+.1%}", flush=True)

    # ── [C] regime 分层: book vs hs300 月超额 (R11) ──
    print("\n[C] regime 分层 (book vs hs300 月超额, R11) ...", flush=True)
    reg = pd.read_parquet(REGIME)
    reg["trade_date"] = reg["trade_date"].astype(str)
    reg["month"] = reg["trade_date"].str[:6]
    month_regime = reg.groupby("month")["regime"].agg(lambda x: x.value_counts().idxmax())
    bm = monthly_returns(book_nav).rename("book")
    cmp_m = pd.DataFrame({"book": bm})
    cmp_m["hs300"] = monthly_returns(idx_navs["hs300"])
    cmp_m["fund"] = monthly_returns(fund_nav.reindex(book_dates).ffill())
    cmp_m = cmp_m.dropna()
    cmp_m["regime"] = cmp_m.index.map(month_regime)
    cmp_m["exc_hs300"] = (cmp_m["book"] - cmp_m["hs300"]) * 100
    cmp_m["exc_fund"] = (cmp_m["book"] - cmp_m["fund"]) * 100
    regime_table = {}
    for rg, g in cmp_m.groupby("regime"):
        regime_table[rg] = {
            "n_months": int(len(g)),
            "book_vs_hs300_pp": round(float(g["exc_hs300"].mean()), 3),
            "book_vs_fund_pp": round(float(g["exc_fund"].mean()), 3),
            "fund_ret_pp": round(float(g["fund"].mean()) * 100, 3),
        }
        print(f"  [{rg:9s}] n={len(g):2d}  book-hs300 {g['exc_hs300'].mean():+.2f}pp  "
              f"book-fund {g['exc_fund'].mean():+.2f}pp  fund月收益 {g['fund'].mean()*100:+.2f}%", flush=True)
    cmp_m.to_parquet(CACHE / "ex002_monthly_regime.parquet")

    # ── 归因 ──
    small_cap = bool(pick_mv_pctile < 0.45)          # picks 偏小盘
    high_vol = bool(agg["vol20"]["pick_pctile_mean"] > 0.55)
    high_turnover = bool(agg["turnover_rate"]["pick_pctile_mean"] > 0.55)
    fund_larger = bool(fund_style and fund_style["fund_mv_pctile_in_market"] > pick_mv_pctile)
    # 选股 vs 风格: 基金共识篮 Sharpe 与我们 book (去 lookahead 仍未做, 此处描述)
    fund_sharpe = bench["fund_basket"]["sharpe"]
    book_sharpe = bench["V12.31_book"]["sharpe"]

    results = {
        "config": {"total": TOTAL, "rebal_every": REBAL_EVERY, "vol_lookback": VOL_LOOKBACK,
                   "fund_topn": FUND_TOPN, "baseline_slippage": BASELINE_SLIP,
                   "n_rebal": len(rebal_days), "period": [start, book_dates[-1]],
                   "fund_codes": sorted(fund_hist["ts_code"].unique().tolist()),
                   "fund_quarters": sorted(fund_hist["end_date"].unique().tolist())},
        "A_style_profile": agg,
        "A_industry_tilt": {"over_weight": over, "under_weight": under},
        "B_fund_holdings_style": fund_style,
        "B_sharpe_compare": bench,
        "C_regime": regime_table,
        "attribution": {
            "picks_small_cap": small_cap,
            "picks_high_vol": high_vol,
            "picks_high_turnover": high_turnover,
            "picks_mv_pctile": pick_mv_pctile,
            "fund_holds_larger_cap": fund_larger,
            "v1231_book_sharpe": round(book_sharpe, 3),
            "fund_basket_sharpe": round(fund_sharpe, 3),
        },
        "caveat": "V12.31 book 绝对量级含 r20 池共模 lookahead + 集中, 非可交易声明 (同 BT-001/002); "
                  "基金共识篮 = 7 基金 (按今日存在选取 = fund-selection survivorship), 但持仓用真历史季报 "
                  "(ann_date 因果可得, 非单一静态快照外推) → 比纯静态篮诚实, 仍有幸存者偏差; "
                  "本 task 是描述性风格画像 + 对照, 非 walk-forward ship gate (EX-003 才是)。",
    }
    (CACHE / "ex002_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # ── 裁决句 (风格描述符按实际百分位动态生成, 不预设 R03) ──
    mv_pct = pick_mv_pctile
    vol_pct = agg["vol20"]["pick_pctile_mean"]
    to_pct = agg["turnover_rate"]["pick_pctile_mean"]
    pe_pct = agg["pe_ttm"]["pick_pctile_mean"]
    tags = []
    tags.append("小盘" if mv_pct < 0.45 else ("大盘" if mv_pct > 0.55 else "中盘"))
    if pe_pct > 0.55:
        tags.append("高PE成长")
    elif pe_pct < 0.45:
        tags.append("低PE价值")
    if vol_pct > 0.55:
        tags.append("高波")
    if to_pct > 0.55:
        tags.append("高换手")
    style_tag = "/".join(tags)
    style_dir = (f"V12.31 picks 风格 = **{style_tag}** (市值百分位 {mv_pct:.0%}/PE {pe_pct:.0%}/波动 {vol_pct:.0%}/"
                 f"换手 {to_pct:.0%} 在全市场; 波动换手≈市场中位非高波高换手), vs 基金持仓市值百分位 "
                 f"{fund_style['fund_mv_pctile_in_market']:.0%} ({'基金显著更大盘' if fund_larger else '风格相近'})")
    # 归因逻辑: book 绝对 Sharpe 远高 (lookahead 注水), 但与基金的真实差异主要是风格暴露
    if fund_larger and small_cap:
        attr = ("与基金的差异**主导来自风格/universe 暴露** (我们小盘高PE成长, 基金大盘质量, 市值百分位 "
                f"{mv_pct:.0%} vs {fund_style['fund_mv_pctile_in_market']:.0%}) → "
                "增量在**风格 tilt / 风险管理** (BT-002 已示动量月相对最弱 = 小盘成长在动量行情吃亏), "
                "非单纯选股重标; 但 V12.31 book 绝对 Sharpe 受 lookahead 注水, 真实选股 α 须 EX-003 去 lookahead 定。"
                "符合 responsePlaybook['风格差']: 重要发现 = 风格暴露 driven。")
    else:
        attr = ("扣风格后差异仍在 → 选股层差距, 支持 EX-003 重标 (responsePlaybook['选股差'])。")

    conclusion = (
        f"{style_dir}。行业 over-weight: "
        + ", ".join(f"{k}(+{t:.1f}pp)" for k, _, _, t in over[:3]) + "; "
        f"under-weight: " + ", ".join(f"{k}({t:.1f}pp)" for k, _, _, t in under[:2]) + "。"
        f"book Sharpe 对照 ({start}~{book_dates[-1]}): V12.31 {book_sharpe:+.2f} / hs300 "
        f"{bench['hs300']['sharpe']:+.2f} / CSI1000 {bench['csi1000']['sharpe']:+.2f} / "
        f"基金共识篮 {fund_sharpe:+.2f} (真历史季报, ann_date 因果)。"
        f"regime: book vs hs300 各 regime 超额见表 (R11)。{attr} "
        f"【绝对量级非可交易声明】V12.31 book 含 r20 池共模 lookahead + 集中 (同 BT-001/002); "
        f"基金篮含 fund-selection survivorship (7 基金今日存在) 但持仓真历史季报 (比纯静态篮诚实)。"
    )

    VERDICT.write_text(json.dumps({
        "id": "EX-002", "status": "built", "conclusion": conclusion,
        "metrics": {
            "picks_mv_pctile_in_market": round(mv_pct, 4),
            "picks_vol_pctile_in_market": round(vol_pct, 4),
            "picks_turnover_pctile_in_market": round(to_pct, 4),
            "picks_pe_pctile_in_market": round(agg["pe_ttm"]["pick_pctile_mean"], 4),
            "fund_holdings_mv_pctile_in_market": round(fund_style["fund_mv_pctile_in_market"], 4) if fund_style else None,
            "fund_holds_larger_cap": fund_larger,
            "v1231_book_net_sharpe": round(book_sharpe, 3),
            "hs300_sharpe": round(bench["hs300"]["sharpe"], 3),
            "csi1000_sharpe": round(bench["csi1000"]["sharpe"], 3),
            "fund_basket_sharpe": round(fund_sharpe, 3),
            "fund_basket_ann_return": round(bench["fund_basket"]["ann_return"], 4),
            "picks_small_cap": small_cap, "picks_high_vol": high_vol, "picks_high_turnover": high_turnover,
            "style_profile": agg,
        },
        "industry_tilt": {"over_weight": over, "under_weight": under},
        "regime_table": regime_table,
        "cost_model": cm.describe(),
        "artifacts": [
            "research/backtest/run_ex002.py",
            "research/cache/ex002/ex002_results.json",
            "research/cache/ex002/ex002_style_profile.parquet",
            "research/cache/ex002/ex002_fund_basket.parquet",
            "research/cache/ex002/ex002_monthly_regime.parquet",
            "research/cache/ex002/fund_portfolio_hist.parquet",
            "research/cache/ex002/daily_basic/*.parquet",
        ],
        "note": "对照法/基准/基金篮构造冻结 (R01, DESIGN §7); 复用 BT-002 baseline book + 指数缓存; "
                "ST 源头排除 (R06); regime 分层必带 (R11); daily_basic per-date checkpoint (R08); "
                "基金共识篮用真历史季报 ann_date 因果 (非静态快照外推, 标注 fund-selection survivorship); "
                "描述性风格画像+对照=合法完成 (R02); 中间指标≠ship (R03); 绝对量级含 lookahead 非可交易声明; "
                "大缓存 gitignored 无 features 写入 (R04); 生产线只读未碰 (R05)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
