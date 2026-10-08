#!/usr/bin/env python
"""Sector rotation as structure: do sectors move together or against each other, and what does that do to the lists?

The user's hypothesis (2026-10-06): relative to the broad index a sector is a beta, but sectors
also reinforce or offset one another; the state of those relations, by sector category, should
decide how the lists do, and through them the single names.

Everything is measured at the signal-day close from Shenwan level-1 indices and the CSI 300:

  co-movement      mean pairwise correlation of the 31 sector daily returns over 20 sessions; share of
                   their variance carried by the first principal component
  category strength four a-priori categories (growth/tech, cyclical, financial/defensive, consumer),
                   each an equal-weight index of its sectors: 20-day return minus the CSI 300
  growth-vs-defensive coupling   20-session correlation of the growth and the financial/defensive
                   category daily returns: positive = moving together, negative = offsetting
  leading category the category with the highest 20-day return

Outcomes: the frozen (trend-chasing) and v2 (non-chasing) offensive lists' daily mean exit return
on the 251 days neither model saw. Name level: a listed name's own category (Tushare industry
mapped by keyword) against the leading category.

Factor list and category maps are fixed before running. Descriptive only; the reserved window is
not read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_cross_filter_v2 import nw_se  # noqa: E402
from s20_pure_history import outcomes  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, select  # noqa: E402

OUT = ROOT / "output/experiments/sector_structure_20261006"
CACHE = ROOT / "output/tushare_cache"
SCORES = {"frozen": ROOT / "output/experiments/s20_frozen_rescore_20261005/frozen_scores.parquet",
          "v2": ROOT / "output/experiments/s20_unified_v2_20261005/predictions.parquet"}
OOS = lambda d: (d < "20250901") | (d > "20260126")  # noqa: E731
SW_CATEGORY = {
    "growth": ["电子", "计算机", "通信", "传媒", "电力设备", "医药生物", "国防军工"],
    "cyclical": ["有色金属", "钢铁", "基础化工", "石油石化", "煤炭", "建筑材料", "机械设备", "汽车"],
    "defensive": ["银行", "非银金融", "公用事业", "交通运输", "房地产", "建筑装饰", "环保"],
    "consumer": ["食品饮料", "家用电器", "商贸零售", "社会服务", "农林牧渔", "纺织服饰", "轻工制造", "美容护理", "综合"],
}
NAME_KEYWORDS = {
    "growth": ["半导体", "元器件", "IT设备", "软件", "互联网", "通信", "电信", "电气设备", "新型电力", "生物制药", "化学制药", "中成药", "医疗保健", "医药商业",
               "影视", "出版", "广告", "航空", "船舶", "运输设备", "电器仪表"],
    "cyclical": ["钢", "铜", "铝", "铅锌", "黄金", "小金属", "化工", "化纤", "塑料", "橡胶", "染料", "煤炭", "焦炭", "石油", "水泥", "玻璃", "陶瓷", "矿物", "其他建材",
                 "机械", "机床", "汽车", "摩托车", "造纸"],
    "defensive": ["银行", "保险", "证券", "多元金融", "地产", "园区", "房产", "建筑工程", "装修", "路桥", "公路", "铁路", "港口", "水运", "空运", "机场", "公共交通",
                  "仓储", "火力", "水力", "供气", "水务", "环境保护"],
    "consumer": ["白酒", "啤酒", "红黄酒", "软饮料", "食品", "乳制品", "饲料", "种植", "渔业", "林业", "农业", "农药", "家用电器", "家居", "电器连锁", "百货", "超市", "商品城",
                 "批发", "商贸", "其他商业", "旅游", "酒店", "文教", "服饰", "纺织", "日用化工", "综合类"],
}


def category_of_industry(name: str) -> str | None:
    for cat, words in NAME_KEYWORDS.items():
        if any(w in name for w in words):
            return cat
    return None


def context() -> pd.DataFrame:
    cl = pd.read_parquet(CACHE / "sw_index_classify.parquet")
    l1 = cl[cl.level == "L1"].set_index("industry_name").index_code
    sw = pd.concat([pd.read_parquet(p, columns=["ts_code", "trade_date", "close"]) for p in sorted((CACHE / "sw_daily").glob("*.parquet"))])
    close = sw[sw.ts_code.isin(set(l1))].pivot(index="trade_date", columns="ts_code", values="close").sort_index()
    ret = close.pct_change()
    hs = pd.read_parquet(CACHE / "index_daily/000300.SH.parquet", columns=["trade_date", "close"]).set_index("trade_date").close.sort_index()
    hs_ret = hs.pct_change()
    f = pd.DataFrame(index=ret.index)
    pair, pc1, beta_disp = [], [], []
    for i in range(len(ret)):
        if i < 20:
            pair.append(np.nan); pc1.append(np.nan); beta_disp.append(np.nan); continue
        w = ret.iloc[i - 19:i + 1].dropna(axis=1)
        c = w.corr().to_numpy()
        pair.append((c.sum() - len(c)) / (len(c) * (len(c) - 1)))
        ev = np.linalg.eigvalsh(np.cov(w.to_numpy().T))
        pc1.append(ev[-1] / ev.sum())
        h = hs_ret.reindex(w.index).to_numpy()
        betas = [np.cov(w[col].to_numpy(), h)[0, 1] / np.var(h) for col in w.columns]
        beta_disp.append(np.std(betas))
    f["sector_pair_corr_20"] = pair
    f["sector_pc1_share_20"] = pc1
    f["sector_beta_dispersion_20"] = beta_disp
    cat_ret = {}
    for cat, names in SW_CATEGORY.items():
        codes = [l1[n] for n in names if n in l1.index]
        cat_ret[cat] = ret[codes].mean(axis=1)
    cat_ret = pd.DataFrame(cat_ret)
    level = (1 + cat_ret.fillna(0)).cumprod()
    r20 = level / level.shift(20) - 1
    hs20 = hs / hs.shift(20) - 1
    for cat in SW_CATEGORY:
        f[f"{cat}_rel_20"] = r20[cat] - hs20.reindex(r20.index)
    valid = r20.dropna(how="all")
    f["leading_category"] = valid.idxmax(axis=1).reindex(f.index)
    f["growth_defensive_corr_20"] = cat_ret.growth.rolling(20).corr(cat_ret.defensive)
    f["growth_cyclical_corr_20"] = cat_ret.growth.rolling(20).corr(cat_ret.cyclical)
    f["growth_minus_defensive_20"] = r20.growth - r20.defensive
    f["growth_hs300_corr_60"] = cat_ret.growth.rolling(60).corr(hs_ret.reindex(cat_ret.index))
    return f


def table(day: pd.DataFrame, col: str, label: str) -> str:
    x = day[col]
    cut = pd.qcut(x, 3, labels=["low", "mid", "high"]) if x.nunique() > 10 else x
    parts = []
    for name in ("frozen", "v2"):
        rho = day[name].corr(x.rank() if x.dtype != object else x.astype("category").cat.codes, method="spearman") if x.dtype != object else float("nan")
        terc = day.groupby(cut, observed=True)[name].mean()
        parts.append(f"{name} rho {rho:+.2f} | " + " ".join(f"{k}:{v:+.2f}({(cut == k).sum()})" for k, v in terc.items()))
    return f"{label:34s}: " + "  ||  ".join(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    out = outcomes()[["ts_code", "trade_date", "ret_v1", "state"]]
    basic = pd.read_parquet(CACHE / "stock_basic.parquet", columns=["ts_code", "industry"])
    lists, day = {}, {}
    for name, path in SCORES.items():
        picked = select(pd.read_parquet(path), PureConfig(), "U15D10")[["ts_code", "trade_date"]].merge(out).merge(basic, on="ts_code", how="left")
        lists[name] = picked[OOS(picked.trade_date)]
        day[name] = lists[name].groupby("trade_date").ret_v1.mean()
    ctx = context()
    d = pd.DataFrame(day).join(ctx).dropna(subset=["sector_pair_corr_20"])
    lines = ["Sector structure against the S20 offensive lists, 251 days outside the frozen model's fit and tuning windows.",
             "Daily list return = +15%/-10% exit, list basis. Terciles show mean(return)(days). Fixed factor list; no tuning.", "",
             "== A. do sectors move together or apart? =="]
    for col, label in (("sector_pair_corr_20", "mean pairwise sector corr (20d)"), ("sector_pc1_share_20", "variance share of 1st PC (20d)"),
                       ("sector_beta_dispersion_20", "dispersion of sector betas to CSI300"), ("growth_defensive_corr_20", "growth vs defensive corr (20d)"),
                       ("growth_cyclical_corr_20", "growth vs cyclical corr (20d)"), ("growth_hs300_corr_60", "growth category corr with CSI300 (60d)")):
        lines.append(table(d, col, label))
    lines += ["", "== B. which category leads, and the list's result =="]
    for cat in SW_CATEGORY:
        lines.append(table(d, f"{cat}_rel_20", f"{cat} 20d return minus CSI300"))
    lines.append(table(d, "growth_minus_defensive_20", "growth minus defensive, 20d"))
    lines.append("")
    for cat, g in d.groupby("leading_category"):
        lines.append(f"leading category = {cat:9s}: days {len(g):3d} | frozen {g.frozen.mean():+5.2f} | v2 {g.v2.mean():+5.2f}")
    lines += ["", "== C. reinforce or offset: growth leads x sectors co-move (2x2) =="]
    d["growth_leads"] = np.where(d.growth_rel_20 > 0, "growth leads", "growth lags")
    d["comove"] = np.where(d.sector_pair_corr_20 > d.sector_pair_corr_20.median(), "sectors co-move (high corr)", "sectors diverge (low corr)")
    for (a, b), g in d.groupby(["growth_leads", "comove"]):
        lines.append(f"{a:13s} & {b:28s}: days {len(g):3d} | frozen {g.frozen.mean():+5.2f} | v2 {g.v2.mean():+5.2f}")
    d["gd"] = np.where(d.growth_defensive_corr_20 > 0, "growth & defensive together", "growth & defensive opposed")
    for (a, b), g in d.groupby(["growth_leads", "gd"]):
        lines.append(f"{a:13s} & {b:28s}: days {len(g):3d} | frozen {g.frozen.mean():+5.2f} | v2 {g.v2.mean():+5.2f}")
    lines += ["", "== D. name level: the listed name's own category against the leading category (same-day paired) =="]
    for name in ("frozen", "v2"):
        q = lists[name].copy()
        q["cat"] = q.industry.fillna("").map(category_of_industry)
        q = q.join(ctx[["leading_category", "growth_rel_20"]], on="trade_date")
        cover = q.cat.notna().mean()
        q = q.dropna(subset=["cat", "leading_category"])
        share = q.cat.value_counts(normalize=True).round(2).to_dict()
        lines.append(f"[{name}] names mapped to a category: {100 * cover:.0f}% ; category mix {share}")
        for cat, g in q.groupby("cat"):
            lines.append(f"    {cat:9s}: n {len(g):5d} ret {g.ret_v1.mean():+5.2f} pure_down {100 * (g.state == 'pure_down').mean():4.1f}")
        q["aligned"] = q.cat == q.leading_category
        a, b = q[q.aligned], q[~q.aligned]
        diff = (a.groupby("trade_date").ret_v1.mean() - b.groupby("trade_date").ret_v1.mean()).dropna()
        lines.append(f"    in the leading category {100 * q.aligned.mean():.0f}% of names; ret {a.ret_v1.mean():+.2f} vs others {b.ret_v1.mean():+.2f}; "
                     f"same-day paired {diff.mean():+.2f} (t {diff.mean() / nw_se(diff.to_numpy()):+.2f}, days {len(diff)})")
        gq = q[q.cat == "growth"]
        for state, g in gq.groupby(np.where(gq.growth_rel_20 > 0, "growth category leads CSI300", "growth category lags CSI300")):
            lines.append(f"    growth-category names when {state:28s}: n {len(g):5d} ret {g.ret_v1.mean():+5.2f} pure_down {100 * (g.state == 'pure_down').mean():4.1f}")
        ng = q[q.cat != "growth"]
        for state, g in ng.groupby(np.where(ng.growth_rel_20 > 0, "growth category leads CSI300", "growth category lags CSI300")):
            lines.append(f"    non-growth names when         {state:28s}: n {len(g):5d} ret {g.ret_v1.mean():+5.2f} pure_down {100 * (g.state == 'pure_down').mean():4.1f}")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "report.txt").write_text(text + "\n", encoding="utf-8")
    d.to_csv(OUT / "daily_sector_context.csv")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
