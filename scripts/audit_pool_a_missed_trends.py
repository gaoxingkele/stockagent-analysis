"""Audit stable recent winners that Pool A did not select.

The price screen uses close-to-close return and close-based maximum drawdown.
Pool A rejection reasons are reconstructed with the production V7c hard rules.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from stockagent_analysis.v12_scoring import V12Scorer


ROOT = Path(__file__).resolve().parents[1]


def _price_screen(date: str) -> pd.DataFrame:
    daily_dir = ROOT / "output" / "tushare_cache" / "daily"
    files = [p for p in sorted(daily_dir.glob("*.parquet")) if p.stem <= date][-21:]
    if len(files) < 21:
        raise RuntimeError(f"need 21 trading-day files through {date}, found {len(files)}")

    prices = pd.concat(
        [pd.read_parquet(p, columns=["ts_code", "trade_date", "close"]) for p in files],
        ignore_index=True,
    )
    prices = prices[prices["ts_code"].astype(str).str.endswith((".SH", ".SZ"))]
    rows: list[dict] = []
    for code, group in prices.groupby("ts_code"):
        group = group.sort_values("trade_date")
        if str(group.iloc[-1]["trade_date"]) != date:
            continue
        closes = pd.to_numeric(group["close"], errors="coerce").to_numpy(dtype=float)
        row: dict = {"ts_code": code}
        for days in (10, 20):
            ok = len(closes) >= days + 1 and np.isfinite(closes[-(days + 1):]).all()
            if not ok:
                row.update({f"return_{days}d": np.nan, f"max_drawdown_{days}d": np.nan,
                            f"up_day_ratio_{days}d": np.nan, f"pass_{days}d": False})
                continue
            window = closes[-(days + 1):]
            ret = window[-1] / window[0] - 1
            drawdown = window / np.maximum.accumulate(window) - 1
            row.update({
                f"return_{days}d": ret,
                f"max_drawdown_{days}d": float(drawdown.min()),
                f"up_day_ratio_{days}d": float((np.diff(window) > 0).mean()),
                f"pass_{days}d": bool(ret > 0.15 and drawdown.min() > -0.10),
            })
        if row["pass_10d"] or row["pass_20d"]:
            rows.append(row)
    return pd.DataFrame(rows)


def _history(codes: set[str], date: str) -> pd.DataFrame:
    daily_paths = [
        p for p in sorted((ROOT / "output" / "tushare_cache" / "daily").glob("*.parquet"))
        if p.stem <= date
    ][-21:]
    start_date = daily_paths[0].stem
    score_paths = sorted((ROOT / "output" / "daily_pick").glob("scores_*.parquet"))
    score_paths = [p for p in score_paths if start_date <= p.stem[-8:] <= date]
    state = {code: {"score_dates_checked": 0, "v7c_signal_dates": [], "pool_a_dates": []}
             for code in codes}
    for path in score_paths:
        signal_date = path.stem[-8:]
        scores = pd.read_parquet(path, columns=["ts_code", "v7c_recommend"])
        subset = scores[scores["ts_code"].isin(codes)]
        for record in subset.to_dict("records"):
            item = state[record["ts_code"]]
            item["score_dates_checked"] += 1
            if bool(record["v7c_recommend"]):
                item["v7c_signal_dates"].append(signal_date)
        pool_path = (ROOT / "output" / "daily_pick" / f"dashboard_{signal_date}"
                     / "poolA_system.csv")
        if pool_path.exists():
            pool_codes = set(pd.read_csv(pool_path)["ts_code"].astype(str))
            for code in codes & pool_codes:
                state[code]["pool_a_dates"].append(signal_date)
    return pd.DataFrame([
        {"ts_code": code, "score_dates_checked": item["score_dates_checked"],
         "v7c_signal_count": len(item["v7c_signal_dates"]),
         "v7c_signal_dates": ",".join(item["v7c_signal_dates"]),
         "pool_a_count": len(item["pool_a_dates"]),
         "pool_a_dates": ",".join(item["pool_a_dates"])}
        for code, item in state.items()
    ])


def _rule_audit(date: str) -> pd.DataFrame:
    scorer = V12Scorer.get(ROOT)
    factors = scorer.load_factors_for_date(date)
    factors = factors.copy()
    factors["r20_pred"] = scorer.predict_one(factors, "r20_v16_all")
    factors = scorer._enrich_zombie(factors, date)
    factors["v7c_eligible"] = scorer._v7c_eligibility_mask(factors)

    eligible = factors["v7c_eligible"]
    factors["r20_rank_pct"] = np.nan
    factors.loc[eligible, "r20_rank_pct"] = factors.loc[eligible, "r20_pred"].rank(
        pct=True, method="first"
    )
    pyr_cutoff = pd.to_numeric(
        factors.loc[eligible, "pyr_velocity_20_60"], errors="coerce"
    ).quantile(0.35)
    factors["pass_r20_top5"] = factors["r20_rank_pct"] >= 0.95
    factors["pass_pyr_velocity"] = factors["pyr_velocity_20_60"] < pyr_cutoff
    factors["pass_f1_f2"] = ((factors["f1_neg1"].abs() < 0.005)
                              & (factors["f2_pos1"].abs() < 0.005))
    factors["pass_not_zombie"] = ~factors["is_zombie"].fillna(False).astype(bool)
    factors["pass_industry_mom"] = (factors["industry_mom_60d_rank"].isna()
                                     | (factors["industry_mom_60d_rank"] >= 0.10))
    factors["v7c_recommend_rebuilt"] = scorer._apply_v7c_rules(factors)

    def reasons(row: pd.Series) -> str:
        failed = []
        if not bool(row["v7c_eligible"]): failed.append("硬规则输入缺失/非有限值")
        if not bool(row["pass_r20_top5"]): failed.append("r20预测未进当日前5%")
        if not bool(row["pass_pyr_velocity"]): failed.append("20/60速度形态未进较低35%")
        if not bool(row["pass_f1_f2"]): failed.append("F1/F2形态阈值不通过")
        if not bool(row["pass_not_zombie"]): failed.append("僵尸横盘过滤")
        if not bool(row["pass_industry_mom"]): failed.append("行业60日动量后10%")
        return "；".join(failed) if failed else "通过V7c；若未入最终表则为行业cap/TopN排序"

    factors["latest_rejection_reasons"] = factors.apply(reasons, axis=1)
    cols = ["ts_code", "industry", "r20_pred", "r20_rank_pct", "pyr_velocity_20_60",
            "f1_neg1", "f2_pos1", "is_zombie", "industry_mom_60d_rank",
            "v7c_eligible", "pass_r20_top5", "pass_pyr_velocity", "pass_f1_f2",
            "pass_not_zombie", "pass_industry_mom", "v7c_recommend_rebuilt",
            "latest_rejection_reasons"]
    result = factors[[c for c in cols if c in factors.columns]].copy()
    result.attrs["pyr_velocity_p35"] = float(pyr_cutoff)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None, help="YYYYMMDD; defaults to latest daily cache")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    daily_files = sorted((ROOT / "output" / "tushare_cache" / "daily").glob("*.parquet"))
    date = args.date or daily_files[-1].stem

    trend = _price_screen(date)
    basic = pd.read_parquet(ROOT / "output" / "tushare_cache" / "stock_basic.parquet")[
        ["ts_code", "name", "industry"]
    ]
    audit = trend.merge(basic, on="ts_code", how="left")
    rules = _rule_audit(date)
    pyr_cutoff = rules.attrs["pyr_velocity_p35"]
    audit = audit.merge(rules, on="ts_code", how="left", suffixes=("", "_scored"))
    audit["latest_rejection_reasons"] = audit["latest_rejection_reasons"].fillna(
        "未进入评分宇宙（通常为ST或基础因子缺失）"
    )
    audit = audit.merge(_history(set(audit["ts_code"]), date), on="ts_code", how="left")
    audit["window_group"] = np.select(
        [audit["pass_10d"] & audit["pass_20d"], audit["pass_20d"], audit["pass_10d"]],
        ["10日和20日均通过", "仅20日通过", "仅10日通过"], default="",
    )
    audit = audit.sort_values(["pass_20d", "return_20d", "return_10d"], ascending=False)

    output = Path(args.output) if args.output else (
        ROOT / "output" / "daily_pick" / f"pool_a_missed_trends_{date}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"date={date}")
    print(f"candidates={len(audit)} pass10={int(audit['pass_10d'].sum())} "
          f"pass20={int(audit['pass_20d'].sum())} "
          f"both={int((audit['pass_10d'] & audit['pass_20d']).sum())}")
    print(f"pyr_velocity_p35={pyr_cutoff:.8f}")
    print(audit["latest_rejection_reasons"].value_counts().head(15).to_string())
    print(f"output={output}")


if __name__ == "__main__":
    main()
