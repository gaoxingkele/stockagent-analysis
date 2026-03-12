# -*- coding: utf-8 -*-
"""信号记录与回测验证模块。

每次运行结束后自动记录信号到 output/signal_history.csv，
支持后续回填实际涨跌并统计各维度预测准确率(IC)。
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

SIGNAL_DB = Path("output/signal_history.csv")

COLUMNS = [
    "date", "symbol", "name", "final_score", "decision",
    "close_price", "pe_ttm", "momentum_20", "volatility_20",
    # 各维度分数
    "TREND", "TECH", "CAPITAL_FLOW", "FUNDAMENTAL", "KLINE_PATTERN",
    "DIVERGENCE", "SUPPORT_RESISTANCE", "CHANLUN", "DERIV_MARGIN",
    "VOLUME_PRICE", "TIMEFRAME_RESONANCE", "RELATIVE_STRENGTH",
    # 狙击点位（P2）
    "ideal_buy", "stop_loss", "take_profit_1", "direction_expected",
    # 后续涨跌（回验时填入）
    "ret_5d", "ret_10d", "ret_20d",
]


def record_signal(
    final_decision: dict[str, Any],
    detail: list[dict[str, Any]],
    snap: dict[str, Any],
    features: dict[str, Any],
) -> None:
    """运行结束后记录信号到 signal_history.csv。"""
    row: dict[str, Any] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "symbol": snap.get("symbol"),
        "name": snap.get("name"),
        "final_score": final_decision.get("score"),
        "decision": final_decision.get("decision"),
        "close_price": snap.get("close"),
        "pe_ttm": features.get("pe_ttm"),
        "momentum_20": features.get("momentum_20"),
        "volatility_20": features.get("volatility_20"),
    }
    # 各维度分数
    for d in detail:
        dim = d.get("dim_code", "")
        if dim in COLUMNS:
            row[dim] = d.get("score_0_100")
    # 狙击点位
    sp = final_decision.get("sniper_points", {})
    row["ideal_buy"] = sp.get("ideal_buy", "")
    row["stop_loss"] = sp.get("stop_loss", "")
    row["take_profit_1"] = sp.get("take_profit_1", "")
    decision = final_decision.get("decision", "")
    score = float(final_decision.get("score", 50))
    if decision in ("buy",) or score >= 65:
        row["direction_expected"] = "up"
    elif decision in ("sell",) or score < 40:
        row["direction_expected"] = "down"
    else:
        row["direction_expected"] = "neutral"
    # 后验字段留空
    row["ret_5d"] = ""
    row["ret_10d"] = ""
    row["ret_20d"] = ""

    SIGNAL_DB.parent.mkdir(parents=True, exist_ok=True)
    file_exists = SIGNAL_DB.exists()
    with open(SIGNAL_DB, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def backfill_returns() -> None:
    """读取 signal_history.csv 中 ret_5d/10d/20d 为空的记录，用实际数据回填。"""
    import pandas as pd

    if not SIGNAL_DB.exists():
        print("[回验] signal_history.csv 不存在")
        return

    df = pd.read_csv(SIGNAL_DB, dtype=str)
    empty_mask = df["ret_5d"].isna() | (df["ret_5d"] == "")
    filled = 0

    for idx, row in df[empty_mask].iterrows():
        symbol = str(row["symbol"])
        base_close = float(row["close_price"]) if row.get("close_price") else None
        if not base_close:
            continue
        hist = _fetch_daily_after(symbol, str(row["date"]), days=25)
        if hist is None or hist.empty:
            continue
        if len(hist) >= 5:
            df.at[idx, "ret_5d"] = str(round((float(hist.iloc[4]["close"]) / base_close - 1) * 100, 2))
        if len(hist) >= 10:
            df.at[idx, "ret_10d"] = str(round((float(hist.iloc[9]["close"]) / base_close - 1) * 100, 2))
        if len(hist) >= 20:
            df.at[idx, "ret_20d"] = str(round((float(hist.iloc[19]["close"]) / base_close - 1) * 100, 2))
        filled += 1

    df.to_csv(SIGNAL_DB, index=False, encoding="utf-8")
    print(f"[回验] 已回填 {filled} 条记录")


def evaluate_accuracy() -> None:
    """统计信号准确率和各维度IC。"""
    import pandas as pd

    if not SIGNAL_DB.exists():
        print("[评估] signal_history.csv 不存在")
        return

    df = pd.read_csv(SIGNAL_DB)
    df["ret_10d"] = pd.to_numeric(df["ret_10d"], errors="coerce")
    df = df.dropna(subset=["ret_10d"])

    if df.empty:
        print("[评估] 无已回填的记录可评估")
        return

    # 整体胜率
    buy_signals = df[df["decision"] == "buy"]
    if len(buy_signals) > 0:
        win_rate = (buy_signals["ret_10d"] > 0).mean()
        print(f"Buy信号胜率(10日): {win_rate:.1%} ({len(buy_signals)}条)")
    else:
        print("尚无Buy信号")

    # final_score 与 ret_10d 相关性
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    valid_score = df[["final_score", "ret_10d"]].dropna()
    if len(valid_score) > 5:
        ic = valid_score["final_score"].corr(valid_score["ret_10d"])
        print(f"综合评分IC(10日): {ic:.4f}")

    # 各维度IC
    dim_cols = ["TREND", "TECH", "CAPITAL_FLOW", "FUNDAMENTAL", "KLINE_PATTERN",
                "DIVERGENCE", "SUPPORT_RESISTANCE", "CHANLUN", "DERIV_MARGIN",
                "VOLUME_PRICE", "TIMEFRAME_RESONANCE"]
    ic_report = {}
    for col in dim_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            valid = df[[col, "ret_10d"]].dropna()
            if len(valid) > 5:
                ic_report[col] = round(float(valid[col].corr(valid["ret_10d"])), 4)
    if ic_report:
        print("各维度IC(10日):")
        for dim, ic_val in sorted(ic_report.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {dim}: {ic_val:+.4f}")

    print(f"信号总数: {len(df)}")


def infer_direction(decision: str, score: float) -> str:
    """从决策推断预期方向。"""
    if decision in ("buy", "strong_buy") or score >= 65:
        return "up"
    elif decision in ("sell", "strong_sell") or score < 40:
        return "down"
    return "neutral"


def evaluate_single(row: dict) -> dict:
    """评估单条信号的准确性。"""
    direction = row.get("direction_expected") or infer_direction(
        str(row.get("decision", "")), float(row.get("final_score", 50) or 50)
    )
    ret_10d = float(row.get("ret_10d", 0) or 0)
    stop_loss = float(row.get("stop_loss", 0) or 0)
    take_profit = float(row.get("take_profit_1", 0) or 0)
    close = float(row.get("close_price", 0) or 0)

    direction_correct = (
        (direction == "up" and ret_10d > 0)
        or (direction == "down" and ret_10d < 0)
        or (direction == "neutral" and abs(ret_10d) < 2)
    )

    stop_hit = stop_loss > 0 and close > 0 and close * (1 + ret_10d / 100) <= stop_loss
    tp_hit = take_profit > 0 and close > 0 and close * (1 + ret_10d / 100) >= take_profit

    if direction == "up":
        sim_return = ret_10d
    elif direction == "down":
        sim_return = -ret_10d
    else:
        sim_return = 0.0

    return {
        "direction_correct": direction_correct,
        "stop_hit": stop_hit,
        "tp_hit": tp_hit,
        "simulated_return_pct": round(sim_return, 2),
        "outcome": "win" if sim_return > 0 else "loss" if sim_return < 0 else "flat",
    }


def compute_summary() -> dict:
    """计算整体回测统计。"""
    import pandas as pd

    if not SIGNAL_DB.exists():
        return {"error": "signal_history.csv 不存在"}

    df = pd.read_csv(SIGNAL_DB)
    total = len(df)
    if total == 0:
        return {"error": "无信号记录"}

    df["ret_10d"] = pd.to_numeric(df.get("ret_10d", pd.Series(dtype=float)), errors="coerce")
    evaluated = df.dropna(subset=["ret_10d"])
    n = len(evaluated)
    if n == 0:
        return {"total_signals": total, "evaluated": 0, "note": "尚无回填数据"}

    results = [evaluate_single(row.to_dict()) for _, row in evaluated.iterrows()]

    win_count = sum(1 for r in results if r["outcome"] == "win")
    direction_correct = sum(1 for r in results if r["direction_correct"])
    stop_hits = sum(1 for r in results if r["stop_hit"])
    tp_hits = sum(1 for r in results if r["tp_hit"])
    avg_return = sum(r["simulated_return_pct"] for r in results) / n

    buy_mask = evaluated["decision"].isin(["buy", "strong_buy"]) if "decision" in evaluated.columns else pd.Series(dtype=bool)
    sell_mask = evaluated["decision"].isin(["sell", "strong_sell"]) if "decision" in evaluated.columns else pd.Series(dtype=bool)

    return {
        "total_signals": total,
        "evaluated": n,
        "win_rate": round(win_count / n * 100, 1),
        "direction_accuracy": round(direction_correct / n * 100, 1),
        "stop_loss_hit_rate": round(stop_hits / n * 100, 1),
        "take_profit_hit_rate": round(tp_hits / n * 100, 1),
        "avg_simulated_return": round(avg_return, 2),
        "buy_count": int(buy_mask.sum()),
        "buy_win_rate": round(float((evaluated.loc[buy_mask, "ret_10d"] > 0).mean() * 100), 1) if buy_mask.any() else 0,
        "sell_count": int(sell_mask.sum()),
        "sell_win_rate": round(float((evaluated.loc[sell_mask, "ret_10d"] < 0).mean() * 100), 1) if sell_mask.any() else 0,
    }


def _fetch_daily_after(symbol: str, date_str: str, days: int = 25):
    """获取指定日期之后的日线数据。"""
    try:
        import akshare as ak
        from datetime import datetime, timedelta

        start = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
        end = start + timedelta(days=days + 15)
        df = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        if df is not None and not df.empty:
            col_map = {"收盘": "close", "日期": "date"}
            df = df.rename(columns=col_map)
            return df.head(days)
    except Exception:
        pass
    return None
