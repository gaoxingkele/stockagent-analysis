# -*- coding: utf-8 -*-
"""盘中异常预警：对比前日评分与实时走势，检测严重背离。"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .backtest import SIGNAL_DB


@dataclass
class AnomalyResult:
    symbol: str
    name: str
    signal_date: str
    signal_score: float
    signal_decision: str
    close_at_signal: float
    current_price: float
    intraday_pct: float
    volume_ratio: float | None
    level: str        # "severe" / "warning" / "normal"
    message: str


# ────────────────────────────────────────────────
# 1. 读取历史信号
# ────────────────────────────────────────────────

def load_recent_signals(
    symbols: list[str] | None = None,
    date_filter: str | None = None,
) -> list[dict[str, Any]]:
    """从 signal_history.csv 读取最近信号，按 (date,symbol) 去重取均值。"""
    if not SIGNAL_DB.exists():
        return []

    rows: list[dict[str, str]] = []
    with open(SIGNAL_DB, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))

    if not rows:
        return []

    # 确定目标日期
    if date_filter:
        target_date = date_filter
    else:
        all_dates = sorted(set(r.get("date", "") for r in rows if r.get("date")))
        if not all_dates:
            return []
        target_date = all_dates[-1]

    # 过滤日期
    filtered = [r for r in rows if r.get("date") == target_date]
    if symbols:
        sym_set = set(s.strip() for s in symbols)
        filtered = [r for r in filtered if r.get("symbol", "").strip() in sym_set]

    # 按 symbol 去重，取均值评分
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in filtered:
        groups[r.get("symbol", "")].append(r)

    results: list[dict[str, Any]] = []
    for sym, group in groups.items():
        scores = []
        for r in group:
            try:
                scores.append(float(r.get("final_score", 0)))
            except (ValueError, TypeError):
                pass
        avg_score = sum(scores) / len(scores) if scores else 0
        first = group[0]
        try:
            close_price = float(first.get("close_price", 0))
        except (ValueError, TypeError):
            close_price = 0.0
        # 根据均值评分重新推断决策
        if avg_score >= 65:
            decision = "buy"
        elif avg_score < 40:
            decision = "sell"
        else:
            decision = "hold"
        results.append({
            "symbol": sym,
            "name": first.get("name", ""),
            "date": target_date,
            "final_score": round(avg_score, 1),
            "decision": decision,
            "close_price": close_price,
        })

    return results


# ────────────────────────────────────────────────
# 2. 获取实时快照（批量）
# ────────────────────────────────────────────────

_spot_cache: Any = None


def _get_spot_df():
    """获取全市场实时行情（缓存，只调一次）。"""
    global _spot_cache
    if _spot_cache is not None:
        return _spot_cache
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and not df.empty:
            _spot_cache = df
            return df
    except Exception:
        pass
    return None


def fetch_realtime_snapshot(symbol: str) -> dict[str, Any]:
    """获取单只股票实时快照。"""
    df = _get_spot_df()
    if df is None:
        return {}

    # akshare 代码列为 "代码"
    code_col = "代码" if "代码" in df.columns else "code"
    match = df[df[code_col].astype(str).str.strip() == symbol.strip()]
    if match.empty:
        return {}

    row = match.iloc[0]
    try:
        result: dict[str, Any] = {
            "current_price": float(row.get("最新价", 0) or 0),
            "pct_change": float(row.get("涨跌幅", 0) or 0),
            "volume_ratio": None,
            "prev_close": float(row.get("昨收", 0) or 0),
        }
        vr = row.get("量比")
        if vr is not None and str(vr).strip() not in ("", "-", "nan"):
            result["volume_ratio"] = float(vr)
        return result
    except (ValueError, TypeError):
        return {}


# ────────────────────────────────────────────────
# 3. 异常分类
# ────────────────────────────────────────────────

_DEFAULT_THRESHOLDS = {
    "buy_warn_drop_pct": 3.0,
    "buy_severe_drop_pct": 5.0,
    "sell_warn_rise_pct": 5.0,
    "sell_severe_rise_pct": 8.0,
    "buy_score_min": 70,
    "sell_score_max": 40,
}


def classify_anomaly(
    signal: dict[str, Any],
    realtime: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> AnomalyResult:
    """将信号+实时数据分类为异常等级。"""
    th = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    score = float(signal.get("final_score", 50))
    pct = float(realtime.get("pct_change", 0))
    current_price = float(realtime.get("current_price", 0))
    vr = realtime.get("volume_ratio")

    buy_min = float(th["buy_score_min"])
    sell_max = float(th["sell_score_max"])

    level, message = "normal", "走势符合预期"

    if score >= buy_min:
        if pct <= -float(th["buy_severe_drop_pct"]):
            level = "severe"
            message = f"评分{score:.0f}(买入)但盘中跌{pct:.1f}%，严重背离"
        elif pct <= -float(th["buy_warn_drop_pct"]):
            level = "warning"
            message = f"评分{score:.0f}(买入)但盘中跌{pct:.1f}%，需关注"
    elif score < sell_max:
        if pct >= float(th["sell_severe_rise_pct"]):
            level = "severe"
            message = f"评分{score:.0f}(卖出)但盘中涨{pct:.1f}%，严重背离"
        elif pct >= float(th["sell_warn_rise_pct"]):
            level = "warning"
            message = f"评分{score:.0f}(卖出)但盘中涨{pct:.1f}%，需关注"

    return AnomalyResult(
        symbol=signal.get("symbol", ""),
        name=signal.get("name", ""),
        signal_date=signal.get("date", ""),
        signal_score=score,
        signal_decision=signal.get("decision", ""),
        close_at_signal=float(signal.get("close_price", 0)),
        current_price=current_price,
        intraday_pct=pct,
        volume_ratio=vr,
        level=level,
        message=message,
    )


# ────────────────────────────────────────────────
# 4. 主流程
# ────────────────────────────────────────────────

def run_intraday_check(
    symbols: list[str] | None = None,
    date_filter: str | None = None,
    config: dict[str, Any] | None = None,
) -> list[AnomalyResult]:
    """执行盘中异常预警。"""
    cfg = (config or {}).get("intraday_check", {})
    thresholds = cfg.get("thresholds", {})

    signals = load_recent_signals(symbols=symbols, date_filter=date_filter)
    if not signals:
        print("[盘中预警] 未找到匹配的历史信号记录", flush=True)
        return []

    # 市场时段检查
    now = datetime.now()
    hour = now.hour
    if hour < 9 or hour >= 16:
        print(f"[盘中预警] 当前时间 {now.strftime('%H:%M')} 不在交易时段(9:30-15:00)，数据可能为昨日收盘价", flush=True)

    results: list[AnomalyResult] = []
    for sig in signals:
        realtime = fetch_realtime_snapshot(sig["symbol"])
        if not realtime:
            results.append(AnomalyResult(
                symbol=sig["symbol"], name=sig.get("name", ""),
                signal_date=sig["date"], signal_score=sig["final_score"],
                signal_decision=sig["decision"],
                close_at_signal=sig.get("close_price", 0),
                current_price=0, intraday_pct=0, volume_ratio=None,
                level="unknown", message="无法获取实时数据",
            ))
            continue
        results.append(classify_anomaly(sig, realtime, thresholds))

    # 排序：severe > warning > normal
    order = {"severe": 0, "warning": 1, "unknown": 2, "normal": 3}
    results.sort(key=lambda r: (order.get(r.level, 9), -r.signal_score))
    return results


# ────────────────────────────────────────────────
# 5. 格式化输出
# ────────────────────────────────────────────────

_DECISION_CN = {
    "buy": "买入", "sell": "卖出", "hold": "观望",
    "strong_buy": "强买", "strong_sell": "强卖",
}

_LEVEL_ICON = {
    "severe": "🔴严重",
    "warning": "⚠️预警",
    "normal": "✅正常",
    "unknown": "❓未知",
}


def print_anomaly_table(results: list[AnomalyResult]) -> None:
    """格式化输出异常预警表。"""
    if not results:
        print("[盘中预警] 无结果")
        return

    print()
    print("=" * 90)
    print(f"  盘中异常预警  |  信号日期: {results[0].signal_date}  |  检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 90)

    # 表头
    header = f"{'#':>2}  {'代码':<8} {'名称':<8} {'评分':>5} {'建议':<5} {'信号价':>8} {'现价':>8} {'涨跌幅':>7} {'量比':>5} {'状态':<8} {'说明'}"
    print(header)
    print("-" * 90)

    for i, r in enumerate(results, 1):
        decision_cn = _DECISION_CN.get(r.signal_decision, r.signal_decision)
        level_icon = _LEVEL_ICON.get(r.level, r.level)
        vr_str = f"{r.volume_ratio:.1f}" if r.volume_ratio is not None else " -"
        price_str = f"{r.current_price:.2f}" if r.current_price > 0 else "  N/A"
        pct_str = f"{r.intraday_pct:+.1f}%" if r.current_price > 0 else "  N/A"
        close_str = f"{r.close_at_signal:.2f}" if r.close_at_signal > 0 else "  N/A"

        print(
            f"{i:>2}  {r.symbol:<8} {r.name:<8} {r.signal_score:>5.1f} {decision_cn:<5} "
            f"{close_str:>8} {price_str:>8} {pct_str:>7} {vr_str:>5} {level_icon:<8} {r.message}"
        )

    print("-" * 90)

    # 汇总
    severe = sum(1 for r in results if r.level == "severe")
    warning = sum(1 for r in results if r.level == "warning")
    normal = sum(1 for r in results if r.level == "normal")
    unknown = sum(1 for r in results if r.level == "unknown")
    parts = []
    if severe:
        parts.append(f"🔴严重:{severe}")
    if warning:
        parts.append(f"⚠️预警:{warning}")
    if normal:
        parts.append(f"✅正常:{normal}")
    if unknown:
        parts.append(f"❓未知:{unknown}")
    print(f"  合计: {len(results)}只 | {' | '.join(parts)}")
    print()
