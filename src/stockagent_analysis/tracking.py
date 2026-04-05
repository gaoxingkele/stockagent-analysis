# -*- coding: utf-8 -*-
"""评分追踪与校准系统。

评分后持续跟踪股票涨跌，验证评分系统准确性，输出权重修正建议。

分数-仓位映射:
  95+ → 满仓    期望 ret_5d > +5%
  90  → 80%     期望 ret_5d > +3%
  85  → 65%     期望 ret_5d > +2%
  80  → 50%     期望 ret_5d > +1%
  75  → 35%     期望 ret_5d >  0%
  70  → 25%     期望 ret_5d > -1%
  65  → 15%     期望 ret_5d > -2%
  45~65 → 观望   期望 |ret_5d| < 5%
  40  → 卖15%   期望 ret_5d < +2%
  35  → 卖30%   期望 ret_5d <  0%
  30  → 卖50%   期望 ret_5d < -1%
  25  → 卖70%   期望 ret_5d < -3%
  <25 → 清仓    期望 ret_5d < -5%
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

TRACKING_DB = Path("output/tracking.csv")

COLUMNS = [
    "eval_date",        # 评分日期
    "data_date",        # 数据日期（K线最后日期）
    "symbol",
    "name",
    "final_score",
    "decision",
    "decision_level",
    "base_close",       # 评分当日收盘价
    # 12个智能体维度分数
    "trend_momentum", "capital_liquidity", "fundamental", "tech_quant",
    "pattern", "sentiment_flow", "chanlun", "divergence",
    "kline_vision", "resonance", "volume_structure", "deriv_margin",
    # T+N 收盘价
    "close_1d", "close_2d", "close_3d", "close_5d", "close_10d", "close_20d",
    # T+N 涨跌幅%
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    # 区间最大回撤% 和最大涨幅%
    "max_drawdown_5d", "max_gain_5d",
    "max_drawdown_10d", "max_gain_10d",
    # 结果判定
    "score_band",       # 所属分数段
    "expected_dir",     # 预期方向 up/neutral/down
    "outcome",          # correct / neutral / wrong
    "filled_days",      # 已回填交易日数
    # 来源
    "run_dir",
    "providers",
]

# ── 分数段定义 ──────────────────────────────────────────────
SCORE_BANDS = [
    # (下限, 上限, 标签, 仓位%, 预期方向, ret_5d阈值用于判定correct)
    (95, 100, "S95+",  100, "up",      5.0),
    (90,  95, "S90",    80, "up",      3.0),
    (85,  90, "S85",    65, "up",      2.0),
    (80,  85, "S80",    50, "up",      1.0),
    (75,  80, "S75",    35, "up",      0.0),
    (70,  75, "S70",    25, "up",     -1.0),
    (65,  70, "S65",    15, "up",     -2.0),
    (45,  65, "WATCH",   0, "neutral", None),  # 观望区, |ret| < 5%
    (40,  45, "S40",   -15, "down",    2.0),
    (35,  40, "S35",   -30, "down",    0.0),
    (30,  35, "S30",   -50, "down",   -1.0),
    (25,  30, "S25",   -70, "down",   -3.0),
    ( 0,  25, "S<25", -100, "down",   -5.0),
]


def _get_band(score: float) -> tuple[str, str, float | None]:
    """返回 (band_label, expected_dir, threshold)。"""
    for lo, hi, label, _pos, direction, threshold in SCORE_BANDS:
        if lo <= score < hi or (hi == 100 and score >= 95):
            return label, direction, threshold
    return "WATCH", "neutral", None


def _judge_outcome(score: float, ret_5d: float) -> str:
    """根据分数段和实际5日收益判定 correct/neutral/wrong。"""
    label, direction, threshold = _get_band(score)

    if direction == "up":
        # 买入区: ret >= threshold → correct, ret在[-5%, threshold)→ neutral, ret < -5% → wrong
        if ret_5d >= threshold:
            return "correct"
        elif ret_5d >= -5.0:
            return "neutral"
        else:
            return "wrong"
    elif direction == "down":
        # 卖出区: ret <= threshold → correct, ret在(threshold, 5%] → neutral, ret > 5% → wrong
        if ret_5d <= threshold:
            return "correct"
        elif ret_5d <= 5.0:
            return "neutral"
        else:
            return "wrong"
    else:
        # 观望区: |ret| <= 5% → correct, else → wrong
        if abs(ret_5d) <= 5.0:
            return "correct"
        else:
            return "wrong"


# ── 记录器 ──────────────────────────────────────────────────
AGENT_DIM_MAP = {
    "trend_momentum_agent": "trend_momentum",
    "capital_liquidity_agent": "capital_liquidity",
    "fundamental_agent": "fundamental",
    "tech_quant_agent": "tech_quant",
    "pattern_agent": "pattern",
    "sentiment_flow_agent": "sentiment_flow",
    "chanlun_agent": "chanlun",
    "divergence_agent": "divergence",
    "kline_vision_agent": "kline_vision",
    "resonance_agent": "resonance",
    "volume_structure_agent": "volume_structure",
    "deriv_margin_agent": "deriv_margin",
}


def record_evaluation(output: dict[str, Any], run_dir: str | Path = "") -> None:
    """评分完成后记录一行到 tracking.csv。"""
    score = float(output.get("final_score", 0))
    band_label, expected_dir, _ = _get_band(score)

    # 从 analysis_features 取收盘价
    features = output.get("analysis_features", {})
    base_close = features.get("close", 0)

    row: dict[str, Any] = {
        "eval_date": datetime.now().strftime("%Y-%m-%d"),
        "data_date": output.get("data_date", ""),
        "symbol": output.get("symbol", ""),
        "name": output.get("name", ""),
        "final_score": round(score, 2),
        "decision": output.get("final_decision", ""),
        "decision_level": output.get("decision_level", ""),
        "base_close": base_close,
        "score_band": band_label,
        "expected_dir": expected_dir,
        "outcome": "",
        "filled_days": 0,
        "run_dir": str(run_dir),
        "providers": ",".join(output.get("multi_eval_providers", [])),
    }

    # 12个智能体维度: 取所有模型的加权平均分
    model_scores = output.get("model_scores", {})
    if model_scores:
        # model_scores = {provider: {agent_id: score, ...}, ...}
        agent_sums: dict[str, list[float]] = {}
        for _provider, agents in model_scores.items():
            if not isinstance(agents, dict):
                continue
            for agent_id, s in agents.items():
                v = _safe_float(s)
                if v > 0:
                    agent_sums.setdefault(agent_id, []).append(v)
        for agent_id, col_name in AGENT_DIM_MAP.items():
            vals = agent_sums.get(agent_id, [])
            row[col_name] = round(sum(vals) / len(vals), 1) if vals else ""
    else:
        for col_name in AGENT_DIM_MAP.values():
            row[col_name] = ""

    # T+N 字段留空
    for n in [1, 2, 3, 5, 10, 20]:
        row[f"close_{n}d"] = ""
        row[f"ret_{n}d"] = ""
    for n in [5, 10]:
        row[f"max_drawdown_{n}d"] = ""
        row[f"max_gain_{n}d"] = ""

    TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)
    file_exists = TRACKING_DB.exists()
    with open(TRACKING_DB, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# ── 回填器 ──────────────────────────────────────────────────
def backfill_returns(force: bool = False) -> dict[str, int]:
    """批量回填 tracking.csv 中未完成的记录。

    返回 {"total": N, "filled": M, "skipped": K}。
    """
    import pandas as pd

    if not TRACKING_DB.exists():
        print("[追踪] tracking.csv 不存在，无需回填")
        return {"total": 0, "filled": 0, "skipped": 0}

    df = pd.read_csv(TRACKING_DB, dtype=str)
    if df.empty:
        return {"total": 0, "filled": 0, "skipped": 0}

    filled = 0
    skipped = 0
    today = datetime.now()

    for idx, row in df.iterrows():
        filled_days = int(row.get("filled_days") or 0)
        if filled_days >= 20 and not force:
            skipped += 1
            continue

        symbol = str(row.get("symbol", ""))
        data_date = str(row.get("data_date") or row.get("eval_date", ""))
        base_close = _safe_float(row.get("base_close"))
        if not symbol or not data_date or base_close <= 0:
            skipped += 1
            continue

        # 标准化日期格式
        data_date = data_date.replace("/", "-")
        if len(data_date) == 8 and "-" not in data_date:  # "20260302" → "2026-03-02"
            data_date = f"{data_date[:4]}-{data_date[4:6]}-{data_date[6:]}"

        # 至少需要1个交易日才回填
        eval_dt = datetime.strptime(data_date, "%Y-%m-%d")
        if (today - eval_dt).days < 1:
            skipped += 1
            continue

        hist = _fetch_daily_after(symbol, data_date, days=25)
        if hist is None or hist.empty:
            skipped += 1
            continue

        n_days = len(hist)
        closes = hist["close"].astype(float).tolist()

        # 回填 close_Nd 和 ret_Nd
        for n in [1, 2, 3, 5, 10, 20]:
            if n <= n_days:
                c = closes[n - 1]
                df.at[idx, f"close_{n}d"] = str(round(c, 2))
                df.at[idx, f"ret_{n}d"] = str(round((c / base_close - 1) * 100, 2))

        # 区间最大回撤和最大涨幅
        for window in [5, 10]:
            if window <= n_days:
                window_closes = closes[:window]
                rets = [(c / base_close - 1) * 100 for c in window_closes]
                df.at[idx, f"max_drawdown_{window}d"] = str(round(min(rets), 2))
                df.at[idx, f"max_gain_{window}d"] = str(round(max(rets), 2))

        df.at[idx, "filled_days"] = str(min(n_days, 20))

        # 判定 outcome（需要至少5天数据）
        if n_days >= 5:
            score = _safe_float(row.get("final_score"))
            ret_5d = _safe_float(df.at[idx, "ret_5d"])
            df.at[idx, "outcome"] = _judge_outcome(score, ret_5d)

        filled += 1

    df.to_csv(TRACKING_DB, index=False, encoding="utf-8")
    print(f"[追踪] 回填完成: {filled} 条更新, {skipped} 条跳过")
    return {"total": len(df), "filled": filled, "skipped": skipped}


# ── 校准器 ──────────────────────────────────────────────────
def calibrate(min_samples: int = 10) -> dict[str, Any]:
    """分析评分-收益相关性，输出校准报告。

    返回结构:
    {
        "total_evaluated": N,
        "overall_ic_5d": float,    # 评分 vs 5日收益 rank correlation
        "overall_ic_10d": float,
        "band_stats": {            # 每个分数段的统计
            "S95+": {"count": N, "avg_ret_5d": X, "correct_rate": Y, ...},
            ...
        },
        "agent_ic": {              # 每个智能体维度的 IC
            "trend_momentum": {"ic_5d": X, "ic_10d": Y},
            ...
        },
        "weight_suggestions": {    # 权重修正建议
            "trend_momentum": {"current_ic": X, "action": "increase/decrease/keep"},
            ...
        },
    }
    """
    import pandas as pd

    report: dict[str, Any] = {}

    if not TRACKING_DB.exists():
        return {"error": "tracking.csv 不存在"}

    df = pd.read_csv(TRACKING_DB)
    # 只用有5日数据的记录
    df["ret_5d"] = pd.to_numeric(df["ret_5d"], errors="coerce")
    df["ret_10d"] = pd.to_numeric(df["ret_10d"], errors="coerce")
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    evaluated = df.dropna(subset=["ret_5d", "final_score"])

    if len(evaluated) < min_samples:
        return {"error": f"样本不足: {len(evaluated)}/{min_samples}"}

    report["total_evaluated"] = len(evaluated)

    # 1. 整体 IC（Spearman rank correlation）
    report["overall_ic_5d"] = round(float(
        evaluated["final_score"].corr(evaluated["ret_5d"], method="spearman")
    ), 4)
    valid_10d = evaluated.dropna(subset=["ret_10d"])
    if len(valid_10d) > 5:
        report["overall_ic_10d"] = round(float(
            valid_10d["final_score"].corr(valid_10d["ret_10d"], method="spearman")
        ), 4)

    # 2. 分数段统计
    band_stats = {}
    for lo, hi, label, position_pct, direction, threshold in SCORE_BANDS:
        mask = (evaluated["final_score"] >= lo) & (evaluated["final_score"] < hi)
        if label == "S95+":
            mask = evaluated["final_score"] >= 95
        subset = evaluated[mask]
        if len(subset) == 0:
            continue
        n = len(subset)
        avg_ret_5d = round(float(subset["ret_5d"].mean()), 2)
        avg_ret_10d = round(float(subset["ret_10d"].mean()), 2) if "ret_10d" in subset and subset["ret_10d"].notna().any() else None

        # correct rate
        outcomes = subset["outcome"].value_counts()
        correct_n = int(outcomes.get("correct", 0))
        wrong_n = int(outcomes.get("wrong", 0))
        neutral_n = int(outcomes.get("neutral", 0))

        # 模拟收益 = 仓位% × ret_5d
        sim_return = round(abs(position_pct) / 100 * avg_ret_5d, 2) if position_pct != 0 else 0

        band_stats[label] = {
            "count": n,
            "position_pct": position_pct,
            "direction": direction,
            "threshold": threshold,
            "avg_ret_5d": avg_ret_5d,
            "avg_ret_10d": avg_ret_10d,
            "correct": correct_n,
            "neutral": neutral_n,
            "wrong": wrong_n,
            "correct_rate": round(correct_n / n * 100, 1) if n > 0 else 0,
            "wrong_rate": round(wrong_n / n * 100, 1) if n > 0 else 0,
            "simulated_return": sim_return,
        }
    report["band_stats"] = band_stats

    # 3. 各智能体维度 IC
    agent_ic = {}
    for col in AGENT_DIM_MAP.values():
        if col not in evaluated.columns:
            continue
        evaluated[col] = pd.to_numeric(evaluated[col], errors="coerce")
        valid = evaluated[[col, "ret_5d", "ret_10d"]].dropna(subset=[col, "ret_5d"])
        if len(valid) < 5:
            continue
        ic_5d = round(float(valid[col].corr(valid["ret_5d"], method="spearman")), 4)
        ic_10d = None
        valid_10 = valid.dropna(subset=["ret_10d"])
        if len(valid_10) > 5:
            ic_10d = round(float(valid_10[col].corr(valid_10["ret_10d"], method="spearman")), 4)
        agent_ic[col] = {"ic_5d": ic_5d, "ic_10d": ic_10d}
    report["agent_ic"] = agent_ic

    # 4. 权重修正建议
    suggestions = {}
    for col, ics in agent_ic.items():
        ic5 = ics["ic_5d"]
        if ic5 > 0.15:
            action = "increase"
        elif ic5 < -0.05:
            action = "decrease"
        else:
            action = "keep"
        suggestions[col] = {"ic_5d": ic5, "action": action}
    report["weight_suggestions"] = suggestions

    return report


def print_calibration_report(report: dict[str, Any]) -> None:
    """终端打印校准报告。"""
    if "error" in report:
        print(f"[校准] {report['error']}")
        return

    print()
    print("=" * 100)
    print("评分追踪校准报告")
    print("=" * 100)
    print(f"样本数: {report['total_evaluated']}")
    print(f"整体IC(5日): {report.get('overall_ic_5d', 'N/A')}")
    print(f"整体IC(10日): {report.get('overall_ic_10d', 'N/A')}")

    # 分数段表格
    print()
    print("-" * 100)
    print(f"{'分数段':>6} | {'数量':>4} | {'仓位':>5} | {'方向':>7} | {'5日均涨':>7} | {'10日均涨':>8} | "
          f"{'正确':>4} | {'中性':>4} | {'错误':>4} | {'正确率':>6} | {'模拟收益':>8}")
    print("-" * 100)

    band_stats = report.get("band_stats", {})
    # 按分数段从高到低排
    band_order = ["S95+", "S90", "S85", "S80", "S75", "S70", "S65",
                  "WATCH", "S40", "S35", "S30", "S25", "S<25"]
    total_sim = 0
    for label in band_order:
        s = band_stats.get(label)
        if not s:
            continue
        pos = f"{s['position_pct']:+d}%" if s['position_pct'] != 0 else "观望"
        ret10 = f"{s['avg_ret_10d']:+.2f}%" if s['avg_ret_10d'] is not None else "N/A"
        total_sim += s['simulated_return']
        print(f"{label:>6} | {s['count']:>4} | {pos:>5} | {s['direction']:>7} | "
              f"{s['avg_ret_5d']:+7.2f}% | {ret10:>8} | "
              f"{s['correct']:>4} | {s['neutral']:>4} | {s['wrong']:>4} | "
              f"{s['correct_rate']:>5.1f}% | {s['simulated_return']:+7.2f}%")
    print("-" * 100)
    print(f"{'合计模拟收益':>50} {total_sim:+.2f}%")

    # 智能体IC表
    agent_ic = report.get("agent_ic", {})
    if agent_ic:
        print()
        print("-" * 60)
        print(f"{'智能体维度':<20} | {'IC(5日)':>8} | {'IC(10日)':>8} | {'建议':>8}")
        print("-" * 60)
        suggestions = report.get("weight_suggestions", {})
        sorted_agents = sorted(agent_ic.items(), key=lambda x: abs(x[1]["ic_5d"]), reverse=True)
        for col, ics in sorted_agents:
            ic10 = f"{ics['ic_10d']:+.4f}" if ics['ic_10d'] is not None else "N/A"
            action = suggestions.get(col, {}).get("action", "")
            action_cn = {"increase": "加权", "decrease": "降权", "keep": "保持"}.get(action, "")
            print(f"{col:<20} | {ics['ic_5d']:>+8.4f} | {ic10:>8} | {action_cn:>8}")

    print()
    print("=" * 100)


# ── 工具函数 ────────────────────────────────────────────────
_proxy_disabled = False


def _disable_proxy():
    """彻底禁用代理（包括 Windows 注册表级系统代理）。"""
    global _proxy_disabled
    if _proxy_disabled:
        return
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
              "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    # Patch urllib.request.getproxies 返回空（绕过 Windows 注册表代理）
    import urllib.request
    urllib.request.getproxies = lambda: {}
    # Patch requests.Session 禁用 trust_env 和代理
    try:
        import requests
        _orig_init = requests.Session.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.trust_env = False
            self.proxies = {"http": None, "https": None}

        requests.Session.__init__ = _patched_init
    except ImportError:
        pass
    _proxy_disabled = True


def _safe_float(v) -> float:
    try:
        return float(v) if v is not None and str(v).strip() not in ("", "nan") else 0.0
    except (ValueError, TypeError):
        return 0.0


def _fetch_daily_after(symbol: str, date_str: str, days: int = 25):
    """获取指定日期之后的日线数据。优先TDX本地，fallback到akshare。"""
    import pandas as pd

    start_dt = datetime.strptime(date_str, "%Y-%m-%d")

    # 方式1: TDX 本地数据（最快最可靠）
    try:
        from .data_backend import DataBackend
        backend = DataBackend(mode="combined", default_sources=["tdx", "akshare"])
        df = backend._fetch_kline_tdx(symbol, "day", limit=250)
        if df is not None and not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
            after = df[df["ts"] > pd.Timestamp(start_dt)].head(days)
            if not after.empty:
                after = after.rename(columns={"ts": "date"})
                return after.reset_index(drop=True)
    except Exception:
        pass

    # 方式2: akshare（需网络）
    try:
        _disable_proxy()
        import akshare as ak
        start = start_dt + timedelta(days=1)
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
    except Exception as e:
        print(f"  [追踪] {symbol} 数据拉取失败: {e}")
    return None


# ── 历史数据补录（从已有 output/history/ 导入） ──────────────
def import_from_history(history_dir: Path | str = "output/history") -> int:
    """将 output/history/{symbol}/{date}.json 中的历史评分导入 tracking.csv。
    仅导入 tracking.csv 中不存在的记录。返回导入条数。
    """
    history_dir = Path(history_dir)
    if not history_dir.exists():
        print("[追踪] history 目录不存在")
        return 0

    # 已有记录的去重集合
    existing = set()
    if TRACKING_DB.exists():
        with open(TRACKING_DB, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row.get('symbol')}_{row.get('data_date') or row.get('eval_date')}"
                existing.add(key)

    imported = 0
    for symbol_dir in sorted(history_dir.iterdir()):
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name
        for json_file in sorted(symbol_dir.glob("*.json")):
            date_str = json_file.stem  # "2026-04-03"
            key = f"{symbol}_{date_str}"
            if key in existing:
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                score = _safe_float(data.get("final_score"))
                if score <= 0:
                    continue
                # 构造 output dict 兼容 record_evaluation
                # history JSON 的 features 字段名可能是 features 或 analysis_features
                features = data.get("analysis_features") or data.get("features") or {}
                pseudo_output = {
                    "symbol": symbol,
                    "name": data.get("name", ""),
                    "final_score": score,
                    "final_decision": data.get("final_decision", ""),
                    "decision_level": data.get("decision_level", ""),
                    "data_date": data.get("data_date", date_str),
                    "analysis_features": features,
                    "model_scores": data.get("model_scores", {}),
                    "multi_eval_providers": data.get("multi_eval_providers", []),
                }
                record_evaluation(pseudo_output, run_dir=data.get("run_dir", ""))
                imported += 1
            except Exception as e:
                print(f"  [追踪] 导入 {json_file} 失败: {e}")

    print(f"[追踪] 从历史目录导入 {imported} 条记录")
    return imported
