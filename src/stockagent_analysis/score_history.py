# -*- coding: utf-8 -*-
"""评分历史记录与逐日对比分析模块。

每次分析完成后，将评分快照存储到 output/history/{symbol}/{date}.json，
支持跨日对比、可解释性分析和趋势追踪。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


HISTORY_DIR = Path("output/history")


def _get_data_date(run_dir: str | Path) -> str:
    """从run目录的日线K线CSV获取最后有效数据日期。"""
    kline_csv = Path(run_dir) / "data" / "kline" / "day.csv"
    if kline_csv.exists():
        with open(kline_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if last_line:
                    return last_line.split(",")[0]  # 第一列是日期
    return datetime.now().strftime("%Y-%m-%d")


def save_score_snapshot(
    run_dir: str | Path,
    output: dict[str, Any],
) -> Path:
    """将本次分析的评分快照保存到历史目录。

    存储路径: output/history/{symbol}/{data_date}.json
    data_date: 最新日线K线的日期（非运行日期）

    保存内容:
    - 有效数据日期、运行时间
    - 最终评分、决策、阈值
    - 各Provider的加权总分
    - 各Provider×各Agent的评分矩阵
    - 归一化后的Agent权重
    - 关键特征值（close, pct_chg, pe_ttm, volume_ratio等）
    - 本地评分（_simple_policy的30维分数）
    """
    symbol = output.get("symbol", "unknown")
    data_date = _get_data_date(run_dir)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    features = output.get("analysis_features", {})

    # 提取本地评分（来自_simple_policy）
    local_scores = {}
    agent_votes = output.get("agent_votes", [])
    if isinstance(agent_votes, list):
        for av in agent_votes:
            if isinstance(av, dict):
                dim = av.get("dim_code", "")
                local_s = av.get("local_score")
                if dim and local_s is not None:
                    local_scores[dim] = local_s

    snapshot = {
        "symbol": symbol,
        "name": output.get("name", ""),
        "data_date": data_date,
        "run_time": run_time,
        # 最终结果
        "final_score": output.get("final_score"),
        "final_decision": output.get("final_decision"),
        "decision_level": output.get("decision_level"),
        "thresholds": output.get("thresholds"),
        # 各Provider总分
        "model_totals": output.get("model_totals", {}),
        # 各Provider×各Agent评分矩阵
        "model_scores": output.get("model_scores", {}),
        # 归一化权重
        "model_weights": {
            p: w for p, w in (list(output.get("model_weights", {}).items())[:1])
        } if output.get("model_weights") else {},
        # 关键特征
        "features": {
            k: features.get(k) for k in [
                "close", "pct_chg", "pe_ttm", "turnover_rate", "pb",
                "total_mv", "momentum_20", "volatility_20", "drawdown_60",
                "volume_ratio_5_20", "trend_strength", "news_sentiment",
                "news_count", "relative_strength",
            ] if features.get(k) is not None
        },
        # 本地30维评分
        "local_scores": local_scores,
    }

    # 保存
    sym_dir = HISTORY_DIR / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    out_path = sym_dir / f"{data_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    print(f"[历史] 已保存 {symbol} {data_date} 评分快照 → {out_path}", flush=True)
    return out_path


def load_history(symbol: str, last_n: int = 5) -> list[dict[str, Any]]:
    """加载某只股票最近N天的评分历史，按日期倒序。"""
    sym_dir = HISTORY_DIR / symbol
    if not sym_dir.exists():
        return []
    files = sorted(sym_dir.glob("*.json"), reverse=True)
    results = []
    for f in files[:last_n]:
        with open(f, "r", encoding="utf-8") as fp:
            results.append(json.load(fp))
    return results


def compare_scores(symbol: str, last_n: int = 3) -> Optional[str]:
    """生成逐日评分对比分析报告。

    对比最近N天的评分变化，找出变化最大的Agent维度，
    给出可解释性的分数变化分析。
    """
    history = load_history(symbol, last_n)
    if len(history) < 2:
        return None

    lines: list[str] = []
    lines.append(f"## {symbol} {history[0].get('name', '')} 逐日评分对比")
    lines.append("")

    # 总分趋势
    lines.append("### 总分趋势")
    lines.append("| 日期 | 综合评分 | 变化 | 决策 | 收盘价 | 涨跌幅 |")
    lines.append("|------|--------:|-----:|------|-------:|-------:|")
    for i, h in enumerate(history):
        score = h.get("final_score", 0)
        delta = ""
        if i + 1 < len(history):
            prev = history[i + 1].get("final_score", 0)
            d = score - prev
            delta = f"{d:+.1f}"
        decision = h.get("decision_level", "")
        close = h.get("features", {}).get("close", "")
        pct = h.get("features", {}).get("pct_chg", "")
        pct_str = f"{pct:+.2f}%" if isinstance(pct, (int, float)) else ""
        lines.append(f"| {h['data_date']} | {score:.1f} | {delta} | {decision} | {close} | {pct_str} |")

    # 逐日Agent评分变化（取最近两天）
    curr = history[0]
    prev = history[1]
    lines.append("")
    lines.append(f"### Agent评分变化 ({prev['data_date']} → {curr['data_date']})")

    # 取所有Provider的平均分对比
    curr_scores = _avg_agent_scores(curr.get("model_scores", {}))
    prev_scores = _avg_agent_scores(prev.get("model_scores", {}))

    all_agents = sorted(set(list(curr_scores.keys()) + list(prev_scores.keys())))
    changes = []
    for agent in all_agents:
        c = curr_scores.get(agent, 0)
        p = prev_scores.get(agent, 0)
        d = c - p
        if abs(d) >= 0.5:
            changes.append((agent, p, c, d))

    changes.sort(key=lambda x: abs(x[3]), reverse=True)

    if changes:
        lines.append("| Agent | 前日 | 当日 | 变化 | 影响 |")
        lines.append("|-------|-----:|-----:|-----:|------|")
        for agent, p, c, d in changes[:15]:
            impact = "利多" if d > 0 else "利空"
            emoji = "↑" if d > 0 else "↓"
            lines.append(f"| {agent} | {p:.1f} | {c:.1f} | {d:+.1f}{emoji} | {impact} |")

    # 关键特征变化
    lines.append("")
    lines.append("### 关键指标变化")
    curr_feat = curr.get("features", {})
    prev_feat = prev.get("features", {})
    feat_labels = {
        "close": "收盘价", "pct_chg": "涨跌幅%",
        "volume_ratio_5_20": "量比(5/20)", "momentum_20": "20日动量",
        "volatility_20": "20日波动率", "trend_strength": "趋势强度",
    }
    feat_changes = []
    for key, label in feat_labels.items():
        cv = curr_feat.get(key)
        pv = prev_feat.get(key)
        if cv is not None and pv is not None:
            try:
                cv, pv = float(cv), float(pv)
                feat_changes.append((label, pv, cv, cv - pv))
            except (ValueError, TypeError):
                pass
    if feat_changes:
        lines.append("| 指标 | 前日 | 当日 | 变化 |")
        lines.append("|------|-----:|-----:|-----:|")
        for label, pv, cv, d in feat_changes:
            lines.append(f"| {label} | {pv:.4g} | {cv:.4g} | {d:+.4g} |")

    # 如果有3天数据，加趋势判断
    if len(history) >= 3:
        scores_3d = [h.get("final_score", 0) for h in history[:3]]
        lines.append("")
        if scores_3d[0] > scores_3d[1] > scores_3d[2]:
            lines.append("**趋势**: 连续上升 ↑↑")
        elif scores_3d[0] < scores_3d[1] < scores_3d[2]:
            lines.append("**趋势**: 连续下降 ↓↓")
        elif scores_3d[0] > scores_3d[1] and scores_3d[1] < scores_3d[2]:
            lines.append("**趋势**: V型反弹 ↓↑")
        elif scores_3d[0] < scores_3d[1] and scores_3d[1] > scores_3d[2]:
            lines.append("**趋势**: 倒V回落 ↑↓")
        else:
            lines.append("**趋势**: 震荡")

    return "\n".join(lines)


def _avg_agent_scores(model_scores: dict) -> dict[str, float]:
    """计算所有Provider的Agent平均分。"""
    if not model_scores:
        return {}
    all_agents: dict[str, list[float]] = {}
    for provider, agents in model_scores.items():
        if not isinstance(agents, dict):
            continue
        for agent, score in agents.items():
            try:
                s = float(score)
            except (ValueError, TypeError):
                continue
            all_agents.setdefault(agent, []).append(s)
    return {a: sum(v) / len(v) for a, v in all_agents.items() if v}
