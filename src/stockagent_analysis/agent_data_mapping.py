# -*- coding: utf-8 -*-
"""Agent 与本地/云端数据对照关系。调用大模型前检查各 Agent 所需数据是否从本地提供。"""

from __future__ import annotations

from typing import Any


# required_data_points -> 本地数据来源
LOCAL_DATA_MAPPING: dict[str, list[str]] = {
    "日/周/月K线": ["kline_indicators", "kline:month,week,day"],
    "30min/60min/日/周/月K线": ["kline_indicators", "kline:month,week,day"],  # 兼容旧配置，现统一为日/周/月
    "HH/HL/LH/LL结构点": ["kline_indicators", "features.trend_strength"],
    "RSI": ["kline_indicators", "features"],
    "MACD": ["kline_indicators", "features"],
    "KDJ": ["kline_indicators", "features"],
    "布林带": ["kline_indicators", "features"],
    "MA5-250": ["kline_indicators", "historical_daily"],
    "营收利润": ["fundamentals", "snapshot"],
    "ROE/毛利率": ["fundamentals"],
    "估值PE/PB": ["fundamentals", "snapshot.pe_ttm", "snapshot"],
    "资产负债结构": ["fundamentals"],
    "主力净流入": ["features.volume_ratio", "snapshot"],
    "超大单/大单": ["features", "snapshot"],
    "筹码峰": ["fundamentals", "features"],
    "北向资金": ["fundamentals", "news"],
    "新闻热度": ["news", "features.news_sentiment", "features.news_count"],
    "社媒情绪": ["news", "features.news_sentiment"],
    "公告舆情": ["news"],
    "事件冲击": ["news"],
    "利率": ["fundamentals", "news"],
    "汇率": ["fundamentals", "news"],
    "大宗商品": ["news"],
    "宏观事件": ["news"],
    "沪深300相关性": ["features", "kline_indicators"],
    "行业Beta": ["fundamentals", "features"],
    "指数回归系数": ["features", "kline_indicators"],
    "因子暴露": ["features", "kline_indicators"],
    "回测片段": ["historical_daily", "kline_indicators"],
    "短中期信号": ["features", "kline_indicators"],
    "问询函": ["news"],
    "处罚公告": ["news"],
    "合规事件": ["news"],
    "政策约束": ["news"],
    "股东户数": ["fundamentals"],
    "前十大股东": ["fundamentals"],
    "机构持仓变化": ["fundamentals"],
    "盘口异动": ["snapshot", "features.volume_ratio"],
    "拉升回落模式": ["kline_indicators", "features"],
    "1h/2h/日线/周线K线": ["kline_indicators", "kline:day,week,month"],  # 现统一为日/周/月
    "头肩顶/M顶/长上影线等顶部形态": ["kline_indicators", "features"],
    "W底/长下影线阳线等底部形态": ["kline_indicators", "features"],
    "StochRSI": ["kline_indicators", "features"],
    "波动率": ["kline_indicators", "features.volatility_20"],
    "3-5根K线组合": ["kline_indicators", "features"],
    "K线形态组合": ["kline_indicators", "features"],
    "长期趋势线": ["kline_indicators", "features"],
    "主力控盘迹象": ["features", "snapshot"],
    "分时资金流": ["snapshot", "features"],
    "主力大单轨迹": ["features", "snapshot"],
    "异常成交": ["snapshot", "features"],
    "融资余额": ["fundamentals", "news"],
    "融券余额": ["fundamentals", "news"],
    "融资净买入": ["fundamentals", "news"],
    "期权IV(可选)": ["fundamentals"],
    "新闻文本": ["news"],
    "社媒文本": ["news"],
    "主题情绪网络": ["news", "features.news_sentiment"],
    "行业指数": ["fundamentals", "features"],
    "政策公告": ["news"],
    "监管新闻": ["news"],
    "板块轮动": ["features", "news"],
    "大宗交易折溢价": ["fundamentals", "news"],
    "融券套利窗口": ["fundamentals", "news"],
    "价差结构": ["kline_indicators", "features"],
    "行业景气度": ["fundamentals", "news"],
    "上下游价格": ["fundamentals", "news"],
    "供需结构": ["fundamentals", "news"],
    "盈利质量": ["fundamentals"],
    "现金流质量": ["fundamentals"],
    "护城河指标": ["fundamentals"],
    "成交额": ["snapshot", "features"],
    "换手率": ["snapshot", "fundamentals.turnover_rate"],
    "盘口深度": ["snapshot", "features"],
    "委比量比": ["snapshot", "features.volume_ratio"],
    # 背离/量价/支撑阻力 智能体
    "MACD/RSI背离信号": ["kline_indicators", "features"],
    "量价关系信号": ["kline_indicators", "features"],
    "支撑阻力位": ["kline_indicators", "features"],
    "缠论买卖点信号": ["kline_indicators", "features"],
    "图形形态信号": ["kline_indicators", "features"],
    # 相对强弱多层对标
    "个股vs沪深300相对强弱": ["features.relative_strength", "features"],
    "超额收益": ["features.relative_strength", "features"],
    "RS趋势": ["features.relative_strength", "features"],
    "个股vs行业板块相对强弱": ["features.rs_vs_industry", "features"],
    "个股vs板块龙头相对强弱": ["features.rs_vs_leaders", "features"],
    "个股vs行业ETF相对强弱": ["features.rs_vs_etf", "features"],
    # K线视觉智能体专用
    "1h/日/周/月K线图像": ["chart_files", "kline_indicators"],
    "日线K线图像": ["chart_files", "kline_indicators"],
    "周线K线图像": ["chart_files", "kline_indicators"],
    "月线K线图像": ["chart_files", "kline_indicators"],
}


def get_agent_data_mapping(agent_cfg: dict[str, Any], analysis_context: dict[str, Any]) -> dict[str, Any]:
    """获取单个 Agent 的本地/云端数据对照。"""
    ds = agent_cfg.get("data_sources", {})
    required = ds.get("required_data_points", [])
    integrity = analysis_context.get("data_integrity", {})
    features = analysis_context.get("features", {})
    kline_indicators = features.get("kline_indicators", {})

    local_ok = []
    local_missing = []
    cloud_only = []

    chart_files = analysis_context.get("chart_files", {})

    for point in required:
        local_sources = LOCAL_DATA_MAPPING.get(point, [])
        if not local_sources:
            cloud_only.append(point)
            continue
        # 简化检查：根据 data_integrity 和 features 判断
        has_kline = integrity.get("kline_ok", False) and any(v.get("ok") for v in kline_indicators.values() if isinstance(v, dict))
        has_fundamentals = integrity.get("fundamentals_ok", False)
        has_news = integrity.get("news_ok", False)
        has_snapshot = integrity.get("snapshot_ok", False)
        has_charts = bool(chart_files)

        provided = False
        for src in local_sources:
            if "chart_files" in src and has_charts:
                provided = True
                break
            if "kline" in src and has_kline:
                provided = True
                break
            if "fundamentals" in src and has_fundamentals:
                provided = True
                break
            if "news" in src and has_news:
                provided = True
                break
            if "snapshot" in src and has_snapshot:
                provided = True
                break
            if "features" in src and features:
                provided = True
                break

        if provided:
            local_ok.append((point, local_sources))
        else:
            local_missing.append((point, local_sources))

    return {
        "agent_id": agent_cfg.get("agent_id", ""),
        "role": agent_cfg.get("role", ""),
        "required_data_points": required,
        "local_provided": local_ok,
        "local_missing": local_missing,
        "cloud_only": cloud_only,
    }


def build_agent_data_table(agent_configs: list[dict], analysis_context: dict[str, Any]) -> list[dict[str, Any]]:
    """构建所有 Agent 的本地/云端数据对照表。"""
    return [get_agent_data_mapping(cfg, analysis_context) for cfg in agent_configs]


def format_agent_data_table(mappings: list[dict[str, Any]]) -> str:
    """格式化输出 Agent 与本地/云端数据对照表。"""
    lines = ["\n[Agent 与本地/云端数据对照]", "-" * 60]
    for m in mappings:
        aid = m.get("agent_id", "")
        role = m.get("role", "")
        lines.append(f"\n{aid} ({role})")
        ok_list = m.get("local_provided", [])
        miss_list = m.get("local_missing", [])
        cloud_list = m.get("cloud_only", [])
        if ok_list:
            for point, srcs in ok_list:
                lines.append(f"  [OK] 本地: {point} <- {', '.join(srcs)}")
        if miss_list:
            for point, srcs in miss_list:
                lines.append(f"  [X] 缺失: {point} (期望: {', '.join(srcs)})")
        if cloud_list:
            for point in cloud_list:
                lines.append(f"  [~] 云端: {point} (无本地映射)")
    lines.append("-" * 60)
    return "\n".join(lines)


def check_agent_data_ready(mapping: dict[str, Any]) -> bool:
    """检查 Agent 所需数据是否已从本地提供（核心数据需本地有）。"""
    miss = mapping.get("local_missing", [])
    required = mapping.get("required_data_points", [])
    if not required:
        return True
    # 若核心 required 有缺失，则未就绪
    return len(miss) == 0
