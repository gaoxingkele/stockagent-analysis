# -*- coding: utf-8 -*-
from pathlib import Path


TARGET_TEMPLATE = """# 中国股市多智能体定义文档（转换版）

## 原文来源
- 文件: {source_name}

## 转换目标
- 标的市场: 中国 A 股
- 决策标签: buy / hold / sell
- 核心流程: 多智能体分析 -> 辩论 -> 投票 -> 管理智能体裁决

## 角色定义（建议）
- manager: 编排与投票汇总
- fundamental_agent: 基本面分析
- technical_agent: 技术面分析
- sentiment_agent: 情绪面分析
- risk_agent: 风险控制分析

## 数据源定义
- 全局: AKShare + Tushare（可切换/组合）
- 智能体可配置专用数据源

## 通信与审计
- 结论提交: submissions/*.json
- 辩论消息: messages/*.json
- 独立日志: logs/<agent_id>.log

## 从原文提取片段（供人工复核）
{snippet}
"""


def convert_other_domain_doc(input_path: Path, output_dir: Path) -> Path:
    text = input_path.read_text(encoding="utf-8", errors="replace")
    snippet = text[:3000]
    out = TARGET_TEMPLATE.format(source_name=input_path.name, snippet=snippet)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_cn_stock.md"
    output_path.write_text(out, encoding="utf-8")
    return output_path
