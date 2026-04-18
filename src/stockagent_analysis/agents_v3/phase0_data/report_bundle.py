"""Phase 0 报告集合 - 整合 6 份 markdown 报告。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ReportBundle:
    """Phase 0 产出: 6 份客观事实报告 (markdown 字符串)。

    下游 LLM 角色从这里读取相应的报告作为 prompt 输入。
    """
    symbol: str
    name: str
    technical: str = ""          # 技术面报告
    structure: str = ""          # K线结构报告 (缠论/Donchian/Ichimoku)
    wave: str = ""               # 波浪理论报告
    capital: str = ""            # 资金/筹码报告
    fundamental: str = ""        # 基本面报告
    sentiment: str = ""          # 舆情/环境报告
    meta: dict[str, Any] = field(default_factory=dict)

    def as_markdown(self) -> str:
        """合并为单个大 markdown 字符串(供整体预览)。"""
        sections = [
            f"# Phase 0 量化事实报告 · {self.symbol} {self.name}",
            "",
            "## 技术面报告", self.technical, "",
            "## K线结构报告", self.structure, "",
            "## 波浪结构报告", self.wave, "",
            "## 资金/筹码报告", self.capital, "",
            "## 基本面报告", self.fundamental, "",
            "## 舆情环境报告", self.sentiment, "",
        ]
        return "\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get_for_role(self, role: str) -> str:
        """按角色侧重返回相关报告组合。

        - structure_expert: technical + structure
        - wave_expert: structure + wave
        - intraday_t: technical + capital
        - martingale: technical + capital + sentiment
        - bull/bear: 全部
        - trader: 全部
        """
        role = role.lower()
        parts: list[str] = []
        if role in ("structure_expert", "structure"):
            parts = [self.technical, self.structure]
        elif role in ("wave_expert", "wave"):
            parts = [self.structure, self.wave]
        elif role in ("intraday_t", "intraday"):
            parts = [self.technical, self.capital]
        elif role in ("martingale", "grid"):
            parts = [self.technical, self.capital, self.sentiment]
        else:
            parts = [
                self.technical, self.structure, self.wave,
                self.capital, self.fundamental, self.sentiment,
            ]
        return "\n\n---\n\n".join(p for p in parts if p)


def build_all_reports(symbol: str, name: str, ctx: dict[str, Any]) -> ReportBundle:
    """从 analysis_context 生成全部 6 份报告。

    Args:
        symbol: 股票代码
        name: 股票名称
        ctx: 完整 analysis_context(orchestrator 已收集)

    Returns:
        ReportBundle 对象
    """
    from .technical_report import build_technical_report
    from .structure_report import build_structure_report
    from .wave_report import build_wave_report
    from .capital_report import build_capital_report
    from .fundamental_report import build_fundamental_report
    from .sentiment_report import build_sentiment_report

    bundle = ReportBundle(symbol=symbol, name=name)
    bundle.technical = build_technical_report(symbol, name, ctx)
    bundle.structure = build_structure_report(symbol, name, ctx)
    bundle.wave = build_wave_report(symbol, name, ctx)
    bundle.capital = build_capital_report(symbol, name, ctx)
    bundle.fundamental = build_fundamental_report(symbol, name, ctx)
    bundle.sentiment = build_sentiment_report(symbol, name, ctx)

    snap = ctx.get("snapshot", {})
    bundle.meta = {
        "close": snap.get("close"),
        "pct_chg": snap.get("pct_chg"),
        "source": snap.get("source"),
        "data_quality": ctx.get("features", {}).get("data_quality_score"),
    }
    return bundle
