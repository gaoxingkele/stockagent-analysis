"""Phase 0: 量化事实层 - 6 份客观报告生成器(纯 Python, 无 LLM)。"""
from .technical_report import build_technical_report
from .structure_report import build_structure_report
from .wave_report import build_wave_report
from .capital_report import build_capital_report
from .fundamental_report import build_fundamental_report
from .sentiment_report import build_sentiment_report
from .report_bundle import ReportBundle, build_all_reports

__all__ = [
    "build_technical_report",
    "build_structure_report",
    "build_wave_report",
    "build_capital_report",
    "build_fundamental_report",
    "build_sentiment_report",
    "ReportBundle",
    "build_all_reports",
]
