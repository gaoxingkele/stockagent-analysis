"""Phase 2: 多空辩论 - Bull/Bear 3 轮 + Research Judge 仲裁。"""
from .bull_analyst import run_bull
from .bear_analyst import run_bear
from .research_judge import run_judge
from .debate_loop import run_investment_debate, InvestmentPlan

__all__ = [
    "run_bull",
    "run_bear",
    "run_judge",
    "run_investment_debate",
    "InvestmentPlan",
]
