"""短线做 T 分析师。"""
from .base_expert import BaseExpert


class IntradayTExpert(BaseExpert):
    role = "intraday_t_expert"
    role_cn = "短线做 T 分析师"
    prompt_file = "expert_intraday_t"
    bundle_role_key = "intraday_t"
