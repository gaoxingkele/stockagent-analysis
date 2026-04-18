"""马丁策略/网格交易员。"""
from .base_expert import BaseExpert


class MartingaleExpert(BaseExpert):
    role = "martingale_expert"
    role_cn = "马丁网格策略交易员"
    prompt_file = "expert_martingale"
    bundle_role_key = "martingale"
    max_tokens = 1200
