"""波浪理论分析师。"""
from .base_expert import BaseExpert


class WaveExpert(BaseExpert):
    role = "wave_expert"
    role_cn = "波浪理论分析师"
    prompt_file = "expert_wave"
    bundle_role_key = "wave_expert"
    max_tokens = 1200   # 波浪推断需要更长空间
