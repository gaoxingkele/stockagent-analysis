"""K 走势结构分析师。"""
from .base_expert import BaseExpert


class StructureExpert(BaseExpert):
    role = "structure_expert"
    role_cn = "K 走势结构分析师"
    prompt_file = "expert_structure"
    bundle_role_key = "structure_expert"
