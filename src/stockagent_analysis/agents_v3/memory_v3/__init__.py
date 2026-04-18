"""角色记忆库 - 支持 embedding 向量召回 + Jaccard fallback。"""
from .role_memory import RoleMemory, get_memory, MemoryRecord
from .embedding import embed_texts, cosine_sim, cosine_sim_matrix, get_backend_name

__all__ = [
    "RoleMemory", "get_memory", "MemoryRecord",
    "embed_texts", "cosine_sim", "cosine_sim_matrix", "get_backend_name",
]
