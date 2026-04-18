"""角色记忆库 - 简化版 JSONL + 关键词召回。

每个角色(bull/bear/judge/trader/pm)独立一个 .jsonl 文件。
首期使用关键词匹配 + 情境 embedding 简化召回, 后续可升级向量库。

每条记录:
{
  "ts": "2026-04-18",
  "symbol": "000876",
  "situation": "当时情境摘要(报告浓缩)",
  "decision": "当时决策内容",
  "outcome": "N 日后结果(回测填入)",
  "lesson": "教训(如果失败)"
}
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_DEFAULT_MEM_DIR = _PROJECT_ROOT / "output" / "memory_v3"


@dataclass
class MemoryRecord:
    ts: str
    symbol: str
    situation: str
    decision: str
    outcome: str = ""
    lesson: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "symbol": self.symbol,
            "situation": self.situation,
            "decision": self.decision,
            "outcome": self.outcome,
            "lesson": self.lesson,
        }


def _tokenize(text: str) -> set[str]:
    """简单的中文+英文分词(按非字符拆).用于关键词召回。"""
    if not text:
        return set()
    # 按非字母数字中文拆
    toks = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower())
    return set(toks)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


class RoleMemory:
    """单个角色的记忆库。

    底层: JSONL 文件(每行一条记录)。
    召回: Jaccard 关键词相似度(情境 token 重合度)。
    """

    def __init__(self, role: str, mem_dir: Path | None = None):
        self.role = role
        self.mem_dir = Path(mem_dir) if mem_dir else _DEFAULT_MEM_DIR
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.mem_dir / f"{role}.jsonl"

    def add(self, record: MemoryRecord) -> None:
        """追加一条记录。"""
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("[RoleMemory:%s] 写入失败: %s", self.role, e)

    def all_records(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        out: list[MemoryRecord] = []
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    out.append(MemoryRecord(
                        ts=d.get("ts", ""),
                        symbol=d.get("symbol", ""),
                        situation=d.get("situation", ""),
                        decision=d.get("decision", ""),
                        outcome=d.get("outcome", ""),
                        lesson=d.get("lesson", ""),
                    ))
                except (json.JSONDecodeError, TypeError):
                    continue
        except Exception as e:
            logger.warning("[RoleMemory:%s] 读取失败: %s", self.role, e)
        return out

    def retrieve(
        self,
        situation: str,
        n: int = 3,
        min_sim: float = 0.05,
        method: str = "auto",
    ) -> list[MemoryRecord]:
        """召回 top-n 相似历史记录。

        Args:
            situation: 当前情境摘要
            n: 返回 top-n
            min_sim: 相似度阈值(embedding 方法下通常 0.3+)
            method: "auto"=优先 embedding, 失败降级 jaccard; "embedding"=强制向量; "jaccard"=强制关键词
        """
        all_recs = self.all_records()
        if not all_recs or not situation:
            return []

        if method in ("auto", "embedding"):
            try:
                from .embedding import embed_texts, cosine_sim_matrix, get_backend_name
                texts = [situation] + [r.situation for r in all_recs]
                vecs = embed_texts(texts)
                if vecs.shape[0] == len(texts) and vecs.shape[1] > 1:
                    sims = cosine_sim_matrix(vecs[0], vecs[1:])
                    backend = get_backend_name()
                    # 不同 backend 的合理阈值差异
                    thr = {"sentence-transformers": 0.35, "openai": 0.30, "tfidf": 0.05}.get(backend, min_sim)
                    scored = [(rec, float(s)) for rec, s in zip(all_recs, sims) if s >= thr]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    if scored:
                        return [r for r, _ in scored[:n]]
                    # 无匹配但 embedding 成功: 若 method="embedding" 就返空; 否则 auto 降级到 jaccard
                    if method == "embedding":
                        return []
            except Exception as e:
                logger.warning("[RoleMemory:%s] embedding 召回失败, 降级 jaccard: %s", self.role, e)

        # Jaccard fallback
        q_tokens = _tokenize(situation)
        if not q_tokens:
            return []
        scored = [(rec, _jaccard(q_tokens, _tokenize(rec.situation))) for rec in all_recs]
        scored = [(r, s) for r, s in scored if s >= min_sim]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:n]]

    def format_for_prompt(self, records: list[MemoryRecord]) -> str:
        if not records:
            return ""
        lines = [f"(共 {len(records)} 条历史类似情境)"]
        for i, r in enumerate(records, 1):
            lines.append(f"{i}. [{r.ts} {r.symbol}] 情境: {r.situation[:180]}")
            lines.append(f"   决策: {r.decision[:180]}")
            if r.outcome:
                lines.append(f"   结果: {r.outcome[:120]}")
            if r.lesson:
                lines.append(f"   教训: {r.lesson[:150]}")
        return "\n".join(lines)


_CACHE: dict[str, RoleMemory] = {}


def get_memory(role: str, mem_dir: Path | None = None) -> RoleMemory:
    """获取角色记忆库(单例)。"""
    key = f"{role}:{str(mem_dir) if mem_dir else 'default'}"
    if key not in _CACHE:
        _CACHE[key] = RoleMemory(role, mem_dir)
    return _CACHE[key]
