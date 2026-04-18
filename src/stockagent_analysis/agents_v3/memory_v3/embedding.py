"""Embedding 抽象层 - 自动选最合适的 backend。

优先级 (运行时动态选择):
  1. sentence-transformers (本地 BGE-M3, 最准)
  2. OpenAI embedding API (云, 需 OPENAI_API_KEY)
  3. TF-IDF + cosine (纯 numpy, 中文字符 3-gram, 无外部模型)
  4. Jaccard (最末位 fallback)

接口:
  - embed_texts(texts: list[str]) -> np.ndarray, shape=(N, D)
  - cosine_sim(a, b) -> float
  - get_backend_name() -> str
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Backend 检测
# ─────────────────────────────────────────────────────────────────

_BACKEND: str | None = None
_BACKEND_MODEL: Any = None


def get_backend_name() -> str:
    """返回当前使用的 backend 名称。"""
    global _BACKEND
    if _BACKEND is None:
        _detect_backend()
    return _BACKEND or "jaccard"


def _detect_backend() -> None:
    """运行时探测可用 backend 并缓存。"""
    global _BACKEND, _BACKEND_MODEL

    # 1. sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        _BACKEND_MODEL = SentenceTransformer(model_name)
        _BACKEND = "sentence-transformers"
        logger.info("[Embedding] 使用 sentence-transformers: %s", model_name)
        return
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[Embedding] sentence-transformers 加载失败: %s", e)

    # 2. OpenAI embedding
    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai  # type: ignore
            _BACKEND_MODEL = openai.OpenAI()
            _BACKEND = "openai"
            logger.info("[Embedding] 使用 OpenAI embedding")
            return
        except Exception as e:
            logger.warning("[Embedding] OpenAI 初始化失败: %s", e)

    # 3. TF-IDF (纯 numpy)
    _BACKEND = "tfidf"
    _BACKEND_MODEL = None
    logger.info("[Embedding] 使用 TF-IDF 字符 n-gram(无外部模型)")


# ─────────────────────────────────────────────────────────────────
# TF-IDF 字符 3-gram backend
# ─────────────────────────────────────────────────────────────────

def _char_ngrams(text: str, ns: tuple[int, ...] = (1, 2, 3)) -> list[str]:
    """中文/英文混合文本的混合字符 n-gram (默认 1+2+3, 中文友好)。"""
    text = re.sub(r"\s+", "", text.lower())
    if not text:
        return []
    out: list[str] = []
    for n in ns:
        if len(text) < n:
            continue
        out.extend(text[i:i + n] for i in range(len(text) - n + 1))
    return out


def _tfidf_embed(texts: list[str]) -> np.ndarray:
    """基于提供的文本集, 用 TF-IDF 构造稀疏向量(以 ndarray 返回)。"""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    # 1) 抽取所有 n-gram
    doc_tokens = [_char_ngrams(t) for t in texts]
    vocab = sorted({tok for toks in doc_tokens for tok in toks})
    vocab_idx = {tok: i for i, tok in enumerate(vocab)}
    V = len(vocab)
    N = len(texts)
    if V == 0 or N == 0:
        return np.zeros((N, 1), dtype=np.float32)

    # 2) TF
    tf = np.zeros((N, V), dtype=np.float32)
    for i, toks in enumerate(doc_tokens):
        for tok in toks:
            tf[i, vocab_idx[tok]] += 1
        if len(toks) > 0:
            tf[i] /= len(toks)

    # 3) IDF
    df = (tf > 0).sum(axis=0)  # shape (V,)
    idf = np.log((N + 1) / (df + 1)) + 1
    tfidf = tf * idf

    # 4) L2 归一化
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tfidf / norms


# ─────────────────────────────────────────────────────────────────
# 主接口
# ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """把一批文本编码为向量矩阵, shape=(N, D)。

    自动选择最优 backend。若失败则返回零矩阵(同 shape)。
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    backend = get_backend_name()

    try:
        if backend == "sentence-transformers":
            vecs = _BACKEND_MODEL.encode(texts, normalize_embeddings=True,
                                          show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)

        if backend == "openai":
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            resp = _BACKEND_MODEL.embeddings.create(model=model, input=texts)
            vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
            # 手动归一化
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms

        # TF-IDF
        return _tfidf_embed(texts)

    except Exception as e:
        logger.warning("[Embedding] backend=%s 编码失败 %s, 降级到 TF-IDF", backend, e)
        try:
            return _tfidf_embed(texts)
        except Exception:
            return np.zeros((len(texts), 1), dtype=np.float32)


def cosine_sim_matrix(q: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """q (1,D) 对 docs (N,D) 的余弦相似度, 返回 (N,)。

    前提: 两者已 L2 归一化。
    """
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if docs.ndim == 1:
        docs = docs.reshape(1, -1)
    if q.shape[1] != docs.shape[1]:
        return np.zeros(docs.shape[0], dtype=np.float32)
    return (docs @ q.T).flatten()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度。"""
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / (na * nb))
