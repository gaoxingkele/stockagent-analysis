# -*- coding: utf-8 -*-
"""BM25 记忆系统 - 情景分析与建仓决策持久化 + 相似召回"""
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

try:
    from rank_bm25 import BM25Okapi

    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False


class BM25Memory:
    """离线 BM25 记忆：存入 + 召回相似历史情境。

    每次个股分析完成后调用 add_decision() 存入记忆。
    分析时可调用 get_context_for_prompt() 获取 Top-N 相似历史供 Prompt 参考。
    """

    _INSTANCE: Optional["BM25Memory"] = None

    @classmethod
    def get_instance(cls, memory_dir: str = "output/memory") -> "BM25Memory":
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(memory_dir)
        return cls._INSTANCE

    def __init__(self, memory_dir: str = "output/memory"):
        self.memory_dir = Path(memory_dir)
        self.situations: list[dict] = []
        self.bm25: Any = None
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ── 内部工具 ────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """中英文混合 tokenization。"""
        tokens = re.findall(r"[\w\.\-]+", text.lower())
        return tokens

    def _load(self) -> None:
        path = self.memory_dir / "situations.json"
        if path.exists():
            try:
                self.situations = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                self.situations = []
        if _BM25_AVAILABLE and self.situations:
            tokenized = [self._tokenize(s["situation_text"]) for s in self.situations]
            self.bm25 = BM25Okapi(tokenized)

    def _save(self) -> None:
        path = self.memory_dir / "situations.json"
        path.write_text(
            json.dumps(self.situations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── 公开 API ───────────────────────────────────────────

    def add_decision(self, fd: dict, features: dict, analysis_features: dict) -> None:
        """分析完成后调用，存入当前情境 + 决策建议。"""
        # 基本信息
        score = float(fd.get("final_score", 0) or 0)
        decision = str(fd.get("final_decision", ""))
        decision_level = str(fd.get("decision_level", ""))

        feat = features or {}
        af = analysis_features or {}

        # 从 analysis_features 提取技术指标
        day_ma = af.get("kline_indicators", {}).get("day", {}).get("ma_system", {}) or {}
        ma5_pct = float(day_ma.get("ma5", {}).get("pct_above", 0) or 0)
        ma5_val = float(day_ma.get("ma5", {}).get("value", 0) or 0)
        ma20_val = float(day_ma.get("ma20", {}).get("value", 0) or 0)

        kl = af.get("key_levels") or {}
        if isinstance(kl, dict):
            band_high = kl.get("band_high", "")
            band_low = kl.get("band_low", "")
        else:
            band_high = band_low = ""

        # 从 features 提取
        momentum = float(feat.get("momentum_20", 0) or 0)
        volatility = float(feat.get("volatility_20", 0) or 0)
        vol_ratio = float(feat.get("volume_ratio_5_20", 0) or 0)
        pe = float(feat.get("pe_ttm", 0) or 0)
        pct_chg = float(feat.get("pct_chg", 0) or 0)
        trend = float(feat.get("trend_strength", 0) or 0)

        # 构建 situation_text（供 BM25 检索）
        situation_text = (
            f"{score:.1f}分{decision} {decision_level} "
            f"MA5乖离{ma5_pct:+.1f}% MA5={ma5_val:.2f} MA20={ma20_val:.2f} "
            f"趋势{trend:.2f} 动量{momentum:.2f} 波动率{volatility:.2f} "
            f"量比{vol_ratio:.2f} PE={pe:.1f} 涨跌{pct_chg:+.1f}% "
            f"压力位{band_high} 支撑位{band_low}"
        )

        # 构建 recommendation_text（召回时展示）
        sp = fd.get("sniper_points") or {}
        scenarios = fd.get("scenarios") or {}
        ideal_buy = float(sp.get("ideal_buy", 0) or 0)
        stop_loss = float(sp.get("stop_loss", 0) or 0)
        tp1 = float(sp.get("take_profit_1", 0) or 0)
        opt_reason = str(scenarios.get("optimistic", {}).get("reason", ""))[:50]
        neu_reason = str(scenarios.get("neutral", {}).get("reason", ""))[:50]
        pess_reason = str(scenarios.get("pessimistic", {}).get("reason", ""))[:50]
        short_term = str(fd.get("short_term_hold", ""))
        medium_term = str(fd.get("medium_long_term_hold", ""))

        recommendation_text = (
            f"{decision_level} | {score:.1f}分 "
            f"理想买{ideal_buy:.2f} 止损{stop_loss:.2f} 止盈1{tp1:.2f} "
            f"乐观:{opt_reason} 中性:{neu_reason} 悲观:{pess_reason} "
            f"{short_term} {medium_term}"
        )

        entry = {
            "id": f"{fd.get('symbol', '')}_{int(time.time())}",
            "date": time.strftime("%Y-%m-%d"),
            "symbol": fd.get("symbol", ""),
            "name": fd.get("name", ""),
            "situation_text": situation_text.strip(),
            "recommendation_text": recommendation_text.strip(),
            "final_score": score,
            "final_decision": decision,
            "sniper_points": sp,
            "scenarios": scenarios,
        }

        self.situations.append(entry)
        self._save()

        # 重建 BM25 索引
        if _BM25_AVAILABLE:
            tokenized = [self._tokenize(s["situation_text"]) for s in self.situations]
            self.bm25 = BM25Okapi(tokenized)

    def recall(self, situation_text: str, n: int = 3) -> list[dict]:
        """返回 Top-N 相似历史情境（按 BM25 得分降序）。"""
        if not self.bm25 or not self.situations:
            return []
        tokens = self._tokenize(situation_text)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # 按得分降序排列，取 Top-N
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n]
        return [
            {
                "situation": self.situations[i]["situation_text"],
                "recommendation": self.situations[i]["recommendation_text"],
                "score": self.situations[i]["final_score"],
                "symbol": self.situations[i]["symbol"],
                "name": self.situations[i]["name"],
                "date": self.situations[i]["date"],
                "sniper_points": self.situations[i].get("sniper_points", {}),
                "bm25_score": round(scores[i], 2),
            }
            for i in top_indices
        ]

    def get_context_for_prompt(
        self, current_situation: str, n: int = 3
    ) -> str:
        """拼成字符串，注入到 LLM Prompt 中。"""
        memories = self.recall(current_situation, n)
        if not memories:
            return ""
        lines = ["\n【相似历史情境参考】"]
        for m in memories:
            lines.append(
                f"- [{m['symbol']} {m['name']} @{m['date']}] "
                f"情境:{m['situation']} → 决策:{m['recommendation']}"
            )
        return "\n".join(lines)
