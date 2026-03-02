# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .data_backend import DataBackend, MarketSnapshot
from .io_utils import dump_json, get_agent_logger
from .llm_client import LLMRouter, _supports_vision, _DEFAULT_VISION_FALLBACK


@dataclass
class AgentResult:
    agent_id: str
    dim_code: str
    vote: str
    score_0_100: float
    confidence_0_1: float
    reason: str
    llm_multi_evaluations: dict[str, str]


@dataclass
class AgentBaseResult:
    """本地策略分析结果（无LLM调用），用于并行Provider执行前的基础数据。"""
    agent_id: str
    dim_code: str
    agent_type: str          # "analyst" | "kline_vision"
    role: str
    weight: float
    vote: str
    score_0_100: float
    confidence_0_1: float
    reason: str              # _simple_policy 的本地结论
    data_context: str        # 预构建的数据摘要，供LLM参考
    snap: MarketSnapshot     # 快照，用于后续提交
    # kline_vision 专用：
    timeframe: str | None = None
    image_b64: str | None = None
    vision_prompt: str | None = None


class AnalystAgent:
    def __init__(
        self,
        cfg: dict[str, Any],
        run_dir: Path,
        backend: DataBackend,
        llm_routers: dict[str, LLMRouter] | None = None,
    ):
        self.cfg = cfg
        self.agent_id = cfg["agent_id"]
        self.dim_code = cfg.get("dim_code", "")
        self.role = cfg["role"]
        self.weight = float(cfg.get("weight", 0.0))
        self.preferred_sources = cfg.get("preferred_sources", [])
        self.data_sources = cfg.get("data_sources", {})
        self.logger = get_agent_logger(run_dir, self.agent_id)
        self.run_dir = run_dir
        self.backend = backend
        self.llm_routers = llm_routers or {}

    def _build_snapshot(self, symbol: str, name: str, ctx: dict[str, Any]) -> MarketSnapshot:
        """从 analysis_context 构建 MarketSnapshot，或从后端获取。"""
        ctx_snap = ctx.get("snapshot", {})
        if ctx_snap:
            return MarketSnapshot(
                symbol=str(ctx_snap.get("symbol", symbol)),
                name=str(ctx_snap.get("name", name)),
                close=float(ctx_snap.get("close", 0.0) or 0.0),
                pct_chg=float(ctx_snap.get("pct_chg", 0.0) or 0.0),
                pe_ttm=ctx_snap.get("pe_ttm"),
                turnover_rate=ctx_snap.get("turnover_rate"),
                source=str(ctx_snap.get("source", "context")),
            )
        return self.backend.fetch_snapshot(symbol, name, self.preferred_sources)

    def analyze_local(
        self,
        symbol: str,
        name: str,
        analysis_context: dict[str, Any] | None = None,
    ) -> AgentBaseResult:
        """本地策略分析（无LLM调用），返回 AgentBaseResult 供并行Provider执行使用。"""
        ctx = analysis_context or {}
        snap = self._build_snapshot(symbol, name, ctx)
        vote, score_0_100, confidence, reason = self._simple_policy(snap, ctx)
        data_context = self._build_data_context(ctx) if ctx else ""
        return AgentBaseResult(
            agent_id=self.agent_id, dim_code=self.dim_code,
            agent_type="analyst", role=self.role, weight=self.weight,
            vote=vote, score_0_100=score_0_100, confidence_0_1=confidence,
            reason=reason, data_context=data_context, snap=snap,
        )

    def analyze(
        self,
        symbol: str,
        name: str,
        analysis_context: dict[str, Any] | None = None,
        progress_cb=None,
    ) -> AgentResult:
        ctx = analysis_context or {}
        snap = self._build_snapshot(symbol, name, ctx)
        vote, score_0_100, confidence, reason = self._simple_policy(snap, ctx)
        llm_evals: dict[str, str] = {}
        data_context = self._build_data_context(ctx) if ctx else None
        if self.llm_routers:
            for provider, router in self.llm_routers.items():
                ctx_len = len(data_context or "")
                ctx_brief = (data_context or "").replace("\n", " | ")[:120]
                if progress_cb:
                    progress_cb("大模型评估", f"{self.agent_id} → {provider} | 提交本地数据 {ctx_len}字符")
                self.logger.info(
                    "[LLM提交] 研判增强 agent=%s provider=%s | 本地数据 %d字符: %s",
                    self.agent_id, provider, ctx_len, ctx_brief,
                )
                text = router.enrich_reason(self.role, symbol, reason, data_context=data_context)
                if text:
                    llm_evals[provider] = text.strip()
            if llm_evals:
                reason = self._merge_multi_eval(llm_evals, fallback=reason)
        self.logger.info(
            "dim=%s snapshot=%s vote=%s score=%.2f conf=%.2f providers=%s reason=%s",
            self.dim_code, snap, vote, score_0_100, confidence, list(llm_evals.keys()), reason
        )
        result = AgentResult(self.agent_id, self.dim_code, vote, score_0_100, confidence, reason, llm_evals)
        self._submit(result, snap, ctx)
        return result

    @staticmethod
    def _build_data_context(ctx: dict[str, Any]) -> str:
        """从 analysis_context 构建供大模型参考的本地数据摘要（控制长度）。"""
        parts = []
        snap = ctx.get("snapshot", {})
        if snap:
            close = snap.get("close")
            pct = snap.get("pct_chg")
            pe = snap.get("pe_ttm")
            turn = snap.get("turnover_rate")
            line = f"行情: 收盘={close}, 涨跌幅={pct}%"
            if pe is not None:
                line += f", PE(TTM)={pe}"
            if turn is not None:
                line += f", 换手率={turn}%"
            parts.append(line)
        f = ctx.get("features", {})
        if f:
            mom = f.get("momentum_20")
            vol = f.get("volatility_20")
            dd = f.get("drawdown_60")
            vr = f.get("volume_ratio_5_20")
            trend = f.get("trend_strength")
            news_sent = f.get("news_sentiment")
            news_cnt = f.get("news_count")
            feats = []
            if mom is not None:
                feats.append(f"20日动量={mom:.1f}%")
            if vol is not None:
                feats.append(f"20日波动={vol:.1f}%")
            if dd is not None:
                feats.append(f"60日回撤={dd:.1f}%")
            if vr is not None:
                feats.append(f"量比5/20={vr:.2f}")
            if trend is not None:
                feats.append(f"趋势强度={trend:.1f}%")
            if news_sent is not None:
                feats.append(f"新闻情绪={news_sent:.1f}")
            if news_cnt is not None:
                feats.append(f"新闻条数={news_cnt}")
            if feats:
                parts.append("指标: " + ", ".join(feats))
            kli = f.get("kline_indicators", {})
            if kli:
                ok_tfs = [f"{tf}({v.get('rows',0)}根)" for tf, v in kli.items() if isinstance(v, dict) and v.get("ok")]
                if ok_tfs:
                    parts.append("K线: " + ", ".join(ok_tfs))
        news = ctx.get("news", [])[:5]
        if news:
            titles = [str(n.get("title", ""))[:40] for n in news if n.get("title")]
            if titles:
                parts.append("近期新闻: " + " | ".join(titles))
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _merge_multi_eval(llm_evals: dict[str, str], fallback: str) -> str:
        if not llm_evals:
            return fallback
        parts = []
        for p in ("grok", "gemini", "kimi"):
            if p in llm_evals:
                short = llm_evals[p].replace("\n", " ").strip()
                parts.append(f"{p}: {short[:90]}")
        if not parts:
            for p, txt in llm_evals.items():
                parts.append(f"{p}: {txt.replace(chr(10), ' ')[:90]}")
        return "综合评价（grok+gemini+kimi）：" + "；".join(parts)

    def _simple_policy(self, snap: MarketSnapshot, analysis_context: dict[str, Any]) -> tuple[str, float, float, str]:
        f = analysis_context.get("features", {})
        pct = float(f.get("pct_chg", snap.pct_chg or 0.0))
        mom = float(f.get("momentum_20", 0.0))
        vol = float(f.get("volatility_20", 0.0))
        dd = float(f.get("drawdown_60", 0.0))
        vr = float(f.get("volume_ratio_5_20", 1.0))
        trend = float(f.get("trend_strength", 0.0))
        news = float(f.get("news_sentiment", 0.0))
        pe = f.get("pe_ttm")
        pb = f.get("pb")
        turn = f.get("turnover_rate")

        pe_bias = 0.0
        if pe is not None:
            pe_f = float(pe)
            if pe_f < 20:
                pe_bias = 5.0
            elif pe_f > 60:
                pe_bias = -6.0
        pb_bias = 0.0
        if pb is not None:
            pb_f = float(pb)
            if pb_f < 2:
                pb_bias = 2.0
            elif pb_f > 8:
                pb_bias = -2.5

        base_dim = {
            "TREND": 50 + 0.45 * mom + 0.5 * trend + 0.15 * pct + 0.25 * dd,
            "TECH": 50 + 0.35 * mom + 0.3 * trend + (vr - 1.0) * 8 - 0.2 * vol,
            "LIQ": 50 + (vr - 1.0) * 10 - 0.2 * abs(pct) - 0.1 * vol + (1.5 if turn is not None else -1.0),
            "CAPITAL_FLOW": 50 + 0.25 * pct + (vr - 1.0) * 10 + 0.08 * news + 0.15 * mom,
            "SECTOR_POLICY": 50 + 0.07 * news + 0.1 * mom,
            "BETA": 50 + 0.2 * trend + 0.12 * pct - 0.25 * vol,
            "SENTIMENT": 50 + 0.45 * news + 0.1 * pct,
            "FUNDAMENTAL": 50 + pe_bias + pb_bias + 0.1 * mom,
            "QUANT": 50 + 0.3 * mom - 0.15 * vol + (vr - 1.0) * 6,
            "MACRO": 50 + 0.1 * mom - 0.1 * vol + 0.05 * news,
            "INDUSTRY": 50 + 0.08 * news + 0.12 * mom,
            "FLOW_DETAIL": 50 + (vr - 1.0) * 12 + 0.15 * pct,
            "MM_BEHAVIOR": 50 + (vr - 1.0) * 8 + 0.2 * news - 0.15 * vol,
            "NLP_SENTIMENT": 50 + 0.55 * news + 0.1 * mom,
            # 融资融券：价格上涨利好（多头借力），成交量激增偏空（追高风险），高波动减分，动量微加
            "DERIV_MARGIN": 50 + 0.2 * pct - (vr - 1.0) * 4 - 0.1 * vol + 0.05 * mom,
        }
        # TOP_STRUCTURE / BOTTOM_STRUCTURE：基于 kline_indicators 多周期结构研判（仅日线/周线/月线）
        kli = f.get("kline_indicators", {})
        top_signals, bot_signals = 0.0, 0.0
        for tf in ("day", "week", "month"):
            ind = kli.get(tf)
            if not isinstance(ind, dict) or not ind.get("ok"):
                continue
            upper = float(ind.get("upper_shadow_ratio", 0))
            lower = float(ind.get("lower_shadow_ratio", 0))
            mom_tf = float(ind.get("momentum_10", 0))
            amp = float(ind.get("amplitude_20", 0))
            # 顶部结构：长上影线+负动量 或 高振幅+负动量（头肩顶/M顶/长上影线）
            if upper > 40 and mom_tf < 0:
                top_signals += 10
            elif upper > 35:
                top_signals += 5
            elif mom_tf < -5 and amp > 15:
                top_signals += 6
            elif mom_tf < 0:
                top_signals += 2
            # 底部结构：长下影线+正动量 或 正动量反弹（W底/长下影线阳线）
            if lower > 40 and mom_tf > 0:
                bot_signals += 10
            elif lower > 35:
                bot_signals += 5
            elif mom_tf > 5:
                bot_signals += 6
            elif mom_tf > 0:
                bot_signals += 2
        base_dim["TOP_STRUCTURE"] = 50.0 - top_signals + bot_signals * 0.3
        base_dim["BOTTOM_STRUCTURE"] = 50.0 + bot_signals - top_signals * 0.3
        # K线视觉智能体的文字降级评分（无图时使用综合技术信号）
        tech_score = float(base_dim.get("TECH", 50.0))
        base_dim["KLINE_DAY"] = tech_score
        base_dim["KLINE_WEEK"] = (tech_score + float(base_dim.get("TREND", 50.0))) / 2.0
        base_dim["KLINE_MONTH"] = (tech_score + float(base_dim.get("TREND", 50.0)) + float(base_dim.get("FUNDAMENTAL", 50.0))) / 3.0
        base_dim["KLINE_1H"] = 50 + (vr - 1.0) * 8 + 0.2 * pct - 0.1 * vol
        score_0_100 = float(base_dim.get(self.dim_code, 50.0))
        score_0_100 = max(0.0, min(100.0, score_0_100))

        data_quality = float(f.get("data_quality_score", 0.6))
        confidence = 0.45 + 0.45 * data_quality
        if snap.source == "mock":
            confidence -= 0.18
        confidence = max(0.1, min(0.98, confidence))
        if score_0_100 >= 70:
            return "buy", score_0_100, confidence, f"{self.role}({self.dim_code})：偏强，评分{score_0_100:.1f}"
        if score_0_100 < 50:
            return "sell", score_0_100, confidence, f"{self.role}({self.dim_code})：偏弱，评分{score_0_100:.1f}"
        return "hold", score_0_100, confidence, f"{self.role}({self.dim_code})：中性，评分{score_0_100:.1f}"

    def _submit(self, result: AgentResult, snap: MarketSnapshot, analysis_context: dict[str, Any]) -> None:
        now = datetime.now().isoformat()
        payload = {
            "timestamp": now,
            "agent_id": result.agent_id,
            "dim_code": result.dim_code,
            "role": self.role,
            "vote": result.vote,
            "score_0_100": result.score_0_100,
            "confidence_0_1": result.confidence_0_1,
            "reason": result.reason,
            "llm_multi_evaluations": result.llm_multi_evaluations,
            "data_source_config": self.data_sources,
            "analysis_features": analysis_context.get("features", {}),
            "snapshot": {
                "symbol": snap.symbol,
                "name": snap.name,
                "close": snap.close,
                "pct_chg": snap.pct_chg,
                "pe_ttm": snap.pe_ttm,
                "turnover_rate": snap.turnover_rate,
                "source": snap.source
            },
            "to": "manager"
        }
        path = self.run_dir / "submissions" / f"{self.agent_id}.json"
        dump_json(path, payload)


def write_message(
    run_dir: Path,
    from_agent: str,
    to_agent: str,
    round_idx: int,
    content: str
) -> None:
    now = datetime.now().isoformat()
    payload = {
        "timestamp": now,
        "round": round_idx,
        "from": from_agent,
        "to": to_agent,
        "content": content,
        "protocol": "agent_json_message_v1"
    }
    ts = datetime.now().strftime("%H%M%S_%f")
    dump_json(run_dir / "messages" / f"r{round_idx}_{from_agent}_to_{to_agent}_{ts}.json", payload)


# K线视觉智能体的 dim_code 与周期对应关系
_KLINE_VISION_DIMS = {
    "1h":    "KLINE_1H",
    "day":   "KLINE_DAY",
    "week":  "KLINE_WEEK",
    "month": "KLINE_MONTH",
}


def _parse_vision_response(text: str) -> tuple[str, float, str]:
    """从视觉模型回复中解析 建议/评分/核心判断。
    返回 (vote, score_0_100, core_reason)。解析失败时返回默认值。
    """
    vote = "hold"
    score = 50.0
    core = text.strip()[:200] if text else ""

    # 建议解析
    vote_match = re.search(r"建议[：:]\s*(buy|hold|sell|买入|持有|观望|卖出|减仓)", text or "", re.I)
    if vote_match:
        v = vote_match.group(1).lower()
        if v in ("buy", "买入"):
            vote = "buy"
        elif v in ("sell", "卖出", "减仓"):
            vote = "sell"
        else:
            vote = "hold"

    # 评分解析（兼容 "评分：72" / "72/100" / "72分"）
    for pat in [
        r"评分[：:]\s*([0-9]{1,3}\.?\d*)",
        r"([0-9]{1,3}\.?\d*)\s*/\s*100",
        r"([0-9]{1,3}\.?\d*)\s*分",
    ]:
        m = re.search(pat, text or "")
        if m:
            v = float(m.group(1))
            if 0 <= v <= 100:
                score = v
                break

    # 核心判断
    core_match = re.search(r"核心判断[：:]\s*(.+?)(?:\n|$)", text or "", re.S)
    if core_match:
        core = core_match.group(1).strip()[:300]
    elif text:
        core = text.strip()[:300]

    return vote, score, core


class KlineVisionAgent(AnalystAgent):
    """多周期K线视觉智能体：把K线图作为图像输入给支持视觉的LLM进行辅助研判。
    每个实例对应一个K线周期（1h/day/week/month）。
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        run_dir: Path,
        backend: DataBackend,
        timeframe: str,
        llm_routers: dict[str, LLMRouter] | None = None,
    ):
        super().__init__(cfg, run_dir, backend, llm_routers=llm_routers)
        self.timeframe = timeframe  # "1h" / "day" / "week" / "month"
        self.dim_code = _KLINE_VISION_DIMS.get(timeframe, f"KLINE_{timeframe.upper()}")

    def analyze_local(
        self,
        symbol: str,
        name: str,
        analysis_context: dict[str, Any] | None = None,
    ) -> AgentBaseResult:
        """本地策略分析 + 加载K线图 + 构建视觉prompt，不调用LLM。"""
        from .chart_generator import load_image_base64, TIMEFRAME_LABEL

        base = super().analyze_local(symbol, name, analysis_context)
        base.agent_type = "kline_vision"
        base.dim_code = self.dim_code
        base.timeframe = self.timeframe

        ctx = analysis_context or {}
        chart_files = ctx.get("chart_files", {})
        chart_path = chart_files.get(self.timeframe)
        if chart_path:
            image_b64 = load_image_base64(chart_path)
            if image_b64:
                base.image_b64 = image_b64
                tf_label = TIMEFRAME_LABEL.get(self.timeframe, self.timeframe)
                base.vision_prompt = (
                    f"你是中国股市专业技术分析师。请仔细分析以下{base.snap.symbol} {name}的{tf_label}K线综合图"
                    f"（包含均线MA5/10/20/60/120/250、布林带、成交量、MACD、RSI、KDJ及趋势线）。\n\n"
                    f"当前价格：{base.snap.close}，今日涨跌幅：{base.snap.pct_chg}%\n\n"
                    f"请从以下角度给出研判：\n"
                    f"1. 趋势方向（上升/下降/震荡盘整）\n"
                    f"2. 关键支撑/压力位\n"
                    f"3. 技术指标信号（MACD/RSI/KDJ超买超卖等）\n"
                    f"4. 量价配合情况\n"
                    f"5. 趋势线信号（突破或压制）\n\n"
                    f"请严格按如下格式输出：\n"
                    f"建议：buy/hold/sell\n"
                    f"评分：[0-100]\n"
                    f"核心判断：[2-3句简洁分析]\n"
                )
        return base

    def analyze(
        self,
        symbol: str,
        name: str,
        analysis_context: dict[str, Any] | None = None,
        progress_cb=None,
    ) -> AgentResult:
        from .chart_generator import load_image_base64, TIMEFRAME_LABEL

        ctx = analysis_context or {}
        chart_files: dict[str, str] = ctx.get("chart_files", {})
        chart_path = chart_files.get(self.timeframe)

        # 筛选支持视觉的 router
        vision_routers = {p: r for p, r in self.llm_routers.items() if r.supports_vision()}

        # 若无视觉 router，尝试用默认视觉回退提供商（默认 kimi）补位
        if not vision_routers and chart_path:
            fallback_p = os.getenv("VISION_FALLBACK_PROVIDER", _DEFAULT_VISION_FALLBACK)
            fallback_key = os.getenv(f"{fallback_p.upper()}_API_KEY", "").strip()
            if fallback_key and _supports_vision(fallback_p):
                ref = next(iter(self.llm_routers.values()), None)
                fallback_router = LLMRouter(
                    provider=fallback_p,
                    temperature=ref.temperature if ref else 0.3,
                    max_tokens=ref.max_tokens if ref else 600,
                    request_timeout_sec=ref.request_timeout_sec if ref else 25.0,
                )
                vision_routers = {fallback_p: fallback_router}
                self.logger.info(
                    "[KlineVision] 当前providers无视觉能力，回退到 %s 视觉模型 (timeframe=%s)",
                    fallback_p, self.timeframe,
                )
                if progress_cb:
                    progress_cb("K线视觉回退", f"{self.agent_id} 无视觉provider → 使用 {fallback_p} 视觉模型")

        # 若仍无视觉 router 或无图表，降级到父类文字分析
        if not chart_path or not vision_routers:
            self.logger.info(
                "[KlineVision] no chart or no vision provider, fallback to text. "
                "timeframe=%s chart=%s vision_providers=%s",
                self.timeframe, chart_path, list(vision_routers.keys()),
            )
            result = super().analyze(symbol, name, analysis_context=analysis_context, progress_cb=progress_cb)
            result.dim_code = self.dim_code
            return result

        image_b64 = load_image_base64(chart_path)
        if not image_b64:
            result = super().analyze(symbol, name, analysis_context=analysis_context, progress_cb=progress_cb)
            result.dim_code = self.dim_code
            return result

        # 快照
        ctx_snap = ctx.get("snapshot", {})
        if ctx_snap:
            snap = MarketSnapshot(
                symbol=str(ctx_snap.get("symbol", symbol)),
                name=str(ctx_snap.get("name", name)),
                close=float(ctx_snap.get("close", 0.0) or 0.0),
                pct_chg=float(ctx_snap.get("pct_chg", 0.0) or 0.0),
                pe_ttm=ctx_snap.get("pe_ttm"),
                turnover_rate=ctx_snap.get("turnover_rate"),
                source=str(ctx_snap.get("source", "context")),
            )
        else:
            snap = self.backend.fetch_snapshot(symbol, name, self.preferred_sources)

        tf_label = TIMEFRAME_LABEL.get(self.timeframe, self.timeframe)
        prompt = (
            f"你是中国股市专业技术分析师。请仔细分析以下{symbol} {name}的{tf_label}K线综合图"
            f"（包含均线MA5/10/20/60/120/250、布林带、成交量、MACD、RSI、KDJ及趋势线）。\n\n"
            f"当前价格：{snap.close}，今日涨跌幅：{snap.pct_chg}%\n\n"
            f"请从以下角度给出研判：\n"
            f"1. 趋势方向（上升/下降/震荡盘整）\n"
            f"2. 关键支撑/压力位\n"
            f"3. 技术指标信号（MACD/RSI/KDJ超买超卖等）\n"
            f"4. 量价配合情况\n"
            f"5. 趋势线信号（突破或压制）\n\n"
            f"请严格按如下格式输出：\n"
            f"建议：buy/hold/sell\n"
            f"评分：[0-100]\n"
            f"核心判断：[2-3句简洁分析]\n"
        )

        llm_evals: dict[str, str] = {}
        all_votes, all_scores = [], []

        for provider, router in vision_routers.items():
            if progress_cb:
                progress_cb("K线视觉分析", f"{self.agent_id}({tf_label}) → {provider} | 提交图像 {len(image_b64)//1024}KB")
            self.logger.info(
                "[LLM提交] K线视觉分析 agent=%s provider=%s timeframe=%s | 图像 %dKB",
                self.agent_id, provider, self.timeframe, len(image_b64) // 1024,
            )
            text = router.chat_with_image(prompt, image_b64)
            if text:
                llm_evals[provider] = text.strip()
                v, s, _ = _parse_vision_response(text)
                all_votes.append(v)
                all_scores.append(s)

        if not llm_evals:
            # 所有视觉调用失败，降级
            result = super().analyze(symbol, name, analysis_context=analysis_context, progress_cb=progress_cb)
            result.dim_code = self.dim_code
            return result

        # 综合多个视觉模型结果
        avg_score = sum(all_scores) / len(all_scores)
        avg_score = max(0.0, min(100.0, avg_score))
        # 投票决定最终方向
        from collections import Counter
        vote_counts = Counter(all_votes)
        final_vote = vote_counts.most_common(1)[0][0]
        if avg_score >= 70:
            final_vote = "buy"
        elif avg_score < 50:
            final_vote = "sell"
        else:
            final_vote = "hold"

        # 合并多模型评价文本
        parts = []
        for p in ("gemini", "grok", "claude", "openai"):
            if p in llm_evals:
                short = llm_evals[p].replace("\n", " ").strip()[:120]
                parts.append(f"{p}: {short}")
        if not parts:
            for p, txt in llm_evals.items():
                parts.append(f"{p}: {txt.replace(chr(10), ' ')[:120]}")
        reason = f"【{tf_label}K线视觉研判】" + "；".join(parts)

        data_quality = float(ctx.get("features", {}).get("data_quality_score", 0.7))
        confidence = 0.5 + 0.4 * data_quality
        confidence = max(0.1, min(0.98, confidence))

        self.logger.info(
            "dim=%s timeframe=%s vote=%s score=%.2f conf=%.2f providers=%s",
            self.dim_code, self.timeframe, final_vote, avg_score, confidence, list(llm_evals.keys()),
        )

        result = AgentResult(
            agent_id=self.agent_id,
            dim_code=self.dim_code,
            vote=final_vote,
            score_0_100=avg_score,
            confidence_0_1=confidence,
            reason=reason,
            llm_multi_evaluations=llm_evals,
        )
        self._submit(result, snap, ctx)
        return result


def create_agent(
    cfg: dict[str, Any],
    run_dir: Path,
    backend: DataBackend,
    llm_routers: dict[str, LLMRouter] | None = None,
) -> AnalystAgent:
    """Agent 工厂：根据 cfg 中的 agent_type 决定返回 KlineVisionAgent 或普通 AnalystAgent。"""
    agent_type = cfg.get("agent_type", "analyst")
    if agent_type == "kline_vision":
        timeframe = cfg.get("timeframe", "day")
        return KlineVisionAgent(cfg, run_dir, backend, timeframe=timeframe, llm_routers=llm_routers)
    return AnalystAgent(cfg, run_dir, backend, llm_routers=llm_routers)
