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
from core.router import LLMRouter, _supports_vision, _DEFAULT_VISION_FALLBACK


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

        # ── 新闻情绪截断：原始值可能 -100~100，截断至 ±10 避免单因子主导 ──
        news_c = max(-10.0, min(10.0, news * 0.10))

        # ── PE/PB 多级偏差（放大信号，增加估值区分度） ──
        pe_bias = 0.0
        if pe is not None:
            pe_f = float(pe)
            if pe_f < 0:
                pe_bias = -10.0    # 亏损股
            elif pe_f < 15:
                pe_bias = 12.0     # 深度价值
            elif pe_f < 25:
                pe_bias = 5.0      # 合理偏低
            elif pe_f > 100:
                pe_bias = -10.0    # 严重高估
            elif pe_f > 60:
                pe_bias = -8.0     # 高估
            elif pe_f > 40:
                pe_bias = -4.0     # 偏高
        pb_bias = 0.0
        if pb is not None:
            pb_f = float(pb)
            if pb_f < 1:
                pb_bias = 8.0      # 破净
            elif pb_f < 2:
                pb_bias = 4.0      # 低估
            elif pb_f > 50:
                pb_bias = -8.0     # 极度高估
            elif pb_f > 15:
                pb_bias = -5.0     # 高估
            elif pb_f > 8:
                pb_bias = -3.0     # 偏高

        # ── 15维度评分公式（放大系数，覆盖 [15,85] 区间） ──
        base_dim = {
            "TREND":         50 + 1.0 * mom + 0.8 * trend + 0.3 * pct + 0.12 * dd,
            "TECH":          50 + 0.7 * mom + 0.6 * trend + (vr - 1.0) * 12 - 0.4 * vol,
            "LIQ":           50 + (vr - 1.0) * 18 - 0.3 * abs(pct) - 0.2 * vol + (3.0 if turn is not None else -2.0),
            "CAPITAL_FLOW":  50 + 0.5 * pct + (vr - 1.0) * 16 + news_c * 0.5 + 0.35 * mom,
            "SECTOR_POLICY": 50 + news_c * 0.8 + 0.3 * mom,
            "BETA":          50 + 0.6 * trend + 0.5 * pct - 0.8 * vol + 0.15 * mom,
            "SENTIMENT":     50 + news_c + 0.3 * pct,
            "FUNDAMENTAL":   50 + pe_bias + pb_bias + 0.25 * mom,
            "QUANT":         50 + 0.7 * mom - 0.35 * vol + (vr - 1.0) * 10,
            "MACRO":         50 + 0.25 * mom - 0.25 * vol + news_c * 0.5,
            "INDUSTRY":      50 + news_c * 0.8 + 0.3 * mom,
            "FLOW_DETAIL":   50 + (vr - 1.0) * 20 + 0.4 * pct,
            "MM_BEHAVIOR":   50 + (vr - 1.0) * 14 + news_c * 0.6 - 0.35 * vol,
            "NLP_SENTIMENT": 50 + news_c * 1.2 + 0.2 * mom,
            "DERIV_MARGIN":  50 + 0.4 * pct - (vr - 1.0) * 8 - 0.2 * vol + 0.1 * mom,
        }
        # TOP_STRUCTURE / BOTTOM_STRUCTURE：基于 kline_indicators 多周期结构研判
        # 评分语义：TOP高分=顶部信号强(卖出警示)，BOTTOM高分=底部信号强(买入参考)
        # 两者独立评估，交叉印证；通常只有一个结构显著
        kli = f.get("kline_indicators", {})
        top_signals, bot_signals = 0.0, 0.0
        top_tf_detail: dict[str, float] = {}
        bot_tf_detail: dict[str, float] = {}
        tf_label_map = {"day": "日线", "week": "周线", "month": "月线"}
        for tf in ("day", "week", "month"):
            ind = kli.get(tf)
            if not isinstance(ind, dict) or not ind.get("ok"):
                continue
            upper = float(ind.get("upper_shadow_ratio", 0))
            lower = float(ind.get("lower_shadow_ratio", 0))
            mom_tf = float(ind.get("momentum_10", 0))
            amp = float(ind.get("amplitude_20", 0))
            # 顶部结构：长上影线+负动量 或 高振幅+负动量（头肩顶/M顶/长上影线）
            tf_top = 0.0
            if upper > 40 and mom_tf < 0:
                tf_top = 10
            elif upper > 35:
                tf_top = 5
            elif mom_tf < -5 and amp > 15:
                tf_top = 6
            elif mom_tf < 0:
                tf_top = 2
            top_signals += tf_top
            if tf_top > 0:
                top_tf_detail[tf_label_map[tf]] = tf_top
            # 底部结构：长下影线+正动量 或 正动量反弹（W底/长下影线阳线）
            tf_bot = 0.0
            if lower > 40 and mom_tf > 0:
                tf_bot = 10
            elif lower > 35:
                tf_bot = 5
            elif mom_tf > 5:
                tf_bot = 6
            elif mom_tf > 0:
                tf_bot = 2
            bot_signals += tf_bot
            if tf_bot > 0:
                bot_tf_detail[tf_label_map[tf]] = tf_bot
        # TOP: 高分=顶部信号强(卖出警示)；在最终加权中需用 100-score 反转
        base_dim["TOP_STRUCTURE"] = 50.0 + top_signals * 1.5 - bot_signals * 0.4
        # BOTTOM: 高分=底部信号强(买入参考)
        base_dim["BOTTOM_STRUCTURE"] = 50.0 + bot_signals * 1.5 - top_signals * 0.4
        # per-tf detail 可从 kline_indicators 重算，不需要存到 self
        # K线视觉智能体的文字降级评分（无图时使用综合技术信号）
        tech_score = float(base_dim.get("TECH", 50.0))
        base_dim["KLINE_DAY"] = tech_score
        base_dim["KLINE_WEEK"] = (tech_score + float(base_dim.get("TREND", 50.0))) / 2.0
        base_dim["KLINE_MONTH"] = (tech_score + float(base_dim.get("TREND", 50.0)) + float(base_dim.get("FUNDAMENTAL", 50.0))) / 3.0
        base_dim["KLINE_1H"] = 50 + (vr - 1.0) * 14 + 0.4 * pct - 0.2 * vol
        # KLINE_PATTERN：多周期K线形态组合评分（月线0.40 + 周线0.35 + 日线0.25）
        kp_score = 50.0
        kp_total_w = 0.0
        for _tf, _w in (("month", 0.40), ("week", 0.35), ("day", 0.25)):
            _td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            _pats = _td.get("kline_patterns", []) if isinstance(_td, dict) else []
            if not _pats:
                continue
            _bull = sorted([(float(p.get("confidence", 50)) - 50) for p in _pats if p.get("direction") == "bullish"], reverse=True)
            _bear = sorted([(float(p.get("confidence", 50)) - 50) for p in _pats if p.get("direction") == "bearish"], reverse=True)
            _net = ((_bull[0] if _bull else 0.0) + sum(_bull[1:]) * 0.25
                    - (_bear[0] if _bear else 0.0) - sum(_bear[1:]) * 0.25)
            _net = max(-35.0, min(35.0, _net))
            kp_score += _net * _w
            kp_total_w += _w
        if kp_total_w > 0:
            kp_score = 50.0 + (kp_score - 50.0) / kp_total_w
        base_dim["KLINE_PATTERN"] = max(10.0, min(90.0, kp_score))

        # DIVERGENCE：MACD/RSI背离信号（多周期加权: 月0.40 + 周0.35 + 日0.25）
        div_score = 50.0
        div_total_w = 0.0
        for _tf, _w in (("month", 0.40), ("week", 0.35), ("day", 0.25)):
            _td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            _dv = _td.get("divergence", {}) if isinstance(_td, dict) else {}
            _ds = float(_dv.get("divergence_score", 0))
            if _ds != 0:
                div_score += _ds * _w * 0.8  # 放大缩放
                div_total_w += _w
        if div_total_w > 0:
            div_score = 50.0 + (div_score - 50.0) / div_total_w
        base_dim["DIVERGENCE"] = max(10.0, min(90.0, div_score))

        # VOLUME_PRICE：量价关系信号（多周期加权: 日0.50 + 周0.30 + 月0.20）
        vp_score = 50.0
        vp_total_w = 0.0
        for _tf, _w in (("day", 0.50), ("week", 0.30), ("month", 0.20)):
            _td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            _vp = _td.get("volume_price", {}) if isinstance(_td, dict) else {}
            _vs = float(_vp.get("volume_price_score", 0))
            if _vs != 0:
                vp_score += _vs * _w * 1.0
                vp_total_w += _w
        if vp_total_w > 0:
            vp_score = 50.0 + (vp_score - 50.0) / vp_total_w
        base_dim["VOLUME_PRICE"] = max(10.0, min(90.0, vp_score))

        # SUPPORT_RESISTANCE：支撑阻力位信号（仅日线,最直接）
        sr_data = (kli.get("day", {}) or {}).get("support_resistance", {})
        sr_raw = float(sr_data.get("sr_score", 0)) if isinstance(sr_data, dict) else 0
        base_dim["SUPPORT_RESISTANCE"] = max(10.0, min(90.0, 50.0 + sr_raw))

        # CHANLUN：缠论买卖点信号（多周期加权: 日0.45 + 周0.35 + 月0.20）
        cl_score = 50.0
        cl_total_w = 0.0
        for _tf, _w in (("day", 0.45), ("week", 0.35), ("month", 0.20)):
            _td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            _cl = _td.get("chanlun", {}) if isinstance(_td, dict) else {}
            _cs = float(_cl.get("chanlun_score", 0))
            if _cs != 0:
                cl_score += _cs * _w * 1.0
                cl_total_w += _w
        if cl_total_w > 0:
            cl_score = 50.0 + (cl_score - 50.0) / cl_total_w
        base_dim["CHANLUN"] = max(10.0, min(90.0, cl_score))

        # CHART_PATTERN：大级别图形形态（日0.50 + 周0.30 + 月0.20）
        cp_score = 50.0
        cp_total_w = 0.0
        for _tf, _w in (("day", 0.50), ("week", 0.30), ("month", 0.20)):
            _td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            _cp = _td.get("chart_patterns", {}) if isinstance(_td, dict) else {}
            _cps = float(_cp.get("chart_pattern_score", 0))
            if _cps != 0:
                cp_score += _cps * _w * 0.8
                cp_total_w += _w
        if cp_total_w > 0:
            cp_score = 50.0 + (cp_score - 50.0) / cp_total_w
        base_dim["CHART_PATTERN"] = max(10.0, min(90.0, cp_score))

        # TIMEFRAME_RESONANCE：多周期共振评分
        # 检查日/周/月三周期的趋势斜率和动量方向是否一致
        tf_directions = {}
        for _tf in ("day", "week", "month"):
            _td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(_td, dict) or not _td.get("ok"):
                continue
            _slope = float(_td.get("trend_slope_pct", 0))
            _mom = float(_td.get("momentum_10", 0))
            if _slope > 0.05 and _mom > 0:
                tf_directions[_tf] = "bullish"
            elif _slope < -0.05 and _mom < 0:
                tf_directions[_tf] = "bearish"
            else:
                tf_directions[_tf] = "neutral"
        res_score = 50.0
        if len(tf_directions) >= 2:
            directions = list(tf_directions.values())
            bull_count = sum(1 for d in directions if d == "bullish")
            bear_count = sum(1 for d in directions if d == "bearish")
            if bull_count >= 3:
                res_score = 80.0  # 三周期共振看涨
            elif bull_count == 2 and bear_count == 0:
                res_score = 68.0
            elif bear_count >= 3:
                res_score = 20.0  # 三周期共振看跌
            elif bear_count == 2 and bull_count == 0:
                res_score = 32.0
            elif bull_count >= 1 and bear_count >= 1:
                res_score = 45.0  # 多空分歧
        base_dim["TIMEFRAME_RESONANCE"] = max(10.0, min(90.0, res_score))

        # TRENDLINE：趋势线突破检测
        # 基于日线最近价格与20日趋势线的关系
        tl_score = 50.0
        day_data = kli.get("day", {}) if isinstance(kli, dict) else {}
        if isinstance(day_data, dict) and day_data.get("ok"):
            slope = float(day_data.get("trend_slope_pct", 0))
            day_mom = float(day_data.get("momentum_10", 0))
            day_vr = vr  # volume ratio
            # 趋势线转向+量能配合 = 突破信号
            if slope > 0.1 and day_mom > 3:
                tl_score += 18
                if day_vr > 1.3:
                    tl_score += 10  # 放量突破趋势线
            elif slope < -0.1 and day_mom < -3:
                tl_score -= 18
                if day_vr > 1.3:
                    tl_score -= 10  # 放量跌破趋势线
            # 斜率反转信号
            week_data = kli.get("week", {}) if isinstance(kli, dict) else {}
            if isinstance(week_data, dict) and week_data.get("ok"):
                week_slope = float(week_data.get("trend_slope_pct", 0))
                if slope > 0 and week_slope < 0:
                    tl_score += 8  # 日线拐头向上但周线仍下，可能是反转初期
                elif slope < 0 and week_slope > 0:
                    tl_score -= 8  # 日线拐头向下但周线仍上，可能是调整
        base_dim["TRENDLINE"] = max(10.0, min(90.0, tl_score))
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


class DivergenceAgent(AnalystAgent):
    """MACD/RSI背离检测Agent：检测多周期顶底背离信号。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

        kli = ctx.get("features", {}).get("kline_indicators", {})
        tf_labels = {"day": "日线", "week": "周线", "month": "月线"}
        for tf, label in tf_labels.items():
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            dv = td.get("divergence", {})
            vp = td.get("volume_price", {})
            lines = [f"[{label}] RSI={td.get('rsi', 'N/A')} | MACD DIF={td.get('macd_dif', 'N/A')} | 动量={td.get('momentum_10', 0):.1f}%"]
            if dv:
                if dv.get("macd_top_div"):
                    lines.append(f"  !! {dv.get('macd_div_desc', 'MACD顶背离')}")
                if dv.get("macd_bot_div"):
                    lines.append(f"  !! {dv.get('macd_div_desc', 'MACD底背离')}")
                if dv.get("rsi_top_div"):
                    lines.append(f"  !! {dv.get('rsi_div_desc', 'RSI顶背离')}")
                if dv.get("rsi_bot_div"):
                    lines.append(f"  !! {dv.get('rsi_div_desc', 'RSI底背离')}")
                if not any(dv.get(k) for k in ("macd_top_div", "macd_bot_div", "rsi_top_div", "rsi_bot_div")):
                    lines.append("  无背离信号")
                lines.append(f"  背离综合分={dv.get('divergence_score', 0)}")
            parts.append("\n".join(lines))

        parts.append(
            "\n分析要求：\n1.背离信号可靠性评估（是否真背离vs假背离）\n"
            "2.多周期背离是否共振\n3.结合RSI超买超卖区域判断背离有效性\n"
            "4.给出操作建议和风险提示\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class VolumePriceAgent(AnalystAgent):
    """量价关系分析Agent：检测放量突破、缩量回踩、量能异动等信号。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

        kli = ctx.get("features", {}).get("kline_indicators", {})
        tf_labels = {"day": "日线", "week": "周线", "month": "月线"}
        for tf, label in tf_labels.items():
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            vp = td.get("volume_price", {})
            lines = [f"[{label}] 动量={td.get('momentum_10', 0):.1f}%"]
            if vp:
                flags = []
                if vp.get("volume_breakout"): flags.append("放量突破")
                if vp.get("shrink_pullback"): flags.append("缩量回踩")
                if vp.get("volume_anomaly"): flags.append("底部放量")
                if vp.get("climax_volume"): flags.append("天量滞涨")
                lines.append(f"  信号: {', '.join(flags) if flags else '无特殊信号'}")
                lines.append(f"  OBV趋势: {vp.get('obv_trend', 'N/A')}")
                lines.append(f"  量价评分: {vp.get('volume_price_score', 0)}")
                if vp.get("desc"):
                    lines.append(f"  详情: {vp['desc']}")
            parts.append("\n".join(lines))

        parts.append(
            "\n分析要求：\n1.量价配合是否健康\n"
            "2.是否有主力资金进出信号\n3.成交量变化趋势分析\n"
            "4.给出操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class SupportResistanceAgent(AnalystAgent):
    """支撑阻力位检测Agent：检测关键价格位并评估当前位置。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

        kli = ctx.get("features", {}).get("kline_indicators", {})
        # 主要用日线的支撑阻力
        td = kli.get("day", {}) if isinstance(kli, dict) else {}
        sr = td.get("support_resistance", {}) if isinstance(td, dict) else {}
        if sr:
            parts.append(f"价格位置: {sr.get('price_position', 'N/A')}")
            sups = sr.get("support_levels", [])
            if sups:
                lines = ["支撑位:"]
                for s in sups[:5]:
                    lines.append(f"  {s.get('price', 'N/A')} ({s.get('label', '')})")
                parts.append("\n".join(lines))
            ress = sr.get("resistance_levels", [])
            if ress:
                lines = ["阻力位:"]
                for r in ress[:5]:
                    lines.append(f"  {r.get('price', 'N/A')} ({r.get('label', '')})")
                parts.append("\n".join(lines))
            gaps = sr.get("gaps", [])
            if gaps:
                parts.append(f"跳空缺口: {len(gaps)}个")
            if sr.get("desc"):
                parts.append(f"评估: {sr['desc']}")

        # Fibonacci
        fib = ctx.get("features", {}).get("fibonacci_key_levels", {})
        if fib and fib.get("ok"):
            parts.append(
                f"Fibonacci: 波段高={fib.get('band_high')}, 低={fib.get('band_low')}, "
                f"38.2%={fib.get('retrace_382')}, 50%={fib.get('retrace_50')}, 61.8%={fib.get('retrace_618')}"
            )

        parts.append(
            "\n分析要求：\n1.当前价格距支撑/阻力位的距离和突破可能性\n"
            "2.哪些是强支撑/强阻力（多重验证）\n3.建议的入场区间和止损位\n"
            "4.给出操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class TimeframeResonanceAgent(AnalystAgent):
    """多周期共振评分Agent：检测日/周/月周期信号是否一致共振。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

        kli = ctx.get("features", {}).get("kline_indicators", {})
        tf_labels = {"day": "日线(短期)", "week": "周线(中期)", "month": "月线(长期)"}
        for tf, label in tf_labels.items():
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            slope = td.get("trend_slope_pct", 0)
            mom = td.get("momentum_10", 0)
            rsi = td.get("rsi", "N/A")
            macd_hist = td.get("macd_hist", "N/A")
            parts.append(
                f"[{label}] 趋势斜率={slope:.4f}%/bar | 动量={mom:.1f}% | RSI={rsi} | MACD柱={macd_hist}"
            )

        parts.append(
            "\n分析要求：\n1.日/周/月三周期趋势方向是否一致\n"
            "2.共振程度评估（强共振/弱共振/分歧）\n"
            "3.如有分歧应以哪个周期为主\n"
            "4.给出操作建议和信号强度\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class TrendlineAgent(AnalystAgent):
    """趋势线突破检测Agent：检测价格对趋势线的突破或跌破信号。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

        f = ctx.get("features", {})
        vr = f.get("volume_ratio_5_20", 1.0)
        parts.append(f"量比5/20: {vr:.2f}")

        kli = f.get("kline_indicators", {})
        tf_labels = {"day": "日线", "week": "周线", "month": "月线"}
        for tf, label in tf_labels.items():
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            slope = td.get("trend_slope_pct", 0)
            mom = td.get("momentum_10", 0)
            ma = td.get("ma_system", {})
            ma20 = ma.get("ma20", {}).get("pct_above", "N/A")
            ma60 = ma.get("ma60", {}).get("pct_above", "N/A")
            parts.append(f"[{label}] 趋势斜率={slope:.4f}%/bar | 动量={mom:.1f}% | 偏离MA20={ma20}% | 偏离MA60={ma60}%")

        parts.append(
            "\n分析要求：\n1.趋势线是否被有效突破（需放量确认）\n"
            "2.突破后是否存在回踩确认\n"
            "3.趋势转折还是短暂波动\n"
            "4.给出操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class ChartPatternAgent(AnalystAgent):
    """大级别图形形态Agent：三角形、旗形、箱体、杯柄、圆弧底/顶等20-60根K线级别形态。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

        kli = ctx.get("features", {}).get("kline_indicators", {})
        tf_labels = {"day": "日线", "week": "周线", "month": "月线"}
        for tf, label in tf_labels.items():
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            cp = td.get("chart_patterns", {})
            lines = [f"[{label}图形形态]"]
            if cp and cp.get("patterns"):
                for p in cp["patterns"]:
                    icon = "↑" if p.get("direction") == "bullish" else ("↓" if p.get("direction") == "bearish" else "→")
                    lines.append(f"  {icon}【{p['name']}】置信度{p.get('confidence', 50)}% | {p.get('desc', '')}")
                lines.append(f"  图形评分: {cp.get('chart_pattern_score', 0)}")
            else:
                lines.append("  未检测到明显形态")
            parts.append("\n".join(lines))

        parts.append(
            "\n分析要求：\n1.图形形态的完整度和可靠性\n"
            "2.预期突破方向和目标位\n3.形态失败的止损位\n"
            "4.给出操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class ChanlunAgent(AnalystAgent):
    """缠论买卖点Agent：基于缠中说禅理论检测分型、笔、中枢和三类买卖点。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

        kli = ctx.get("features", {}).get("kline_indicators", {})
        tf_labels = {"day": "日线", "week": "周线", "month": "月线"}
        for tf, label in tf_labels.items():
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            cl = td.get("chanlun", {})
            lines = [f"[{label}缠论]"]
            if cl:
                # 中枢
                zs = cl.get("zhongshu", [])
                if zs:
                    last_zs = zs[-1]
                    lines.append(f"  中枢: [{last_zs.get('low','?')}-{last_zs.get('high','?')}] ({last_zs.get('bi_count',0)}笔)")
                # 笔
                bis = cl.get("bi_list", [])
                if bis:
                    for b in bis[-3:]:
                        lines.append(f"  笔: {b.get('dir','')} {b.get('start_price','?')}→{b.get('end_price','?')}")
                # 买卖点
                for s in cl.get("buy_signals", []):
                    lines.append(f"  ↑ 【{s.get('type','')}】{s.get('desc','')}")
                for s in cl.get("sell_signals", []):
                    lines.append(f"  ↓ 【{s.get('type','')}】{s.get('desc','')}")
                lines.append(f"  缠论评分: {cl.get('chanlun_score', 0)}")
                if cl.get("desc"):
                    lines.append(f"  概述: {cl['desc']}")
            else:
                lines.append("  数据不足")
            parts.append("\n".join(lines))

        parts.append(
            "\n分析要求：\n1.当前处于缠论哪个阶段（上涨/下跌/中枢震荡）\n"
            "2.是否存在明确的买卖点信号\n3.中枢位置对后续走势的指引意义\n"
            "4.结合笔的方向给出操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class KlinePatternAgent(AnalystAgent):
    """多周期K线形态组合风险评估Agent。

    对日/周/月三周期最新3-5根K线进行形态识别，涵盖15+种经典形态（锤子线、
    射击之星、吞噬、孕线、早晨/黄昏之星、红三兵/三乌鸦、W底/M顶、头肩顶底等），
    综合多周期共振判断未来方向，作为独立风险评估维度加入整体框架。
    """

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        """构建K线形态专属数据摘要，供大模型增强分析使用。"""
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

        kli = ctx.get("features", {}).get("kline_indicators", {})
        tf_labels = {"day": "日线(短期)", "week": "周线(中期)", "month": "月线(长期)"}

        for tf, label in tf_labels.items():
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                parts.append(f"[{label}] 数据不足")
                continue
            lines = [f"[{label}] {td.get('rows', 0)}根K线 | 收盘={td.get('close', 'N/A')} "
                     f"| 动量={td.get('momentum_10', 0):.1f}% | RSI={td.get('rsi', 'N/A')} "
                     f"| 趋势斜率={td.get('trend_slope_pct', 0):.4f}%/bar"]
            pats = td.get("kline_patterns", [])
            if pats:
                for p in pats:
                    icon = "↑" if p.get("direction") == "bullish" else ("↓" if p.get("direction") == "bearish" else "→")
                    lines.append(f"  {icon}【{p['name']}】置信度{p.get('confidence', 50)}% | {p.get('desc', '')}")
            ma = td.get("ma_system", {})
            ma_parts = []
            for period in (5, 20, 60):
                mv = ma.get(f"ma{period}", {})
                if mv.get("value"):
                    ma_parts.append(f"MA{period}={mv['value']:.2f}({mv.get('pct_above', 0):+.1f}%)")
            if ma_parts:
                lines.append("  " + " | ".join(ma_parts))
            parts.append("\n".join(lines))

        parts.append(
            "\n分析要求：\n1.各周期形态是否共振（多头/空头/分歧）\n"
            "2.当前最关键形态及其预测含义\n3.结合均线给出支撑/压力位\n"
            "4.预判未来1-3根K线的方向与风险\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


def create_agent(
    cfg: dict[str, Any],
    run_dir: Path,
    backend: DataBackend,
    llm_routers: dict[str, LLMRouter] | None = None,
) -> AnalystAgent:
    """Agent 工厂：根据 cfg 中的 agent_type 决定返回对应 Agent 实例。"""
    agent_type = cfg.get("agent_type", "analyst")
    if agent_type == "kline_vision":
        timeframe = cfg.get("timeframe", "day")
        return KlineVisionAgent(cfg, run_dir, backend, timeframe=timeframe, llm_routers=llm_routers)
    if agent_type == "kline_pattern":
        return KlinePatternAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "divergence":
        return DivergenceAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "volume_price":
        return VolumePriceAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "support_resistance":
        return SupportResistanceAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "chanlun":
        return ChanlunAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "chart_pattern":
        return ChartPatternAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "timeframe_resonance":
        return TimeframeResonanceAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "trendline":
        return TrendlineAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    return AnalystAgent(cfg, run_dir, backend, llm_routers=llm_routers)
