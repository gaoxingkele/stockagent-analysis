# -*- coding: utf-8 -*-
"""v2 Agent体系：12个精简Agent，差异化评分公式。

合并前29个Agent → 12个真正独立的分析维度，每个维度有独立的量化逻辑。
"""
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


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # 通用数据上下文（供LLM参考）
    # ------------------------------------------------------------------
    @staticmethod
    def _build_data_context(ctx: dict[str, Any]) -> str:
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
        # 基本面扩展字段
        fund_parts = []
        roe = f.get("roe")
        if roe is not None:
            fund_parts.append(f"ROE={roe:.1f}%")
        rev_yoy = f.get("revenue_yoy")
        if rev_yoy is not None:
            fund_parts.append(f"营收增速={rev_yoy:.1f}%")
        np_yoy = f.get("netprofit_yoy")
        if np_yoy is not None:
            fund_parts.append(f"净利润增速={np_yoy:.1f}%")
        debt = f.get("debt_to_assets")
        if debt is not None:
            fund_parts.append(f"资产负债率={debt:.1f}%")
        cr = f.get("current_ratio")
        if cr is not None:
            fund_parts.append(f"流动比率={cr:.2f}")
        qr = f.get("quick_ratio")
        if qr is not None:
            fund_parts.append(f"速动比率={qr:.2f}")
        gm = f.get("grossprofit_margin")
        if gm is not None:
            fund_parts.append(f"毛利率={gm:.1f}%")
        if fund_parts:
            parts.append("基本面: " + " | ".join(fund_parts))
        # 同业对比
        peer = f.get("peer_comparison", {})
        if peer.get("peers"):
            ind_avg = peer.get("industry_avg", {})
            peer_names = [p["name"] for p in peer["peers"][:5]]
            peer_line = f"同业({peer.get('industry','')})TOP5: {', '.join(peer_names)}"
            if ind_avg.get("pe_ttm"):
                peer_line += f" | 行业PE均值={ind_avg['pe_ttm']:.1f}"
            if ind_avg.get("roe"):
                peer_line += f" | 行业ROE均值={ind_avg['roe']:.1f}%"
            parts.append(peer_line)
        # 筹码分布
        chip = f.get("chip_distribution", {})
        if chip:
            parts.append(
                f"筹码分布: 获利{chip.get('profit_ratio', 'N/A')}% | "
                f"套牢{chip.get('trapped_ratio', 'N/A')}% | "
                f"集中度{chip.get('concentration', 'N/A')}% | "
                f"平均成本{chip.get('avg_cost', 'N/A')} | "
                f"健康度{chip.get('health_score', 'N/A')}"
            )
        # 市场策略
        strategy = f.get("market_strategy", {})
        if strategy:
            parts.append(
                f"市场策略: {strategy.get('phase_cn', '未知')}阶段 | "
                f"建议最大仓位{strategy.get('position_cap', 0.6) * 100:.0f}% | "
                f"{strategy.get('sector_bias', '')}"
            )
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
        for p in ("grok", "gemini", "kimi", "deepseek", "glm"):
            if p in llm_evals:
                short = llm_evals[p].replace("\n", " ").strip()
                parts.append(f"{p}: {short[:90]}")
        if not parts:
            for p, txt in llm_evals.items():
                parts.append(f"{p}: {txt.replace(chr(10), ' ')[:90]}")
        return "综合评价：" + "；".join(parts)

    # ------------------------------------------------------------------
    # 辅助评分函数
    # ------------------------------------------------------------------
    @staticmethod
    def _calc_fundamental_extra(f: dict) -> float:
        """ROE/成长性/负债率/流动性/同业相对估值评分偏差。"""
        bias = 0.0
        # ── ROE ──
        roe = f.get("roe")
        if roe is not None:
            roe_v = float(roe)
            if roe_v > 20: bias += 8
            elif roe_v > 15: bias += 5
            elif roe_v > 10: bias += 3
            elif roe_v < 0: bias -= 6
        # ── 成长性 ──
        rev_yoy = f.get("revenue_yoy")
        np_yoy = f.get("netprofit_yoy")
        if rev_yoy is not None and np_yoy is not None:
            rv, nv = float(rev_yoy), float(np_yoy)
            if rv > 30 and nv > 30: bias += 10
            elif rv > 15 and nv > 15: bias += 5
            elif rv < -10 or nv < -20: bias -= 8
        # ── 负债率 ──
        debt = f.get("debt_to_assets")
        if debt is not None:
            dv = float(debt)
            if dv > 80: bias -= 6
            elif dv > 70: bias -= 4
            elif dv > 60: bias -= 2
            elif dv < 30: bias += 3
        # ── 流动比率 ──
        cr = f.get("current_ratio")
        if cr is not None:
            cv = float(cr)
            if cv < 1.0: bias -= 4
            elif cv < 1.5: bias -= 2
            elif cv > 2.5: bias += 2
        # ── 速动比率 ──
        qr = f.get("quick_ratio")
        if qr is not None:
            qv = float(qr)
            if qv < 0.8: bias -= 3
            elif qv < 1.0: bias -= 1
            elif qv > 1.5: bias += 2
        # ── 同业相对估值 ──
        peer = f.get("peer_comparison", {})
        ind_avg = peer.get("industry_avg", {})
        if ind_avg:
            pe_self = f.get("pe_ttm")
            pe_avg = ind_avg.get("pe_ttm")
            if pe_self and pe_avg and pe_avg > 0 and pe_self > 0:
                ratio = float(pe_self) / float(pe_avg)
                if ratio < 0.6: bias += 6     # 远低于行业均值
                elif ratio < 0.85: bias += 3  # 略低于行业
                elif ratio > 1.5: bias -= 5   # 远高于行业
                elif ratio > 1.2: bias -= 2   # 略高于行业
        return bias

    @staticmethod
    def _calc_margin_hsgt_bias(f: dict) -> float:
        """融资融券和北向资金偏差。"""
        bias = 0.0
        margin = f.get("margin_data", {})
        rzye_5d = float(margin.get("rzye_change_5d") or 0)
        if rzye_5d > 5: bias += 6
        elif rzye_5d < -5: bias -= 6
        elif rzye_5d > 2: bias += 3
        elif rzye_5d < -2: bias -= 3
        hsgt = f.get("hsgt_data", {})
        hk_chg = float(hsgt.get("hk_hold_change_pct") or 0)
        if hk_chg > 3: bias += 5
        elif hk_chg < -3: bias -= 5
        elif hk_chg > 1: bias += 2
        elif hk_chg < -1: bias -= 2
        return bias

    # ------------------------------------------------------------------
    # v2 差异化评分公式
    # ------------------------------------------------------------------
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
        kli = f.get("kline_indicators", {})
        news_c = max(-10.0, min(10.0, news * 0.10))

        score = self._calc_dim_score(
            pct, mom, vol, dd, vr, trend, news_c, pe, pb, turn, kli, f
        )
        score = max(0.0, min(100.0, score))

        data_quality = float(f.get("data_quality_score", 0.6))
        confidence = 0.45 + 0.45 * data_quality
        if snap.source == "mock":
            confidence -= 0.18
        confidence = max(0.1, min(0.98, confidence))
        if score >= 70:
            return "buy", score, confidence, f"{self.role}({self.dim_code})：偏强，评分{score:.1f}"
        if score < 50:
            return "sell", score, confidence, f"{self.role}({self.dim_code})：偏弱，评分{score:.1f}"
        return "hold", score, confidence, f"{self.role}({self.dim_code})：中性，评分{score:.1f}"

    def _calc_dim_score(self, pct, mom, vol, dd, vr, trend, news_c, pe, pb, turn, kli, f) -> float:
        """按 dim_code 分发到对应评分公式。子类可覆盖此方法。"""
        dim = self.dim_code
        if dim == "TREND_MOMENTUM":
            return self._score_trend_momentum(pct, mom, vol, dd, vr, trend, kli, f)
        if dim == "TECH_QUANT":
            return self._score_tech_quant(pct, mom, vol, vr, kli)
        if dim == "CAPITAL_LIQUIDITY":
            return self._score_capital_liquidity(pct, mom, vr, turn, kli)
        if dim == "FUNDAMENTAL":
            return self._score_fundamental(mom, pe, pb, f)
        if dim == "SENTIMENT_FLOW":
            return self._score_sentiment_flow(pct, mom, vol, vr, news_c, f)
        if dim == "DERIV_MARGIN":
            return 50 + self._calc_margin_hsgt_bias(f) + 0.3 * pct - 0.15 * vol
        if dim == "DIVERGENCE":
            return self._score_divergence(kli)
        if dim == "CHANLUN":
            return self._score_multi_tf(kli, "chanlun", "chanlun_score", 1.0)
        if dim == "PATTERN":
            return self._score_pattern(kli)
        if dim == "VOLUME_STRUCTURE":
            return self._score_volume_structure(pct, vr, kli)
        if dim == "RESONANCE":
            return self._score_resonance(mom, kli, f)
        if dim == "KLINE_VISION":
            return self._score_kline_vision_fallback(mom, vol, vr, pct, kli)
        # 未知维度: 中性
        return 50.0

    # ---------- 1. TREND_MOMENTUM: MA排列+趋势线+乖离率 ----------
    def _score_trend_momentum(self, pct, mom, vol, dd, vr, trend, kli, f) -> float:
        day = kli.get("day", {}) if isinstance(kli, dict) else {}
        ma_sys = day.get("ma_system", {}) if isinstance(day, dict) else {}

        # (a) MA排列评分：统计短>长的相邻对数
        ma_vals = {}
        for p in (5, 10, 20, 60, 120):
            mv = ma_sys.get(f"ma{p}", {})
            v = mv.get("value") if isinstance(mv, dict) else None
            if v is not None:
                ma_vals[p] = float(v)
        periods = sorted(ma_vals.keys())
        ordered = sum(1 for i in range(len(periods) - 1) if ma_vals[periods[i]] > ma_vals[periods[i + 1]])
        reversed_ = sum(1 for i in range(len(periods) - 1) if ma_vals[periods[i]] < ma_vals[periods[i + 1]])
        n_pairs = max(1, len(periods) - 1)
        # 全多头+20, 全空头-20, 线性插值
        ma_score = (ordered - reversed_) / n_pairs * 20.0

        # (b) 趋势斜率: slope_pct → [-15, +15]
        slope = float(day.get("trend_slope_pct", 0)) if isinstance(day, dict) else 0
        slope_score = max(-15.0, min(15.0, slope * 80))

        # (c) 趋势线突破
        tl_bonus = 0.0
        _tl = day.get("trendlines", {}) if isinstance(day, dict) else {}
        if isinstance(_tl, dict):
            _dtl = _tl.get("down_trendline")
            if isinstance(_dtl, dict) and _dtl.get("broken"):
                _bo = _dtl.get("breakout", {})
                if isinstance(_bo, dict) and _bo.get("confirmed"):
                    tl_bonus += 12
                else:
                    tl_bonus += 5
            _utl = _tl.get("up_trendline")
            if isinstance(_utl, dict) and _utl.get("broken"):
                _bo = _utl.get("breakout", {})
                if isinstance(_bo, dict) and _bo.get("confirmed"):
                    tl_bonus -= 12
                else:
                    tl_bonus -= 5

        # (d) 乖离率惩罚
        bias_ma5 = float(ma_sys.get("ma5", {}).get("pct_above", 0) if isinstance(ma_sys.get("ma5"), dict) else 0)
        bias_pen = 0.0
        if abs(bias_ma5) > 8: bias_pen = -25
        elif abs(bias_ma5) > 5: bias_pen = -15
        elif abs(bias_ma5) > 3: bias_pen = -5

        # (e) ADX 趋势强度修正: >25趋势市放大信号, <20震荡市压缩信号
        adx = day.get("adx") if isinstance(day, dict) else None
        adx_adj = 0.0
        if adx is not None:
            adx = float(adx)
            if adx > 40:
                adx_adj = 8.0 if (ma_score + slope_score) > 0 else -8.0  # 强趋势放大方向
            elif adx > 25:
                adx_adj = 4.0 if (ma_score + slope_score) > 0 else -4.0
            elif adx < 15:
                adx_adj = 0.0  # 极弱趋势不加分，但也不强制减分

        score = 50 + ma_score + slope_score + tl_bonus + 0.3 * mom + bias_pen + adx_adj
        if bias_ma5 > 8:
            score = min(score, 35)
        return score

    # ---------- 2. TECH_QUANT: RSI+MACD+布林+量比+动量 ----------
    def _score_tech_quant(self, pct, mom, vol, vr, kli) -> float:
        day = kli.get("day", {}) if isinstance(kli, dict) else {}
        if not isinstance(day, dict):
            day = {}

        # RSI位置评分
        rsi = float(day.get("rsi") or 50)
        rsi_score = 0.0
        if rsi > 80: rsi_score = -12
        elif rsi > 70: rsi_score = -6
        elif rsi < 20: rsi_score = 12
        elif rsi < 30: rsi_score = 6

        # MACD柱状图方向
        macd_hist = float(day.get("macd_hist") or 0)
        macd_score = max(-10.0, min(10.0, macd_hist * 5))

        # KDJ超买超卖
        kdj_k = float(day.get("kdj_k") or 50)
        kdj_score = 0.0
        if kdj_k > 80: kdj_score = -5
        elif kdj_k < 20: kdj_score = 5

        # 量比
        vr_score = max(-10.0, min(10.0, (vr - 1.0) * 12))

        # ATR 波动率修正: ATR/价格 > 5% 高波动扣分, < 2% 低波动小幅加分
        atr_val = day.get("atr") if isinstance(day, dict) else None
        close_val = day.get("close") if isinstance(day, dict) else None
        atr_adj = 0.0
        if atr_val and close_val and float(close_val) > 0:
            atr_pct = float(atr_val) / float(close_val) * 100
            if atr_pct > 5:
                atr_adj = -5.0  # 高波动扣分
            elif atr_pct > 3:
                atr_adj = -2.0
            elif atr_pct < 1.5:
                atr_adj = 2.0   # 低波动稳定加分

        return 50 + rsi_score + macd_score + kdj_score + vr_score + 0.3 * mom - 0.2 * vol + atr_adj

    # ---------- 3. CAPITAL_LIQUIDITY: OBV+量比+换手+量价信号 ----------
    def _score_capital_liquidity(self, pct, mom, vr, turn, kli) -> float:
        day = kli.get("day", {}) if isinstance(kli, dict) else {}
        vp = day.get("volume_price", {}) if isinstance(day, dict) else {}
        if not isinstance(vp, dict):
            vp = {}

        # OBV趋势
        obv_trend = vp.get("obv_trend", "flat")
        obv_score = 12.0 if obv_trend == "up" else (-12.0 if obv_trend == "down" else 0.0)

        # 量比趋势
        vr_score = max(-15.0, min(15.0, (vr - 1.0) * 18))

        # 换手率评估
        turn_score = 0.0
        if turn is not None:
            t = float(turn)
            if t > 15: turn_score = -6      # 过度换手
            elif t > 8: turn_score = -2
            elif t < 0.5: turn_score = -4   # 流动性不足
            elif 1 <= t <= 5: turn_score = 3  # 健康换手

        # 量价信号
        vp_score = 0.0
        if vp.get("volume_breakout"): vp_score += 12
        if vp.get("shrink_pullback"): vp_score += 8
        if vp.get("climax_volume"): vp_score -= 8
        if vp.get("volume_anomaly"): vp_score += 4

        return 50 + obv_score + vr_score + turn_score + vp_score + 0.2 * pct

    # ---------- 4. FUNDAMENTAL: PE高估惩罚(PE>50扣分, <=50中性不占权重) ----------
    def _score_fundamental(self, mom, pe, pb, f) -> float:
        if pe is None:
            return 50.0  # 无PE数据，中性
        pe_f = float(pe)
        if pe_f < 0:
            return 40.0  # 亏损，轻度扣分
        if pe_f <= 50:
            return 50.0  # PE正常范围，中性不干扰
        # PE > 50: 高估惩罚，越高扣越多
        if pe_f > 200:
            return 20.0
        elif pe_f > 100:
            return 30.0
        elif pe_f > 75:
            return 38.0
        else:  # 50 < PE <= 75
            return 44.0

    # ---------- 5. SENTIMENT_FLOW: 新闻+行业+主力行为+量比异动 ----------
    def _score_sentiment_flow(self, pct, mom, vol, vr, news_c, f) -> float:
        # 新闻情绪（主导因子，权重最大）
        news_score = news_c * 1.5

        # 量比异动作为主力行为代理
        mm_score = 0.0
        if vr > 1.8: mm_score = 8      # 大幅放量 → 主力活跃
        elif vr > 1.3: mm_score = 4
        elif vr < 0.5: mm_score = -6   # 极度缩量 → 无人关注
        elif vr < 0.7: mm_score = -3

        # 融资融券趋势作为情绪指标
        margin_sent = 0.0
        margin = f.get("margin_data", {})
        rzye_5d = float(margin.get("rzye_change_5d") or 0)
        if rzye_5d > 3: margin_sent = 4
        elif rzye_5d < -3: margin_sent = -4

        return 50 + news_score + mm_score + margin_sent + 0.2 * mom + 0.15 * pct

    # ---------- 6a. DIVERGENCE: 背离检测 + 无背离时RSI/MACD辅助 ----------
    def _score_divergence(self, kli) -> float:
        # 先用多周期背离检测
        has_signal = False
        score = 50.0
        total_w = 0.0
        for _tf, _w in (("day", 0.50), ("week", 0.35), ("month", 0.15)):
            td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            sub = td.get("divergence", {}) if isinstance(td, dict) else {}
            ds = float(sub.get("divergence_score") or 0) if isinstance(sub, dict) else 0
            if ds != 0:
                score += ds * _w * 1.0
                total_w += _w
                has_signal = True
        if has_signal and total_w > 0:
            score = 50.0 + (score - 50.0) / total_w
            return max(10.0, min(90.0, score))
        # 无背离 — 中性基调，背离agent职责是检测背离而非判多空
        return 50.0

    # ---------- 6b. 多周期加权评分通用 (CHANLUN) ----------
    def _score_multi_tf(self, kli, data_key: str, score_key: str, scale: float) -> float:
        score = 50.0
        total_w = 0.0
        for _tf, _w in (("day", 0.50), ("week", 0.35), ("month", 0.15)):
            td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            sub = td.get(data_key, {}) if isinstance(td, dict) else {}
            ds = float(sub.get(score_key) or 0) if isinstance(sub, dict) else 0
            if ds != 0:
                score += ds * _w * scale
                total_w += _w
        if total_w > 0:
            score = 50.0 + (score - 50.0) / total_w
        return max(10.0, min(90.0, score))

    # ---------- 7. PATTERN: K线形态+图表形态+顶底结构合并 ----------
    def _score_pattern(self, kli) -> float:
        if not isinstance(kli, dict):
            return 50.0

        # (a) K线形态 (原KLINE_PATTERN)
        kp_score = 0.0
        for _tf, _w in (("day", 0.50), ("week", 0.35), ("month", 0.15)):
            td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            pats = td.get("kline_patterns", []) if isinstance(td, dict) else []
            if not pats:
                continue
            bull = sorted([(float(p.get("confidence", 50)) - 50) for p in pats if p.get("direction") == "bullish"], reverse=True)
            bear = sorted([(float(p.get("confidence", 50)) - 50) for p in pats if p.get("direction") == "bearish"], reverse=True)
            net = ((bull[0] if bull else 0) + sum(bull[1:]) * 0.25
                   - (bear[0] if bear else 0) - sum(bear[1:]) * 0.25)
            kp_score += max(-15.0, min(15.0, net)) * _w

        # (b) 图表形态 (原CHART_PATTERN)
        cp_score = 0.0
        for _tf, _w in (("day", 0.50), ("week", 0.35), ("month", 0.15)):
            td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            cp = td.get("chart_patterns", {}) if isinstance(td, dict) else {}
            cps = float(cp.get("chart_pattern_score") or 0) if isinstance(cp, dict) else 0
            if cps != 0:
                cp_score += cps * _w * 0.5

        # (c) 顶底结构 (原TOP/BOTTOM, 内部反转处理)
        top_sig, bot_sig = 0.0, 0.0
        for tf in ("day", "week", "month"):
            ind = kli.get(tf)
            if not isinstance(ind, dict) or not ind.get("ok"):
                continue
            upper = float(ind.get("upper_shadow_ratio") or 0)
            lower = float(ind.get("lower_shadow_ratio") or 0)
            mom_tf = float(ind.get("momentum_10") or 0)
            # 顶部信号
            if upper > 40 and mom_tf < 0: top_sig += 8
            elif upper > 35: top_sig += 4
            elif mom_tf < -5: top_sig += 3
            # 底部信号
            if lower > 40 and mom_tf > 0: bot_sig += 8
            elif lower > 35: bot_sig += 4
            elif mom_tf > 5: bot_sig += 3
        # 顶部信号压制评分, 底部信号抬升评分 (内部完成反转)
        structure_score = (bot_sig - top_sig) * 1.2

        # 连续性修正
        day_td = kli.get("day", {}) if isinstance(kli, dict) else {}
        cs = day_td.get("continuity_stats", {}) if isinstance(day_td, dict) else {}
        cont_bonus = 0.0
        if cs.get("consecutive_bull", 0) >= 3 and cs.get("body_trend") == "escalating":
            cont_bonus += 3
        if cs.get("consecutive_bear", 0) >= 3 and cs.get("body_trend") == "escalating":
            cont_bonus -= 3
        if cs.get("higher_highs", 0) >= 3: cont_bonus += 2
        if cs.get("lower_lows", 0) >= 3: cont_bonus -= 2

        return max(10.0, min(90.0, 50 + kp_score + cp_score + structure_score + cont_bonus))

    # ---------- 8. VOLUME_STRUCTURE: 量价+支撑阻力 ----------
    def _score_volume_structure(self, pct, vr, kli) -> float:
        # 量价关系多周期
        vp_score = 0.0
        for _tf, _w in (("day", 0.50), ("week", 0.35), ("month", 0.15)):
            td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            vp = td.get("volume_price", {}) if isinstance(td, dict) else {}
            vs = float(vp.get("volume_price_score") or 0) if isinstance(vp, dict) else 0
            if vs != 0:
                vp_score += vs * _w

        # 支撑阻力 (日线)
        sr_data = (kli.get("day", {}) or {}).get("support_resistance", {})
        sr_raw = float(sr_data.get("sr_score") or 0) if isinstance(sr_data, dict) else 0

        return max(10.0, min(90.0, 50 + vp_score * 0.6 + sr_raw * 0.4))

    # ---------- 9. RESONANCE: 多周期共振+相对强弱 ----------
    def _score_resonance(self, mom, kli, f) -> float:
        # (a) 多周期共振
        tf_dirs = {}
        for _tf in ("day", "week", "month"):
            td = kli.get(_tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            slope = float(td.get("trend_slope_pct") or 0)
            _mom = float(td.get("momentum_10") or 0)
            if slope > 0.05 and _mom > 0:
                tf_dirs[_tf] = "bullish"
            elif slope < -0.05 and _mom < 0:
                tf_dirs[_tf] = "bearish"
            else:
                tf_dirs[_tf] = "neutral"
        res_score = 50.0
        if len(tf_dirs) >= 2:
            dirs = list(tf_dirs.values())
            bull_c = sum(1 for d in dirs if d == "bullish")
            bear_c = sum(1 for d in dirs if d == "bearish")
            if bull_c >= 3: res_score = 78
            elif bull_c == 2 and bear_c == 0: res_score = 66
            elif bear_c >= 3: res_score = 22
            elif bear_c == 2 and bull_c == 0: res_score = 34
            elif bull_c >= 1 and bear_c >= 1: res_score = 45

        # (b) 相对强弱
        rs = f.get("relative_strength", {})
        rs_adj = 0.0
        if isinstance(rs, dict) and rs.get("ok"):
            ex20 = float(rs.get("excess_return_20d") or 0)
            if ex20 > 15: rs_adj = 10
            elif ex20 > 8: rs_adj = 6
            elif ex20 > 3: rs_adj = 3
            elif ex20 < -15: rs_adj = -10
            elif ex20 < -8: rs_adj = -6
            elif ex20 < -3: rs_adj = -3
            rs_trend = rs.get("rs_trend", "flat")
            if rs_trend == "up": rs_adj += 4
            elif rs_trend == "down": rs_adj -= 4
            # 多层共振
            trend_sigs = []
            for key in ("relative_strength", "rs_vs_industry", "rs_vs_leaders"):
                _rd = f.get(key, {})
                if isinstance(_rd, dict) and _rd.get("ok"):
                    trend_sigs.append(_rd.get("rs_trend", "flat"))
            up_c = sum(1 for t in trend_sigs if t == "up")
            down_c = sum(1 for t in trend_sigs if t == "down")
            if up_c >= 3: rs_adj += 4
            elif down_c >= 3: rs_adj -= 4

        return max(10.0, min(90.0, res_score + rs_adj))

    # ---------- 10. KLINE_VISION fallback (无视觉时的文本降级评分) ----------
    def _score_kline_vision_fallback(self, mom, vol, vr, pct, kli) -> float:
        day = kli.get("day", {}) if isinstance(kli, dict) else {}
        rsi = float(day.get("rsi") or 50) if isinstance(day, dict) else 50
        slope = float(day.get("trend_slope_pct") or 0) if isinstance(day, dict) else 0
        rsi_score = 0.0
        if rsi > 75: rsi_score = -8
        elif rsi < 25: rsi_score = 8
        slope_score = max(-10.0, min(10.0, slope * 60))
        return 50 + slope_score + rsi_score + 0.3 * mom + (vr - 1.0) * 8 - 0.15 * vol

    # ------------------------------------------------------------------
    # 提交结果
    # ------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Agent间消息写入
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# K线视觉Agent (保留，合并为单实例)
# ---------------------------------------------------------------------------

_KLINE_VISION_DIMS = {
    "1h":    "KLINE_1H",
    "day":   "KLINE_DAY",
    "week":  "KLINE_WEEK",
    "month": "KLINE_MONTH",
}


def _parse_vision_response(text: str) -> tuple[str, float, str]:
    """从视觉模型回复中解析 建议/评分/核心判断。"""
    vote = "hold"
    score = 50.0
    core = text.strip()[:200] if text else ""
    vote_match = re.search(r"建议[：:]\s*(buy|hold|sell|买入|持有|观望|卖出|减仓)", text or "", re.I)
    if vote_match:
        v = vote_match.group(1).lower()
        if v in ("buy", "买入"):
            vote = "buy"
        elif v in ("sell", "卖出", "减仓"):
            vote = "sell"
        else:
            vote = "hold"
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
    core_match = re.search(r"核心判断[：:]\s*(.+?)(?:\n|$)", text or "", re.S)
    if core_match:
        core = core_match.group(1).strip()[:300]
    elif text:
        core = text.strip()[:300]
    return vote, score, core


class KlineVisionAgent(AnalystAgent):
    """K线视觉智能体：日线图像输入给支持视觉的LLM进行研判。
    v2: 合并为单实例, 默认使用日线图。
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        run_dir: Path,
        backend: DataBackend,
        timeframe: str = "day",
        llm_routers: dict[str, LLMRouter] | None = None,
    ):
        super().__init__(cfg, run_dir, backend, llm_routers=llm_routers)
        self.timeframe = timeframe
        # v2: 统一dim_code为KLINE_VISION, 不再按周期分
        self.dim_code = cfg.get("dim_code", "KLINE_VISION")

    def analyze_local(
        self,
        symbol: str,
        name: str,
        analysis_context: dict[str, Any] | None = None,
    ) -> AgentBaseResult:
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
        vision_routers = {p: r for p, r in self.llm_routers.items() if r.supports_vision()}

        if not vision_routers and chart_path:
            fallback_p = os.getenv("VISION_FALLBACK_PROVIDER", _DEFAULT_VISION_FALLBACK)
            fallback_key = os.getenv(f"{fallback_p.upper()}_API_KEY", "").strip()
            if fallback_key and _supports_vision(fallback_p):
                ref = next(iter(self.llm_routers.values()), None)
                fallback_router = LLMRouter(
                    provider=fallback_p,
                    temperature=ref.temperature if ref else 0.3,
                    max_tokens=ref.max_tokens if ref else 600,
                    request_timeout_sec=ref.request_timeout_sec if ref else 45.0,
                )
                vision_routers = {fallback_p: fallback_router}

        if not chart_path or not vision_routers:
            result = super().analyze(symbol, name, analysis_context=analysis_context, progress_cb=progress_cb)
            result.dim_code = self.dim_code
            return result

        image_b64 = load_image_base64(chart_path)
        if not image_b64:
            result = super().analyze(symbol, name, analysis_context=analysis_context, progress_cb=progress_cb)
            result.dim_code = self.dim_code
            return result

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
                progress_cb("K线视觉分析", f"{self.agent_id}({tf_label}) → {provider}")
            text = router.chat_with_image(prompt, image_b64)
            if text:
                llm_evals[provider] = text.strip()
                v, s, _ = _parse_vision_response(text)
                all_votes.append(v)
                all_scores.append(s)

        if not llm_evals:
            result = super().analyze(symbol, name, analysis_context=analysis_context, progress_cb=progress_cb)
            result.dim_code = self.dim_code
            return result

        avg_score = sum(all_scores) / len(all_scores)
        avg_score = max(0.0, min(100.0, avg_score))
        if avg_score >= 70:
            final_vote = "buy"
        elif avg_score < 50:
            final_vote = "sell"
        else:
            final_vote = "hold"

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
        confidence = max(0.1, min(0.98, 0.5 + 0.4 * data_quality))

        result = AgentResult(
            agent_id=self.agent_id, dim_code=self.dim_code,
            vote=final_vote, score_0_100=avg_score, confidence_0_1=confidence,
            reason=reason, llm_multi_evaluations=llm_evals,
        )
        self._submit(result, snap, ctx)
        return result


# ---------------------------------------------------------------------------
# 专业Agent子类 (各自覆盖 _build_data_context)
# ---------------------------------------------------------------------------

class DivergenceAgent(AnalystAgent):
    """MACD/RSI背离检测Agent。"""

    _SYSTEM_PROMPT_CACHE: "str | None" = None

    @classmethod
    def _load_system_prompt(cls) -> str:
        if cls._SYSTEM_PROMPT_CACHE is None:
            prompt_path = (
                Path(__file__).resolve().parent.parent.parent
                / "configs" / "prompts" / "divergence_system.txt"
            )
            cls._SYSTEM_PROMPT_CACHE = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        return cls._SYSTEM_PROMPT_CACHE

    def _simple_policy(self, snap, ctx) -> tuple[str, float, float, str]:
        """本地策略：基于多周期背离评分给出投票和置信度。"""
        kli = ctx.get("features", {}).get("kline_indicators", {})
        score = self._score_divergence(kli)

        # 统计各周期背离情况
        bull_tfs, bear_tfs = [], []
        for tf in ("day", "week", "month"):
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            dv = td.get("divergence", {}) if isinstance(td, dict) else {}
            if not isinstance(dv, dict):
                continue
            if dv.get("macd_bot_div") or dv.get("rsi_bot_div"):
                bull_tfs.append(tf)
            if dv.get("macd_top_div") or dv.get("rsi_top_div"):
                bear_tfs.append(tf)

        # 置信度：多周期共振时更高，无背离时极低（让权重自动缩小）
        if len(bull_tfs) >= 2 or len(bear_tfs) >= 2:
            confidence = 0.80
        elif bull_tfs or bear_tfs:
            confidence = 0.60
        else:
            confidence = 0.20

        tf_label = {"day": "日", "week": "周", "month": "月"}
        if bull_tfs:
            signal = "+".join(tf_label[t] for t in bull_tfs) + "线底背离"
        elif bear_tfs:
            signal = "+".join(tf_label[t] for t in bear_tfs) + "线顶背离"
        else:
            signal = "无背离"

        reason = f"背离信号: {signal} | 本地评分={score:.0f}"
        if score >= 70:
            return "buy", score, confidence, reason
        if score <= 30:
            return "sell", score, confidence, reason
        return "hold", score, confidence, reason

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []

        sys_prompt = self._load_system_prompt()
        if sys_prompt:
            parts.append(sys_prompt)
            parts.append("")
            parts.append("======== 六、当前股票实时背离数据 ========")
            parts.append("")

        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")

        kli = ctx.get("features", {}).get("kline_indicators", {})
        has_any_signal = False
        for tf, label in (("day", "日线"), ("week", "周线"), ("month", "月线")):
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            dv = td.get("divergence", {}) if isinstance(td, dict) else {}
            rsi = td.get("rsi")
            macd = td.get("macd", {}) if isinstance(td, dict) else {}
            dif = macd.get("dif") if isinstance(macd, dict) else td.get("macd_dif")
            mom = td.get("momentum_10") or 0

            rsi_str = f"{float(rsi):.1f}" if rsi is not None else "N/A"
            dif_str = f"{float(dif):.4f}" if dif is not None else "N/A"
            lines = [f"[{label}] RSI={rsi_str} | MACD DIF={dif_str} | 动量={float(mom):.1f}%"]

            if isinstance(dv, dict):
                ds = dv.get("divergence_score", 0)
                has_bull = dv.get("macd_bot_div") or dv.get("rsi_bot_div")
                has_bear = dv.get("macd_top_div") or dv.get("rsi_top_div")
                if has_bull or has_bear:
                    has_any_signal = True
                if dv.get("macd_top_div"):
                    lines.append(f"  !! {dv.get('macd_div_desc', 'MACD顶背离')} (看空)")
                if dv.get("macd_bot_div"):
                    lines.append(f"  !! {dv.get('macd_div_desc', 'MACD底背离')} (看多)")
                if dv.get("rsi_top_div"):
                    lines.append(f"  !! {dv.get('rsi_div_desc', 'RSI顶背离')} (看空)")
                if dv.get("rsi_bot_div"):
                    lines.append(f"  !! {dv.get('rsi_div_desc', 'RSI底背离')} (看多)")
                if not (has_bull or has_bear):
                    lines.append("  无背离信号")
                lines.append(f"  背离综合分={ds} ({'底背离强度' if ds > 0 else '顶背离强度' if ds < 0 else '无'})")
            parts.append("\n".join(lines))

        local_score = self._score_divergence(kli)
        parts.append(f"\n[本地综合评分] {local_score:.0f}/100 ({'有背离信号' if has_any_signal else '无背离，空头基调'})")

        return "\n".join(parts)


class ChanlunAgent(AnalystAgent):
    """缠论买卖点Agent。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")
        kli = ctx.get("features", {}).get("kline_indicators", {})
        for tf, label in (("day", "日线"), ("week", "周线"), ("month", "月线")):
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            cl = td.get("chanlun", {})
            lines = [f"[{label}缠论]"]
            if cl:
                zs = cl.get("zhongshu", [])
                if zs:
                    last_zs = zs[-1]
                    lines.append(f"  中枢: [{last_zs.get('low','?')}-{last_zs.get('high','?')}] ({last_zs.get('bi_count',0)}笔)")
                bis = cl.get("bi_list", [])
                if bis:
                    for b in bis[-3:]:
                        lines.append(f"  笔: {b.get('dir','')} {b.get('start_price','?')}→{b.get('end_price','?')}")
                for s in cl.get("buy_signals", []):
                    lines.append(f"  ↑ 【{s.get('type','')}】{s.get('desc','')}")
                for s in cl.get("sell_signals", []):
                    lines.append(f"  ↓ 【{s.get('type','')}】{s.get('desc','')}")
                lines.append(f"  缠论评分: {cl.get('chanlun_score', 0)}")
            else:
                lines.append("  数据不足")
            parts.append("\n".join(lines))
        parts.append(
            "\n分析要求：\n1.当前缠论阶段（上涨/下跌/中枢震荡）\n"
            "2.是否有买卖点信号\n3.中枢位置的指引意义\n4.操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class PatternAgent(AnalystAgent):
    """形态综合Agent: 合并K线形态+图表形态+顶底结构。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")
        kli = ctx.get("features", {}).get("kline_indicators", {})
        for tf, label in (("day", "日线"), ("week", "周线"), ("month", "月线")):
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            lines = [f"[{label}形态分析]"]
            # K线形态
            pats = td.get("kline_patterns", [])
            if pats:
                for p in pats:
                    icon = "↑" if p.get("direction") == "bullish" else ("↓" if p.get("direction") == "bearish" else "→")
                    pos = p.get("position_pct")
                    pos_str = f" ({pos:.0f}%位)" if pos is not None else ""
                    lines.append(f"  {icon}【{p['name']}】置信度{p.get('confidence', 50)}%{pos_str}")
            # 图表形态
            cp = td.get("chart_patterns", {})
            if cp and cp.get("patterns"):
                for p in cp["patterns"]:
                    icon = "↑" if p.get("direction") == "bullish" else ("↓" if p.get("direction") == "bearish" else "→")
                    lines.append(f"  {icon}〖{p['name']}〗置信度{p.get('confidence', 50)}%")
            # 顶底信号
            upper = float(td.get("upper_shadow_ratio") or 0)
            lower = float(td.get("lower_shadow_ratio") or 0)
            mom_tf = float(td.get("momentum_10") or 0)
            if upper > 35 and mom_tf < 0:
                lines.append(f"  ⚠ 顶部信号: 上影线{upper:.0f}%+负动量")
            if lower > 35 and mom_tf > 0:
                lines.append(f"  ★ 底部信号: 下影线{lower:.0f}%+正动量")
            # 连续性
            cs = td.get("continuity_stats", {})
            if cs:
                cs_parts = []
                if cs.get("consecutive_bull", 0) > 0:
                    cs_parts.append(f"连阳{cs['consecutive_bull']}日")
                elif cs.get("consecutive_bear", 0) > 0:
                    cs_parts.append(f"连阴{cs['consecutive_bear']}日")
                if cs_parts:
                    lines.append("  连续性: " + " | ".join(cs_parts))
            if len(lines) == 1:
                lines.append("  未检测到明显形态")
            parts.append("\n".join(lines))
        parts.append(
            "\n分析要求：\n1.各周期形态是否共振\n2.最关键形态及其预测含义\n"
            "3.是否有顶底反转信号\n4.操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class VolumeStructureAgent(AnalystAgent):
    """量价结构Agent: 合并量价关系+支撑阻力。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")
        kli = ctx.get("features", {}).get("kline_indicators", {})
        # 量价
        for tf, label in (("day", "日线"), ("week", "周线"), ("month", "月线")):
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            vp = td.get("volume_price", {})
            lines = [f"[{label}量价]"]
            if vp:
                flags = []
                if vp.get("volume_breakout"): flags.append("放量突破")
                if vp.get("shrink_pullback"): flags.append("缩量回踩")
                if vp.get("volume_anomaly"): flags.append("底部放量")
                if vp.get("climax_volume"): flags.append("天量滞涨")
                lines.append(f"  信号: {', '.join(flags) if flags else '无'}")
                lines.append(f"  OBV: {vp.get('obv_trend', 'N/A')} | 量价评分: {vp.get('volume_price_score', 0)}")
            parts.append("\n".join(lines))
        # 支撑阻力 (日线)
        td = kli.get("day", {}) if isinstance(kli, dict) else {}
        sr = td.get("support_resistance", {}) if isinstance(td, dict) else {}
        if sr:
            parts.append(f"价格位置: {sr.get('price_position', 'N/A')}")
            sups = sr.get("support_levels", [])
            if sups:
                lines = ["支撑位:"]
                for s in sups[:3]:
                    lines.append(f"  {s.get('price', 'N/A')} ({s.get('label', '')})")
                parts.append("\n".join(lines))
            ress = sr.get("resistance_levels", [])
            if ress:
                lines = ["阻力位:"]
                for r in ress[:3]:
                    lines.append(f"  {r.get('price', 'N/A')} ({r.get('label', '')})")
                parts.append("\n".join(lines))
        parts.append(
            "\n分析要求：\n1.量价配合是否健康\n2.当前距支撑/阻力位距离\n"
            "3.操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class CapitalLiquidityAgent(AnalystAgent):
    """资金流动性Agent: 合并资金流向+流动性+量价信号。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")
        f = ctx.get("features", {})
        vr = f.get("volume_ratio_5_20")
        turn = f.get("turnover_rate")
        parts.append(f"量比5/20: {vr:.2f}" if vr else "量比: N/A")
        if turn is not None:
            parts.append(f"换手率: {turn}%")
        kli = f.get("kline_indicators", {})
        day = kli.get("day", {}) if isinstance(kli, dict) else {}
        vp = day.get("volume_price", {}) if isinstance(day, dict) else {}
        if vp:
            flags = []
            if vp.get("volume_breakout"): flags.append("放量突破")
            if vp.get("shrink_pullback"): flags.append("缩量回踩")
            if vp.get("climax_volume"): flags.append("天量滞涨")
            parts.append(f"量价信号: {', '.join(flags) if flags else '无'}")
            parts.append(f"OBV趋势: {vp.get('obv_trend', 'N/A')}")
        chip = f.get("chip_distribution", {})
        if chip:
            parts.append(f"筹码: 获利{chip.get('profit_ratio', 'N/A')}% | 集中度{chip.get('concentration', 'N/A')}%")
        parts.append(
            "\n分析要求：\n1.资金流向判断（净流入/出）\n2.量价配合分析\n"
            "3.流动性风险评估\n4.操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class SentimentFlowAgent(AnalystAgent):
    """情绪舆情Agent: 合并新闻情绪+行业政策+主力行为。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")
        f = ctx.get("features", {})
        news_sent = f.get("news_sentiment")
        news_cnt = f.get("news_count")
        if news_sent is not None:
            parts.append(f"新闻情绪: {news_sent:.1f} ({news_cnt or 0}条)")
        vr = f.get("volume_ratio_5_20")
        if vr:
            parts.append(f"量比(主力活跃度): {vr:.2f}")
        margin = f.get("margin_data", {})
        if margin:
            parts.append(f"融资余额5日变化: {margin.get('rzye_change_5d', 'N/A')}%")
        hsgt = f.get("hsgt_data", {})
        if hsgt:
            parts.append(f"北向持仓变化: {hsgt.get('hk_hold_change_pct', 'N/A')}%")
        strategy = f.get("market_strategy", {})
        if strategy:
            parts.append(f"市场阶段: {strategy.get('phase_cn', '未知')}")
        # v2: 优先使用LLM新闻分析结果
        na = ctx.get("news_analysis", {})
        if na:
            if na.get("summary"):
                parts.append(f"新闻情绪总结: {na['summary']}")
            if na.get("sentiment_score") is not None:
                parts.append(f"LLM情绪分: {na['sentiment_score']} (-100~100)")
            events = na.get("key_events", [])
            if events:
                parts.append("关键事件: " + " | ".join(str(e)[:50] for e in events[:3]))
            risks = na.get("risk_alerts", [])
            if risks:
                parts.append("风险预警: " + " | ".join(str(r)[:50] for r in risks))
            catalysts = na.get("positive_catalysts", [])
            if catalysts:
                parts.append("利好催化: " + " | ".join(str(c)[:50] for c in catalysts))
        else:
            news = ctx.get("news", [])[:5]
            if news:
                titles = [str(n.get("title", ""))[:50] for n in news if n.get("title")]
                if titles:
                    parts.append("近期新闻: " + " | ".join(titles))
        parts.append(
            "\n分析要求：\n1.市场情绪偏多还是偏空\n2.是否有重大利好/利空消息\n"
            "3.主力资金态度（融资+北向）\n4.操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class ResonanceAgent(AnalystAgent):
    """多周期共振Agent: 合并多周期共振+相对强弱。"""

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        parts: list[str] = []
        snap = ctx.get("snapshot", {})
        if snap:
            parts.append(f"行情: 收盘={snap.get('close')}, 涨跌幅={snap.get('pct_chg')}%")
        f = ctx.get("features", {})
        kli = f.get("kline_indicators", {})
        for tf, label in (("day", "日线(短期)"), ("week", "周线(中期)"), ("month", "月线(长期)")):
            td = kli.get(tf, {}) if isinstance(kli, dict) else {}
            if not isinstance(td, dict) or not td.get("ok"):
                continue
            parts.append(
                f"[{label}] 斜率={td.get('trend_slope_pct', 0):.4f}%/bar | "
                f"动量={td.get('momentum_10') or 0:.1f}% | RSI={td.get('rsi', 'N/A')}"
            )
        # 相对强弱
        rs = f.get("relative_strength", {})
        if isinstance(rs, dict) and rs.get("ok"):
            parts.append(
                f"[vs沪深300] 超额收益20日={rs.get('excess_return_20d')}% | "
                f"RS趋势={rs.get('rs_trend', 'unknown')}"
            )
        rs_ind = f.get("rs_vs_industry", {})
        if isinstance(rs_ind, dict) and rs_ind.get("ok"):
            parts.append(f"[vs行业] 超额收益20日={rs_ind.get('excess_return_20d')}%")
        parts.append(
            "\n分析要求：\n1.日/周/月趋势是否一致共振\n2.个股相对大盘/行业强弱\n"
            "3.共振信号强度和可靠性\n4.操作建议\n\n"
            "请严格按格式输出：\n建议：buy/hold/sell\n评分：[0-100]\n核心判断：[2-3句分析]"
        )
        return "\n".join(parts)


class ChannelReversalAgent(AnalystAgent):
    """动态通道反弹/反转Agent — 基于平滑Donchian通道的阶段状态机评分。"""

    def _load_day_kline(self) -> "pd.DataFrame | None":
        """从 run_dir 加载日线CSV。"""
        import pandas as pd
        csv_path = self.run_dir / "data" / "kline" / "day.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        col_map = {
            "收盘": "close", "开盘": "open", "最高": "high",
            "最低": "low", "成交量": "volume", "日期": "date",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _simple_policy(self, snap, ctx):
        """本地策略：运行通道反转状态机，返回最新评分。"""
        try:
            from .channel_reversal import compute_channel, detect_phases
            df = self._load_day_kline()
            if df is None or len(df) < 160:
                return "hold", 50.0, 0.3, "日线数据不足，无法计算通道指标"
            df = compute_channel(df)
            df = detect_phases(df)
            last = df.iloc[-1]
            phase = last.get("phase", "0D")
            score = float(last.get("cr_score", 50))
            days = int(last.get("phase_days", 0))
            rsi = last.get("rsi")
            vol_r = last.get("vol_ratio")
            import math
            rsi_str = f"{rsi:.1f}" if rsi is not None and not math.isnan(rsi) else "N/A"
            vol_str = f"{vol_r:.2f}" if vol_r is not None and not math.isnan(vol_r) else "N/A"

            phase_labels = {
                "0U": "上半区(多头)", "0D": "下半区(空头)",
                "1": "破下轨(超卖)", "2": "回到轨道(反弹启动)",
                "3A": "弱反弹(大概率破新低)", "3B": "过中轨(看多确认)",
                "4A": "有效反转(突破上轨)", "4B": "上轨整理(蓄力)",
            }
            label = phase_labels.get(phase, phase)

            if score >= 70:
                vote = "buy"
            elif score <= 30:
                vote = "sell"
            else:
                vote = "hold"

            confidence = 0.6
            if phase in ("4A", "3A"):
                confidence = 0.8
            elif phase in ("3B", "1"):
                confidence = 0.7

            reason = (
                f"通道阶段: {label}(持续{days}日) | "
                f"RSI={rsi_str} | 量比={vol_str} | 通道评分={score:.0f}"
            )
            return vote, score, confidence, reason
        except Exception as e:
            self.logger.warning("channel_reversal local policy failed: %s", e)
            return "hold", 50.0, 0.3, f"通道计算异常: {e}"

    def _build_data_context(self, ctx: dict[str, Any]) -> str:
        """为LLM构建通道反转完整分析上下文（系统提示词+实时数据）。"""
        from ._cr_data_context import build_data_context as _bdc
        return _bdc(self, ctx)


# ---------------------------------------------------------------------------
# Agent 工厂
# ---------------------------------------------------------------------------

def create_agent(
    cfg: dict[str, Any],
    run_dir: Path,
    backend: DataBackend,
    llm_routers: dict[str, LLMRouter] | None = None,
) -> AnalystAgent:
    """v2 Agent 工厂：12个Agent类型。"""
    agent_type = cfg.get("agent_type", "analyst")
    if agent_type == "kline_vision":
        timeframe = cfg.get("timeframe", "day")
        return KlineVisionAgent(cfg, run_dir, backend, timeframe=timeframe, llm_routers=llm_routers)
    if agent_type == "divergence":
        return DivergenceAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "chanlun":
        return ChanlunAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "pattern":
        return PatternAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "volume_structure":
        return VolumeStructureAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "capital_liquidity":
        return CapitalLiquidityAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "sentiment_flow":
        return SentimentFlowAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "resonance":
        return ResonanceAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    if agent_type == "channel_reversal":
        return ChannelReversalAgent(cfg, run_dir, backend, llm_routers=llm_routers)
    return AnalystAgent(cfg, run_dir, backend, llm_routers=llm_routers)
