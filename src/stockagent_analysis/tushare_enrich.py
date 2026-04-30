"""Tushare 高级数据增强模块 - 用真实数据替代本地估算。

集成接口:
  1. stk_factor_pro  - 60+ 预计算技术指标(多复权), 替代本地 MACD/RSI/KDJ/BOLL
  2. cyq_perf        - 真实筹码绩效(winner_rate 获利盘 + 成本分位 + 加权均价)
  3. cyq_chips       - 筹码分布(价位+占比数组, 计算集中度)
  4. moneyflow       - 4 档主力资金(小/中/大/超大单)
  5. stk_holdernumber- 股东户数(判断筹码集中趋势)

入口:
  enrich_with_tushare(symbol, run_dir) -> dict
    返回 {'tushare_factors':..., 'tushare_cyq':..., 'tushare_moneyflow':..., 'tushare_holders':...}
    同时缓存到 run_dir/data/tushare_*.json
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Tushare code 归一化
# ─────────────────────────────────────────────────────────────────

def _normalize_ts_code(symbol: str) -> str:
    """6 位代码 → ts_code(如 000876 → 000876.SZ, 600519 → 600519.SH, 688xxx → .SH, 920xxx → .BJ)"""
    s = (symbol or "").strip().upper()
    if "." in s:
        return s
    if len(s) != 6:
        return f"{s}.SH"
    prefix = s[0]
    if prefix in ("6", "5", "9"):   # 6xx 主板, 5xx 基金, 9xx 国债
        if s.startswith("688"):
            return f"{s}.SH"   # 科创板
        return f"{s}.SH"
    if prefix == "0" or prefix == "3":
        if s.startswith("300") or s.startswith("301") or s.startswith("302"):
            return f"{s}.SZ"   # 创业板
        return f"{s}.SZ"         # 深主板
    if prefix == "4" or prefix == "8" or s.startswith("920"):
        return f"{s}.BJ"         # 北交所
    return f"{s}.SH"


# ─────────────────────────────────────────────────────────────────
# Tushare client 单例
# ─────────────────────────────────────────────────────────────────

_PRO = None


def _get_pro():
    global _PRO
    if _PRO is not None:
        return _PRO
    try:
        import tushare as ts
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            logger.warning("[tushare_enrich] TUSHARE_TOKEN 缺失")
            return None
        _PRO = ts.pro_api(token=token, timeout=30)
        return _PRO
    except ImportError:
        logger.warning("[tushare_enrich] tushare 未安装")
        return None
    except Exception as e:
        logger.warning("[tushare_enrich] pro_api 初始化失败: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────
# 抓取函数 (每个独立 try-except, 失败返 None 不中断)
# ─────────────────────────────────────────────────────────────────

def fetch_industry(ts_code: str) -> str | None:
    """拿股票行业 (stock_basic), 用于 sparse_layered 上下文."""
    pro = _get_pro()
    if not pro:
        return None
    try:
        df = pro.stock_basic(ts_code=ts_code, fields="ts_code,industry")
        if df is None or df.empty:
            return None
        ind = df.iloc[0].get("industry")
        return str(ind).strip() if ind else None
    except Exception as e:
        logger.warning("[tushare_enrich] stock_basic industry 失败 %s: %s", ts_code, e)
        return None


def fetch_stk_factor_pro(ts_code: str, days: int = 60) -> list[dict] | None:
    """拿近 N 天的扩展技术因子。

    Returns:
        list[dict] (按日期升序, 最后一个是最新), 或 None(失败)
    """
    pro = _get_pro()
    if not pro:
        return None
    try:
        df = pro.stk_factor_pro(ts_code=ts_code)
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date").tail(days)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning("[tushare_enrich] stk_factor_pro 失败 %s: %s", ts_code, e)
        return None


def fetch_cyq_perf(ts_code: str, days: int = 10) -> list[dict] | None:
    """筹码绩效(含 winner_rate 获利盘, 成本分位)。"""
    pro = _get_pro()
    if not pro:
        return None
    try:
        df = pro.cyq_perf(ts_code=ts_code, limit=days)
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning("[tushare_enrich] cyq_perf 失败 %s: %s", ts_code, e)
        return None


def fetch_cyq_chips(ts_code: str) -> dict | None:
    """筹码分布(最新一天每个价位的占比), 计算集中度汇总值。"""
    pro = _get_pro()
    if not pro:
        return None
    try:
        # 取最近 1-2 天的 chips(一天内所有价位, 可能几百行)
        df = pro.cyq_chips(ts_code=ts_code, limit=500)
        if df is None or df.empty:
            return None
        # 取最新一天
        latest_date = df["trade_date"].max()
        df_latest = df[df["trade_date"] == latest_date].copy()
        if df_latest.empty:
            return None

        # 计算集中度: 占比 top 20% 价位的累计占比
        df_latest = df_latest.sort_values("percent", ascending=False)
        top20 = df_latest.head(max(1, len(df_latest) // 5))
        top20_pct = float(top20["percent"].sum())

        # 价格分布统计
        prices = df_latest["price"].values
        pcts = df_latest["percent"].values
        weighted_price = float((prices * pcts).sum() / pcts.sum()) if pcts.sum() > 0 else None

        return {
            "trade_date": str(latest_date),
            "price_count": int(len(df_latest)),
            "top20_concentration": round(top20_pct, 4),
            "weighted_avg_price": round(weighted_price, 2) if weighted_price else None,
            "min_price": float(df_latest["price"].min()),
            "max_price": float(df_latest["price"].max()),
        }
    except Exception as e:
        logger.warning("[tushare_enrich] cyq_chips 失败 %s: %s", ts_code, e)
        return None


def fetch_moneyflow(ts_code: str, days: int = 10) -> list[dict] | None:
    """资金流 4 档分层。优先 moneyflow_dc (含 rate 字段), fallback 旧接口。"""
    pro = _get_pro()
    if not pro:
        return None
    # moneyflow_dc: buy_xxx_amount 已是净值(买-卖), 含 rate 字段
    try:
        df = pro.moneyflow_dc(ts_code=ts_code, limit=days)
        if df is not None and not df.empty:
            df = df.sort_values("trade_date")
            rows = df.to_dict(orient="records")
            for r in rows:
                r["_source"] = "dc"
            return rows
    except Exception as e:
        logger.debug("[tushare_enrich] moneyflow_dc 不可用 %s, 回退旧接口: %s", ts_code, e)
    # fallback: 旧接口(buy/sell 分开)
    try:
        df = pro.moneyflow(ts_code=ts_code, limit=days)
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date")
        rows = df.to_dict(orient="records")
        for r in rows:
            r["_source"] = "legacy"
        return rows
    except Exception as e:
        logger.warning("[tushare_enrich] moneyflow 失败 %s: %s", ts_code, e)
        return None


def fetch_holdernumber(ts_code: str, periods: int = 4) -> list[dict] | None:
    """股东户数(通常季度数据)。"""
    pro = _get_pro()
    if not pro:
        return None
    try:
        df = pro.stk_holdernumber(ts_code=ts_code, limit=periods)
        if df is None or df.empty:
            return None
        df = df.sort_values("end_date")
        return df.to_dict(orient="records")
    except Exception as e:
        logger.warning("[tushare_enrich] stk_holdernumber 失败 %s: %s", ts_code, e)
        return None


# ─────────────────────────────────────────────────────────────────
# 从 stk_factor_pro 提取关键技术指标(qfq 版本)
# ─────────────────────────────────────────────────────────────────

def summarize_factors(factor_rows: list[dict]) -> dict:
    """从 stk_factor_pro 提取最新日的关键技术指标汇总(qfq 前复权)。"""
    if not factor_rows:
        return {}
    latest = factor_rows[-1]

    def g(key: str, default=None):
        v = latest.get(key)
        try:
            return round(float(v), 4) if v is not None else default
        except (ValueError, TypeError):
            return default

    return {
        "trade_date": str(latest.get("trade_date", "")),
        # 收盘(qfq) + 基础
        "close_qfq": g("close_qfq"),
        "pre_close": g("pre_close"),
        "pct_chg": g("pct_chg"),
        "turnover_rate": g("turnover_rate"),
        "turnover_rate_f": g("turnover_rate_f"),
        "volume_ratio": g("volume_ratio"),
        "pe_ttm": g("pe_ttm"),
        "pb": g("pb"),
        "total_mv": g("total_mv"),
        "circ_mv": g("circ_mv"),
        # 均线 (qfq)
        "ma5": g("ma_qfq_5"), "ma10": g("ma_qfq_10"), "ma20": g("ma_qfq_20"),
        "ma30": g("ma_qfq_30"), "ma60": g("ma_qfq_60"), "ma90": g("ma_qfq_90"),
        "ma250": g("ma_qfq_250"),
        # EMA
        "ema5": g("ema_qfq_5"), "ema20": g("ema_qfq_20"), "ema60": g("ema_qfq_60"),
        # 布林
        "boll_upper": g("boll_upper_qfq"),
        "boll_mid": g("boll_mid_qfq"),
        "boll_lower": g("boll_lower_qfq"),
        # MACD
        "macd_dif": g("macd_dif_qfq"),
        "macd_dea": g("macd_dea_qfq"),
        "macd_hist": g("macd_qfq"),
        # KDJ
        "kdj_k": g("kdj_k_qfq"), "kdj_d": g("kdj_d_qfq"), "kdj_j": g("kdj_qfq"),
        # RSI
        "rsi6": g("rsi_qfq_6"), "rsi12": g("rsi_qfq_12"), "rsi24": g("rsi_qfq_24"),
        # DMI 趋势强度
        "dmi_adx": g("dmi_adx_qfq"),
        "dmi_adxr": g("dmi_adxr_qfq"),
        "dmi_pdi": g("dmi_pdi_qfq"),
        "dmi_mdi": g("dmi_mdi_qfq"),
        # 其他
        "atr": g("atr_qfq"),
        "bias1": g("bias1_qfq"), "bias2": g("bias2_qfq"), "bias3": g("bias3_qfq"),
        "cci": g("cci_qfq"),
        "mfi": g("mfi_qfq"),
        "mtm": g("mtm_qfq"),
        "obv": g("obv_qfq"),
        "roc": g("roc_qfq"),
        "wr": g("wr_qfq"),
        "trix": g("trix_qfq"),
        "updays": g("updays"), "downdays": g("downdays"),
        "topdays": g("topdays"), "lowdays": g("lowdays"),
    }


def summarize_moneyflow(mf_rows: list[dict]) -> dict:
    """汇总资金流分层，含 3日MA平滑 + 主力/散户背离信号。

    DC版: buy_xxx_amount 已是净值, 含 rate 字段
    旧版: 需要 buy - sell 计算净值
    """
    if not mf_rows:
        return {}

    source = mf_rows[0].get("_source", "legacy")
    n = len(mf_rows)
    latest = mf_rows[-1]

    def _net_cat(r: dict, cat: str) -> float:
        if source == "dc":
            return float(r.get(f"buy_{cat}_amount", 0) or 0)
        return float(r.get(f"buy_{cat}_amount", 0) or 0) - float(r.get(f"sell_{cat}_amount", 0) or 0)

    def _rate_cat(r: dict, cat: str) -> float | None:
        if source == "dc":
            v = r.get(f"buy_{cat}_amount_rate")
            return float(v) if v is not None else None
        return None

    # 逐日序列
    daily_main, daily_retail = [], []
    daily_main_rate, daily_retail_rate = [], []
    for r in mf_rows:
        elg, lg, sm = _net_cat(r, "elg"), _net_cat(r, "lg"), _net_cat(r, "sm")
        daily_main.append(elg + lg)
        daily_retail.append(sm)
        r_elg, r_lg = _rate_cat(r, "elg"), _rate_cat(r, "lg")
        r_sm = _rate_cat(r, "sm")
        daily_main_rate.append((r_elg + r_lg) if r_elg is not None and r_lg is not None else None)
        daily_retail_rate.append(r_sm)

    def _ma(series: list, window: int = 3) -> float | None:
        vals = [x for x in series[-window:] if x is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # 连续主力净流入天数(从最新日往前)
    consecutive_main_days = 0
    for v in reversed(daily_main):
        if v > 0:
            consecutive_main_days += 1
        else:
            break

    # 近3日方向一致性 → 背离信号
    main_pos = sum(1 for v in daily_main[-3:] if v > 0)
    retail_pos = sum(1 for v in daily_retail[-3:] if v > 0)
    main_trend = "inflow" if main_pos >= 2 else "outflow"
    retail_trend = "inflow" if retail_pos >= 2 else "outflow"

    if main_trend == "inflow" and retail_trend == "outflow":
        divergence = "smart_accumulating"   # 主力吸筹, 散户出货
    elif main_trend == "outflow" and retail_trend == "inflow":
        divergence = "distribution"         # 主力派发, 散户接盘 → 高危
    elif main_trend == "inflow":
        divergence = "consensus_buy"
    elif main_trend == "outflow":
        divergence = "consensus_sell"
    else:
        divergence = "neutral"

    # N日累计
    total_net_key = "net_amount" if source == "dc" else "net_mf_amount"
    total_net = sum(float(r.get(total_net_key, 0) or 0) for r in mf_rows)
    total_main = sum(daily_main)

    return {
        "days": n,
        "source": source,
        "trade_date_latest": str(latest.get("trade_date", "")),
        # 最新一日
        "latest_main_net": round(daily_main[-1], 2),
        "latest_retail_net": round(daily_retail[-1], 2),
        "latest_super_large_net": round(_net_cat(latest, "elg"), 2),
        "latest_large_net": round(_net_cat(latest, "lg"), 2),
        "latest_medium_net": round(_net_cat(latest, "md"), 2),
        # 3日MA平滑(主力净 / 散户净 / rate)
        "main_net_ma3": _ma(daily_main, 3),
        "retail_net_ma3": _ma(daily_retail, 3),
        "main_rate_ma3": _ma(daily_main_rate, 3),    # 主力净流入率%, None if legacy
        "retail_rate_ma3": _ma(daily_retail_rate, 3),
        # N日累计
        f"sum_{n}d_net_total": round(total_net, 2),
        f"sum_{n}d_main_net": round(total_main, 2),
        # 信号
        "divergence": divergence,
        "consecutive_main_days": consecutive_main_days,
        "main_trend": main_trend,
        "retail_trend": retail_trend,
    }


def summarize_cyq(cyq_rows: list[dict]) -> dict:
    """汇总筹码绩效指标。"""
    if not cyq_rows:
        return {}
    latest = cyq_rows[-1]

    def g(key: str):
        v = latest.get(key)
        try:
            return round(float(v), 4) if v is not None else None
        except (ValueError, TypeError):
            return None

    d = {
        "trade_date": str(latest.get("trade_date", "")),
        "winner_rate": g("winner_rate"),          # 获利盘 %
        "weight_avg_cost": g("weight_avg"),       # 加权平均成本
        "cost_5pct": g("cost_5pct"),              # 最低 5% 成本
        "cost_15pct": g("cost_15pct"),
        "cost_50pct": g("cost_50pct"),            # 中位成本
        "cost_85pct": g("cost_85pct"),
        "cost_95pct": g("cost_95pct"),            # 最高 5% 成本
        "his_high": g("his_high"),                # 历史最高
        "his_low": g("his_low"),                  # 历史最低
    }
    # 计算筹码分散度 = (cost_95 - cost_5) / cost_50 (越大越分散)
    if d.get("cost_95pct") and d.get("cost_5pct") and d.get("cost_50pct"):
        d["dispersion"] = round((d["cost_95pct"] - d["cost_5pct"]) / d["cost_50pct"], 4)
    return d


# ─────────────────────────────────────────────────────────────────
# Quant Score - 4 维 deterministic 量化打分 (ADX/winner/mainflow/holders)
# ─────────────────────────────────────────────────────────────────

def compute_quant_score(enrich: dict[str, Any]) -> dict[str, Any]:
    """基于 Tushare 增强数据打量化分(50 中性 ±30)。

    因子设计:
      1. ADX 趋势强度           [-8, +10]
      2. winner_rate 筹码获利盘  [-10, +12]
      3. 主力资金 (3日MA rate优先, 绝对量fallback)
         3a. 背离信号            [-12, +8]   主力派发/吸筹
         3b. 强度/rate          [-10, +10]
         3c. 连续性             [0, +5]
      4. 股东户数变化%           [-6, +8]
      5. 市值×PE 分层因子(2026-04-29 加, 来自 102 万样本回测):
         5a. ma_ratio_60 反转/动量(按市值段反向)  [-6, +6]
         5b. mfi_14 动量(仅大盘 / 中盘正向)       [-3, +5]
         5c. sump_20 累计涨幅(按市值段反向)       [-5, +5]
         5d. ht_trendmode(全段反向)               [-3, +3]

    有数据才打分; 缺数据不参与不惩罚。
    总偏移累加到 50 基准上, clamp [0, 100]。
    """
    adjustments: list[dict] = []

    def _add(factor: str, delta: float, reason: str):
        adjustments.append({"factor": factor, "delta": round(delta, 2), "reason": reason})

    # 1. ADX 趋势强度
    tsf = enrich.get("tushare_factors") or {}
    adx, pdi, mdi = tsf.get("dmi_adx"), tsf.get("dmi_pdi"), tsf.get("dmi_mdi")
    if adx is not None and pdi is not None and mdi is not None:
        try:
            adx_f, pdi_f, mdi_f = float(adx), float(pdi), float(mdi)
            up = pdi_f > mdi_f
            if adx_f >= 30 and up:
                _add("adx", +10.0, f"强趋势向上 ADX={adx_f:.1f} +DI>-DI")
            elif adx_f >= 25 and up:
                _add("adx", +6.0, f"趋势向上 ADX={adx_f:.1f}")
            elif adx_f >= 25 and not up:
                _add("adx", -8.0, f"强趋势向下 ADX={adx_f:.1f} -DI>+DI")
            elif adx_f < 20:
                _add("adx", -2.0, f"无趋势震荡 ADX={adx_f:.1f}")
        except (ValueError, TypeError):
            pass

    # 2. winner_rate 筹码获利盘
    cyq = enrich.get("tushare_cyq") or {}
    wr = cyq.get("winner_rate")
    if wr is not None:
        try:
            wr_f = float(wr)
            if wr_f <= 20:
                _add("winner_rate", +12.0, f"深度套牢 winner={wr_f:.1f}% (底部筹码利好)")
            elif wr_f <= 35:
                _add("winner_rate", +6.0, f"多数套牢 winner={wr_f:.1f}%")
            elif wr_f >= 85:
                _add("winner_rate", -10.0, f"获利盘过高 winner={wr_f:.1f}% (顶部风险)")
            elif wr_f >= 75:
                _add("winner_rate", -5.0, f"获利盘偏高 winner={wr_f:.1f}%")
        except (ValueError, TypeError):
            pass

    # 3. 主力资金(3日MA + 背离信号)
    mf = enrich.get("tushare_moneyflow") or {}
    divergence = mf.get("divergence", "neutral")
    consecutive_main = int(mf.get("consecutive_main_days", 0) or 0)
    main_rate_ma3 = mf.get("main_rate_ma3")   # 主力净流入率% 3日MA, DC版才有
    main_net_ma3 = mf.get("main_net_ma3")     # 主力净流入额(万) 3日MA

    # 3a. 背离信号(最高优先级, 主力行为先行指标)
    if divergence == "distribution":
        _add("mf_divergence", -12.0, "主力派发+散户接盘 (高危顶部特征)")
    elif divergence == "smart_accumulating":
        _add("mf_divergence", +8.0, "主力吸筹+散户出货 (底部建仓特征)")
    elif divergence == "consensus_buy":
        _add("mf_divergence", +3.0, "主力散户同向净流入")
    elif divergence == "consensus_sell":
        _add("mf_divergence", -5.0, "主力散户同向净流出")

    # 3b. 趋势强度 — rate优先(跨市值可比), 绝对量fallback
    if main_rate_ma3 is not None:
        try:
            r = float(main_rate_ma3)
            if r >= 3.0:
                _add("mf_strength", +10.0, f"主力强力净流入率 {r:.1f}%/日(MA3)")
            elif r >= 1.0:
                _add("mf_strength", +5.0, f"主力净流入率 {r:.1f}%/日(MA3)")
            elif r <= -3.0:
                _add("mf_strength", -10.0, f"主力强力净流出率 {r:.1f}%/日(MA3)")
            elif r <= -1.0:
                _add("mf_strength", -5.0, f"主力净流出率 {r:.1f}%/日(MA3)")
        except (ValueError, TypeError):
            pass
    elif main_net_ma3 is not None:
        try:
            mn = float(main_net_ma3)
            if mn >= 5000:
                _add("mf_strength", +10.0, f"主力大幅净流入(MA3) {mn:.0f}万")
            elif mn >= 1000:
                _add("mf_strength", +5.0, f"主力净流入(MA3) {mn:.0f}万")
            elif mn <= -5000:
                _add("mf_strength", -10.0, f"主力大幅净流出(MA3) {mn:.0f}万")
            elif mn <= -1000:
                _add("mf_strength", -5.0, f"主力净流出(MA3) {mn:.0f}万")
        except (ValueError, TypeError):
            pass
    else:
        # 兼容旧版 sum_Nd_main_net
        for k, v in mf.items():
            if k.startswith("sum_") and k.endswith("d_main_net"):
                try:
                    mn = float(v)
                    if mn >= 5000:
                        _add("mf_strength", +10.0, f"主力大幅净流入 {mn:.0f}万")
                    elif mn >= 1000:
                        _add("mf_strength", +5.0, f"主力净流入 {mn:.0f}万")
                    elif mn <= -5000:
                        _add("mf_strength", -10.0, f"主力大幅净流出 {mn:.0f}万")
                    elif mn <= -1000:
                        _add("mf_strength", -5.0, f"主力净流出 {mn:.0f}万")
                except (ValueError, TypeError):
                    pass
                break

    # 3c. 连续性加分(持续建仓更可信)
    if consecutive_main >= 5:
        _add("mf_consecutive", +5.0, f"主力连续净流入 {consecutive_main} 日")
    elif consecutive_main >= 3:
        _add("mf_consecutive", +3.0, f"主力连续净流入 {consecutive_main} 日")

    # 4. 股东户数变化%(首期→末期)
    # guard: |pct| > 500% 视为异常(通常是新股首期披露基数极小,
    #         如科创板/北交所新股上市首季披露 1000 户, 第二季 60 万户 → +59000%)
    holders = enrich.get("tushare_holders") or []
    if isinstance(holders, list) and len(holders) >= 2:
        try:
            first = float(holders[0].get("holder_num") or 0)
            last = float(holders[-1].get("holder_num") or 0)
            if first > 0:
                pct = (last - first) / first * 100
                if abs(pct) > 500:
                    # 异常数据,跳过不打分(新股基数过小导致的虚假信号)
                    pass
                elif pct <= -5:
                    _add("holders", +8.0, f"股东户数 {pct:+.1f}% (筹码集中利好)")
                elif pct <= -2:
                    _add("holders", +4.0, f"股东户数 {pct:+.1f}% (轻度集中)")
                elif pct >= 5:
                    _add("holders", -6.0, f"股东户数 {pct:+.1f}% (筹码分散利空)")
                elif pct >= 2:
                    _add("holders", -3.0, f"股东户数 {pct:+.1f}% (轻度分散)")
        except (ValueError, TypeError, AttributeError):
            pass

    # 5. 市值×因子 分层规则 (来自 2026-04-29 全因子回测, Q5-Q1 胜率差 ≥10pp)
    # mv 段: 小盘<100亿反转 / 中盘 100-1000 弱反转 / 大盘≥1000亿动量
    total_mv_wan = tsf.get("total_mv")    # 单位: 万元
    if total_mv_wan is not None:
        try:
            mv_bil = float(total_mv_wan) / 1e4   # 转亿元
            close = tsf.get("close_qfq")
            ma60 = tsf.get("ma60")
            ma120 = tsf.get("ma250") or tsf.get("ma90")  # 用 250 或 90 替代
            mfi = tsf.get("mfi")
            rsi24 = tsf.get("rsi24")
            trix = tsf.get("trix")

            # 5a. ma_ratio_60 = close/MA60 - 1 → 偏离均线
            if close and ma60 and ma60 > 0:
                ma_ratio = close / ma60 - 1
                if mv_bil < 100:   # 小盘反转: 偏离少利好, 偏离多利空
                    if ma_ratio < -0.04:
                        _add("layer_ma_ratio60", +6.0,
                             f"小盘股({mv_bil:.0f}亿)偏离60均线{ma_ratio*100:+.1f}% (跌透反弹利好,Q1胜率61.7%)")
                    elif ma_ratio > 0.10:
                        _add("layer_ma_ratio60", -6.0,
                             f"小盘股({mv_bil:.0f}亿)偏离60均线{ma_ratio*100:+.1f}% (涨多必跌,Q5胜率45.7%)")
                    elif ma_ratio > 0.04:
                        _add("layer_ma_ratio60", -3.0,
                             f"小盘股({mv_bil:.0f}亿)偏离60均线{ma_ratio*100:+.1f}% (轻度高位)")
                elif mv_bil >= 1000:   # 大盘动量: 偏离多利好
                    if ma_ratio > 0.04:
                        _add("layer_ma_ratio60", +5.0,
                             f"大盘股({mv_bil:.0f}亿)偏离60均线{ma_ratio*100:+.1f}% (强者恒强,IC=+0.090)")
                    elif ma_ratio < -0.04:
                        _add("layer_ma_ratio60", -3.0,
                             f"大盘股({mv_bil:.0f}亿)偏离60均线{ma_ratio*100:+.1f}% (弱势)")

            # 5b. mfi_14 — 大盘 / 中盘动量 (Q5-Q1 大盘+6.6pp / 中大盘+4.7pp)
            if mfi is not None:
                try:
                    mfi_f = float(mfi)
                    if mv_bil >= 300:
                        if mfi_f >= 75:
                            _add("layer_mfi", +5.0, f"中大盘({mv_bil:.0f}亿) MFI={mfi_f:.0f} (动量强,大盘Q5胜率52.2%)")
                        elif mfi_f >= 60:
                            _add("layer_mfi", +3.0, f"中大盘({mv_bil:.0f}亿) MFI={mfi_f:.0f}")
                    elif mv_bil < 100:
                        # 小盘反向: MFI 高反而是顶部
                        if mfi_f >= 80:
                            _add("layer_mfi", -3.0, f"小盘股 MFI={mfi_f:.0f} 偏高 (Q5-Q1=-8.9pp)")
                except (ValueError, TypeError):
                    pass

            # 5c. RSI24 — 大盘动量 (1000亿+ Q5-Q1=+1.8pp, 大盘 IC=+0.114)
            if rsi24 is not None and mv_bil >= 1000:
                try:
                    rsi_f = float(rsi24)
                    if rsi_f >= 65:
                        _add("layer_rsi24", +4.0, f"大盘股({mv_bil:.0f}亿) RSI24={rsi_f:.0f} (强者恒强,IC=+0.114)")
                    elif rsi_f >= 55:
                        _add("layer_rsi24", +2.0, f"大盘股 RSI24={rsi_f:.0f}")
                    elif rsi_f <= 35:
                        _add("layer_rsi24", -2.0, f"大盘股 RSI24={rsi_f:.0f} (弱势)")
                except (ValueError, TypeError):
                    pass

            # 5d. TRIX — 全市场反向 (小盘 Q5-Q1=-16.5pp, 大盘动量 Q5-Q1=-4.1pp 但 IC 反转)
            # 简化为: 小盘 trix>0 利空, 大盘不参与 (IC 区分力弱)
            if trix is not None and mv_bil < 100:
                try:
                    trix_f = float(trix)
                    if trix_f > 0.5:
                        _add("layer_trix", -3.0, f"小盘股 TRIX={trix_f:.2f} >0 (反转风险)")
                    elif trix_f < -0.5:
                        _add("layer_trix", +3.0, f"小盘股 TRIX={trix_f:.2f} <0 (反弹利好)")
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError):
            pass

    # 6. PE × ATR 分层 — PE 0-15 价值股波动正向, PE 100+ 题材股波动反向
    pe_ttm = tsf.get("pe_ttm")
    atr_pct = None
    try:
        atr_v = tsf.get("atr")
        close_v = tsf.get("close_qfq")
        if atr_v and close_v and float(close_v) > 0:
            atr_pct = float(atr_v) / float(close_v)
    except (ValueError, TypeError):
        pass

    if pe_ttm is not None and atr_pct is not None:
        try:
            pe_f = float(pe_ttm)
            if 0 < pe_f < 15:   # 低估值价值股, 波动正向 (Q5-Q1=-3.4pp 实际反向但弱, 用作弱信号)
                if atr_pct > 0.04:
                    _add("layer_pe_atr", +2.0, f"低PE({pe_f:.0f})价值股波动率{atr_pct*100:.1f}% (启动信号弱+)")
            elif pe_f >= 100 or pe_f < 0:   # 高估值/亏损股, 波动反向 (Q5-Q1=-19.7pp 强反转)
                if atr_pct > 0.05:
                    _add("layer_pe_atr", -4.0, f"高PE/亏损股波动率{atr_pct*100:.1f}% (Q5胜率仅41%,反转风险)")
        except (ValueError, TypeError):
            pass

    total_delta = sum(a["delta"] for a in adjustments)
    quant_score = max(0.0, min(100.0, 50.0 + total_delta))
    return {
        "quant_score": round(quant_score, 2),
        "total_delta": round(total_delta, 2),
        "adjustments": adjustments,
        "has_data": bool(adjustments),
    }


# ─────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────

def enrich_with_tushare(symbol: str, run_dir: Path | None = None,
                        use_cache: bool = True) -> dict[str, Any]:
    """抓取 Tushare 全套高级数据, 返回可直接并入 features 的汇总字典。

    返回结构:
        {
          "tushare_factors": {...},            # summarize_factors 输出
          "tushare_factors_raw": [...],         # 近 60 天原始 stk_factor_pro
          "tushare_cyq": {...},                 # summarize_cyq
          "tushare_cyq_chips": {...},           # 筹码分布集中度
          "tushare_moneyflow": {...},           # summarize_moneyflow
          "tushare_holders": [...],             # 股东户数时序
          "ts_code": "..."
        }
    """
    ts_code = _normalize_ts_code(symbol)
    result: dict[str, Any] = {"ts_code": ts_code}

    cache_dir = None
    if run_dir:
        cache_dir = Path(run_dir) / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)

    # 缓存读
    if use_cache and cache_dir:
        cache_file = cache_dir / "tushare_enrich.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                # 兼容旧 cache: 缺 industry 字段就补一次, 不影响主流程
                if cached and not cached.get("industry"):
                    industry = fetch_industry(ts_code)
                    if industry:
                        cached["industry"] = industry
                        try:
                            cache_file.write_text(
                                json.dumps(cached, ensure_ascii=False, indent=2),
                                encoding="utf-8")
                        except Exception:
                            pass
                return cached
            except Exception:
                pass

    # 1) 技术因子
    factors_raw = fetch_stk_factor_pro(ts_code, days=60)
    if factors_raw:
        result["tushare_factors_raw"] = factors_raw
        result["tushare_factors"] = summarize_factors(factors_raw)

    # 2) 筹码绩效
    cyq_rows = fetch_cyq_perf(ts_code, days=10)
    if cyq_rows:
        result["tushare_cyq_series"] = cyq_rows
        result["tushare_cyq"] = summarize_cyq(cyq_rows)

    # 3) 筹码分布
    chips = fetch_cyq_chips(ts_code)
    if chips:
        result["tushare_cyq_chips"] = chips

    # 4) 资金流
    mf_rows = fetch_moneyflow(ts_code, days=10)
    if mf_rows:
        result["tushare_moneyflow_series"] = mf_rows
        result["tushare_moneyflow"] = summarize_moneyflow(mf_rows)

    # 5) 股东户数
    holders = fetch_holdernumber(ts_code, periods=4)
    if holders:
        result["tushare_holders"] = holders

    # 6) 行业 (用于 sparse_layered 上下文)
    industry = fetch_industry(ts_code)
    if industry:
        result["industry"] = industry

    # 写缓存
    if cache_dir:
        try:
            (cache_dir / "tushare_enrich.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("[tushare_enrich] 缓存写入失败: %s", e)

    return result
