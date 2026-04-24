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
    """资金流 4 档分层(小/中/大/超大单)。"""
    pro = _get_pro()
    if not pro:
        return None
    try:
        df = pro.moneyflow(ts_code=ts_code, limit=days)
        if df is None or df.empty:
            return None
        df = df.sort_values("trade_date")
        return df.to_dict(orient="records")
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
    """汇总 5/10 日主力资金分层。"""
    if not mf_rows:
        return {}

    def _sum(key: str) -> float:
        return sum(float(r.get(key, 0) or 0) for r in mf_rows)

    latest = mf_rows[-1]
    n = len(mf_rows)

    # 最新一日
    latest_net = float(latest.get("net_mf_amount", 0) or 0)
    latest_elg = float(latest.get("buy_elg_amount", 0) or 0) - float(latest.get("sell_elg_amount", 0) or 0)
    latest_lg = float(latest.get("buy_lg_amount", 0) or 0) - float(latest.get("sell_lg_amount", 0) or 0)
    latest_md = float(latest.get("buy_md_amount", 0) or 0) - float(latest.get("sell_md_amount", 0) or 0)
    latest_sm = float(latest.get("buy_sm_amount", 0) or 0) - float(latest.get("sell_sm_amount", 0) or 0)

    # N 日累计
    total_net = _sum("net_mf_amount")
    total_elg_net = _sum("buy_elg_amount") - _sum("sell_elg_amount")
    total_lg_net = _sum("buy_lg_amount") - _sum("sell_lg_amount")
    main_net_total = total_elg_net + total_lg_net   # 主力 = 大+超大

    return {
        "days": n,
        "trade_date_latest": str(latest.get("trade_date", "")),
        # 最新一日(单位: 万元, Tushare 默认)
        "latest_net_total": round(latest_net, 2),
        "latest_super_large_net": round(latest_elg, 2),   # 超大单净流入
        "latest_large_net": round(latest_lg, 2),           # 大单净流入
        "latest_medium_net": round(latest_md, 2),          # 中单净流入
        "latest_small_net": round(latest_sm, 2),           # 小单净流入
        "latest_main_net": round(latest_elg + latest_lg, 2),   # 主力单(大+超大)
        # N 日累计
        f"sum_{n}d_net_total": round(total_net, 2),
        f"sum_{n}d_main_net": round(main_net_total, 2),
        f"sum_{n}d_super_large_net": round(total_elg_net, 2),
        f"sum_{n}d_large_net": round(total_lg_net, 2),
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
    """基于 Tushare 增强数据打 4 维量化分(50 中性 ±30)。

    因子设计:
      1. ADX 趋势强度        范围 [-8, +10]   (阈值 20/25/30, 区分方向)
      2. winner_rate 筹码     范围 [-10, +12] (<=20 深度套牢加分, >=85 顶部扣分)
      3. sum_Nd_main_net      范围 [-10, +10] (超大+大单累计, 单位万元)
      4. holder_num 变化%     范围 [-6, +8]   (同比减少=筹码集中利好)

    有数据才打分的因子才累加; 缺数据则不参与不惩罚。
    总偏移累加到 50 基准上, clamp 到 [0, 100]。

    Returns:
        {
          "quant_score": float,           # 最终 0-100 分
          "total_delta": float,           # 累计偏移
          "adjustments": list[dict],      # 每个因子的 delta+reason
          "has_data": bool,               # 是否至少命中一个因子
        }
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

    # 3. N 日主力资金累计(大+超大单)
    mf = enrich.get("tushare_moneyflow") or {}
    main_net = None
    for k, v in mf.items():
        if k.startswith("sum_") and k.endswith("d_main_net"):
            main_net = v
            break
    if main_net is not None:
        try:
            mn = float(main_net)
            if mn >= 5000:
                _add("main_net", +10.0, f"主力大幅净流入 {mn:.0f} 万")
            elif mn >= 1000:
                _add("main_net", +5.0, f"主力净流入 {mn:.0f} 万")
            elif mn <= -5000:
                _add("main_net", -10.0, f"主力大幅净流出 {mn:.0f} 万")
            elif mn <= -1000:
                _add("main_net", -5.0, f"主力净流出 {mn:.0f} 万")
        except (ValueError, TypeError):
            pass

    # 4. 股东户数变化%(首期→末期)
    holders = enrich.get("tushare_holders") or []
    if isinstance(holders, list) and len(holders) >= 2:
        try:
            first = float(holders[0].get("holder_num") or 0)
            last = float(holders[-1].get("holder_num") or 0)
            if first > 0:
                pct = (last - first) / first * 100
                if pct <= -5:
                    _add("holders", +8.0, f"股东户数 {pct:+.1f}% (筹码集中利好)")
                elif pct <= -2:
                    _add("holders", +4.0, f"股东户数 {pct:+.1f}% (轻度集中)")
                elif pct >= 5:
                    _add("holders", -6.0, f"股东户数 {pct:+.1f}% (筹码分散利空)")
                elif pct >= 2:
                    _add("holders", -3.0, f"股东户数 {pct:+.1f}% (轻度分散)")
        except (ValueError, TypeError, AttributeError):
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
                return json.loads(cache_file.read_text(encoding="utf-8"))
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

    # 写缓存
    if cache_dir:
        try:
            (cache_dir / "tushare_enrich.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("[tushare_enrich] 缓存写入失败: %s", e)

    return result
