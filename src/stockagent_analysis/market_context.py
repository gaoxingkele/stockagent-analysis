# -*- coding: utf-8 -*-
"""市场环境感知模块 — 大盘状态 + 板块热度 + ETF走势 + 多模态视觉研判。

会话级缓存: 同一批次分析多只股票时, 大盘/板块数据只拉一次。
"""
from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("stockagent.market_ctx")

# 确保国内数据源直连(不走海外LLM代理)
try:
    from .data_backend import _ensure_akshare_no_proxy
    _ensure_akshare_no_proxy()
except Exception:
    pass

# ── 常量 ────────────────────────────────────────────────────────────

# 主要宽基指数
MAJOR_INDICES: dict[str, str] = {
    "sh000001": "上证指数",
    "sh000300": "沪深300",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000905": "中证500",
    "sh000852": "中证1000",
}

# 趋势状态枚举
TREND_STATES = {
    "uptrend":            "上涨通道",
    "uptrend_breakdown":  "跌破上涨通道",
    "sideways":           "横盘整理",
    "trend_center":       "走势中枢",
    "ma_convergence":     "多均线缠绕",
    "downtrend":          "下跌通道",
    "downtrend_breakout": "向上突破下跌通道",
    "unknown":            "未知",
}


# ── 数据类 ───────────────────────────────────────────────────────────

@dataclass
class TrendState:
    """单个标的(指数/ETF)的趋势状态。"""
    code: str
    name: str
    state: str           # TREND_STATES key
    state_cn: str        # 中文标签
    close: float = 0.0
    ma5: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    ma20_slope: float = 0.0   # MA20 5日斜率(%)
    ma_spread: float = 0.0    # 均线离散度(%)
    bband_width: float = 0.0  # 布林带宽度(%)
    ret_5d: float = 0.0
    ret_20d: float = 0.0
    vol_20d: float = 0.0      # 20日波动率(%)
    volume_ratio: float = 1.0  # 5日/20日量比
    rsi: float = 50.0


@dataclass
class SectorHeat:
    """概念/行业板块热度。"""
    sector_name: str
    rank: int = 0          # 热度排名(1=最热)
    pct_chg_1d: float = 0.0
    pct_chg_5d: float = 0.0
    pct_chg_20d: float = 0.0
    lead_stock: str = ""   # 领涨股名称
    related_etfs: list[str] = field(default_factory=list)


@dataclass
class MarketContext:
    """完整市场环境上下文。"""
    generated_at: str = ""
    # 大盘指数状态
    index_states: list[TrendState] = field(default_factory=list)
    # 综合大盘评估
    market_score: float = 50.0   # 0-100, 50=中性
    market_phase: str = "balanced"  # offensive/balanced/defensive
    market_phase_cn: str = "平衡"
    # 个股关联板块热度
    sector_heats: list[SectorHeat] = field(default_factory=list)
    # 关联ETF走势
    etf_states: list[TrendState] = field(default_factory=list)
    # 视觉分析摘要(大盘+ETF)
    vision_summary: str = ""
    # 全市场资金流方向信号
    mkt_flow_signal: str = "neutral"   # distribution/smart_money_buying/consensus_buy/consensus_sell/neutral
    mkt_flow_detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "index_states": [asdict(s) for s in self.index_states],
            "market_score": self.market_score,
            "market_phase": self.market_phase,
            "market_phase_cn": self.market_phase_cn,
            "sector_heats": [asdict(s) for s in self.sector_heats],
            "etf_states": [asdict(s) for s in self.etf_states],
            "vision_summary": self.vision_summary,
            "mkt_flow_signal": self.mkt_flow_signal,
            "mkt_flow_detail": self.mkt_flow_detail,
        }


# ── 趋势状态分类器(纯本地计算) ───────────────────────────────────────

def classify_trend_state(df: pd.DataFrame) -> TrendState:
    """基于日线DataFrame分类趋势状态。

    df 至少需要 close/volume 列, 60+行。
    返回 TrendState (code/name 由调用方填充)。
    """
    ts = TrendState(code="", name="", state="unknown", state_cn="未知")
    if df is None or len(df) < 60:
        return ts

    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series([1.0] * len(df))

    c = float(close.iloc[-1])
    ts.close = round(c, 2)

    # 均线
    ts.ma5 = round(float(close.rolling(5).mean().iloc[-1]), 2)
    ts.ma10 = round(float(close.rolling(10).mean().iloc[-1]), 2)
    ts.ma20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
    ts.ma60 = round(float(close.rolling(60).mean().iloc[-1]), 2)

    # MA20 斜率: 最近5日MA20变化率(%)
    ma20_series = close.rolling(20).mean()
    ma20_5ago = float(ma20_series.iloc[-6]) if len(ma20_series) > 5 else ts.ma20
    ts.ma20_slope = round((ts.ma20 / ma20_5ago - 1) * 100, 3) if ma20_5ago > 0 else 0

    # 均线离散度: (max(MAs) - min(MAs)) / close * 100
    mas = [ts.ma5, ts.ma10, ts.ma20, ts.ma60]
    ts.ma_spread = round((max(mas) - min(mas)) / c * 100, 2) if c > 0 else 0

    # 布林带宽度
    std20 = float(close.rolling(20).std().iloc[-1])
    ts.bband_width = round(4 * std20 / ts.ma20 * 100, 2) if ts.ma20 > 0 else 0

    # 收益率
    if len(close) >= 6:
        ts.ret_5d = round((c / float(close.iloc[-6]) - 1) * 100, 2)
    if len(close) >= 21:
        ts.ret_20d = round((c / float(close.iloc[-21]) - 1) * 100, 2)

    # 波动率
    ts.vol_20d = round(float(close.pct_change().tail(20).std() * 100), 2)

    # 量比
    vol_5 = float(volume.tail(5).mean())
    vol_20 = float(volume.tail(20).mean())
    ts.volume_ratio = round(vol_5 / vol_20, 2) if vol_20 > 0 else 1.0

    # RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100
    ts.rsi = round(100 - 100 / (1 + rs), 1)

    # ── 状态分类逻辑 ──
    state = _classify_state(c, ts.ma5, ts.ma10, ts.ma20, ts.ma60,
                            ts.ma20_slope, ts.ma_spread, ts.bband_width)
    ts.state = state
    ts.state_cn = TREND_STATES.get(state, "未知")
    return ts


def _classify_state(c: float, ma5: float, ma10: float, ma20: float, ma60: float,
                    ma20_slope: float, ma_spread: float, bband_width: float) -> str:
    """核心状态分类逻辑。"""

    # 多均线缠绕: 均线离散度极低(< 2%)
    if ma_spread < 2.0:
        return "ma_convergence"

    # 上涨通道: price > MA20 > MA60, MA20斜率正
    if c > ma20 > ma60 and ma20_slope > 0.05:
        # 检查是否刚跌破上涨通道 (price < MA5 且接近MA20)
        if c < ma5 and (c - ma20) / ma20 * 100 < 1.0:
            return "uptrend_breakdown"
        return "uptrend"

    # 下跌通道: price < MA20 < MA60, MA20斜率负
    if c < ma20 < ma60 and ma20_slope < -0.05:
        # 检查是否向上突破 (price > MA5 且接近MA20)
        if c > ma5 and (ma20 - c) / ma20 * 100 < 1.0:
            return "downtrend_breakout"
        return "downtrend"

    # 横盘整理: 布林带收窄 + MA20斜率平
    if bband_width < 6.0 and abs(ma20_slope) < 0.15:
        return "sideways"

    # 走势中枢: price在MA60附近震荡(偏离<3%)
    if abs(c - ma60) / ma60 * 100 < 3.0:
        return "trend_center"

    # 跌破上涨通道的宽松判断: MA20 > MA60 但 price < MA20
    if ma20 > ma60 and c < ma20:
        return "uptrend_breakdown"

    # 向上突破下跌通道的宽松判断: MA20 < MA60 但 price > MA20
    if ma20 < ma60 and c > ma20:
        return "downtrend_breakout"

    return "sideways"


# ── 大盘综合评分 ──────────────────────────────────────────────────────

def _compute_market_score(index_states: list[TrendState]) -> tuple[float, str, str]:
    """基于多指数状态计算综合大盘评分(0-100)和阶段。"""
    if not index_states:
        return 50.0, "balanced", "平衡"

    state_scores = {
        "uptrend": 80,
        "uptrend_breakdown": 55,
        "downtrend_breakout": 65,
        "sideways": 50,
        "trend_center": 48,
        "ma_convergence": 50,
        "downtrend": 20,
        "unknown": 50,
    }

    # 加权: 沪深300权重最大
    weights = {
        "sh000001": 0.20,  # 上证
        "sh000300": 0.30,  # 沪深300
        "sz399001": 0.15,  # 深成指
        "sz399006": 0.15,  # 创业板
        "sh000905": 0.10,  # 中证500
        "sh000852": 0.10,  # 中证1000
    }

    total_w = 0.0
    weighted_score = 0.0
    for s in index_states:
        w = weights.get(s.code, 0.1)
        base = state_scores.get(s.state, 50)
        # 用动量微调: ret_20d 贡献 +-10分
        momentum_adj = max(-10, min(10, s.ret_20d * 0.5))
        # RSI超买超卖修正
        rsi_adj = 0
        if s.rsi > 75:
            rsi_adj = -5
        elif s.rsi < 25:
            rsi_adj = 5
        score = max(0, min(100, base + momentum_adj + rsi_adj))
        weighted_score += score * w
        total_w += w

    market_score = round(weighted_score / total_w, 1) if total_w > 0 else 50.0

    if market_score >= 65:
        phase, phase_cn = "offensive", "进攻"
    elif market_score <= 35:
        phase, phase_cn = "defensive", "防守"
    else:
        phase, phase_cn = "balanced", "平衡"

    return market_score, phase, phase_cn


# ── 指数/ETF K线获取(走多级降级链) ─────────────────────────────────────

def _get_mkt_moneyflow_signal(days: int = 5) -> dict:
    """拉全市场 moneyflow_mkt_dc, 5日MA计算主力/散户方向。

    返回: {"signal": str, "detail": str, "score_adj": float}
    signal 值: distribution / smart_money_buying / consensus_buy / consensus_sell / neutral
    score_adj: 建议对 market_score 的微调量(-5 ~ +5)
    """
    try:
        from .tushare_enrich import _get_pro
        pro = _get_pro()
        if not pro:
            return {"signal": "neutral", "detail": "", "score_adj": 0}
        df = pro.moneyflow_mkt_dc(limit=days)
        if df is None or df.empty:
            return {"signal": "neutral", "detail": "", "score_adj": 0}
        df = df.sort_values("trade_date")
        rows = df.to_dict(orient="records")

        # 逐日主力(ELG+LG)净流入 和 散户(SM)净流入
        daily_main = [
            float(r.get("buy_elg_amount", 0) or 0) + float(r.get("buy_lg_amount", 0) or 0)
            for r in rows
        ]
        daily_retail = [float(r.get("buy_sm_amount", 0) or 0) for r in rows]

        # 5日MA
        main_ma5 = sum(daily_main) / len(daily_main)
        retail_ma5 = sum(daily_retail) / len(daily_retail)

        # 近3日方向一致性
        main_pos = sum(1 for v in daily_main[-3:] if v > 0)
        retail_pos = sum(1 for v in daily_retail[-3:] if v > 0)
        main_trend = "inflow" if main_pos >= 2 else "outflow"
        retail_trend = "inflow" if retail_pos >= 2 else "outflow"

        unit = 1e8  # 转亿
        if main_trend == "outflow" and retail_trend == "inflow":
            signal = "distribution"
            detail = f"大盘主力净流出(5日均{main_ma5/unit:.0f}亿), 散户仍在承接"
            score_adj = -5.0
        elif main_trend == "inflow" and retail_trend == "outflow":
            signal = "smart_money_buying"
            detail = f"大盘主力净流入(5日均{main_ma5/unit:.0f}亿), 散户减仓"
            score_adj = +4.0
        elif main_trend == "inflow":
            signal = "consensus_buy"
            detail = f"市场整体净流入(主力5日均{main_ma5/unit:.0f}亿)"
            score_adj = +2.0
        else:
            signal = "consensus_sell"
            detail = f"市场整体净流出(主力5日均{main_ma5/unit:.0f}亿)"
            score_adj = -3.0

        return {
            "signal": signal,
            "detail": detail,
            "score_adj": score_adj,
            "main_ma5": round(main_ma5, 2),
            "retail_ma5": round(retail_ma5, 2),
        }
    except Exception as e:
        logger.debug("[market_context] mkt_moneyflow_signal 失败: %s", e)
        return {"signal": "neutral", "detail": "", "score_adj": 0}


def _fetch_index_daily(code_with_market: str, days: int = 120) -> pd.DataFrame | None:
    """获取指数日线数据。
    code_with_market 格式: sh000001 / sz399001 (兼容旧调用)
    或者 6位纯数字(自动推断市场)
    """
    if code_with_market.startswith(("sh", "sz", "bj")):
        market = code_with_market[:2]
        code = code_with_market[2:]
    else:
        market = ""
        code = code_with_market
    df, source = _fetch_klines_multi_source(code, prefer_market=market, days=days)
    if df is not None:
        logger.debug("index %s → %s", code_with_market, source)
    return df


def _fetch_etf_daily(code: str, days: int = 120) -> pd.DataFrame | None:
    """获取ETF日线数据。code为6位纯数字(如515290)。"""
    # ETF 市场判定: 5xxxxx=sh, 1xxxxx=sz
    market = "sh" if code.startswith("5") else "sz"
    df, source = _fetch_klines_multi_source(code, prefer_market=market, days=days)
    if df is not None:
        logger.debug("etf %s → %s", code, source)
    return df


# ── 概念板块查询 ───────────────────────────────────────────────────────

def _no_proxy_call(func, *args, **kwargs):
    """临时清空代理环境变量执行函数,避免海外LLM代理拦截国内站点。"""
    import os as _os
    saved = {k: _os.environ.pop(k, None) for k in
             ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
              "http_proxy", "https_proxy", "all_proxy")}
    try:
        return func(*args, **kwargs)
    finally:
        for k, v in saved.items():
            if v is not None:
                _os.environ[k] = v


# ── TDX 本地二进制读取(零延迟,无网络依赖) ──────────────────────────────

def _guess_tdx_market(code: str) -> str:
    """根据6位代码猜测TDX市场目录。
    规则:
      sh: 60xxxx(股), 68xxxx(科创), 5xxxxx(沪市ETF), 000xxx(上证指数), 00xxxx非深指时也有误差
      sz: 30xxxx(创业), 00xxxx(深股), 1xxxxx(深市ETF/基金), 399xxx(深证指数)
      bj: 8/4/9开头
    特殊: 指数优先走sh(上证), 399开头走sz(深证)
    """
    if code.startswith(("8", "4", "9")):
        return "bj"
    if code.startswith("399"):
        return "sz"
    if code.startswith("000"):  # 000001=上证指数(sh) / 000001=平安银行(sz)
        # 上证指数区间: 000001-000099 是上证综指及衍生
        # 深市股票区间: 000001-000999 是A股
        # 冲突! 需要根据场景判断: 这里默认走sh(指数优先)
        # 调用方应该用 sh_or_sz 参数覆盖
        return "sh"
    if code.startswith(("5", "6")):
        return "sh"
    if code.startswith(("0", "1", "2", "3")):
        return "sz"
    return "sh"


def _read_tdx_day_file(market: str, code: str, days: int = 120) -> pd.DataFrame | None:
    """直接解析 vipdoc/{market}/lday/{market}{code}.day 二进制文件。

    TDX .day 格式: 每条32字节 = date(u4) open(u4) high(u4) low(u4) close(u4) amount(f4/u4) volume(u4) reserved(u4)
    指数/股票/ETF 统一: 价格字段除以100。amount字段对指数是 u4(万元), 对股票/ETF是 f4。
    这里统一按 u4 解,因为对趋势分析无影响(只用 close/volume)。
    """
    import os as _os
    import struct
    import datetime as _dt

    tdx_dir = _os.getenv("TDX_DIR", "D:/tdx")
    day_path = _os.path.join(tdx_dir, "vipdoc", market, "lday", f"{market}{code}.day")
    if not _os.path.isfile(day_path):
        return None

    records = []
    try:
        # 只读取最后 days * 32 字节,加速大文件
        with open(day_path, "rb") as f:
            f.seek(0, 2)  # EOF
            size = f.tell()
            read_bytes = min(size, (days + 10) * 32)
            f.seek(-read_bytes, 2)
            data = f.read(read_bytes)
        offset = 0
        while offset + 32 <= len(data):
            fields = struct.unpack_from("<8I", data, offset)
            offset += 32
            date_int = fields[0]
            year = date_int // 10000
            month = (date_int % 10000) // 100
            day = date_int % 100
            try:
                dt = _dt.date(year, month, day)
            except ValueError:
                continue
            records.append({
                "date": dt,
                "open": fields[1] / 100.0,
                "high": fields[2] / 100.0,
                "low": fields[3] / 100.0,
                "close": fields[4] / 100.0,
                "volume": float(fields[6]),
            })
    except Exception as e:
        logger.warning("read tdx day file failed %s/%s: %s", market, code, e)
        return None

    if not records or len(records) < 30:
        return None
    df = pd.DataFrame(records).tail(days).reset_index(drop=True)
    return df


def _fetch_klines_multi_source(code: str,
                               prefer_market: str = "",
                               days: int = 120) -> tuple[pd.DataFrame | None, str]:
    """多级降级数据源链:
      TDX本地 → akshare东财 → tushare → 腾讯 → 同花顺

    对指数: code如 000001(上证) / 399001(深证) / 000300(沪深300)
        prefer_market="sh" 或 "sz" 指定TDX子目录,也决定akshare symbol前缀
    对ETF/股票: code如 515290(银行ETF) / 600519(个股)
        prefer_market可为空,自动猜测

    返回 (df, source_name), df为None时source_name='failed'。
    df 包含列: date, open, high, low, close, volume
    """
    market = prefer_market or _guess_tdx_market(code)

    # 1. TDX 本地
    df = _read_tdx_day_file(market, code, days=days)
    if df is not None:
        return df, f"tdx:{market}"

    # 另一市场重试(指数 000xxx 冲突 sh/sz 时)
    if not prefer_market:
        other = "sz" if market == "sh" else "sh"
        df = _read_tdx_day_file(other, code, days=days)
        if df is not None:
            return df, f"tdx:{other}"

    # 2. akshare 东财
    akshare_code = f"{market}{code}" if market in ("sh", "sz") else code
    df = _fetch_via_akshare(akshare_code, code, days)
    if df is not None:
        return df, "akshare"

    # 3. tushare
    df = _fetch_via_tushare(code, market, days)
    if df is not None:
        return df, "tushare"

    # 4. 腾讯
    df = _fetch_via_tencent(code, market, days)
    if df is not None:
        return df, "tencent"

    # 5. 同花顺(可选,仅部分指数支持)
    df = _fetch_via_ths(code, market, days)
    if df is not None:
        return df, "ths"

    return None, "failed"


def _fetch_via_akshare(full_code: str, code: str, days: int) -> pd.DataFrame | None:
    """akshare 多接口尝试: 指数 stock_zh_index_daily, ETF fund_etf_hist_em。"""
    try:
        import akshare as ak
    except ImportError:
        return None

    # 指数走 stock_zh_index_daily (需要带 sh/sz 前缀)
    if code.startswith(("000", "399")):
        try:
            time.sleep(0.3)
            df = _no_proxy_call(ak.stock_zh_index_daily, symbol=full_code)
            if df is not None and len(df) >= 30:
                df = df.rename(columns={c: c.lower() for c in df.columns})
                df["close"] = df["close"].astype(float)
                if "volume" in df.columns:
                    df["volume"] = df["volume"].astype(float)
                return df.tail(days).reset_index(drop=True)
        except Exception as e:
            logger.debug("akshare index %s failed: %s", full_code, e)

    # ETF (5/1 开头的6位代码)
    if code.startswith(("5", "1")) and len(code) == 6:
        try:
            time.sleep(0.3)
            end_d = datetime.now().strftime("%Y%m%d")
            start_d = (datetime.now() - timedelta(days=days + 60)).strftime("%Y%m%d")
            df = _no_proxy_call(
                ak.fund_etf_hist_em,
                symbol=code, period="daily",
                start_date=start_d, end_date=end_d, adjust="",
            )
            if df is not None and len(df) >= 30:
                rename = {"日期": "date", "开盘": "open", "最高": "high",
                          "最低": "low", "收盘": "close", "成交量": "volume"}
                df = df.rename(columns=rename)
                df["close"] = df["close"].astype(float)
                if "volume" in df.columns:
                    df["volume"] = df["volume"].astype(float)
                return df.tail(days).reset_index(drop=True)
        except Exception as e:
            logger.debug("akshare etf %s failed: %s", code, e)

    return None


def _fetch_via_tushare(code: str, market: str, days: int) -> pd.DataFrame | None:
    """tushare 获取指数/ETF日线。"""
    try:
        import tushare as ts
        import os as _os
        token = _os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            return None
        pro = ts.pro_api(token)

        # 构造 ts_code
        if code.startswith(("000", "399")) and market:
            ts_code = f"{code}.{market.upper()}"
            time.sleep(0.5)
            df = pro.index_daily(ts_code=ts_code)
        elif code.startswith(("5", "1")) and len(code) == 6:
            ts_code = f"{code}.{market.upper() if market else ('SH' if code.startswith('5') else 'SZ')}"
            time.sleep(0.5)
            df = pro.fund_daily(ts_code=ts_code)
        else:
            return None

        if df is None or len(df) < 30:
            return None
        df = df.rename(columns={"trade_date": "date", "vol": "volume"})
        df["close"] = df["close"].astype(float)
        if "volume" in df.columns:
            df["volume"] = df["volume"].astype(float)
        df = df.sort_values("date").reset_index(drop=True)
        return df.tail(days)
    except Exception as e:
        logger.debug("tushare %s failed: %s", code, e)
        return None


def _fetch_via_tencent(code: str, market: str, days: int) -> pd.DataFrame | None:
    """腾讯财经直连: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"""
    try:
        import requests
        mkt_prefix = market or ("sh" if code.startswith(("5", "6", "000", "688")) else "sz")
        symbol = f"{mkt_prefix}{code}"
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={symbol},day,,,{days + 20},qfq"
        # 直连,禁用代理
        import os as _os
        proxies = {"http": None, "https": None}
        r = requests.get(url, proxies=proxies, timeout=8)
        if r.status_code != 200:
            return None
        data = r.json()
        # 路径: data.{symbol}.day 或 data.{symbol}.qfqday
        day_list = None
        sec = data.get("data", {}).get(symbol, {})
        for key in ("qfqday", "day"):
            if key in sec:
                day_list = sec[key]
                break
        if not day_list or len(day_list) < 30:
            return None
        records = []
        for item in day_list:
            records.append({
                "date": item[0],
                "open": float(item[1]),
                "close": float(item[2]),
                "high": float(item[3]),
                "low": float(item[4]),
                "volume": float(item[5]) if len(item) > 5 else 0.0,
            })
        df = pd.DataFrame(records).tail(days).reset_index(drop=True)
        return df
    except Exception as e:
        logger.debug("tencent %s failed: %s", code, e)
        return None


def _fetch_via_ths(code: str, market: str, days: int) -> pd.DataFrame | None:
    """同花顺(通过 akshare 的 ths 接口)。仅作为最后备选。"""
    try:
        import akshare as ak
        # 同花顺指数接口(部分支持)
        if code.startswith(("000", "399")):
            try:
                df = _no_proxy_call(ak.stock_zh_index_hist_csindex, symbol=code)
                if df is not None and len(df) >= 30:
                    df = df.rename(columns={"日期": "date", "收盘": "close", "成交量": "volume"})
                    df["close"] = df["close"].astype(float)
                    if "volume" in df.columns:
                        df["volume"] = df["volume"].astype(float)
                    return df.tail(days).reset_index(drop=True)
            except Exception:
                pass
    except Exception as e:
        logger.debug("ths %s failed: %s", code, e)
    return None


def _fetch_stock_concepts(symbol: str) -> list[str]:
    """查询个股所属行业/概念板块名称列表(多源降级)。

    降级链:
      1. tushare stock_basic (取行业,稳定)
      2. akshare stock_individual_info_em (个股信息含行业,东财接口不稳)
      3. 返回空列表
    """
    concepts = []

    # 1. tushare
    try:
        import tushare as ts
        import os as _os
        token = _os.getenv("TUSHARE_TOKEN", "").strip()
        if token:
            pro = ts.pro_api(token)
            time.sleep(0.3)
            if symbol.startswith("6"):
                ts_code = f"{symbol}.SH"
            elif symbol.startswith(("8", "4", "9")):
                ts_code = f"{symbol}.BJ"
            else:
                ts_code = f"{symbol}.SZ"
            df = pro.stock_basic(ts_code=ts_code, fields="ts_code,industry,name")
            if df is not None and not df.empty:
                industry = str(df.iloc[0].get("industry", "")).strip()
                if industry:
                    concepts.append(industry)
    except Exception as e:
        logger.debug("tushare stock_basic for %s failed: %s", symbol, e)

    # 2. akshare 东财(不稳定,作为补充)
    try:
        import akshare as ak
        time.sleep(0.3)
        info = _no_proxy_call(ak.stock_individual_info_em, symbol=symbol)
        if info is not None and not info.empty:
            for _, row in info.iterrows():
                item = str(row.get("item", ""))
                value = str(row.get("value", "")).strip()
                if ("行业" in item or "板块" in item) and value and value not in concepts:
                    concepts.append(value)
    except Exception as e:
        logger.debug("akshare individual_info for %s failed: %s", symbol, e)

    if not concepts:
        logger.warning("fetch concepts for %s: all sources failed", symbol)
    return concepts


def _fetch_concept_board_ranking(top_n: int = 20) -> list[dict]:
    """获取板块涨幅排行(多源降级)。

    降级链:
      1. akshare stock_board_industry_name_em (东财行业榜,较稳)
      2. akshare stock_board_concept_name_em (东财概念榜,经常被限流)
      3. tushare ths_index + ths_daily (同花顺指数,需权限)
      4. 返回空列表
    """
    # 1. 东财行业榜
    try:
        import akshare as ak
        time.sleep(0.5)
        df = _no_proxy_call(ak.stock_board_industry_name_em)
        if df is not None and not df.empty:
            if "涨跌幅" in df.columns:
                df = df.sort_values("涨跌幅", ascending=False)
            result = []
            for i, (_, row) in enumerate(df.head(top_n).iterrows()):
                result.append({
                    "rank": i + 1,
                    "name": str(row.get("板块名称", "")),
                    "pct_chg": float(row.get("涨跌幅", 0) or 0),
                    "lead_stock": str(row.get("领涨股票", "")),
                    "source": "em_industry",
                })
            if result:
                return result
    except Exception as e:
        logger.debug("akshare industry ranking failed: %s", e)

    # 2. 东财概念榜
    try:
        import akshare as ak
        time.sleep(0.5)
        df = _no_proxy_call(ak.stock_board_concept_name_em)
        if df is not None and not df.empty:
            if "涨跌幅" in df.columns:
                df = df.sort_values("涨跌幅", ascending=False)
            result = []
            for i, (_, row) in enumerate(df.head(top_n).iterrows()):
                result.append({
                    "rank": i + 1,
                    "name": str(row.get("板块名称", "")),
                    "pct_chg": float(row.get("涨跌幅", 0) or 0),
                    "lead_stock": str(row.get("领涨股票", "")),
                    "source": "em_concept",
                })
            if result:
                return result
    except Exception as e:
        logger.debug("akshare concept ranking failed: %s", e)

    # 3. tushare ths_index (同花顺指数)
    try:
        import tushare as ts
        import os as _os
        token = _os.getenv("TUSHARE_TOKEN", "").strip()
        if token:
            pro = ts.pro_api(token)
            time.sleep(0.5)
            # 获取概念/行业指数列表
            df_idx = pro.ths_index(exchange="A", type="N")  # N=概念指数
            if df_idx is not None and not df_idx.empty:
                result = []
                for i, (_, row) in enumerate(df_idx.head(top_n).iterrows()):
                    result.append({
                        "rank": i + 1,
                        "name": str(row.get("name", "")),
                        "pct_chg": 0,  # ths_daily需额外调用
                        "lead_stock": "",
                        "source": "ths",
                    })
                if result:
                    return result
    except Exception as e:
        logger.debug("tushare ths_index failed: %s", e)

    logger.warning("fetch concept ranking: all sources failed")
    return []


# ── 概念板块→ETF映射(扩展) ──────────────────────────────────────────

# 主题型ETF映射(补充行业ETF之外的概念ETF)
_CONCEPT_ETF_MAP: dict[str, list[str]] = {
    "人工智能": ["515070", "159819"],
    "机器人": ["562500", "159770"],
    "芯片": ["512480", "159995"],
    "半导体": ["512480", "159995"],
    "新能源": ["516160", "159875"],
    "光伏": ["515790", "159857"],
    "锂电池": ["159755", "516510"],
    "储能": ["159566", "516610"],
    "军工": ["512660", "515850"],
    "医药": ["512010", "159929"],
    "消费": ["159928", "510150"],
    "白酒": ["512690"],
    "银行": ["515290", "512800"],
    "证券": ["512880", "512000"],
    "房地产": ["512200", "159768"],
    "数字经济": ["159658", "560800"],
    "信创": ["562030", "159839"],
    "汽车": ["516110", "159845"],
    "传媒": ["516880"],
    "游戏": ["516010"],
}


def _find_related_etfs(industry: str, concepts: list[str]) -> list[str]:
    """根据行业+概念找到所有相关ETF(去重)。"""
    from .data_backend import _INDUSTRY_ETF_MAP
    etfs = set()
    # 行业ETF
    if industry:
        code = _INDUSTRY_ETF_MAP.get(industry)
        if code:
            etfs.add(code)
        # 模糊匹配
        for key, code in _INDUSTRY_ETF_MAP.items():
            if key in industry or industry in key:
                etfs.add(code)
    # 概念ETF
    for concept in concepts:
        for key, codes in _CONCEPT_ETF_MAP.items():
            if key in concept or concept in key:
                etfs.update(codes)
    return list(etfs)[:6]  # 最多6个,控制API调用


# ── 视觉分析(指数/ETF图表→多模态LLM) ──────────────────────────────────

def _generate_index_chart(df: pd.DataFrame, code: str, name: str,
                          save_dir: Path) -> str | None:
    """为指数/ETF生成日线图表,复用chart_generator。"""
    try:
        from .chart_generator import generate_kline_chart
        save_path = save_dir / f"market_{code}.png"
        png = generate_kline_chart(df, "day", code, name, save_path=save_path)
        return str(save_path) if png else None
    except Exception as e:
        logger.warning("generate chart for %s failed: %s", code, e)
        return None


def _vision_analyze_charts(chart_paths: dict[str, str],
                           llm_routers: dict | None = None) -> str:
    """调用多模态LLM分析指数/ETF图表,返回综合研判摘要。

    chart_paths: {label: path} 如 {"沪深300": "/path/to/chart.png"}
    """
    if not chart_paths or not llm_routers:
        return ""

    from .chart_generator import load_image_base64

    prompt_parts = ["请分析以下A股大盘指数和相关ETF的K线图,给出综合研判:\n"]
    images = []
    for label, path in chart_paths.items():
        b64 = load_image_base64(path)
        if b64:
            prompt_parts.append(f"图表: {label}")
            images.append(b64)

    if not images:
        return ""

    prompt_parts.append(
        "\n请从以下角度分析:\n"
        "1. 各指数/ETF当前趋势方向(上涨/横盘/下跌)\n"
        "2. 关键支撑位和压力位\n"
        "3. 量价配合情况\n"
        "4. 技术形态识别(头肩顶/底、双底/顶、三角形整理等)\n"
        "5. 综合研判: 短期(1-5日)和中期(1-4周)方向\n"
        "回复控制在200字以内,直接给结论。"
    )
    prompt = "\n".join(prompt_parts)

    # 尝试找一个支持视觉的router
    router = None
    for name, r in llm_routers.items():
        if hasattr(r, "chat_with_image"):
            try:
                if r._supports_vision(r.provider):
                    router = r
                    break
            except Exception:
                pass
    # fallback: 用第一个router尝试
    if router is None and llm_routers:
        router = next(iter(llm_routers.values()), None)

    if router is None:
        return ""

    try:
        # 只用第一张图(最重要的指数), 多图拼接太大
        resp = router.chat_with_image(prompt, images[0])
        if resp:
            return resp.strip()[:500]
    except Exception as e:
        logger.warning("vision analysis failed: %s", e)

    return ""


# ── 主入口: MarketContextAnalyzer ────────────────────────────────────

class MarketContextAnalyzer:
    """市场环境分析器 — 带会话级缓存。

    同一个 Analyzer 实例在一次批量分析中共享,
    大盘指数状态只拉一次, 板块/ETF按股票不同各取。
    """

    def __init__(self):
        self._lock = threading.Lock()
        # 会话级缓存
        self._index_states: list[TrendState] | None = None
        self._market_score: float | None = None
        self._market_phase: str | None = None
        self._market_phase_cn: str | None = None
        self._concept_ranking: list[dict] | None = None
        self._etf_cache: dict[str, TrendState] = {}  # etf_code → TrendState
        self._index_dfs: dict[str, pd.DataFrame] = {}  # code → df (for chart gen)
        self._mkt_flow: dict | None = None  # 全市场资金流信号缓存

    def analyze_market_indices(self) -> list[TrendState]:
        """分析6大宽基指数趋势状态(带缓存)。"""
        with self._lock:
            if self._index_states is not None:
                return self._index_states

        states = []
        for code, name in MAJOR_INDICES.items():
            df = _fetch_index_daily(code, days=120)
            if df is not None:
                ts = classify_trend_state(df)
                ts.code = code
                ts.name = name
                states.append(ts)
                self._index_dfs[code] = df
            else:
                states.append(TrendState(code=code, name=name, state="unknown", state_cn="未知"))

        with self._lock:
            self._index_states = states
            score, phase, phase_cn = _compute_market_score(states)
            self._market_score = score
            self._market_phase = phase
            self._market_phase_cn = phase_cn

        logger.info("market indices analyzed: score=%.1f phase=%s states=%s",
                     score, phase, [(s.name, s.state_cn) for s in states])
        return states

    def get_market_score(self) -> tuple[float, str, str]:
        """返回 (market_score, phase, phase_cn)。需先调用 analyze_market_indices。"""
        if self._market_score is None:
            self.analyze_market_indices()
        return self._market_score, self._market_phase, self._market_phase_cn

    def analyze_stock_context(
        self,
        symbol: str,
        industry: str,
        run_dir: str | Path | None = None,
        llm_routers: dict | None = None,
    ) -> MarketContext:
        """为特定个股生成完整市场环境上下文。

        包含: 大盘状态 + 所属板块热度 + 关联ETF走势 + 视觉分析。
        """
        # 1. 大盘指数(缓存)
        index_states = self.analyze_market_indices()
        market_score, phase, phase_cn = self.get_market_score()

        # 2. 概念板块
        concepts = _fetch_stock_concepts(symbol)
        sector_heats = self._analyze_sector_heats(industry, concepts)

        # 3. 关联ETF走势
        related_etfs = _find_related_etfs(industry, concepts)
        etf_states = self._analyze_etfs(related_etfs)

        # 4. 视觉分析(可选,有run_dir且有router时才做)
        vision_summary = ""
        if run_dir and llm_routers:
            vision_summary = self._run_vision_analysis(
                run_dir, index_states, etf_states, llm_routers
            )

        # 5. 全市场资金流信号(会话级缓存, 失败不阻断)
        with self._lock:
            if self._mkt_flow is None:
                self._mkt_flow = _get_mkt_moneyflow_signal(days=5)
        mkt_flow = self._mkt_flow or {}
        mkt_flow_signal = mkt_flow.get("signal", "neutral")
        mkt_flow_detail = mkt_flow.get("detail", "")

        # 资金流信号对 market_score 的微调(±5分, 不触发单独 phase 翻转)
        score_adj = float(mkt_flow.get("score_adj", 0))
        adj_score = max(0.0, min(100.0, market_score + score_adj))
        if adj_score >= 65:
            adj_phase, adj_phase_cn = "offensive", "进攻"
        elif adj_score <= 35:
            adj_phase, adj_phase_cn = "defensive", "防守"
        else:
            adj_phase, adj_phase_cn = "balanced", "平衡"
        if score_adj != 0:
            logger.info(
                "[market_context] 大盘资金流=%s adj=%+.1f → score %.1f→%.1f phase=%s",
                mkt_flow_signal, score_adj, market_score, adj_score, adj_phase,
            )

        ctx = MarketContext(
            generated_at=datetime.now().isoformat(),
            index_states=index_states,
            market_score=adj_score,
            market_phase=adj_phase,
            market_phase_cn=adj_phase_cn,
            sector_heats=sector_heats,
            etf_states=etf_states,
            vision_summary=vision_summary,
            mkt_flow_signal=mkt_flow_signal,
            mkt_flow_detail=mkt_flow_detail,
        )
        return ctx

    def _analyze_sector_heats(self, industry: str, concepts: list[str]) -> list[SectorHeat]:
        """分析个股关联板块的热度。"""
        # 获取全市场概念排行(缓存)
        with self._lock:
            if self._concept_ranking is None:
                self._concept_ranking = _fetch_concept_board_ranking(top_n=30)
            ranking = self._concept_ranking

        # 找出与该股相关的板块
        related_names = set()
        if industry:
            related_names.add(industry)
        related_names.update(concepts)

        heats = []
        for item in ranking:
            board_name = item["name"]
            # 模糊匹配
            is_related = False
            for rn in related_names:
                if rn in board_name or board_name in rn:
                    is_related = True
                    break
            if is_related:
                etfs = []
                for key, codes in _CONCEPT_ETF_MAP.items():
                    if key in board_name or board_name in key:
                        etfs.extend(codes)
                heats.append(SectorHeat(
                    sector_name=board_name,
                    rank=item["rank"],
                    pct_chg_1d=item.get("pct_chg", 0),
                    lead_stock=item.get("lead_stock", ""),
                    related_etfs=list(set(etfs))[:3],
                ))

        # 如果没匹配到,把行业本身加进去(rank=0表示未上榜)
        if not heats and industry:
            heats.append(SectorHeat(sector_name=industry, rank=0))

        return heats[:5]

    def _analyze_etfs(self, etf_codes: list[str]) -> list[TrendState]:
        """分析多个ETF的趋势状态(带缓存)。"""
        states = []
        for code in etf_codes:
            with self._lock:
                if code in self._etf_cache:
                    states.append(self._etf_cache[code])
                    continue

            df = _fetch_etf_daily(code, days=120)
            if df is not None:
                ts = classify_trend_state(df)
                ts.code = code
                ts.name = f"ETF:{code}"
                states.append(ts)
                with self._lock:
                    self._etf_cache[code] = ts
            else:
                ts = TrendState(code=code, name=f"ETF:{code}", state="unknown", state_cn="未知")
                states.append(ts)

        return states

    def _run_vision_analysis(
        self,
        run_dir: str | Path,
        index_states: list[TrendState],
        etf_states: list[TrendState],
        llm_routers: dict,
    ) -> str:
        """为关键指数和ETF生成图表并调用视觉LLM。"""
        charts_dir = Path(run_dir) / "data" / "charts" / "market"
        charts_dir.mkdir(parents=True, exist_ok=True)

        chart_paths: dict[str, str] = {}

        # 选最重要的指数: 沪深300
        key_index = "sh000300"
        if key_index in self._index_dfs:
            path = _generate_index_chart(
                self._index_dfs[key_index], key_index,
                MAJOR_INDICES.get(key_index, "沪深300"), charts_dir
            )
            if path:
                chart_paths["沪深300指数"] = path

        # 选前2个ETF
        for ts in etf_states[:2]:
            code = ts.code
            df = _fetch_etf_daily(code, days=120)
            if df is not None:
                path = _generate_index_chart(df, code, ts.name, charts_dir)
                if path:
                    chart_paths[ts.name] = path

        if not chart_paths:
            return ""

        return _vision_analyze_charts(chart_paths, llm_routers)


# ── 评分调节器 ────────────────────────────────────────────────────────

def compute_market_adjustment(market_ctx: MarketContext, stock_score: float) -> float:
    """基于市场环境计算个股评分调节量(加减分)。

    逻辑:
    - 大盘强势(score>65) + 板块热门(rank<10) → 加分(最多+8)
    - 大盘弱势(score<35) + 板块冷门(rank>20或0) → 减分(最多-8)
    - ETF上涨通道 → 小幅加分; 下跌通道 → 小幅减分
    - 个股分数已经极端(>85或<15)时调节减半,避免过度偏移
    - 弱势市场下,高分股(≥70)不削弱 — 保留逆势优秀标的的区分力
    """
    adj = 0.0

    # 1. 大盘环境调节 (±5)
    ms = market_ctx.market_score
    if ms >= 65:
        adj += min(5, (ms - 65) * 0.15)
    elif ms <= 35:
        adj -= min(5, (35 - ms) * 0.15)

    # 2. 板块热度调节 (±3)
    if market_ctx.sector_heats:
        best_rank = min((s.rank for s in market_ctx.sector_heats if s.rank > 0), default=999)
        if best_rank <= 5:
            adj += 3
        elif best_rank <= 10:
            adj += 1.5
        elif best_rank > 20:
            adj -= 1.5

    # 3. ETF趋势调节 (±3)
    if market_ctx.etf_states:
        etf_bullish = sum(1 for s in market_ctx.etf_states if s.state in ("uptrend", "downtrend_breakout"))
        etf_bearish = sum(1 for s in market_ctx.etf_states if s.state in ("downtrend", "uptrend_breakdown"))
        n = len(market_ctx.etf_states)
        if n > 0:
            ratio = (etf_bullish - etf_bearish) / n
            adj += ratio * 3

    # 极端分数时调节减半
    if stock_score > 85 or stock_score < 15:
        adj *= 0.5

    # 弱势市场不削弱高分逆势股: 评分≥70时,负向调节清零
    if adj < 0 and stock_score >= 70:
        adj = 0.0

    return round(max(-8, min(8, adj)), 2)


# ── 文本摘要(给LLM上下文用) ───────────────────────────────────────────

def market_context_summary(ctx: MarketContext) -> str:
    """生成市场环境文本摘要,供LLM prompt使用。"""
    parts = []

    # 大盘状态
    parts.append(f"大盘环境: {ctx.market_phase_cn}阶段 (评分{ctx.market_score:.0f}/100)")
    idx_lines = []
    for s in ctx.index_states:
        if s.state != "unknown":
            idx_lines.append(f"{s.name}={s.state_cn}(5日{s.ret_5d:+.1f}%/20日{s.ret_20d:+.1f}%)")
    if idx_lines:
        parts.append("指数: " + " | ".join(idx_lines))

    # 板块热度
    if ctx.sector_heats:
        heat_lines = []
        for s in ctx.sector_heats:
            rank_str = f"热度#{s.rank}" if s.rank > 0 else "未上榜"
            heat_lines.append(f"{s.sector_name}({rank_str},今日{s.pct_chg_1d:+.1f}%)")
        parts.append("关联板块: " + " | ".join(heat_lines))

    # ETF走势
    if ctx.etf_states:
        etf_lines = []
        for s in ctx.etf_states:
            if s.state != "unknown":
                etf_lines.append(f"{s.code}={s.state_cn}(5日{s.ret_5d:+.1f}%)")
        if etf_lines:
            parts.append("关联ETF: " + " | ".join(etf_lines))

    # 全市场资金流信号
    _flow_cn = {
        "distribution": "主力派发+散户接盘(高危)",
        "smart_money_buying": "主力净流入+散户减仓(积极)",
        "consensus_buy": "主力散户同向净流入",
        "consensus_sell": "主力散户同向净流出",
    }.get(ctx.mkt_flow_signal, "")
    if _flow_cn:
        detail = f" — {ctx.mkt_flow_detail}" if ctx.mkt_flow_detail else ""
        parts.append(f"大盘资金流: {_flow_cn}{detail}")

    # 视觉研判
    if ctx.vision_summary:
        parts.append(f"图形研判: {ctx.vision_summary[:200]}")

    return "\n".join(parts)
