"""动态稀疏因子矩阵评分 (sparse_layered_score, v1).

设计参考 docs/sparse_factor_matrix_plan_2026_04_30.md.

核心思想:
  - 每只股票只有"在自己上下文下 active"的因子参与打分
  - 上下文 = 市值段 / PE 段 / 行业 / ETF 持有 / 资金流状态
  - 三维分层胜率几何平均 → effective win rate
  - 大于阈值才激活
  - ETF / regime 作为全局乘数
  - 资金流作为门控信号
  - 借 DS 证据理论的 K 冲突度作为置信度评估

输入:
  features: {factor_name: factor_value}
  context: {mv_seg, pe_seg, industry, etf_held(bool), mf_state(str)}
  validity_matrix: 从 output/factor_lab/validity_matrix.json 加载
  regime (可选): {trend: 'slow_bull', dispersion: 'high_industry'}

输出:
  {
    'layered_score': 50 + sum(deltas) clamp [0, 100],
    'sum_delta': float,
    'active_factors': [...]   每个 dict 含 name, q_bucket, w_eff, delta, reason
    'silent_factors': [...]   被静默的因子 + 原因
    'conflict_K': float       0~1, DS 冲突度
    'confidence': 'high'|'med'|'low',
    'gates_applied': [...],
    'regime_multipliers': {...},
  }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 默认参数
DEFAULT_MATRIX_PATH = Path(__file__).resolve().parents[2] / "output" / "factor_lab" / "validity_matrix.json"
DEFAULT_HOLD_BASE = 0.55          # 全市场基准 D+20 胜率
DEFAULT_SCORE_SCALE = 30          # 因子超额胜率 → 分数比例尺
DEFAULT_MAX_DELTA_PER_FACTOR = 8  # 每个因子最多 ±8 分 (避免单因子主导)


# ────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────

_CACHE: dict[str, Any] = {}


def load_validity_matrix(path: Path | str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_MATRIX_PATH
    key = str(p.resolve())
    if key in _CACHE:
        return _CACHE[key]
    if not p.exists():
        raise FileNotFoundError(f"validity_matrix.json 不存在: {p}. 先跑 factor_lab.py --phase 3")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    _CACHE[key] = data
    return data


# ────────────────────────────────────────────────────────
# 上下文分桶 (保持与 factor_lab.py 一致)
# ────────────────────────────────────────────────────────

MV_LABELS = ["20-50亿", "50-100亿", "100-300亿", "300-1000亿", "1000亿+"]
PE_LABELS = ["亏损", "0-15", "15-30", "30-50", "50-100", "100+"]


def bucket_mv(total_mv_wan: float | None) -> str | None:
    """total_mv 单位万元 → 市值段标签."""
    if total_mv_wan is None:
        return None
    bil = total_mv_wan / 1e4
    for (lo, hi), lab in zip([(0, 50), (50, 100), (100, 300),
                              (300, 1000), (1000, 1e9)], MV_LABELS):
        if lo <= bil < hi:
            return lab
    return None


def bucket_pe(pe: float | None) -> str | None:
    if pe is None:
        return None
    if pe < 0: return "亏损"
    if pe < 15: return "0-15"
    if pe < 30: return "15-30"
    if pe < 50: return "30-50"
    if pe < 100: return "50-100"
    return "100+"


# ────────────────────────────────────────────────────────
# Q 桶判定
# ────────────────────────────────────────────────────────

def find_q_bucket(value: float, q_thresholds: list[float] | None) -> int | None:
    """根据 [P20, P40, P60, P80] 边界返回 Q1-Q5."""
    if q_thresholds is None or len(q_thresholds) < 4 or value is None:
        return None
    if value < q_thresholds[0]: return 1
    if value < q_thresholds[1]: return 2
    if value < q_thresholds[2]: return 3
    if value < q_thresholds[3]: return 4
    return 5


# ────────────────────────────────────────────────────────
# 多维度合并
# ────────────────────────────────────────────────────────

def geometric_mean(values: list[float]) -> float:
    """忽略 None, 几何平均."""
    valid = [v for v in values if v is not None and v > 0]
    if not valid:
        return 0.0
    prod = 1.0
    for v in valid:
        prod *= v
    return prod ** (1.0 / len(valid))


def lookup_factor_segments(matrix: dict, factor: str, context: dict) -> dict:
    """返回该因子在 mv/pe/industry 三个维度上的 segment 数据 (如果存在)."""
    f_data = matrix.get("factors", {}).get(factor, {})
    out = {}

    if context.get("mv_seg"):
        seg = f_data.get("mv", {}).get(context["mv_seg"])
        if seg:
            out["mv"] = seg

    if context.get("pe_seg"):
        seg = f_data.get("pe", {}).get(context["pe_seg"])
        if seg:
            out["pe"] = seg

    if context.get("industry"):
        seg = f_data.get("industry", {}).get(context["industry"])
        if seg:
            out["industry"] = seg

    return out


def evaluate_q_signal(q_bucket: int | None, q_wins: list[float | None],
                       q3_win: float | None,
                       threshold: float = 0.04) -> tuple[bool, int, float]:
    """根据该股股实际 q 桶胜率 vs Q3 基线判定信号方向.

    返回 (是否激活, sign +1/-1/0, 该股实际 q 桶胜率)

    threshold = 该桶胜率 - Q3 基线 ≥ threshold 才算看多 (反之看空)
    避免 best/worst 桶误判 (大盘段胜率非单调 U 形)
    """
    if q_bucket is None or q_wins is None or len(q_wins) < q_bucket or q3_win is None:
        return False, 0, 0.0
    actual_w = q_wins[q_bucket - 1]
    if actual_w is None:
        return False, 0, 0.0
    diff = actual_w - q3_win
    if diff >= threshold:
        return True, +1, actual_w
    if diff <= -threshold:
        return True, -1, actual_w
    return False, 0, actual_w


# ────────────────────────────────────────────────────────
# 资金流门控
# ────────────────────────────────────────────────────────

def apply_mf_gate(delta: float, mf_state: str | None, factor_name: str) -> tuple[float, str | None]:
    """资金流门控 — 只对反转/动量类因子起作用.

    Returns: (调整后的 delta, gate_applied_label)
    """
    if mf_state is None:
        return delta, None

    # 反转类因子 (Q1 反弹) 需要主力净流入确认
    reversal_factors = {"ma_ratio_60", "ma_ratio_120", "channel_pos_60",
                         "ma20_ma60", "trix", "sump_20", "ma_ratio_20"}
    # 动量类因子 (Q5 强势) 同样需要主力流入确认
    momentum_factors = {"macd_hist", "rsi_24", "mfi_14", "ppo", "kama_dist"}

    if factor_name not in (reversal_factors | momentum_factors):
        return delta, None

    # 信号确认
    if delta > 0:
        if mf_state in ("main_inflow", "main_inflow_3d"):
            return delta, "mf_inflow_confirm"
        if mf_state == "main_outflow_3d":
            return 0.0, "mf_outflow_veto"   # 主力出货 → 看多信号否决
        return delta * 0.5, "mf_neutral"
    elif delta < 0:
        if mf_state == "main_outflow_3d":
            return delta, "mf_outflow_confirm"
        if mf_state in ("main_inflow", "main_inflow_3d"):
            return 0.0, "mf_inflow_veto"
        return delta * 0.5, "mf_neutral"

    return delta, None


# ────────────────────────────────────────────────────────
# 全局调节 (regime / ETF)
# ────────────────────────────────────────────────────────

# 因子分类 (粗分, 用于 regime 调节)
MOMENTUM_FACTORS = {"macd_hist", "macd", "rsi_24", "rsi_14", "rsi_6",
                     "mfi_14", "ppo", "trix", "kama_dist",
                     "ma5_ma20", "kdj_k", "kdj_d", "stochrsi_k",
                     "lr_slope_20", "lr_slope_60", "lr_angle_20",
                     "cmo_14", "aroon_up", "aroon_osc",
                     "roc_10", "roc_20"}
REVERSAL_FACTORS = {"ma_ratio_60", "ma_ratio_120", "channel_pos_60",
                     "ma20_ma60", "sump_20", "sumn_20", "rank_20",
                     "rsv_20", "wr_14", "qtlu_20", "qtld_20"}


def regime_multiplier(factor_name: str, regime: dict | None) -> float:
    if regime is None:
        return 1.0
    m = 1.0
    trend = regime.get("trend")
    if trend == "slow_bull":
        # 慢牛: 动量因子加权, 反转因子降权
        if factor_name in MOMENTUM_FACTORS:
            m *= 1.10
        elif factor_name in REVERSAL_FACTORS:
            m *= 0.90
    elif trend == "fast_bull":
        if factor_name in MOMENTUM_FACTORS:
            m *= 1.20
        elif factor_name in REVERSAL_FACTORS:
            m *= 0.80
    elif trend == "bear":
        if factor_name in MOMENTUM_FACTORS:
            m *= 0.80
        elif factor_name in REVERSAL_FACTORS:
            m *= 1.20
    return m


def etf_multiplier(factor_name: str, mv_seg: str | None, etf_held: bool) -> float:
    if not etf_held:
        return 1.0
    if mv_seg in ("300-1000亿", "1000亿+"):
        # 大盘股 + ETF 持有 → 动量因子增强 (机构惯性)
        if factor_name in MOMENTUM_FACTORS:
            return 1.30
    elif mv_seg in ("20-50亿", "50-100亿"):
        # 小盘 + ETF → 流动性好, 反转效应减弱
        if factor_name in REVERSAL_FACTORS:
            return 0.70
    return 1.0


# ────────────────────────────────────────────────────────
# DS 证据理论 — 借 K 冲突度
# ────────────────────────────────────────────────────────

def compute_conflict_K(active_factors: list[dict]) -> float:
    """计算 DS 冲突度 K = sum of m_i(long) × m_j(short) for i!=j.

    每个激活因子换算成 mass:
      mass(做多) = max(0, +delta) / max_score
      mass(做空) = max(0, -delta) / max_score
      mass(未知) = 1 - long - short

    K 越大 = 因子间冲突越大 = 信号越不可靠.
    """
    if len(active_factors) < 2:
        return 0.0

    max_d = max(abs(f["delta"]) for f in active_factors)
    if max_d <= 0:
        return 0.0

    masses = []
    for f in active_factors:
        d = f["delta"]
        m_long = max(0, d) / max_d
        m_short = max(0, -d) / max_d
        masses.append((m_long, m_short))

    K = 0.0
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            K += masses[i][0] * masses[j][1]
            K += masses[i][1] * masses[j][0]

    # 归一化: K 上界是 ~N*(N-1)/2
    n = len(masses)
    K_max = n * (n - 1) / 2.0
    return min(K / K_max, 1.0) if K_max > 0 else 0.0


def confidence_label(K: float, n_active: int) -> str:
    if n_active == 0:
        return "none"
    if n_active < 3:
        return "low"      # 激活太少
    if K > 0.45:
        return "low"      # 冲突大
    if K > 0.25:
        return "med"
    return "high"


# ────────────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────────────

def compute_sparse_layered_score(
    features: dict[str, float],
    context: dict[str, Any],
    matrix: dict | None = None,
    regime: dict | None = None,
    mf_state: str | None = None,
    score_scale: float = DEFAULT_SCORE_SCALE,
    max_delta_per_factor: float = DEFAULT_MAX_DELTA_PER_FACTOR,
) -> dict[str, Any]:
    """主打分入口.

    features: 因子值字典 {ma_ratio_60: -0.06, mfi_14: 75, ...}
    context:  {mv_seg, pe_seg, industry, etf_held}
    matrix:   validity_matrix.json 内容; 不传则自动加载
    regime:   {trend, dispersion}
    mf_state: 资金流状态字符串
    """
    if matrix is None:
        matrix = load_validity_matrix()

    factors_data = matrix.get("factors", {})
    base_win = matrix.get("meta", {}).get("base_win_rate", DEFAULT_HOLD_BASE)

    active_factors: list[dict] = []
    silent_factors: list[dict] = []
    gates_applied: list[str] = []

    for fname, fvalue in features.items():
        if fname not in factors_data:
            continue
        if fvalue is None:
            continue

        segs = lookup_factor_segments(matrix, fname, context)
        if not segs:
            silent_factors.append({"name": fname, "reason": "无任何分层段数据"})
            continue

        # 对每个维度: 该股实际 q 桶胜率 - Q3 基线 = 超额(带符号)
        seg_excesses = []      # 各 dim 的 (actual_w - q3_w), 正=看多, 负=看空
        seg_actual_wins = []   # 各 dim 的 actual_w (展示用)
        seg_q3_wins = []       # 各 dim 的 q3_w (展示用)
        seg_q_buckets = []
        any_active_segment = False

        for dim_name in ("mv", "industry", "pe"):
            if dim_name not in segs:
                continue
            seg_data = segs[dim_name]
            q_thresholds = seg_data.get("q_thresholds")
            if not q_thresholds:
                continue
            q_b = find_q_bucket(fvalue, q_thresholds)
            if q_b is None:
                continue
            q_wins_list = seg_data.get("q_wins") or []
            if len(q_wins_list) < q_b:
                continue
            actual_w = q_wins_list[q_b - 1]
            q3_w = seg_data.get("q3_win")
            if actual_w is None or q3_w is None:
                continue
            seg_q_buckets.append(q_b)
            diff = actual_w - q3_w
            # 显著差距才记录
            if abs(diff) >= 0.04:
                seg_excesses.append(diff)
                seg_actual_wins.append(actual_w)
                seg_q3_wins.append(q3_w)
                if seg_data.get("active"):
                    any_active_segment = True

        if not seg_excesses:
            actual_q_str = ",".join(f"Q{q}" for q in seg_q_buckets) if seg_q_buckets else "?"
            silent_factors.append({
                "name": fname,
                "actual_q": actual_q_str,
                "reason": "该股 q 桶胜率与 Q3 基线差距 < 4pp (非显著极端桶)",
            })
            continue

        # 检查方向一致性: 全部正 (看多) 或全部负 (看空)
        all_pos = all(d > 0 for d in seg_excesses)
        all_neg = all(d < 0 for d in seg_excesses)
        if not (all_pos or all_neg):
            silent_factors.append({
                "name": fname,
                "actual_q": ",".join(f"Q{q}" for q in seg_q_buckets),
                "excesses": [round(d * 100, 1) for d in seg_excesses],
                "reason": "维度间方向不一致 (内部冲突)",
            })
            continue

        # 至少 1 维度激活 (validity_matrix Q3+5pp 规则), 提高信噪比
        if not any_active_segment:
            silent_factors.append({
                "name": fname,
                "reason": "无任何 segment 激活",
            })
            continue

        sign = 1 if all_pos else -1
        actual_q_bucket = seg_q_buckets[0]

        # delta = 算术平均超额 × score_scale (excess 已带符号, 不再 × sign)
        mean_excess = sum(seg_excesses) / len(seg_excesses)
        w_eff = sum(seg_actual_wins) / len(seg_actual_wins)   # 展示用算术均值
        q3_eff = sum(seg_q3_wins) / len(seg_q3_wins)
        raw_delta = mean_excess * score_scale

        # 全局调节: regime + ETF
        m_regime = regime_multiplier(fname, regime)
        m_etf = etf_multiplier(fname, context.get("mv_seg"), context.get("etf_held", False))
        delta = raw_delta * m_regime * m_etf

        # 资金流门控
        delta, gate = apply_mf_gate(delta, mf_state, fname)
        if gate:
            gates_applied.append(f"{fname}:{gate}")

        # clamp
        delta = max(-max_delta_per_factor, min(max_delta_per_factor, delta))

        if abs(delta) < 0.1:
            silent_factors.append({
                "name": fname, "reason": f"delta 太小 ({delta:+.2f})",
            })
            continue

        active_factors.append({
            "name": fname,
            "q_bucket": f"Q{actual_q_bucket}",
            "w_eff": round(w_eff, 4),
            "q3_eff": round(q3_eff, 4),
            "delta": round(delta, 2),
            "sign": sign,
            "regime_mul": round(m_regime, 2),
            "etf_mul": round(m_etf, 2),
            "reason": _build_reason(fname, context, actual_q_bucket, sign,
                                     w_eff, q3_eff, m_regime, m_etf, gate),
        })

    # 综合
    sum_delta = sum(f["delta"] for f in active_factors)
    layered_score = max(0, min(100, 50 + sum_delta))

    K = compute_conflict_K(active_factors)
    conf = confidence_label(K, len(active_factors))

    return {
        "layered_score": round(layered_score, 2),
        "sum_delta": round(sum_delta, 2),
        "n_active": len(active_factors),
        "n_silent": len(silent_factors),
        "active_factors": active_factors,
        "silent_factors": silent_factors[:30],   # 限制展示, 避免爆炸
        "conflict_K": round(K, 4),
        "confidence": conf,
        "gates_applied": gates_applied,
        "context": context,
        "regime": regime or {},
        "mf_state": mf_state,
    }


# ────────────────────────────────────────────────────────
# Features 提取 (从 enrich 字典)
# ────────────────────────────────────────────────────────

def _safe_float(v):
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:   # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def extract_features_from_enrich(enrich: dict) -> dict[str, float]:
    """从 tushare_enrich 字典抽取 sparse_layered 评分需要的核心因子值.

    覆盖 ~17 个 high-激活率因子 (Top 25 中能从 enrich 拿到的部分).
    enrich["tushare_factors"]: 最新一天的 stk_factor_pro 汇总
    enrich["tushare_factors_raw"]: 近 60 天原始数据 (用于现算时序因子)
    """
    features: dict[str, float] = {}
    tsf = enrich.get("tushare_factors") or {}
    raw = enrich.get("tushare_factors_raw") or []

    close = _safe_float(tsf.get("close_qfq"))
    ma5 = _safe_float(tsf.get("ma5"))
    ma20 = _safe_float(tsf.get("ma20"))
    ma60 = _safe_float(tsf.get("ma60"))
    ma250 = _safe_float(tsf.get("ma250"))

    # 均线偏离类
    if close and ma60 and ma60 > 0:
        features["ma_ratio_60"] = close / ma60 - 1
    if close and ma250 and ma250 > 0:
        features["ma_ratio_120"] = close / ma250 - 1   # 用 250 替代 120
    if ma20 and ma60 and ma60 > 0:
        features["ma20_ma60"] = ma20 / ma60 - 1
    if ma5 and ma20 and ma20 > 0:
        features["ma5_ma20"] = ma5 / ma20 - 1

    # MACD / RSI / 动量
    for src, dst in [
        ("macd_hist", "macd_hist"),
        ("macd_dif", "macd"),
        ("macd_dea", "macd_signal"),
        ("rsi24", "rsi_24"),
        ("rsi12", "rsi_14"),    # 12 替代 14
        ("rsi6", "rsi_6"),
        ("mfi", "mfi_14"),
        ("trix", "trix"),
        ("cci", "cci_14"),
        ("wr", "wr_14"),
        ("kdj_k", "kdj_k"),
        ("kdj_d", "kdj_d"),
    ]:
        v = _safe_float(tsf.get(src))
        if v is not None:
            features[dst] = v

    # 波动率
    atr = _safe_float(tsf.get("atr"))
    if atr is not None and close and close > 0:
        features["atr_pct"] = atr / close
        features["natr_14"] = atr / close * 100

    # 布林
    boll_u = _safe_float(tsf.get("boll_upper"))
    boll_l = _safe_float(tsf.get("boll_lower"))
    boll_m = _safe_float(tsf.get("boll_mid"))
    if boll_u and boll_l and boll_m and boll_m > 0:
        features["boll_width"] = (boll_u - boll_l) / boll_m
    if boll_u and boll_l and close and (boll_u - boll_l) > 0:
        features["boll_pct"] = (close - boll_l) / (boll_u - boll_l)

    # 时序类 (从 raw 60 天算)
    if len(raw) >= 21:
        # 取最近 60 天
        rs = sorted(raw, key=lambda r: r.get("trade_date", ""))[-60:]
        # close 序列 (qfq)
        closes = [_safe_float(r.get("close_qfq")) for r in rs]
        closes = [c for c in closes if c is not None]
        # pct_chg 序列
        pcts = [_safe_float(r.get("pct_chg")) for r in rs]

        # sump_20 / sumn_20: 最近 20 天累计正/负涨幅 (注意 pct_chg 单位是%)
        last_20_pcts = [p for p in pcts[-20:] if p is not None]
        if len(last_20_pcts) >= 15:
            features["sump_20"] = sum(p for p in last_20_pcts if p > 0)
            features["sumn_20"] = sum(-p for p in last_20_pcts if p < 0)
            features["sumd_20"] = features["sump_20"] - features["sumn_20"]
            # cntp/cntn/cntd
            features["cntp_20"] = float(sum(1 for p in last_20_pcts if p > 0))
            features["cntn_20"] = float(sum(1 for p in last_20_pcts if p < 0))
            features["cntd_20"] = features["cntp_20"] - features["cntn_20"]

        # channel_pos_60: close 在 60 日 [min, max] 中的位置
        if len(closes) >= 30:
            cmin = min(closes)
            cmax = max(closes)
            if cmax > cmin and close is not None:
                features["channel_pos_60"] = (close - cmin) / (cmax - cmin)

        # rank_20: close 在最近 20 天里的分位
        last_20_closes = [c for c in closes[-20:] if c is not None]
        if len(last_20_closes) >= 15 and close is not None:
            n_lower = sum(1 for c in last_20_closes if c < close)
            features["rank_20"] = n_lower / len(last_20_closes)
            features["rsv_20"] = (close - min(last_20_closes)) / (
                max(last_20_closes) - min(last_20_closes) + 1e-12)

        # qtlu_20 / qtld_20: 20 日 80% / 20% 分位 / close - 1
        if len(last_20_closes) >= 15 and close is not None and close > 0:
            import statistics
            q_sorted = sorted(last_20_closes)
            n_q = len(q_sorted)
            features["qtlu_20"] = q_sorted[int(n_q * 0.8)] / close - 1
            features["qtld_20"] = q_sorted[int(n_q * 0.2)] / close - 1

    # ht_trendmode: enrich 没有, 跳过 (可能未来加 talib.HT_TRENDMODE)
    return features


def derive_context_from_enrich(enrich: dict, industry: str | None = None,
                                etf_held: bool = False) -> dict:
    """从 enrich 抽取 context (mv_seg / pe_seg / industry / etf_held).

    industry 由调用方传入 (一般来自 stock_basic 的 industry 字段).
    """
    tsf = enrich.get("tushare_factors") or {}
    total_mv = _safe_float(tsf.get("total_mv"))
    pe_ttm = _safe_float(tsf.get("pe_ttm"))

    return {
        "mv_seg": bucket_mv(total_mv),
        "pe_seg": bucket_pe(pe_ttm),
        "industry": industry,
        "etf_held": etf_held,
        "_raw": {"total_mv": total_mv, "pe_ttm": pe_ttm},
    }


def derive_mf_state(enrich: dict) -> str | None:
    """从 enrich 的资金流字典推导 mf_state."""
    mf = enrich.get("tushare_moneyflow") or {}
    if not mf:
        return None
    # 优先用主力连续天数
    consec = int(mf.get("consecutive_main_days", 0) or 0)
    main_rate_ma3 = _safe_float(mf.get("main_rate_ma3"))
    main_net_ma3 = _safe_float(mf.get("main_net_ma3"))

    inflow = (main_rate_ma3 is not None and main_rate_ma3 > 0) or \
             (main_net_ma3 is not None and main_net_ma3 > 0)
    outflow = (main_rate_ma3 is not None and main_rate_ma3 < 0) or \
              (main_net_ma3 is not None and main_net_ma3 < 0)

    if consec >= 3 and inflow:
        return "main_inflow_3d"
    if consec >= 2 and outflow:
        return "main_outflow_3d"
    if inflow:
        return "main_inflow"
    if outflow:
        return "main_outflow"
    return None


def _build_reason(fname: str, context: dict, q: int, sign: int,
                   w_eff: float, q3_eff: float, m_regime: float,
                   m_etf: float, gate: str | None) -> str:
    parts = []
    direction = "看多" if sign > 0 else "看空"
    parts.append(f"{direction}@Q{q}")
    if context.get("mv_seg"):
        parts.append(f"mv={context['mv_seg']}")
    if context.get("industry"):
        parts.append(f"行业={context['industry']}")
    parts.append(f"W={w_eff*100:.1f}% vs Q3={q3_eff*100:.1f}%")
    if m_regime != 1.0:
        parts.append(f"regime×{m_regime:.2f}")
    if m_etf != 1.0:
        parts.append(f"etf×{m_etf:.2f}")
    if gate:
        parts.append(f"gate={gate}")
    return " ".join(parts)
