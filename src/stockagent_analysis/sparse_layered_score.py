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
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 优先 3 年数据 (factor_lab_3y), 退化到 1 年 (factor_lab)
_MATRIX_3Y = _PROJECT_ROOT / "output" / "factor_lab_3y" / "validity_matrix.json"
_MATRIX_1Y = _PROJECT_ROOT / "output" / "factor_lab" / "validity_matrix.json"
DEFAULT_MATRIX_PATH = _MATRIX_3Y if _MATRIX_3Y.exists() else _MATRIX_1Y
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
        # A股无做空机制: 负向信号=减仓/回避，主力净流入不应否决减仓建议
        if mf_state == "main_outflow_3d":
            return delta, "mf_outflow_confirm"
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

def eb_shrink_win(win_rate: float, q3_win: float, n_samples: int | None, n0: int = 3000) -> float:
    """Empirical Bayes 收缩: 样本少的格子(industry维度)向Q3基线收缩。

    n0=3000 → N≈6000时收缩33%, N≈300000时收缩1%.
    仅对 industry 维度有实质效果(N~5000-30000),mv/pe维度(N~300000)几乎无影响。
    """
    if not n_samples or n_samples <= 0:
        return win_rate
    lam = n0 / (n_samples + n0)
    return (1.0 - lam) * win_rate + lam * q3_win


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
    use_eb: bool = False,
    use_k_weight: bool = True,
    score_mode: str = "win",
) -> dict[str, Any]:
    """主打分入口.

    features: 因子值字典 {ma_ratio_60: -0.06, mfi_14: 75, ...}
    context:  {mv_seg, pe_seg, industry, etf_held}
    matrix:   validity_matrix.json 内容; 不传则自动加载
    regime:   {trend, dispersion}
    mf_state: 资金流状态字符串
    score_mode: "win" = 用 q_wins (胜率, 默认) | "avg" = 用 q_avgs (平均涨幅%)
                avg 模式下: 阈值 0.5pp, score_scale 自动 /10 适配单位
    """
    if matrix is None:
        matrix = load_validity_matrix()

    # 模式相关参数
    if score_mode == "avg":
        wins_key      = "q_avgs"          # q_avgs 是 % 单位 (4.66 = 4.66%)
        seg_threshold = 1.5               # 1.5pp 涨幅超额才算显著 (q_avgs 噪声大)
        eff_scale     = score_scale / 30  # 1.5pp 涨幅 → delta ~1.5
    else:
        wins_key      = "q_wins"
        seg_threshold = 0.04
        eff_scale     = score_scale

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
            q_wins_list = seg_data.get(wins_key) or []
            if len(q_wins_list) < q_b:
                continue
            actual_w = q_wins_list[q_b - 1]
            # q3 基线: avg 模式从 q_avgs[2] 现算; win 模式直接读 q3_win
            if score_mode == "avg":
                q3_w = q_wins_list[2] if len(q_wins_list) >= 3 else None
            else:
                q3_w = seg_data.get("q3_win")
            if actual_w is None or q3_w is None:
                continue
            # EB 收缩: 对 industry 维度(小样本)收缩向 Q3 基线
            if use_eb:
                q_ns = seg_data.get("q_ns") or []
                n_samp = q_ns[q_b - 1] if len(q_ns) >= q_b else None
                actual_w = eb_shrink_win(actual_w, q3_w, n_samp)
            seg_q_buckets.append(q_b)
            diff = actual_w - q3_w
            # 显著差距才记录 (mode 相关阈值)
            if abs(diff) >= seg_threshold:
                seg_excesses.append(diff)
                seg_actual_wins.append(actual_w)
                seg_q3_wins.append(q3_w)
                # avg 模式: 现算激活 (best_avg - q3_avg >= 1.5pp 且 best_avg >= 0)
                if score_mode == "avg":
                    valid_avgs = [v for v in q_wins_list if v is not None]
                    if valid_avgs:
                        best_avg = max(valid_avgs)
                        if (best_avg - q3_w) >= 1.5 and best_avg >= 0.0:
                            any_active_segment = True
                elif seg_data.get("active"):
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

        # delta = 算术平均超额 × eff_scale (avg 模式自动适配单位)
        mean_excess = sum(seg_excesses) / len(seg_excesses)
        w_eff = sum(seg_actual_wins) / len(seg_actual_wins)
        q3_eff = sum(seg_q3_wins) / len(seg_q3_wins)
        raw_delta = mean_excess * eff_scale

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

    K = compute_conflict_K(active_factors)
    conf = confidence_label(K, len(active_factors))

    # K → 动态权重: 冲突大时整体收缩向中性
    k_weight = 1.0
    if use_k_weight:
        if K > 0.45:
            k_weight = 0.60   # 高冲突, 大幅收缩
        elif K > 0.25:
            k_weight = 0.85   # 中等冲突, 小幅收缩
    sum_delta_adj = sum_delta * k_weight

    layered_score = max(0, min(100, 50 + sum_delta_adj))

    if layered_score >= 65:
        trade_signal = "入场/加仓"
    elif layered_score >= 50:
        trade_signal = "持仓观察"
    elif layered_score >= 35:
        trade_signal = "减仓/观望"
    else:
        trade_signal = "清仓/回避"

    # ── LGBM 预测 (可选) + 双独立评分 (入场 + 回撤风险) ───────────────
    lgbm_result = None
    clean_result = None
    maxgain_result = None
    uptrend_result = None
    risk_ml_result = None
    dual_result = None
    try:
        from . import lgbm_predictor
        lgbm_extras = {}
        raw = context.get("_raw") or {}
        for k in ("total_mv", "pe", "pe_ttm", "market_score_adj",
                  "mf_divergence", "mf_strength", "mf_consecutive"):
            if k in raw:
                lgbm_extras[k] = raw[k]
            elif k in features:
                lgbm_extras[k] = features[k]
        ind = context.get("industry", "")
        lgbm_result    = lgbm_predictor.predict(features, ind, extras=lgbm_extras)
        clean_result   = lgbm_predictor.predict_clean(features, ind, extras=lgbm_extras)
        maxgain_result = lgbm_predictor.predict_maxgain(features, ind, extras=lgbm_extras)
        uptrend_result = lgbm_predictor.predict_uptrend(features, ind, extras=lgbm_extras)
        risk_ml_result = lgbm_predictor.predict_risk(features, ind, extras=lgbm_extras)
        # v2 生产模型: 双向评分 (基于 r10/r20 ALL + sell_10/sell_20)
        dual_result = lgbm_predictor.predict_dual(features, ind, extras=lgbm_extras)
    except Exception as _e:
        pass

    # Moneyflow 信号摘要 (从 features 提取展示用, 不影响评分)
    moneyflow_summary = _extract_moneyflow_summary(features)

    # 两个独立评分 (用户自己结合判断)
    entry_eval = _compute_entry_score(layered_score, lgbm_result, maxgain_result,
                                       clean_result, context, uptrend_result)
    risk_eval  = _compute_risk_score(layered_score, lgbm_result, maxgain_result,
                                       context, risk_ml_result)

    return {
        "layered_score": round(layered_score, 2),
        "trade_signal": trade_signal,
        "lgbm": lgbm_result,
        "clean": clean_result,
        "maxgain": maxgain_result,
        "uptrend": uptrend_result,    # 起涨点检测器 (注意: 之前数据有泄漏)
        "moneyflow": moneyflow_summary,
        "dual": dual_result,           # v2 双向评分 (r10/r20 + sell_10/sell_20, 真实 IC 0.073)
        "entry_score":  entry_eval,    # 0-100, 高=适合买入 (规则推导, 后续可被 dual 替代)
        "risk_score":   risk_eval,     # 0-100, 高=浮亏/回撤风险大
        "sum_delta": round(sum_delta, 2),
        "sum_delta_raw": round(sum_delta, 2),
        "k_weight": round(k_weight, 3),
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


# ────────────────────────────────────────────────────────
# 可解释性辅助 (A): explain() 方法
# ────────────────────────────────────────────────────────

def explain_layered_score(result: dict, verbose: bool = True) -> str:
    """生成完整人类可读的解释报告 (A 功能).

    输入: compute_sparse_layered_score 返回的 dict
    输出: markdown 多行字符串, 可直接打印或写入文件
    """
    if not result:
        return "(空结果)"

    score = result.get("layered_score", 50)
    sum_d = result.get("sum_delta", 0)
    n_a = result.get("n_active", 0)
    n_s = result.get("n_silent", 0)
    K = result.get("conflict_K", 0)
    conf = result.get("confidence", "?")
    ctx = result.get("context", {})
    regime = result.get("regime", {})
    mf_state = result.get("mf_state")

    lines = [
        "=" * 72,
        f"sparse_layered_score 解释报告",
        "=" * 72,
        "",
        f"【综合分】 {score:.1f} = 50 (基线) + Δ {sum_d:+.2f}  →  {result.get('trade_signal', '')}",
        f"【激活】 {n_a} 因子有效 / {n_s} 因子静默",
        f"【信号一致性】 {conf}  (DS 冲突度 K = {K:.3f})",
        "",
        "## 上下文 (本只股票当下属于哪些段)",
        f"  市值段:    {ctx.get('mv_seg', '?')}",
        f"  PE 段:     {ctx.get('pe_seg', '?')}",
        f"  行业:      {ctx.get('industry', '?')}",
        f"  ETF 持有: {ctx.get('etf_held', False)}",
        f"  资金流:    {mf_state or '未推导'}",
    ]
    if regime:
        lines.append(f"  大盘 regime: trend={regime.get('trend')} dispersion={regime.get('dispersion')}")
    lines.append("")

    # 激活因子完整溯源
    if result.get("active_factors"):
        lines.append("## 激活因子 (每项的 delta 都能完整溯源)")
        lines.append("")
        for i, f in enumerate(result["active_factors"], 1):
            sgn_arrow = "↑看多" if f.get("sign", 0) > 0 else "↓看空"
            w = f.get("w_eff", 0)
            q3 = f.get("q3_eff", 0)
            excess = w - q3
            d = f.get("delta", 0)
            mr = f.get("regime_mul", 1.0)
            me = f.get("etf_mul", 1.0)
            lines.append(f"  {i}. {f['name']}  ({sgn_arrow})  delta = {d:+.2f}")
            lines.append(f"     ├─ 该股股 q 桶: {f.get('q_bucket', '?')}")
            lines.append(f"     ├─ 该桶在该上下文胜率: W = {w*100:.1f}%")
            lines.append(f"     ├─ Q3 中位桶基线:     Q3 = {q3*100:.1f}%")
            lines.append(f"     ├─ 超额胜率:           +{excess*100:.1f}pp" if excess > 0 else
                         f"     ├─ 超额胜率:           {excess*100:.1f}pp")
            extras = []
            if mr != 1.0:
                extras.append(f"regime×{mr:.2f}")
            if me != 1.0:
                extras.append(f"etf×{me:.2f}")
            if extras:
                lines.append(f"     ├─ 调节器:            {' + '.join(extras)}")
            lines.append(f"     └─ 解释: {f.get('reason', '')}")
            lines.append("")

    # 静默因子摘要
    if result.get("silent_factors") and verbose:
        silent = result["silent_factors"]
        lines.append(f"## 静默因子 ({len(silent)} 个, 不参与打分)")
        lines.append("")
        # 按 reason 归类
        from collections import Counter
        by_reason = Counter()
        for s in silent:
            r = s.get("reason", "?")
            # 截短 reason 取关键词
            if "差距" in r:
                by_reason["q 桶超额 < 4pp (无显著信号)"] += 1
            elif "无任何分层段" in r:
                by_reason["上下文段缺数据"] += 1
            elif "无任何 segment 激活" in r:
                by_reason["validity_matrix 中段未激活"] += 1
            elif "delta 太小" in r:
                by_reason["调节后 delta 太小"] += 1
            elif "方向不一致" in r:
                by_reason["维度间方向冲突"] += 1
            else:
                by_reason[r[:40]] += 1
        for reason, cnt in by_reason.most_common():
            lines.append(f"  - {reason}: {cnt} 个因子")
        lines.append("")

    # 资金流门控
    if result.get("gates_applied"):
        lines.append("## 资金流门控 (主力进/出场对因子的修正)")
        for g in result["gates_applied"]:
            lines.append(f"  - {g}")
        lines.append("")

    # K 冲突解释
    lines.append("## DS 冲突度 K 解释")
    if n_a < 2:
        lines.append(f"  K = {K:.3f}: 激活因子不足 2 个, 无冲突可言")
    elif K < 0.20:
        lines.append(f"  K = {K:.3f}: 多因子方向高度一致 → 信号可信度高")
    elif K < 0.45:
        lines.append(f"  K = {K:.3f}: 部分因子有冲突 → 信号需结合其他证据判断")
    else:
        lines.append(f"  K = {K:.3f}: 因子间冲突大 → 综合分不可靠, 应等待信号一致后再行动")
    lines.append("")

    # 决策建议
    lines.append("## 决策含义")
    if conf == "none":
        lines.append("  → 没有任何因子激活, 系统无观点, 等待更多数据")
    elif score >= 60 and conf == "high":
        lines.append(f"  → 强多信号 ({score:.0f} 分, 高一致性), 可作为加仓/建仓依据")
    elif score >= 60 and conf == "med":
        lines.append(f"  → 多头信号 ({score:.0f} 分, 中等一致性), 建议结合其他因素")
    elif score >= 60 and conf == "low":
        lines.append(f"  → 因子打分多但内部冲突, **不建议作为单独决策依据**")
    elif score <= 40 and conf == "high":
        lines.append(f"  → 强空信号 ({score:.0f} 分, 高一致性), 可作为减仓/做空依据")
    elif score <= 40:
        lines.append(f"  → 偏空信号, 但一致性 {conf}, 谨慎对待")
    else:
        lines.append(f"  → 中性信号 ({score:.0f} 分), 不构成明确方向")
    lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ────────────────────────────────────────────────────────
# LLM context 增强 (B): 给 4 专家用的 markdown
# ────────────────────────────────────────────────────────

def render_for_llm_prompt(result: dict, max_active: int = 12) -> str:
    """生成 LLM-friendly markdown, 嵌入 4 专家 context.

    比简单的展示更结构化 + 直接告诉 LLM 怎么用这些信号.
    """
    if not result or result.get("n_active", 0) == 0:
        return ""

    score = result.get("layered_score", 50)
    sum_d = result.get("sum_delta", 0)
    n_a = result.get("n_active", 0)
    K = result.get("conflict_K", 0)
    conf = result.get("confidence", "?")
    ctx = result.get("context", {})

    # 分类: 看多 vs 看空因子
    longs = [f for f in result.get("active_factors", []) if f.get("delta", 0) > 0]
    shorts = [f for f in result.get("active_factors", []) if f.get("delta", 0) < 0]

    lgbm    = result.get("lgbm") or {}
    clean   = result.get("clean") or {}
    maxgain = result.get("maxgain") or {}
    dual    = result.get("dual") or {}
    entry   = result.get("entry_score") or {}
    risk    = result.get("risk_score") or {}

    lines = [
        "## 量化双独立评分 (买入信号 + 风险信号, 由用户综合决策)",
        "",
    ]
    # Regime 上下文 (市场背景)
    regime = _load_regime_today()
    if regime and regime.get("regime"):
        regime_cn = {
            "bull_policy":       "🚀 政策催化牛",
            "bull_fast":         "📈 快牛 (流动性驱动)",
            "bull_slow_diverge": "🌊 慢牛分化 (个股分化)",
            "bear":              "🐻 熊市",
            "sideways":          "↔ 震荡市",
            "mixed":             "🔀 过渡期",
        }.get(regime["regime"], regime["regime"])
        r5  = regime.get("ret_5d");  r5s = f"{r5*100:+.1f}%"  if r5  is not None else "?"
        r20 = regime.get("ret_20d"); r20s = f"{r20*100:+.1f}%" if r20 is not None else "?"
        r60 = regime.get("ret_60d"); r60s = f"{r60*100:+.1f}%" if r60 is not None else "?"
        rsi = regime.get("rsi14");   rsis = f"{rsi:.0f}"     if rsi is not None else "?"
        lines.append(f"### 🌐 当前市场状态: **{regime_cn}**")
        lines.append(f"  · 沪深300 5日 {r5s} / 20日 {r20s} / 60日 {r60s} / RSI14 {rsis}")
        # Regime 适用建议 (基于 36 次配对实证 vs 沪深300 真 alpha)
        regime_advice = {
            "bull_policy": "⚠ 政策驱动期, 直接买 ETF 完胜选股 (复利效应大), 系统暂禁用",
            "bull_fast": "⚠ 流动性驱动, 普涨期, 直接买 ETF 收益更佳, 系统价值低",
            "bull_slow_diverge": "★ 主战场 — 配对实证胜率 56.2% (个股分化, 系统有微弱 alpha)",
            "bear": "❌ 熊市训练样本不足, 信号反向, 仅用 risk_score 避雷",
            "sideways": "❌ 震荡市配对胜率仅 33% (反向 alpha), 选股不如持币观望",
            "mixed": "★★ 过渡市配对胜率 63.6%, 系统在此 regime 真 alpha 最强",
        }.get(regime["regime"], "")
        if regime_advice:
            lines.append(f"  · 系统适用性: {regime_advice}")
        lines.append("")
    # v2 双向评分 (基于 r10/r20 ALL, IC 接近 SOTA)
    if dual and dual.get("ok"):
        bs = dual.get("buy_score", 50)
        ss = dual.get("sell_score", 50)
        r10 = dual.get("r10_pred", 0)
        r20 = dual.get("r20_pred", 0)
        sp10 = dual.get("sell_10_prob", 0) * 100
        sp20 = dual.get("sell_20_prob", 0) * 100
        # 买入档位
        bs_label = ("强烈看多" if bs >= 75 else "看多" if bs >= 60 else
                     "中性" if bs >= 45 else "看空" if bs >= 30 else "强烈看空")
        ss_label = ("高风险" if ss >= 65 else "中等风险" if ss >= 45 else "低风险")
        lines.append(f"### 🎯 v2 双向评分 (真实 IC 0.073, RankICIR 0.49)")
        lines.append(f"  📈 **买入分: {bs:.0f}/100** [{bs_label}] — r10预测 {r10:+.2f}%, r20预测 {r20:+.2f}%")
        lines.append(f"  📉 **卖出分: {ss:.0f}/100** [{ss_label}] — 10日跌破-5%概率 {sp10:.0f}%, 20日跌破-8%概率 {sp20:.0f}%")
        # 信号一致性
        if bs >= 65 and ss <= 35:
            lines.append(f"  ✓ 信号一致看多 (买入分高+卖出分低)")
        elif bs <= 35 and ss >= 65:
            lines.append(f"  ✓ 信号一致看空 (买入分低+卖出分高)")
        elif abs(bs - (100 - ss)) < 15:
            lines.append(f"  · 信号一致性中等")
        else:
            lines.append(f"  ⚠ 信号冲突 (买入与卖出评分不匹配, 谨慎决策)")
        lines.append("")

    # 买入评分 (传统规则推导, 保留兼容)
    if entry:
        es = entry.get("score", 50)
        lines.append(f"### 📈 买入评分(规则): **{es:.0f} / 100** — {entry.get('label', '')}")
        for r in entry.get("reasons", [])[:5]:
            lines.append(f"  · {r}")
        lines.append("")
    # 风险评分 (独立)
    if risk:
        rs = risk.get("score", 50)
        lines.append(f"### ⚠ 回撤风险评分: **{rs:.0f} / 100** — {risk.get('label', '')}")
        for r in risk.get("reasons", [])[:4]:
            lines.append(f"  · {r}")
        if risk.get("warning"):
            lines.append(f"  > {risk.get('warning')}")
        lines.append("")
    # 量化引擎细节
    lines.append(
        f"**Sparse 多因子: {score:.1f} 分** [{result.get('trade_signal', '')}] "
        f"(delta {sum_d:+.1f}) | 激活 {n_a} 因子 | K={K:.2f} | 一致性 **{conf}**"
    )
    if lgbm.get("ok"):
        lines.append(
            f"**LGBM r20 预测**: 涨幅 {lgbm.get('pred_r20', 0):+.2f}% · "
            f"上涨概率 **{lgbm.get('winprob', 0)*100:.1f}%** · 确信度 {lgbm.get('conf', '?')}"
        )
    if clean.get("ok"):
        cp = clean.get("clean_prob", 0)
        cp_label = "极强" if cp >= 0.30 else "强" if cp >= 0.20 else "中" if cp >= 0.10 else "弱"
        lines.append(
            f"**干净走势概率**: {cp*100:.1f}% **[{cp_label}]** "
            f"(基线 11%, 阈值: 20日内涨>=20% 且回撤>=-3%)"
        )
    if maxgain.get("ok"):
        pg = maxgain.get("pred_gain", 0)
        pdd = maxgain.get("pred_dd", 0)
        gdr = maxgain.get("gain_dd_ratio")
        line = (f"**ML 预测**: 期间最高涨幅 {pg:+.1f}% / 最大回撤 {pdd:+.1f}%")
        if gdr is not None:
            line += f" / 收益风险比 {gdr:.2f}"
        lines.append(line)
    # Moneyflow 资金分层信号
    mf = result.get("moneyflow")
    if mf:
        lines.append(
            f"**💰 资金分层 (5日)**: 主力 {mf.get('main_net_5d', 0):+.2f}亿 / "
            f"散户 {mf.get('sm_net_5d', 0):+.2f}亿 / "
            f"主力连续 {'流入' if mf.get('main_consec_in', 0) > mf.get('main_consec_out', 0) else '流出'} "
            f"{max(mf.get('main_consec_in', 0), mf.get('main_consec_out', 0))} 天"
        )
        if mf.get("pattern") and mf["pattern"] != "中性":
            lines.append(f"  → 资金模式: {mf['pattern']}")
    uptrend = result.get("uptrend") or {}
    if uptrend.get("ok"):
        up = uptrend.get("uptrend_prob", 0)
        tier = uptrend.get("lift_tier", "")
        # 基线 0.75%
        if up >= 0.30:
            label = f"⭐⭐⭐ {tier} (lift 16-17x, ~12% 真起涨)"
        elif up >= 0.15:
            label = f"⭐⭐ {tier} (lift 13x, ~10% 真起涨)"
        elif up >= 0.08:
            label = f"⭐ {tier} (lift 10x, ~7% 真起涨)"
        elif up >= 0.04:
            label = f"{tier} (lift 5x)"
        else:
            label = "底部 80% (起涨概率极低)"
        lines.append(f"**🎯 起涨点概率**: {up*100:.2f}% [{label}]")
    if lgbm.get("ok") or clean.get("ok") or maxgain.get("ok") or uptrend.get("ok"):
        lines.append("")
    lines.append(
        f"**股票上下文**: 市值={ctx.get('mv_seg')} · PE={ctx.get('pe_seg')} · "
        f"行业={ctx.get('industry')} · 资金流={result.get('mf_state', 'neutral')}"
    )
    lines.append("")

    if longs:
        lines.append(f"### 看多信号 ({len(longs)} 个)")
        lines.append("")
        lines.append("| 因子 | Q 桶 | 该桶胜率 | Q3 基线 | 超额 | delta |")
        lines.append("|---|---|---|---|---|---|")
        for f in longs[:max_active]:
            w = f.get("w_eff", 0)
            q3 = f.get("q3_eff", 0)
            ex = w - q3
            lines.append(f"| {f['name']} | {f.get('q_bucket')} | "
                         f"{w*100:.1f}% | {q3*100:.1f}% | "
                         f"+{ex*100:.1f}pp | +{f.get('delta', 0):.2f} |")
        lines.append("")

    if shorts:
        lines.append(f"### 看空信号 ({len(shorts)} 个)")
        lines.append("")
        lines.append("| 因子 | Q 桶 | 该桶胜率 | Q3 基线 | 超额 | delta |")
        lines.append("|---|---|---|---|---|---|")
        for f in shorts[:max_active]:
            w = f.get("w_eff", 0)
            q3 = f.get("q3_eff", 0)
            ex = w - q3
            lines.append(f"| {f['name']} | {f.get('q_bucket')} | "
                         f"{w*100:.1f}% | {q3*100:.1f}% | "
                         f"{ex*100:.1f}pp | {f.get('delta', 0):.2f} |")
        lines.append("")

    # 给 LLM 的解读提示
    lines.append("### 这些信号告诉你什么")
    if conf == "high" and score >= 60:
        lines.append(f"- 多个独立因子高度一致看多, 该上下文 (mv/pe/行业) 历史胜率显著超基准")
        lines.append(f"- LLM 应在分析中给予 **较强多头权重**")
    elif conf == "high" and score <= 40:
        lines.append(f"- 多个因子一致看空, 历史风险率显著高于基准")
        lines.append(f"- LLM 应在分析中给予 **较强空头权重**")
    elif conf == "low":
        lines.append(f"- 因子内部冲突 (K={K:.2f}), 综合分不可作为单独依据")
        lines.append(f"- LLM 应保持中性, 重点看其他证据 (基本面/资金面)")
    else:
        lines.append(f"- 信号强度中等, LLM 可参考但不应过度依赖")

    if result.get("gates_applied"):
        gates_str = ", ".join(result['gates_applied'][:6])
        lines.append(f"- 资金流门控: {gates_str}")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────
# 对比工具 (C): compare 两只股票
# ────────────────────────────────────────────────────────

def compare_stocks(result_a: dict, label_a: str,
                    result_b: dict, label_b: str) -> str:
    """对比两只股票的因子激活差异."""
    score_a = result_a.get("layered_score", 50)
    score_b = result_b.get("layered_score", 50)
    n_a_a = result_a.get("n_active", 0)
    n_a_b = result_b.get("n_active", 0)
    K_a = result_a.get("conflict_K", 0)
    K_b = result_b.get("conflict_K", 0)
    conf_a = result_a.get("confidence", "?")
    conf_b = result_b.get("confidence", "?")

    af_a = {f["name"]: f for f in result_a.get("active_factors", [])}
    af_b = {f["name"]: f for f in result_b.get("active_factors", [])}

    common = set(af_a) & set(af_b)
    only_a = set(af_a) - set(af_b)
    only_b = set(af_b) - set(af_a)

    lines = [
        "=" * 72,
        f"对比: {label_a}  vs  {label_b}",
        "=" * 72,
        "",
        f"## 总览",
        f"{'指标':<20} {label_a:<20} {label_b:<20}",
        f"{'-'*60}",
        f"{'综合分':<20} {score_a:<20.1f} {score_b:<20.1f}",
        f"{'激活因子数':<20} {n_a_a:<20} {n_a_b:<20}",
        f"{'冲突度 K':<20} {K_a:<20.3f} {K_b:<20.3f}",
        f"{'信号一致性':<20} {conf_a:<20} {conf_b:<20}",
        "",
    ]

    ctx_a = result_a.get("context", {})
    ctx_b = result_b.get("context", {})
    lines.append(f"## 上下文对比")
    for k in ("mv_seg", "pe_seg", "industry"):
        va = str(ctx_a.get(k) or "?")
        vb = str(ctx_b.get(k) or "?")
        diff = "" if va == vb else "  <-- 不同"
        lines.append(f"  {k:<12}  {va:<20} | {vb}{diff}")
    lines.append("")

    # 共同因子的 delta 对比
    if common:
        lines.append(f"## 双方都激活的因子 ({len(common)} 个)")
        lines.append(f"{'因子':<22} {'A delta':<10} {'B delta':<10} {'差异':<10}")
        lines.append("-" * 55)
        for fname in sorted(common):
            d_a = af_a[fname].get("delta", 0)
            d_b = af_b[fname].get("delta", 0)
            diff = d_b - d_a
            lines.append(f"{fname:<22} {d_a:<+10.2f} {d_b:<+10.2f} {diff:<+10.2f}")
        lines.append("")

    if only_a:
        lines.append(f"## 仅 {label_a} 激活的因子 ({len(only_a)} 个)")
        for fname in sorted(only_a):
            f = af_a[fname]
            lines.append(f"  {fname:<22} delta={f.get('delta', 0):+.2f}  ({f.get('reason', '')[:50]})")
        lines.append("")

    if only_b:
        lines.append(f"## 仅 {label_b} 激活的因子 ({len(only_b)} 个)")
        for fname in sorted(only_b):
            f = af_b[fname]
            lines.append(f"  {fname:<22} delta={f.get('delta', 0):+.2f}  ({f.get('reason', '')[:50]})")
        lines.append("")

    # 结论
    lines.append("## 对比结论")
    diff_score = score_b - score_a
    if abs(diff_score) < 3:
        lines.append(f"  - 两只股票综合评分接近 (差 {diff_score:+.1f})")
    elif diff_score > 0:
        lines.append(f"  - {label_b} 评分高 {diff_score:.1f} 分, 总体优于 {label_a}")
    else:
        lines.append(f"  - {label_a} 评分高 {-diff_score:.1f} 分, 总体优于 {label_b}")

    if K_a < K_b:
        lines.append(f"  - {label_a} 信号更一致 (K {K_a:.2f} < {K_b:.2f}), 决策更可信")
    elif K_b < K_a:
        lines.append(f"  - {label_b} 信号更一致 (K {K_b:.2f} < {K_a:.2f}), 决策更可信")

    if ctx_a.get("mv_seg") != ctx_b.get("mv_seg"):
        lines.append(f"  - **市值段不同**, 适用因子集差异显著, 不可直接比较绝对分")

    lines.append("=" * 72)
    return "\n".join(lines)


# 起涨点样本矩阵 (从 extract_uptrend_starts 输出加载)
_SAMPLE_MATRIX: dict | None = None
# Regime 当日标签缓存
_REGIME_CACHE: dict | None = None


def _load_regime_today() -> dict | None:
    """加载今日 regime 标签 (取 daily_regime.parquet 最新行)."""
    global _REGIME_CACHE
    if _REGIME_CACHE is not None:
        return _REGIME_CACHE
    p = _PROJECT_ROOT / "output" / "regimes" / "daily_regime.parquet"
    if not p.exists():
        _REGIME_CACHE = {}
        return _REGIME_CACHE
    try:
        import pandas as pd
        df = pd.read_parquet(p)
        last = df.iloc[-1]
        _REGIME_CACHE = {
            "date": str(last.get("trade_date", "")),
            "regime": str(last.get("regime", "")),
            "regime_id": int(last.get("regime_id", 0)),
            "ret_5d":  float(last.get("ret_5d", 0)) if last.get("ret_5d") == last.get("ret_5d") else None,
            "ret_20d": float(last.get("ret_20d", 0)) if last.get("ret_20d") == last.get("ret_20d") else None,
            "ret_60d": float(last.get("ret_60d", 0)) if last.get("ret_60d") == last.get("ret_60d") else None,
            "rsi14":   float(last.get("rsi14", 0)) if last.get("rsi14") == last.get("rsi14") else None,
        }
    except Exception:
        _REGIME_CACHE = {}
    return _REGIME_CACHE


def _load_sample_matrix() -> dict:
    """加载 mv × pe 桶起涨点样本数 (用于稀疏桶警告)."""
    global _SAMPLE_MATRIX
    if _SAMPLE_MATRIX is not None:
        return _SAMPLE_MATRIX
    p = _PROJECT_ROOT / "output" / "uptrend_starts" / "sample_matrix.json"
    if p.exists():
        _SAMPLE_MATRIX = json.loads(p.read_text(encoding="utf-8"))
    else:
        _SAMPLE_MATRIX = {}
    return _SAMPLE_MATRIX


def _bucket_sample_count(mv_seg: str | None, pe_seg: str | None) -> int | None:
    """查 mv × pe 桶在 3 年训练数据里的有效起涨点样本数."""
    if not mv_seg or not pe_seg: return None
    m = _load_sample_matrix()
    return m.get(mv_seg, {}).get(pe_seg)


def _extract_moneyflow_summary(features: dict) -> dict | None:
    """从 features 抽取资金分层信号摘要 (供 LLM 解读, 不参与评分).

    返回字段 (单位 亿元 / 天数 / 比率):
      main_net_5d / main_net_20d: 主力 5/20 日累计净流入
      sm_net_5d:      散户 5 日累计
      main_consec_in / main_consec_out: 主力连续流入/流出天数
      dispersion_5d:  主力 vs 散户分歧 (-2 = 主力出散户接, +2 = 主力进散户出)
      pattern:        识别到的资金模式 (建仓/出货/分歧/同向)
    """
    keys = ["main_net_5d", "main_net_20d", "sm_net_5d", "lg_net_5d", "elg_net_5d",
             "main_consec_in", "main_consec_out", "dispersion_5d",
             "elg_ratio_5d", "buy_sell_imb_5d"]
    data = {}
    for k in keys:
        v = features.get(k)
        if v is not None and v == v:  # not NaN
            data[k] = round(float(v), 4)
    if not data:
        return None

    # 识别资金模式 (反传统判读, 实证有效)
    pattern = "中性"
    main_5d  = data.get("main_net_5d", 0)
    sm_5d    = data.get("sm_net_5d", 0)
    consec_in = data.get("main_consec_in", 0)
    consec_out = data.get("main_consec_out", 0)
    disp = data.get("dispersion_5d", 0)

    if disp <= -1.5:
        # 主力流出 + 散户流入: 实证 dispersion=-2 后续大涨 (反直觉)
        pattern = "⚡ 散户接盘式 (反直觉看多, 模型实证 +30%以上概率 10%)"
    elif disp >= 1.5:
        pattern = "⚠ 主力建仓散户跑 (传统看多, 但实证不如散户接盘强)"
    elif main_5d > 0.5 and consec_in >= 3:
        pattern = "✓ 主力持续建仓 (主力连续流入 3天+, 累计 0.5亿+)"
    elif main_5d < -0.5 and consec_out >= 3:
        pattern = "❌ 主力持续派发 (主力连续流出 3天+)"

    data["pattern"] = pattern
    return data


def _compute_entry_score(sparse_score: float, lgbm: dict | None,
                          maxgain: dict | None = None,
                          clean: dict | None = None,
                          context: dict | None = None,
                          uptrend: dict | None = None) -> dict:
    """入场评分 (0-100, 高=适合买入).

    新主信号: uptrend_prob (起涨点检测器, AUC=0.965, top 1% lift 16x)
    辅助信号: max_gain 预测, sparse, winprob
    """
    base = 50.0
    reasons = []
    pg  = (maxgain or {}).get("pred_gain")
    pdd = (maxgain or {}).get("pred_dd")
    cp  = (clean   or {}).get("clean_prob")
    wp  = (lgbm    or {}).get("winprob")
    up  = (uptrend or {}).get("uptrend_prob")
    up_tier = (uptrend or {}).get("lift_tier")

    # 0. 起涨点检测器 (新主信号, ±30)
    if up is not None:
        if up >= 0.40:    base += 30; reasons.append(f"⭐ 起涨点概率 {up*100:.1f}% (top 0.5%, lift 17x)")
        elif up >= 0.30:  base += 25; reasons.append(f"⭐ 起涨点概率 {up*100:.1f}% (top 1%, lift 16x)")
        elif up >= 0.15:  base += 18; reasons.append(f"起涨点概率 {up*100:.1f}% (top 5%, lift 13x)")
        elif up >= 0.08:  base += 8;  reasons.append(f"起涨点概率 {up*100:.1f}% (top 10%, lift 10x)")
        elif up >= 0.04:  base += 0
        else:             base -= 8;  reasons.append(f"起涨点概率 {up*100:.1f}% [低]")

    ctx = context or {}
    mv_seg   = ctx.get("mv_seg")
    pe_seg   = ctx.get("pe_seg")
    etf_held = ctx.get("etf_held", False)

    # 1. max_gain 预测 (辅助信号, ±15, 权重降低因为 uptrend 主导)
    if pg is not None:
        if pg >= 17:    base += 15; reasons.append(f"max_gain 预测 {pg:.1f}% [极强]")
        elif pg >= 15:  base += 10; reasons.append(f"max_gain 预测 {pg:.1f}% [强]")
        elif pg >= 13:  base += 5
        elif pg >= 11:  base += 0
        elif pg >= 9:   base -= 3
        else:           base -= 10; reasons.append(f"max_gain 预测 {pg:.1f}% [极弱]")

    # 2. sparse 综合 (±10)
    if sparse_score >= 75:   base += 10; reasons.append(f"sparse {sparse_score:.0f} [强]")
    elif sparse_score >= 60: base += 5
    elif sparse_score >= 50: base += 0
    elif sparse_score >= 35: base -= 5
    else:                    base -= 15; reasons.append(f"sparse {sparse_score:.0f} [低分]")

    # 3. r20 winprob (辅助, ±5)
    if wp is not None:
        if wp >= 0.62:   base += 5
        elif wp >= 0.55: base += 2
        elif wp <  0.45: base -= 5

    # 4. clean_prob (辅助, +5)
    if cp is not None and cp >= 0.20: base += 3
    if cp is not None and cp >= 0.30: base += 2

    # 5. 域限: 实证负 alpha 段降权
    if mv_seg == "1000亿+" and not etf_held:
        base -= 30
        reasons.append("⚠ 域限: 1000亿+ 非 ETF (实证 r20=-0.4%, 几乎不可买)")
    elif mv_seg in ("300-1000亿", "1000亿+") and not etf_held:
        base -= 12
        reasons.append("域限: 大盘非 ETF, sparse 实证负 alpha")

    # 5b. 行业黑名单 (实证 top 50 中假阳性集中区)
    industry = ctx.get("industry", "") or ""
    BANK_INSURE = {"银行", "保险", "保险II"}
    SLOW_INDUSTRIES = {"航运港口", "煤炭开采", "水电", "燃气", "高速公路", "铁路运输",
                        "火电", "其他银行", "公用事业"}
    if industry in BANK_INSURE:
        if mv_seg in ("1000亿+", "300-1000亿"):
            base -= 30
            reasons.append(f"⚠ 黑名单: 大盘 {industry} 业 (实证起涨概率高但不涨)")
        else:
            base -= 15
            reasons.append(f"行业 {industry} 起涨表现弱")
    elif industry in SLOW_INDUSTRIES:
        base -= 10
        reasons.append(f"行业 {industry}: 慢速行业, 起涨幅度有限")

    # 6. 稀疏桶警告: mv×pe 桶 3年内干净起涨样本不足
    n_samples = _bucket_sample_count(mv_seg, pe_seg)
    sparse_warning = None
    if n_samples is not None:
        if n_samples < 50:
            base = min(base, 50.0)   # 数据极稀疏, 评分压回中性
            sparse_warning = f"⚠ 训练样本极稀 (3年仅 {n_samples} 个干净起涨点), 评分不可信"
            reasons.append(sparse_warning)
        elif n_samples < 100:
            sparse_warning = f"训练样本偏稀 ({n_samples} 个), 慎信高分"
            reasons.append(sparse_warning)
        elif n_samples < 200:
            reasons.append(f"训练样本中等 ({n_samples} 个)")

    score = max(0.0, min(100.0, round(base, 1)))
    label = ("强买入信号 (top 20%)" if score >= 80 else
             "中等买入"            if score >= 65 else
             "偏多观察"            if score >= 50 else
             "偏弱观望"            if score >= 35 else
             "不建议买入")
    return {"score": score, "label": label, "reasons": reasons}


def _compute_risk_score(sparse_score: float, lgbm: dict | None,
                         maxgain: dict | None = None,
                         context: dict | None = None,
                         risk_ml: dict | None = None) -> dict:
    """回撤/浮亏风险评分 (0-100, 高=风险大).

    新主信号: ML 风险检测器 (AUC=0.674, dd<=-8% 概率)
    辅助: max_dd 预测, sparse 极低反向, 主力流出
    """
    base = 50.0
    reasons = []
    pdd = (maxgain or {}).get("pred_dd")
    pg  = (maxgain or {}).get("pred_gain")
    rp  = (risk_ml or {}).get("risk_prob")
    rt  = (risk_ml or {}).get("risk_tier")

    # 0. ML 风险检测器 (新主信号, ±25)
    if rp is not None:
        if rp >= 0.75:    base += 25; reasons.append(f"ML 风险概率 {rp*100:.0f}% [极高, 实证 51% 真破-8%]")
        elif rp >= 0.60:  base += 15; reasons.append(f"ML 风险概率 {rp*100:.0f}% [高]")
        elif rp >= 0.40:  base += 5;  reasons.append(f"ML 风险概率 {rp*100:.0f}% [中]")
        elif rp >= 0.25:  base -= 5
        else:             base -= 15; reasons.append(f"ML 风险概率 {rp*100:.0f}% [低, 实证 dd<-15% 仅 0.6%]")

    # 1. max_dd 预测 (辅助信号, ±15, 权重降低因为 ML 风险主导)
    if pdd is not None:
        if pdd <= -12:   base += 15
        elif pdd <= -10: base += 10
        elif pdd <= -8:  base += 5
        elif pdd <= -4:  base -= 3
        else:            base -= 8

    # 2. 收益/风险比 (gain/|dd|) — 比率太低风险高
    if pg is not None and pdd is not None and abs(pdd) > 0.5:
        ratio = pg / abs(pdd)
        if ratio < 1.0:    base += 15; reasons.append(f"收益/风险比 {ratio:.2f} [极差, 涨幅 < 回撤]")
        elif ratio < 1.5:  base += 8
        elif ratio >= 2.5: base -= 8

    # 3. sparse 极端低分 (反向加风险)
    if sparse_score < 30:
        base += 10
        reasons.append(f"sparse {sparse_score:.0f} [因子普遍负面]")

    # 4. 个股波动率特征
    extras = (context or {}).get("_raw") or {}
    # 主力流出连续 → 风险加大
    mf_div = extras.get("mf_divergence")
    mf_consec = extras.get("mf_consecutive")
    if mf_div is not None and mf_div < -0.3:
        base += 5
        reasons.append("主力资金流出")
    if mf_consec is not None and mf_consec <= -3:
        base += 5
        reasons.append(f"主力连续流出 {abs(mf_consec)} 天")

    score = max(0.0, min(100.0, round(base, 1)))
    label = ("极高回撤风险" if score >= 80 else
             "高风险"      if score >= 65 else
             "中等风险"    if score >= 50 else
             "较低风险"    if score >= 35 else
             "低风险")
    warning = None
    if score >= 65:
        dd_str = f"{pdd:.1f}%" if pdd is not None else "?"
        warning = (f"⚠ 高回撤风险: 预测期间最大回撤 {dd_str}. "
                   f"实证显示同档股票 ~45% 概率短期跌破 -8%. "
                   f"持仓建议设止损或分批进场.")
    return {"score": score, "label": label, "warning": warning, "reasons": reasons}


# 旧的 _compute_position 已废弃 — 系统不再给仓位建议.
# 由用户结合 entry_score (买入评分) + risk_score (回撤风险评分) 独立判断.


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
