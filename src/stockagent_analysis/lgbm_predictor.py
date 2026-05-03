"""LightGBM r20 预测器 — 单股推理 API.

输出:
  pred_r20:  预测 20 日涨幅 (%)
  winprob:   上涨概率 (0-1, 来自分类模型)
  conf:      模型确信度标签 high/med/low (基于 |pred|/std)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("stockagent.lgbm_predictor")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DIR   = _PROJECT_ROOT / "output" / "lgbm_r20"
_CLEAN_DIR     = _PROJECT_ROOT / "output" / "lgbm_clean"
_MAXGAIN_DIR   = _PROJECT_ROOT / "output" / "lgbm_maxgain"
_UPTREND_DIR   = _PROJECT_ROOT / "output" / "lgbm_uptrend"
_RISK_DIR      = _PROJECT_ROOT / "output" / "lgbm_risk"

# 模块级缓存
_REG_MODEL = None
_CLS_MODEL = None
_FEAT_META: dict | None = None
_CLEAN_MODEL = None
_CLEAN_META: dict | None = None
_GAIN_MODEL = None
_DD_MODEL   = None
_GAIN_META: dict | None = None
# 起涨点检测器 (二分类, AUC=0.965, top 1% lift=16x)
_UPTREND_MODEL = None
_UPTREND_META: dict | None = None
# 回撤风险检测器 (二分类, AUC=0.674, dd<=-8% 预测)
_RISK_MODEL = None
_RISK_META: dict | None = None


def load(model_dir: Path | str | None = None) -> bool:
    """加载回归 + 分类模型 + 元数据. 返回是否加载成功."""
    global _REG_MODEL, _CLS_MODEL, _FEAT_META
    if _REG_MODEL is not None:
        return True

    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm 未安装, LGBM 预测不可用")
        return False

    d = Path(model_dir) if model_dir else _DEFAULT_DIR
    reg_path = d / "model.txt"
    cls_path = d / "classifier.txt"
    meta_path = d / "feature_meta.json"

    if not reg_path.exists() or not meta_path.exists():
        logger.warning("LGBM 模型未找到: %s", d)
        return False

    _REG_MODEL = lgb.Booster(model_file=str(reg_path))
    if cls_path.exists():
        _CLS_MODEL = lgb.Booster(model_file=str(cls_path))
    _FEAT_META = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info("LGBM 加载完成: %d 特征, residual_std=%.3f%%",
                len(_FEAT_META["feature_cols"]),
                _FEAT_META.get("residual_std", 0))
    return True


def load_clean(model_dir: Path | str | None = None) -> bool:
    """加载干净走势检测器."""
    global _CLEAN_MODEL, _CLEAN_META
    if _CLEAN_MODEL is not None:
        return True
    try:
        import lightgbm as lgb
    except ImportError:
        return False

    d = Path(model_dir) if model_dir else _CLEAN_DIR
    cls_path = d / "classifier.txt"
    meta_path = d / "feature_meta.json"
    if not cls_path.exists() or not meta_path.exists():
        logger.debug("clean detector 未找到: %s", d)
        return False
    _CLEAN_MODEL = lgb.Booster(model_file=str(cls_path))
    _CLEAN_META = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info("clean detector 加载: %d 特征", len(_CLEAN_META["feature_cols"]))
    return True


def _build_row(feat_cols: list, industry_map: dict, industry: str,
                features: dict, extras: dict | None) -> pd.DataFrame:
    """构造单行 DataFrame, 处理缺失 + industry_id 映射."""
    extras = extras or {}
    industry_id = industry_map.get(industry, industry_map.get("unknown", -1))
    row = {}
    for c in feat_cols:
        if c == "industry_id":
            row[c] = industry_id
            continue
        v = features.get(c)
        if v is None:
            v = extras.get(c)
        try:
            row[c] = float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            row[c] = np.nan
    return pd.DataFrame([row], columns=feat_cols)


def predict_clean(features: dict[str, Any], industry: str = "",
                   extras: dict[str, Any] | None = None) -> dict | None:
    """预测"干净上涨"概率 (max_gain_20>=20% AND max_dd_20>=-3%).

    返回: {clean_prob: 0-1, ok: bool}
    """
    if not load_clean():
        return None
    feat_cols = _CLEAN_META["feature_cols"]
    industry_map = _CLEAN_META.get("industry_map", {})
    X = _build_row(feat_cols, industry_map, industry, features, extras)
    prob = float(_CLEAN_MODEL.predict(X)[0])
    return {"clean_prob": round(prob, 4), "ok": True}


def load_maxgain(model_dir: Path | str | None = None) -> bool:
    """加载 max_gain + max_dd 回归器."""
    global _GAIN_MODEL, _DD_MODEL, _GAIN_META
    if _GAIN_MODEL is not None:
        return True
    try:
        import lightgbm as lgb
    except ImportError:
        return False
    d = Path(model_dir) if model_dir else _MAXGAIN_DIR
    g_path = d / "regressor_gain.txt"
    dd_path = d / "regressor_dd.txt"
    meta_path = d / "feature_meta.json"
    if not g_path.exists() or not meta_path.exists():
        logger.debug("maxgain 模型未找到: %s", d)
        return False
    _GAIN_MODEL = lgb.Booster(model_file=str(g_path))
    if dd_path.exists():
        _DD_MODEL = lgb.Booster(model_file=str(dd_path))
    _GAIN_META = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info("maxgain 加载: %d 特征", len(_GAIN_META["feature_cols"]))
    return True


def load_uptrend(model_dir: Path | str | None = None) -> bool:
    """加载起涨点检测器 (二分类). AUC=0.965, top 1% lift 16x."""
    global _UPTREND_MODEL, _UPTREND_META
    if _UPTREND_MODEL is not None:
        return True
    try: import lightgbm as lgb
    except ImportError: return False
    d = Path(model_dir) if model_dir else _UPTREND_DIR
    cls_path = d / "classifier.txt"
    meta_path = d / "feature_meta.json"
    if not cls_path.exists() or not meta_path.exists():
        return False
    _UPTREND_MODEL = lgb.Booster(model_file=str(cls_path))
    _UPTREND_META = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info("uptrend detector 加载: %d 特征", len(_UPTREND_META["feature_cols"]))
    return True


def predict_uptrend(features: dict[str, Any], industry: str = "",
                     extras: dict[str, Any] | None = None) -> dict | None:
    """预测干净起涨点概率.

    起涨点定义: MA5 拐点+放量+未来20天MA5稳定向上+不破起涨点-3%+涨幅>=10%

    返回: {uptrend_prob, lift_tier, ok}
      lift_tier:
        'top0.5%' (prob>=0.4):  ~13% 概率是真起涨点 (lift 17x)
        'top1%'   (prob>=0.3):  ~12% (lift 16x)
        'top5%'   (prob>=0.15): ~10% (lift 13x)
        'top10%'  (prob>=0.08): ~7%  (lift 10x)
        'top20%'  (prob>=0.04): ~4%  (lift 5x)
    """
    if not load_uptrend(): return None
    feat_cols = _UPTREND_META["feature_cols"]
    industry_map = _UPTREND_META.get("industry_map", {})
    X = _build_row(feat_cols, industry_map, industry, features, extras)
    prob = float(_UPTREND_MODEL.predict(X)[0])
    if   prob >= 0.40: tier = "top0.5%"
    elif prob >= 0.30: tier = "top1%"
    elif prob >= 0.15: tier = "top5%"
    elif prob >= 0.08: tier = "top10%"
    elif prob >= 0.04: tier = "top20%"
    else:              tier = "below_top20%"
    return {"uptrend_prob": round(prob, 4), "lift_tier": tier, "ok": True}


def load_risk(model_dir: Path | str | None = None) -> bool:
    """加载回撤风险检测器 (二分类, dd<=-8% 概率)."""
    global _RISK_MODEL, _RISK_META
    if _RISK_MODEL is not None: return True
    try: import lightgbm as lgb
    except ImportError: return False
    d = Path(model_dir) if model_dir else _RISK_DIR
    cls_path = d / "classifier.txt"
    meta_path = d / "feature_meta.json"
    if not cls_path.exists() or not meta_path.exists(): return False
    _RISK_MODEL = lgb.Booster(model_file=str(cls_path))
    _RISK_META = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info("risk detector 加载: %d 特征", len(_RISK_META["feature_cols"]))
    return True


def predict_risk(features: dict[str, Any], industry: str = "",
                  extras: dict[str, Any] | None = None) -> dict | None:
    """预测 20 天内 max_dd <= -8% 的概率.

    OOS 实证: Q5 (top20%) 真实风险率 50.8%, Q1 (bot20%) 仅 9.6%.
    Q1 段 dd<-15% 比例从基线 5% 降到 0.6% (避雷强项).

    返回: {risk_prob, risk_tier, ok}
      tier: 'high' (>=0.7), 'mid' (0.4-0.7), 'low' (<0.4)
    """
    if not load_risk(): return None
    feat_cols = _RISK_META["feature_cols"]
    industry_map = _RISK_META.get("industry_map", {})
    X = _build_row(feat_cols, industry_map, industry, features, extras)
    prob = float(_RISK_MODEL.predict(X)[0])
    if   prob >= 0.70: tier = "high"
    elif prob >= 0.40: tier = "mid"
    else:              tier = "low"
    return {"risk_prob": round(prob, 4), "risk_tier": tier, "ok": True}


def predict_maxgain(features: dict[str, Any], industry: str = "",
                     extras: dict[str, Any] | None = None) -> dict | None:
    """预测期间最高涨幅 (max_gain_20%) + 期间最大回撤 (max_dd_20%).

    返回: {pred_gain: float, pred_dd: float, gain_dd_ratio: float, ok: bool}
          gain_dd_ratio = pred_gain / |pred_dd|, 风险收益比 (越高越好)
    """
    if not load_maxgain():
        return None
    feat_cols = _GAIN_META["feature_cols"]
    industry_map = _GAIN_META.get("industry_map", {})
    X = _build_row(feat_cols, industry_map, industry, features, extras)
    pg = float(_GAIN_MODEL.predict(X)[0])
    pd_ = float(_DD_MODEL.predict(X)[0]) if _DD_MODEL else None
    out = {"pred_gain": round(pg, 3), "ok": True}
    if pd_ is not None:
        out["pred_dd"] = round(pd_, 3)
        out["gain_dd_ratio"] = round(pg / abs(pd_), 3) if pd_ != 0 else None
    return out


def predict(features: dict[str, Any], industry: str = "",
             extras: dict[str, Any] | None = None) -> dict | None:
    """对单只股票预测 r20 + winprob.

    features: 因子值字典 {factor_name: value}
    industry: 行业字符串 (训练时一致, 自动映射 industry_id)
    extras:   补充特征 {total_mv, pe, pe_ttm, market_score_adj, mf_divergence,
              mf_strength, mf_consecutive} — LGBM 训练时这些被当作特征,
              如果 features 里没包含, 必须从这里提供.

    返回: {pred_r20, winprob, conf, ok}, 失败 None
    """
    if not load():
        return None
    feat_cols = _FEAT_META["feature_cols"]
    industry_map = _FEAT_META.get("industry_map", {})
    residual_std = _FEAT_META.get("residual_std", 6.0)

    extras = extras or {}
    industry_id = industry_map.get(industry, industry_map.get("unknown", -1))
    row = {}
    for c in feat_cols:
        if c == "industry_id":
            row[c] = industry_id
            continue
        v = features.get(c)
        if v is None:
            v = extras.get(c)
        try:
            row[c] = float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            row[c] = np.nan

    X = pd.DataFrame([row], columns=feat_cols)

    pred_r20 = float(_REG_MODEL.predict(X)[0])
    if _CLS_MODEL is not None:
        winprob = float(_CLS_MODEL.predict(X)[0])
    else:
        # 回退: 用残差标准差 + 正态分布算
        try:
            from scipy.stats import norm
            winprob = float(1 - norm.cdf(0, loc=pred_r20, scale=residual_std))
        except Exception:
            winprob = 0.5

    # 确信度: |pred| 相对 std 越大越可信
    z = abs(pred_r20) / max(residual_std, 0.5)
    if z >= 1.0:
        conf = "high"
    elif z >= 0.5:
        conf = "med"
    else:
        conf = "low"

    return {
        "pred_r20": round(pred_r20, 3),
        "winprob": round(winprob, 4),
        "conf": conf,
        "ok": True,
    }


def predict_with_sparse(features: dict, industry: str,
                         sparse_score: float | None = None) -> dict:
    """方便版: 同时跑预测并附上 sparse_score 综合评估."""
    out = predict(features, industry) or {"ok": False}
    if sparse_score is not None:
        out["sparse_score"] = round(float(sparse_score), 2)
    return out
