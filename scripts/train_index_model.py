#!/usr/bin/env python3
"""训练指数专属评分模型 (独立于 V12.31).

特征: 纯技术面 (RSI/MA偏离/波动率/量比/动量/MACD/布林带)
标签:
  - r20_reg: 未来20日收益率 (回归 → r20_score)
  - pump_cls: 未来20日涨幅 > +3% (二分类 → pump_score)
  - pump_down_cls: 未来20日跌幅 < -3% (二分类 → pump_down_score)

模型: LightGBM × 3 (回归 + 2分类)
数据: 5大指数合并, 2010-2026, ~18000样本
回测: 滚动OOS (train 2010-2022, test 2023-2026)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb


def _rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def _roc_auc(y_true, y_score):
    """简易 AUC (Mann-Whitney U)."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_pos, n_neg = len(pos), len(neg)
    # 对每个正例, 计算超过多少负例
    count = 0
    for p in pos:
        count += (neg < p).sum() + 0.5 * (neg == p).sum()
    return count / (n_pos * n_neg)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "index_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 配置 ──
INDEX_CODES = [
    ("000001.SH", "上证指数"),
    ("000300.SH", "沪深300"),
    ("000905.SH", "中证500"),
    ("000852.SH", "中证1000"),
    ("399006.SZ", "创业板指"),
]

ETF_CODES = [
    # 宽基
    ("510050.SH", "上证50ETF", "宽基"),
    ("510300.SH", "沪深300ETF", "宽基"),
    ("510500.SH", "中证500ETF", "宽基"),
    ("512100.SH", "中证1000ETF", "宽基"),
    ("588000.SH", "科创50ETF", "宽基"),
    ("562800.SH", "中证A500ETF", "宽基"),
    ("159915.SZ", "创业板ETF", "宽基"),
    # 行业
    ("512010.SH", "医药ETF", "行业"),
    ("512660.SH", "军工ETF", "行业"),
    ("512880.SH", "证券ETF", "行业"),
    ("515030.SH", "新能源ETF", "行业"),
    ("512480.SH", "半导体ETF", "行业"),
    ("159869.SZ", "游戏ETF", "行业"),
    # 跨境
    ("513100.SH", "纳指ETF", "跨境"),
    ("513050.SH", "中概互联ETF", "跨境"),
    ("159920.SZ", "恒生ETF", "跨境"),
]
TRAIN_END = "20221231"   # 训练集截止日
TEST_START = "20230101"  # OOS测试集起始
LABEL_WINDOW = 20        # 前瞻20日
PUMP_THRESHOLD = 0.03    # 涨 >3%
DOWN_THRESHOLD = -0.03   # 跌 >3%

# LightGBM 参数 (保守, 防过拟合)
LGB_PARAMS_REG = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "min_child_samples": 100, "bagging_fraction": 0.8, "feature_fraction": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1,
}
LGB_PARAMS_CLS = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
    "min_child_samples": 100, "bagging_fraction": 0.8, "feature_fraction": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1,
    "is_unbalance": True,
}
NUM_ROUNDS = 300


def fetch_all_indices() -> pd.DataFrame:
    """从 tushare 拉取 5 大指数 + 16 ETF 全历史, 合并."""
    import tushare as ts
    pro = ts.pro_api()
    frames = []
    # 指数
    for ts_code, name in INDEX_CODES:
        print(f"  拉取指数 {name} ({ts_code})...")
        df = pro.index_daily(ts_code=ts_code, start_date="20100101", end_date="20260731")
        if df is None or len(df) == 0:
            continue
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["ts_code"] = ts_code
        df["index_name"] = name
        df["asset_type"] = "指数"
        frames.append(df)
    # ETF (fund_daily)
    for ts_code, name, category in ETF_CODES:
        print(f"  拉取ETF {name} ({ts_code})...")
        try:
            df = pro.fund_daily(ts_code=ts_code, start_date="20100101", end_date="20260731")
            if df is None or len(df) == 0:
                continue
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["ts_code"] = ts_code
            df["index_name"] = name
            df["asset_type"] = f"ETF-{category}"
            frames.append(df)
        except Exception as e:
            print(f"    {name} 失败: {e}")
    all_df = pd.concat(frames, ignore_index=True)
    all_df["trade_date"] = all_df["trade_date"].astype(str)
    n_index = all_df[all_df["asset_type"] == "指数"]["ts_code"].nunique()
    n_etf = all_df[all_df["asset_type"] != "指数"]["ts_code"].nunique()
    print(f"  合计 {len(all_df)} 行, {n_index} 指数 + {n_etf} ETF = {n_index + n_etf} 资产")
    return all_df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """对每个指数分组计算技术面特征."""
    feat_frames = []
    for ts_code, grp in df.groupby("ts_code"):
        g = grp.sort_values("trade_date").reset_index(drop=True)
        c = g["close"].values.astype(float)
        o = g["open"].values.astype(float)
        h = g["high"].values.astype(float)
        l = g["low"].values.astype(float)
        v = g["vol"].values.astype(float)
        n = len(g)

        feats = pd.DataFrame(index=g.index)
        feats["ts_code"] = ts_code
        feats["trade_date"] = g["trade_date"]
        feats["index_name"] = g["index_name"]
        feats["close"] = c

        # 收益率
        for w in [1, 5, 10, 20, 60]:
            feats[f"ret_{w}d"] = pd.Series(c).pct_change(w).values

        # MA偏离
        for w in [5, 10, 20, 60, 120]:
            ma = pd.Series(c).rolling(w).mean().values
            feats[f"dev_ma{w}"] = (c / (ma + 1e-10) - 1)

        # RSI-14
        delta = pd.Series(c).diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        feats["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-10))

        # 波动率 (20日收益率标准差)
        feats["vol_20d"] = pd.Series(c).pct_change().rolling(20).std().values

        # 量比
        vol_ma20 = pd.Series(v).rolling(20).mean().values
        feats["vol_ratio"] = v / (vol_ma20 + 1e-10)

        # 动量斜率 (20日线性回归斜率)
        feats["slope_20d"] = pd.Series(c).rolling(20).apply(
            lambda x: np.polyfit(range(20), x, 1)[0] / x.mean() * 100, raw=True).values

        # MACD
        ema12 = pd.Series(c).ewm(span=12).mean().values
        ema26 = pd.Series(c).ewm(span=26).mean().values
        dif = ema12 - ema26
        dea = pd.Series(dif).ewm(span=9).mean().values
        feats["macd_dif"] = dif / (c + 1e-10)  # 归一化
        feats["macd_hist"] = (dif - dea) / (c + 1e-10)

        # 布林带位置 (close 在 ±2σ 中的位置, 0=下轨, 1=上轨)
        bb_mid = pd.Series(c).rolling(20).mean().values
        bb_std = pd.Series(c).rolling(20).std().values
        feats["bb_pos"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

        # 20日涨跌天数比
        changes = pd.Series(c).pct_change()
        feats["up_ratio_20d"] = (changes > 0).rolling(20).sum().values / (
            (changes < 0).rolling(20).sum().values + 1)

        # 振幅 (high-low)/close
        feats["amplitude"] = (h - l) / (c + 1e-10)

        # 缺口 (open-pre_close)/pre_close
        feats["gap"] = (o - g["pre_close"].values.astype(float)) / (g["pre_close"].values.astype(float) + 1e-10)

        feat_frames.append(feats)

    result = pd.concat(feat_frames, ignore_index=True)
    return result


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """计算前瞻标签 (每个指数分组)."""
    label_frames = []
    for ts_code, grp in df.groupby("ts_code"):
        g = grp.sort_values("trade_date").reset_index(drop=True)
        c = g["close"].values.astype(float)
        n = len(g)

        labels = pd.DataFrame(index=g.index)
        # 未来20日收益率
        fwd_ret = np.full(n, np.nan)
        for i in range(n - LABEL_WINDOW):
            fwd_ret[i] = c[i + LABEL_WINDOW] / c[i] - 1
        labels["fwd_ret_20d"] = fwd_ret
        labels["label_pump"] = (fwd_ret > PUMP_THRESHOLD).astype(float)
        labels["label_pump_down"] = (fwd_ret < DOWN_THRESHOLD).astype(float)
        # pump label 未知部分设为 NaN
        labels.loc[labels["fwd_ret_20d"].isna(), ["label_pump", "label_pump_down"]] = np.nan

        label_frames.append(labels)

    return pd.concat(label_frames, ignore_index=True)


FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "dev_ma5", "dev_ma10", "dev_ma20", "dev_ma60", "dev_ma120",
    "rsi_14", "vol_20d", "vol_ratio", "slope_20d",
    "macd_dif", "macd_hist", "bb_pos", "up_ratio_20d",
    "amplitude", "gap",
]


def train_and_evaluate(df: pd.DataFrame):
    """训练 3 个模型 + OOS 回测."""
    train = df[df["trade_date"] <= TRAIN_END].dropna(subset=FEATURE_COLS + ["fwd_ret_20d"])
    test = df[(df["trade_date"] >= TEST_START)].dropna(subset=FEATURE_COLS + ["fwd_ret_20d"])

    print(f"\n训练集: {len(train)} 行 ({train['trade_date'].min()} ~ {train['trade_date'].max()})")
    print(f"测试集: {len(test)} 行 ({test['trade_date'].min()} ~ {test['trade_date'].max()})")

    X_train = train[FEATURE_COLS].values
    X_test = test[FEATURE_COLS].values

    models = {}

    # 1. r20 回归模型
    print("\n[1/3] 训练 r20 回归模型 (未来20日收益率)...")
    y_reg_train = train["fwd_ret_20d"].values
    y_reg_test = test["fwd_ret_20d"].values
    dtrain_reg = lgb.Dataset(X_train, label=y_reg_train, feature_name=FEATURE_COLS)
    reg = lgb.train(LGB_PARAMS_REG, dtrain_reg, num_boost_round=NUM_ROUNDS)
    pred_reg = reg.predict(X_test)
    rmse = _rmse(y_reg_test, pred_reg)
    corr = np.corrcoef(y_reg_test, pred_reg)[0, 1]
    print(f"  OOS RMSE={rmse:.4f}, 相关系数={corr:.4f}")
    models["r20_reg"] = reg

    # 2. pump 分类模型 (涨>3%)
    print("\n[2/3] 训练 pump 分类模型 (20日涨>3%)...")
    pump_train = train.dropna(subset=["label_pump"])
    pump_test = test.dropna(subset=["label_pump"])
    y_pump_train = pump_train["label_pump"].values
    y_pump_test = pump_test["label_pump"].values
    X_pump_train = pump_train[FEATURE_COLS].values
    X_pump_test = pump_test[FEATURE_COLS].values
    pump_pos_rate = y_pump_train.mean()
    print(f"  训练集正例率: {pump_pos_rate:.1%}")
    dtrain_pump = lgb.Dataset(X_pump_train, label=y_pump_train, feature_name=FEATURE_COLS)
    cls_pump = lgb.train(LGB_PARAMS_CLS, dtrain_pump, num_boost_round=NUM_ROUNDS)
    prob_pump = cls_pump.predict(X_pump_test)
    auc_pump = _roc_auc(y_pump_test, prob_pump)
    print(f"  OOS AUC={auc_pump:.4f}")
    models["pump_cls"] = cls_pump

    # 3. pump_down 分类模型 (跌>3%)
    print("\n[3/3] 训练 pump_down 分类模型 (20日跌>3%)...")
    down_train = train.dropna(subset=["label_pump_down"])
    down_test = test.dropna(subset=["label_pump_down"])
    y_down_train = down_train["label_pump_down"].values
    y_down_test = down_test["label_pump_down"].values
    X_down_train = down_train[FEATURE_COLS].values
    X_down_test = down_test[FEATURE_COLS].values
    down_pos_rate = y_down_train.mean()
    print(f"  训练集正例率: {down_pos_rate:.1%}")
    dtrain_down = lgb.Dataset(X_down_train, label=y_down_train, feature_name=FEATURE_COLS)
    cls_down = lgb.train(LGB_PARAMS_CLS, dtrain_down, num_boost_round=NUM_ROUNDS)
    prob_down = cls_down.predict(X_down_test)
    auc_down = _roc_auc(y_down_test, prob_down)
    print(f"  OOS AUC={auc_down:.4f}")
    models["pump_down_cls"] = cls_down

    # OOS 回测: 按指数分组展示
    print("\n" + "=" * 60)
    print("OOS 回测明细 (2023-2026):")
    print("=" * 60)
    test_eval = test.copy()
    test_eval["pred_r20"] = pred_reg
    test_eval["pred_pump"] = prob_pump
    test_eval["pred_pump_down"] = prob_down
    test_eval["pred_ratio"] = prob_pump / (prob_down + 0.01)

    for ts_code, grp in test_eval.groupby("ts_code"):
        g = grp.sort_values("trade_date")
        name = g["index_name"].iloc[0]
        actual = g["fwd_ret_20d"].values
        pred = g["pred_r20"].values
        ic = np.corrcoef(actual, pred)[0, 1]
        # 方向准确率
        dir_acc = ((pred > 0) == (actual > 0)).mean()
        print(f"  {name}: IC={ic:.4f}, 方向准确率={dir_acc:.1%}, "
              f"实际均收益={actual.mean():.2%}, 预测均r20={pred.mean():.4f}")

    # 特征重要性
    print("\n特征重要性 (r20 回归):")
    imp = pd.Series(reg.feature_importance(importance_type="gain"), index=FEATURE_COLS).sort_values(ascending=False)
    for feat, score in imp.head(10).items():
        print(f"  {feat:<20} {score}")

    return models


def save_models(models: dict):
    """保存模型 + 元数据."""
    import json
    for name, model in models.items():
        path = OUT_DIR / f"{name}.txt"
        model.save_model(str(path))
        print(f"  保存: {path}")
    # 元数据
    meta = {
        "feature_cols": FEATURE_COLS,
        "index_codes": [c[0] for c in INDEX_CODES],
        "index_names": {c[0]: c[1] for c in INDEX_CODES},
        "etf_codes": [c[0] for c in ETF_CODES],
        "etf_names": {c[0]: c[1] for c in ETF_CODES},
        "etf_categories": {c[0]: c[2] for c in ETF_CODES},
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "label_window": LABEL_WINDOW,
        "pump_threshold": PUMP_THRESHOLD,
        "down_threshold": DOWN_THRESHOLD,
    }
    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  保存: {meta_path}")


def main():
    print("=" * 60)
    print("指数评分模型训练 (独立于 V12.31)")
    print("=" * 60)

    print("\n[1] 拉取历史数据...")
    raw = fetch_all_indices()

    print("\n[2] 计算特征...")
    feats = compute_features(raw)
    print(f"  特征数: {len(FEATURE_COLS)}, 样本数: {len(feats)}")

    print("\n[3] 计算标签...")
    labels = compute_labels(raw)

    df = feats.join(labels)
    print(f"  合并后: {len(df)} 行, 有效标签: {df['fwd_ret_20d'].notna().sum()}")

    print("\n[4] 训练 + OOS 回测...")
    models = train_and_evaluate(df)

    print("\n[5] 保存模型...")
    save_models(models)

    print("\n[DONE] 模型保存在 output/index_model/")


if __name__ == "__main__":
    main()
