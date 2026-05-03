#!/usr/bin/env python3
"""对比新旧 uptrend 模型预测差异 — 找增强/矛盾样本.

新模型: 加 moneyflow 12 个资金分层特征
旧模型: 没加 moneyflow

输出:
  增强样本 (新模型对了, 旧模型错了): 哪些资金信号让模型改判
  矛盾样本 (旧模型对了, 新模型错了): 资金信号反而误导
  一致预测 (两个都对/都错): 不参考价值
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parent
PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
LABELS = ROOT / "output" / "labels" / "max_gain_labels.parquet"
STARTS = ROOT / "output" / "uptrend_starts" / "starts.parquet"
OLD_MODEL = ROOT / "output" / "lgbm_uptrend_v1"      # 备份的旧模型 (无 moneyflow)
NEW_MODEL = ROOT / "output" / "lgbm_uptrend"          # 新模型 (含 moneyflow)
OUT_DIR = ROOT / "output" / "compare_old_new"
OUT_DIR.mkdir(exist_ok=True)

START = "20250501"
END   = "20260126"


def load_oos():
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= START) & (df["trade_date"] <= END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)

    # 起涨标签
    starts = pd.read_parquet(STARTS, columns=["ts_code","trade_date"])
    starts["trade_date"] = starts["trade_date"].astype(str)
    pos_keys = set(starts["ts_code"]+"|"+starts["trade_date"])
    full["_key"] = full["ts_code"]+"|"+full["trade_date"]
    full["is_uptrend"] = full["_key"].isin(pos_keys).astype(int)

    # max_gain 标签
    labels = pd.read_parquet(LABELS, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    labels["trade_date"] = labels["trade_date"].astype(str)
    full = full.merge(labels, on=["ts_code","trade_date"], how="left")

    # 各种额外 feature
    for path, key in [
        (ROOT / "output" / "regimes" / "daily_regime.parquet", "trade_date"),
        (ROOT / "output" / "amount_features" / "amount_features.parquet",
         ["ts_code","trade_date"]),
        (ROOT / "output" / "regime_extra" / "regime_extra.parquet", "trade_date"),
    ]:
        if path.exists():
            d = pd.read_parquet(path)
            if "trade_date" in d.columns:
                d["trade_date"] = d["trade_date"].astype(str)
            if path.name == "daily_regime.parquet":
                d = d.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                        "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14",
                                        "vol_ratio":"mkt_vol_ratio"})
            full = full.merge(d, on=key, how="left")

    # moneyflow features (新模型独有)
    mf_path = ROOT / "output" / "moneyflow" / "features.parquet"
    if mf_path.exists():
        mf = pd.read_parquet(mf_path)
        mf["trade_date"] = mf["trade_date"].astype(str)
        full = full.merge(mf, on=["ts_code","trade_date"], how="left")

    return full


def predict(model_dir: Path, full: pd.DataFrame) -> np.ndarray:
    booster = lgb.Booster(model_file=str(model_dir / "classifier.txt"))
    meta = json.loads((model_dir / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    indmap = meta["industry_map"]
    full = full.copy()
    full["industry_id"] = full["industry"].fillna("unknown").map(
        lambda x: indmap.get(str(x), -1))
    for c in feat_cols:
        if c not in full.columns: full[c] = np.nan
    return booster.predict(full[feat_cols].astype(float))


def main():
    if not OLD_MODEL.exists():
        print(f"❌ 旧模型不存在: {OLD_MODEL}")
        print("提示: 先备份当前模型为 lgbm_uptrend_v1, 再训练新模型")
        return
    if not NEW_MODEL.exists():
        print(f"❌ 新模型不存在: {NEW_MODEL}")
        return

    print("加载 OOS 数据 + 标签...")
    full = load_oos().dropna(subset=["max_gain_20"])
    full = full.dropna(subset=["r20"])
    print(f"OOS 样本: {len(full)}, 真实起涨 {full['is_uptrend'].sum()}")

    print("旧模型预测 (无 moneyflow)...")
    full["prob_old"] = predict(OLD_MODEL, full)
    print("新模型预测 (含 moneyflow)...")
    full["prob_new"] = predict(NEW_MODEL, full)
    full["delta"] = full["prob_new"] - full["prob_old"]

    # 整体对比
    print(f"\n=== 整体对比 ===")
    for thr in [0.30, 0.15, 0.08]:
        old_top = full[full["prob_old"] >= thr]
        new_top = full[full["prob_new"] >= thr]
        print(f"prob >= {thr}:")
        print(f"  旧模型: n={len(old_top):>5}, 起涨率 {old_top['is_uptrend'].mean()*100:.2f}%, "
              f"max_gain {old_top['max_gain_20'].mean():.2f}%")
        print(f"  新模型: n={len(new_top):>5}, 起涨率 {new_top['is_uptrend'].mean()*100:.2f}%, "
              f"max_gain {new_top['max_gain_20'].mean():.2f}%")

    # 增强样本: 新模型 prob >= 0.30 但旧模型 < 0.10 (新发现)
    print(f"\n=== 增强样本 (新模型识别, 旧模型遗漏) ===")
    enhanced = full[(full["prob_new"] >= 0.30) & (full["prob_old"] < 0.10)]
    print(f"数量: {len(enhanced)}")
    if len(enhanced) > 0:
        is_real_uptrend = enhanced["is_uptrend"].mean() * 100
        max_gain_avg = enhanced["max_gain_20"].mean()
        gain_15 = (enhanced["max_gain_20"] >= 15).mean() * 100
        gain_30 = (enhanced["max_gain_20"] >= 30).mean() * 100
        print(f"  真实起涨率: {is_real_uptrend:.2f}% (vs 基线 0.75%)")
        print(f"  平均 max_gain: {max_gain_avg:.2f}%")
        print(f"  涨过 +15%: {gain_15:.1f}%")
        print(f"  涨过 +30%: {gain_30:.1f}%")

    # 矛盾样本: 旧模型 prob >= 0.30 但新模型 < 0.10 (新模型反对)
    print(f"\n=== 矛盾样本 (旧模型预测起涨, 新模型反对) ===")
    contradicted = full[(full["prob_old"] >= 0.30) & (full["prob_new"] < 0.10)]
    print(f"数量: {len(contradicted)}")
    if len(contradicted) > 0:
        is_real_uptrend = contradicted["is_uptrend"].mean() * 100
        max_gain_avg = contradicted["max_gain_20"].mean()
        gain_15 = (contradicted["max_gain_20"] >= 15).mean() * 100
        print(f"  真实起涨率: {is_real_uptrend:.2f}%")
        print(f"  平均 max_gain: {max_gain_avg:.2f}%")
        print(f"  涨过 +15%: {gain_15:.1f}%")

    # 矛盾的"反转赢家" / "反转输家"
    big_winners_in_contradicted = pd.DataFrame()
    flat = pd.DataFrame()
    if len(contradicted) > 0:
        big_winners_in_contradicted = contradicted[contradicted["max_gain_20"] >= 20]
        print(f"  矛盾中真涨过 +20%: {len(big_winners_in_contradicted)} 个")
        flat = contradicted[contradicted["max_gain_20"] < 5]
        print(f"  矛盾中没涨过 +5%: {len(flat)} 个 (新模型避雷成功)")

    # 写出
    enhanced.to_csv(OUT_DIR / "enhanced.csv", index=False, encoding="utf-8-sig")
    contradicted.to_csv(OUT_DIR / "contradicted.csv", index=False, encoding="utf-8-sig")

    # 显示示例
    print("\n=== 增强样本示例 (top 10) ===")
    show = enhanced.nlargest(10, "max_gain_20")[
        ["ts_code","trade_date","industry","prob_old","prob_new","is_uptrend",
         "max_gain_20","max_dd_20","main_net_5d","sm_net_5d","dispersion_5d"]
    ]
    print(show.to_string(index=False))

    print("\n=== 矛盾样本中真涨股示例 (新模型错过的) ===")
    if len(big_winners_in_contradicted) > 0:
        show2 = big_winners_in_contradicted.nlargest(10, "max_gain_20")[
            ["ts_code","trade_date","industry","prob_old","prob_new",
             "max_gain_20","max_dd_20","main_net_5d","sm_net_5d","dispersion_5d"]
        ]
        print(show2.to_string(index=False))

    # 写报告 markdown
    report = f"""# 新旧模型预测对比报告

## 测试集
- 期间: {START} → {END}
- 样本: {len(full)} 行, 真实起涨 {full['is_uptrend'].sum()} ({full['is_uptrend'].mean()*100:.2f}%)

## 整体效果

| prob 阈值 | 旧模型 (n / 起涨率 / max_gain) | 新模型 (n / 起涨率 / max_gain) |
|---|---|---|
"""
    for thr in [0.30, 0.15, 0.08]:
        ot = full[full["prob_old"] >= thr]
        nt = full[full["prob_new"] >= thr]
        report += (f"| ≥{thr} | {len(ot)} / {ot['is_uptrend'].mean()*100:.2f}% / "
                    f"{ot['max_gain_20'].mean():.2f}% "
                    f"| {len(nt)} / {nt['is_uptrend'].mean()*100:.2f}% / "
                    f"{nt['max_gain_20'].mean():.2f}% |\n")

    report += f"\n## 增强样本 (新发现 {len(enhanced)} 个)\n\n"
    if len(enhanced) > 0:
        report += (f"- 真实起涨率: {enhanced['is_uptrend'].mean()*100:.2f}%\n"
                   f"- 平均 max_gain: {enhanced['max_gain_20'].mean():.2f}%\n"
                   f"- 涨过 +15%: {(enhanced['max_gain_20']>=15).mean()*100:.1f}%\n"
                   f"- 涨过 +30%: {(enhanced['max_gain_20']>=30).mean()*100:.1f}%\n")

    report += f"\n## 矛盾样本 (新模型反对 {len(contradicted)} 个)\n\n"
    if len(contradicted) > 0:
        report += (f"- 真实起涨率: {contradicted['is_uptrend'].mean()*100:.2f}%\n"
                   f"- 平均 max_gain: {contradicted['max_gain_20'].mean():.2f}%\n"
                   f"- 涨过 +20% (新模型错过的): {len(big_winners_in_contradicted)} 个\n"
                   f"- 没涨过 +5% (新模型成功避雷): {len(flat)} 个\n")

    (OUT_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"\n输出: {OUT_DIR}")


if __name__ == "__main__":
    main()
