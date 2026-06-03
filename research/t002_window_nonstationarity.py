# -*- coding: utf-8 -*-
"""T-002 窗口非平稳检验 — ratio 短窗/长窗特征贡献分解 + lead/lag 漂移.

研究问题: pump ratio = P_up/(P_down+eps) 的方向信号, 是被"短窗"还是"长窗"特征驱动?
          这个短/长窗驱动占比是否随 regime (快市/慢市) 漂移? 若漂移显著 → 固定窗口不够
          (window_dynamic_needed); 若稳定 → fixed_ok (并由本任务把 T-004 置 skip)。

方法 (零前视, 全 causal — SIGN-R04):
  1. 分解: 对 v3c 3way 模型跑 TreeSHAP (booster.predict(pred_contrib=True))。ratio 的方向
     信号 = logit_up - logit_down, 故按特征取 signal_shap = shap_up - shap_down。把 247 个
     特征按"回看窗口"分桶 (short<=10 / mid 11-30 / long>=31 / context), 统计每桶 |signal_shap|
     的占比 = 该窗口尺度对 ratio 方向的贡献份额。这是消融/SHAP 口径的"短窗 vs 长窗贡献"。
  2. 漂移: 用 T-001 同口径 regime (过去60日 MA5/MA60 穿越数 ma5_60_cross_60d, p20/p80 切
     trend/mid/choppy)。分 regime 各算一遍贡献份额 → 量化"长窗份额"在快/慢市间的漂移幅度 (pp)。
     再叠加 T-001 已测得的 ignition lead/lag 漂移 (choppy -5d vs trend/mid 0d) 作交叉印证。

预注册判据 (描述性, 非 gate — 真 gate 在 T-005):
  - window_dynamic_needed  iff  max regime 间任一窗口桶份额差 >= 5.0pp  OR  |ignition 漂移| >= 3d
  - 否则 fixed_ok
  T-001 已测 ignition 漂移 = 5d (>=3d), 故漂移侧已触发; SHAP 份额漂移给出第二条独立证据。

checkpoint (SIGN-R08): 采样后的特征+regime 落 research/cache/t002_sample.parquet, 存在即复用。

产出:
  - research/cache/t002_sample.parquet     (采样特征矩阵 + regime, 复用免重算 load_window)
  - research/cache/t002_results.json       (分桶份额 全样本+分 regime + 漂移指标 + 建议 status)
  - 控制台打印窗口贡献份额表
裁决 research/verdicts/T-002.json 由本脚本依真实数字写入。
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window

CACHE = ROOT / "research" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
SAMPLE_CACHE = CACHE / "t002_sample.parquet"
RESULTS = CACHE / "t002_results.json"
VERDICT = ROOT / "research" / "verdicts" / "T-002.json"
PROD = ROOT / "output" / "production"

WIN_START = "20240601"   # 与 T-001 同窗口
WIN_END = "20260101"
EPS = 0.01
SEED = 20260603
N_PER_REGIME = 45000     # 分层采样, 每 regime 上限; SHAP 在 ~13.5w 行上很快

# 预注册描述性判据 (非 gate)
DRIFT_SHARE_PP = 5.0     # 任一窗口桶 regime 间份额差阈值 (百分点)
DRIFT_IGN_DAYS = 3       # ignition lead/lag 漂移阈值 (天), 取自 T-001 结果


def horizon_bucket(name: str) -> tuple[str, int]:
    """把特征名映射到回看窗口桶。返回 (bucket, horizon_int)。
    规则: 取名字里所有整数的最大值作为该特征触及的最长回看窗口。
      short  <=10 ; mid 11-30 ; long >=31 ; 无整数者按语义归 context/short。
    透明可审计: 全部映射会落进 results.json 的 feature_buckets。
    """
    n = name.lower()
    nums = [int(x) for x in re.findall(r"\d+", n)]
    if n.startswith("cdl"):                      # K 线形态: 1-3 根 bar = 短期事件
        return "short", 3
    if nums:
        h = max(nums)
        if h <= 10:
            return "short", h
        if h <= 30:
            return "mid", h
        return "long", h
    # 无窗口数字的: 基本面/市场环境/静态上下文
    CONTEXT = {
        "adx", "winner_rate", "holder_pct", "total_mv", "pe", "pe_ttm", "pb",
        "market_score_adj", "regime_id", "industry_id", "ht_trendmode",
        "mkt_rsi14", "mkt_vol_ratio", "obv", "amount_log", "amount_breakout",
        "bop", "ad", "adosc", "trix", "sar_signal", "vol_price_match",
        "gap_up", "gap_down", "limit_up", "amplitude", "rs_in_decile",
    }
    if n in CONTEXT:
        return "context", 0
    # 兜底: 资金/价量但无显式窗口 → mid (多为 5-20 日动量), 标 horizon=-1 以便审计
    return "mid", -1


def compute_regime() -> pd.DataFrame:
    """T-001 同口径 regime: 60日 MA5/MA60 穿越数, 全 causal。返回 per-day regime 值。"""
    print("[regime] 加载 daily close 算 MA5/MA60 穿越数 ...", flush=True)
    ddir = ROOT / "output" / "tushare_cache" / "daily"
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(ddir.glob("*.parquet"))]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = big.groupby("ts_code")["close"]
    big["ma5"] = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
    big["ma60"] = g.transform(lambda s: s.rolling(60, min_periods=60).mean())
    big["ma5_above"] = (big["ma5"] > big["ma60"]).astype("float")
    big["cross"] = big.groupby("ts_code")["ma5_above"].diff().abs()
    big["ma5_60_cross_60d"] = (big.groupby("ts_code")["cross"]
                               .transform(lambda s: s.rolling(60, min_periods=30).sum()))
    return big[["ts_code", "trade_date", "ma5_60_cross_60d"]]


def build_sample() -> tuple[pd.DataFrame, list[str]]:
    """load_window + 合并 regime + 分层采样。带 checkpoint。返回 (sample_df, feature_cols)。"""
    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json")
                      .read_text(encoding="utf-8"))
    fc = meta["feature_cols"]
    ind_map = meta.get("industry_map", {})

    if SAMPLE_CACHE.exists():
        print(f"[sample] checkpoint 命中 {SAMPLE_CACHE.name}, 跳过 load_window", flush=True)
        return pd.read_parquet(SAMPLE_CACHE), fc

    print(f"[sample] load_window {WIN_START}-{WIN_END} (ST 源头排除, 含 mfk)...", flush=True)
    daily = load_window(WIN_START, WIN_END, with_mfk=True)   # exclude_st=True 默认
    daily["trade_date"] = daily["trade_date"].astype(str)
    if ind_map:
        daily["industry_id"] = (daily["industry"].fillna("unknown")
                                .map(ind_map).fillna(-1).astype(int))
    for c in fc:
        if c not in daily.columns:
            daily[c] = 0.0

    reg = compute_regime()
    daily = daily.merge(reg, on=["ts_code", "trade_date"], how="left")
    daily = daily.dropna(subset=["ma5_60_cross_60d"])
    print(f"[sample] 合并 regime 后 {len(daily):,} 行 / {daily['ts_code'].nunique()} 股", flush=True)

    # regime 切分 (T-001 同口径 p20/p80)
    p20 = daily["ma5_60_cross_60d"].quantile(0.20)
    p80 = daily["ma5_60_cross_60d"].quantile(0.80)
    daily["regime"] = "mid"
    daily.loc[daily["ma5_60_cross_60d"] <= p20, "regime"] = "trend"
    daily.loc[daily["ma5_60_cross_60d"] >= p80, "regime"] = "choppy"

    rng = np.random.RandomState(SEED)
    keep_cols = fc + ["ts_code", "trade_date", "regime", "ma5_60_cross_60d"]
    samples = []
    for r in ["trend", "mid", "choppy"]:
        sub = daily[daily["regime"] == r]
        if len(sub) > N_PER_REGIME:
            idx = rng.choice(sub.index.values, N_PER_REGIME, replace=False)
            sub = sub.loc[idx]
        samples.append(sub[keep_cols])
    sample = pd.concat(samples, ignore_index=True)
    sample.attrs_p20, sample.attrs_p80 = p20, p80   # 仅运行期记录
    sample.to_parquet(SAMPLE_CACHE, index=False)
    print(f"[sample] 落盘 {SAMPLE_CACHE.name} ({len(sample):,} 行; "
          f"regime cut p20={p20:.2f}/p80={p80:.2f})", flush=True)
    return sample, fc


def shap_signal(sample: pd.DataFrame, fc: list[str]) -> np.ndarray:
    """TreeSHAP -> 每特征对 ratio 方向信号(logit_up-logit_down)的 SHAP。返回 (n, F)。"""
    d = PROD / "r5_pump_3way_lgbm_v3c"
    b = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
    F = len(fc)
    X = sample[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"[shap] pred_contrib on {len(X):,} x {F} (3-class TreeSHAP)...", flush=True)
    contrib = b.predict(X, pred_contrib=True)        # (n, n_class*(F+1))
    n_class = contrib.shape[1] // (F + 1)
    assert n_class == 3, f"expected 3 classes, got {n_class}"
    blk = F + 1
    shap_down = contrib[:, 1 * blk: 1 * blk + F]     # class 1 = down
    shap_up = contrib[:, 2 * blk: 2 * blk + F]       # class 2 = up
    return shap_up - shap_down                        # 方向信号的特征贡献


def bucket_shares(sig: np.ndarray, fc: list[str]) -> tuple[dict, dict]:
    """按窗口桶聚合 mean|signal_shap| → 份额。返回 (shares_pct, feature_bucket_map)。"""
    buckets = {f: horizon_bucket(f) for f in fc}
    order = ["short", "mid", "long", "context"]
    absmean = np.abs(sig).mean(axis=0)               # (F,) 每特征平均绝对贡献
    agg = {b: 0.0 for b in order}
    for i, f in enumerate(fc):
        agg[buckets[f][0]] += float(absmean[i])
    tot = sum(agg.values()) or 1.0
    shares = {b: round(100.0 * agg[b] / tot, 2) for b in order}
    fbmap = {f: {"bucket": buckets[f][0], "horizon": buckets[f][1]} for f in fc}
    return shares, fbmap


def main():
    t0 = time.time()
    print("\n=== T-002 窗口非平稳检验 (短窗/长窗贡献 + lead/lag 漂移) ===\n", flush=True)
    sample, fc = build_sample()
    sig = shap_signal(sample, fc)

    # 全样本份额
    overall, fbmap = bucket_shares(sig, fc)
    # 分 regime 份额
    per_regime = {}
    for r in ["trend", "mid", "choppy"]:
        mask = (sample["regime"] == r).to_numpy()
        if mask.sum() < 500:
            per_regime[r] = {"n": int(mask.sum()), "note": "样本不足"}
            continue
        sh, _ = bucket_shares(sig[mask], fc)
        sh["n"] = int(mask.sum())
        per_regime[r] = sh

    # 漂移指标: 各窗口桶在 regime 间的份额极差 (pp)
    drift = {}
    for b in ["short", "mid", "long", "context"]:
        vals = [per_regime[r][b] for r in ["trend", "mid", "choppy"]
                if isinstance(per_regime[r].get(b), (int, float))]
        drift[b] = round(max(vals) - min(vals), 2) if vals else None
    max_share_drift = max(v for v in drift.values() if v is not None)
    # 长窗 vs 短窗 净倾向随 regime 的变化 (慢市 trend 应更偏长窗?)
    long_minus_short = {r: round(per_regime[r]["long"] - per_regime[r]["short"], 2)
                        for r in ["trend", "mid", "choppy"]
                        if "long" in per_regime[r] and "short" in per_regime[r]}

    # T-001 ignition lead/lag 漂移 (复用既有结果, 交叉印证)
    t001 = json.loads((CACHE / "t001_results.json").read_text(encoding="utf-8"))
    igns = t001.get("regime_ignitions", {})
    ign_vals = [v for v in igns.values() if isinstance(v, (int, float))]
    ign_drift = (max(ign_vals) - min(ign_vals)) if ign_vals else 0

    # 描述性判据 (非 gate)
    trigger_share = max_share_drift >= DRIFT_SHARE_PP
    trigger_ign = abs(ign_drift) >= DRIFT_IGN_DAYS
    status = "window_dynamic_needed" if (trigger_share or trigger_ign) else "fixed_ok"

    res = {
        "window": [WIN_START, WIN_END], "seed": SEED, "n_sample": int(len(sample)),
        "n_stocks": int(sample["ts_code"].nunique()),
        "shap_signal_def": "shap_up - shap_down (class2-class1), 驱动 ratio=P_up/(P_down+eps) 方向",
        "bucket_rule": "max int in name; <=10 short / 11-30 mid / >=31 long / cdl*=short / 无窗静态=context",
        "overall_share_pct": overall,
        "per_regime_share_pct": per_regime,
        "share_drift_pp": drift,
        "max_share_drift_pp": round(max_share_drift, 2),
        "long_minus_short_pp_by_regime": long_minus_short,
        "t001_regime_ignitions": igns,
        "ignition_drift_days": int(ign_drift),
        "criteria": {"drift_share_pp_thr": DRIFT_SHARE_PP, "drift_ign_days_thr": DRIFT_IGN_DAYS,
                     "trigger_share": bool(trigger_share), "trigger_ignition": bool(trigger_ign)},
        "suggested_status": status,
        "feature_buckets": fbmap,
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印
    print("\n=== 窗口贡献份额 (%, |signal_shap| 占比) ===", flush=True)
    hdr = f"{'regime':10s} {'n':>8s} {'short':>8s} {'mid':>8s} {'long':>8s} {'context':>8s}"
    print(hdr, flush=True)
    print(f"{'overall':10s} {len(sample):>8d} "
          f"{overall['short']:>8.2f} {overall['mid']:>8.2f} {overall['long']:>8.2f} {overall['context']:>8.2f}",
          flush=True)
    for r in ["trend", "mid", "choppy"]:
        d = per_regime[r]
        if "short" not in d:
            continue
        print(f"{r:10s} {d['n']:>8d} {d['short']:>8.2f} {d['mid']:>8.2f} "
              f"{d['long']:>8.2f} {d['context']:>8.2f}", flush=True)
    print(f"\n窗口份额 regime 极差 (pp): {drift}", flush=True)
    print(f"long-short 净份额 (pp) by regime: {long_minus_short}", flush=True)
    print(f"T-001 ignition 漂移: {igns} → {ign_drift}d", flush=True)
    print(f"\n判据: max份额漂移={max_share_drift:.2f}pp(>= {DRIFT_SHARE_PP}? {trigger_share}) | "
          f"ignition漂移={abs(ign_drift)}d(>= {DRIFT_IGN_DAYS}? {trigger_ign})", flush=True)
    print(f"建议 status = {status}", flush=True)
    print(f"\n[done] -> {RESULTS.relative_to(ROOT)}  耗时 {time.time()-t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    main()
