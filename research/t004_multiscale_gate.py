# -*- coding: utf-8 -*-
"""T-004 多尺度门控启动子 — 最廉价的"动态窗口"实现.

研究动机 (承接 T-001/T-002/T-003):
  T-001: ratio 相对 MA5 上拐点是 sync(确认型), 但 choppy regime ignition 漂移到 -5d。
  T-002: 非平稳来自*相位/时序*而非*窗口配比* (短/长窗 SHAP 份额跨 regime 极差仅 2.42pp)。
  ⇒ 推论: 给"启动子"一个动态有效窗口的*最廉价*办法, 不是连续地重配短/长窗特征权重
     (那一块本就平稳), 而是: 训 K 个不同*预测窗口尺度*的 pump 模型, 再用 regime/run-length
     门控在它们之间硬选择。门控让"有效窗口"随快/慢市切换 = 动态窗口的离散近似。

设计 (K=3 个窗口尺度 = 预测前向窗口 H):
  scale  H(前向天)  上涨阈 g_H   含义
  s3     3          +6%          短窗 — 抓快速启动 (适合 choppy: ignition 领先)
  s5     5          +10%         中窗 — 与生产 v3c 同档 (mid regime 同步)
  s10    10         +15%         长窗 — 抓持续趋势 (适合 trend: 同步且更稳)
  每尺度一个 LGBM 3way (0 中性 / 1 跌 / 2 涨), ratio_sH = P[涨]/(P[跌]+eps)。

门控 (全 causal, 取自 T-001/T-002 实证, 非拟合):
  regime = 过去 60 日 MA5/MA60 穿越数 (ma5_60_cross_60d), p20/p80 切 trend/mid/choppy
           — 与 T-001/T-002/T-003 完全同口径。
  run_len = 连续 MA5 斜率>0 的天数 (causal)。
  规则: 基线尺度 choppy->s3 / mid->s5 / trend->s10 (响应性随快市变高);
        run_len>=10 (老趋势) 升一档更长, run_len<=2 (刚起步) 降一档更短; 夹到 {3,5,10}。
  ratio_dyn = 被选尺度的 ratio_sH (硬选择, 离散动态窗口)。

零前视 / 红线纪律:
  - 特征只用 t 及之前 (load_window causal); *标签*用前向 high/low 仅供训练, 绝不入 features parquet。
  - 三个 pump 模型落 research/models/pump_scale_{H}/, 完全不碰生产模型 (SIGN-R05)。
  - ST 源头排除 (load_window exclude_st=True 默认)。
  - 输出 research/features/dynamic_ratio.parquet 列名自检不撞 forward-field 黑名单 (SIGN-R04)。
  - IC / 选尺度分布仅描述, 非 gate; ship 与否只由 T-005 walk-forward 决定 (SIGN-R03)。

checkpoint (SIGN-R08, >5min 必须):
  - 每尺度模型: research/models/pump_scale_{H}/classifier.txt 存在即跳过训练。
  - 逐尺度推理 ratio: research/cache/t004_scale_ratios.parquet 存在即跳过 load_window+推理,
    只重建门控 (秒级), 便于调门控逻辑而不重算。

产出:
  - research/models/pump_scale_{3,5,10}/   (classifier.txt + meta.json, 3 个 pump 模型)
  - research/cache/t004_scale_ratios.parquet (逐尺度 ratio, 复用)
  - research/features/dynamic_ratio.parquet  (ratio_s3/s5/s10 + regime/run_len/scale_sel/ratio_dyn)
  - research/cache/t004_results.json
裁决 research/verdicts/T-004.json 由本脚本写入 (status=built)。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from train_v15_refresh import load_window

CACHE = ROOT / "research" / "cache"
FEATURES = ROOT / "research" / "features"
MODELS = ROOT / "research" / "models"
CACHE.mkdir(parents=True, exist_ok=True)
FEATURES.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
SCALE_RATIOS = CACHE / "t004_scale_ratios.parquet"
OUT_PARQUET = FEATURES / "dynamic_ratio.parquet"
RESULTS = CACHE / "t004_results.json"
VERDICT = ROOT / "research" / "verdicts" / "T-004.json"
PROD = ROOT / "output" / "production"

WIN_START, WIN_END = "20240601", "20260101"
EPS = 0.01
SEED = 20260603
# 训练用时间切分 (cutoff 之前训练, 之后早停验证; H=10 需前向 ~10 日, cutoff 留足余量)
TRAIN_CUTOFF = "20251001"
N_TRAIN, N_VAL = 500_000, 120_000

# 窗口尺度: H = 前向预测天数; g = 上涨判定阈 (max_gain); DD_CAP = 上涨需浅回撤
SCALES = {3: 0.06, 5: 0.10, 10: 0.15}
DD_CAP = 0.05      # 上涨标签的最大允许回撤 (与 v3c uptrend 浅回撤同精神)

# 门控参数 (取自 T-001/T-002 实证, 非拟合)
RUN_LONG, RUN_FRESH = 10, 2
SCALE_LIST = [3, 5, 10]                 # 从短到长, 用于升/降档
REGIME_BASE = {"choppy": 3, "mid": 5, "trend": 10}


# ───────────────────────── daily close/high/low ─────────────────────────
def load_daily() -> pd.DataFrame:
    print("[daily] 加载 daily close/high/low 全历史 ...", flush=True)
    ddir = ROOT / "output" / "tushare_cache" / "daily"
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close", "high", "low"])
             for f in sorted(ddir.glob("*.parquet"))]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    return big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def build_label(slim: pd.DataFrame, H: int, g: float) -> pd.Series:
    """3way pump 标签 (causal-free 仅训练用): 0 中性 / 1 跌 / 2 涨。
    涨 = 前向 H 日 max_gain>=g 且 max_dd>=-DD_CAP; 跌 = 前向 H 日 max_dd<=-g。"""
    gh = slim.groupby("ts_code")["high"]
    gl = slim.groupby("ts_code")["low"]
    fwd_hi = pd.concat([gh.shift(-k) for k in range(1, H + 1)], axis=1).max(axis=1)
    fwd_lo = pd.concat([gl.shift(-k) for k in range(1, H + 1)], axis=1).min(axis=1)
    gain = fwd_hi / slim["close"] - 1.0
    dd = fwd_lo / slim["close"] - 1.0
    lbl = pd.Series(np.nan, index=slim.index, dtype="float64")
    valid = gain.notna() & dd.notna()
    up = valid & (gain >= g) & (dd >= -DD_CAP)
    down = valid & (dd <= -g)
    lbl[valid] = 0.0
    lbl[down] = 1.0
    lbl[up] = 2.0          # up 优先 (与 down 互斥的概率极小, 但显式后写保证 up 覆盖)
    return lbl


# ───────────────────────── regime / run-length (causal) ─────────────────────────
def compute_gate_inputs(daily: pd.DataFrame) -> pd.DataFrame:
    """T-001/T-002/T-003 同口径 regime + run_len, 全 causal。"""
    print("[gate] 算 MA5/MA60 regime + run_len (causal) ...", flush=True)
    d = daily.copy()
    g = d.groupby("ts_code")["close"]
    d["ma5"] = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
    d["ma60"] = g.transform(lambda s: s.rolling(60, min_periods=60).mean())
    d["ma5_above"] = (d["ma5"] > d["ma60"]).astype("float")
    d["cross"] = d.groupby("ts_code")["ma5_above"].diff().abs()
    d["ma5_60_cross_60d"] = (d.groupby("ts_code")["cross"]
                             .transform(lambda s: s.rolling(60, min_periods=30).sum()))
    # run_len: 连续 MA5 斜率>0 天数 (causal)
    d["ma5_slope"] = d.groupby("ts_code")["ma5"].diff()
    up = (d["ma5_slope"] > 0).astype("int")
    # 分组内连续游程长度: 每遇 up=0 重置
    grp_break = (up == 0).groupby(d["ts_code"]).cumsum()
    d["run_len"] = up.groupby([d["ts_code"], grp_break]).cumsum()
    return d[["ts_code", "trade_date", "ma5_60_cross_60d", "run_len"]]


def assign_regime(s: pd.Series) -> pd.Series:
    p20, p80 = s.quantile(0.20), s.quantile(0.80)
    r = pd.Series("mid", index=s.index)
    r[s <= p20] = "trend"
    r[s >= p80] = "choppy"
    return r, float(p20), float(p80)


def gate_select(regime: pd.Series, run_len: pd.Series) -> pd.Series:
    """regime 基线尺度 + run_len 升/降档, 夹到 {3,5,10}。返回被选 H。"""
    base = regime.map(REGIME_BASE).fillna(5).astype(int)
    idx = base.map({3: 0, 5: 1, 10: 2})
    bump = pd.Series(0, index=regime.index)
    bump[run_len >= RUN_LONG] = 1      # 老趋势 -> 更长
    bump[run_len <= RUN_FRESH] = -1    # 刚起步 -> 更短
    new_idx = (idx + bump).clip(0, 2)
    return new_idx.map({0: 3, 1: 5, 2: 10}).astype(int)


# ───────────────────────── 训练 + 推理 ─────────────────────────
def train_one_scale(daily_feat: pd.DataFrame, fc: list[str], label: pd.Series,
                    H: int) -> lgb.Booster:
    mdir = MODELS / f"pump_scale_{H}"
    mfile = mdir / "classifier.txt"
    if mfile.exists():
        print(f"[train s{H}] checkpoint 命中, 跳过训练", flush=True)
        return lgb.Booster(model_str=mfile.read_text(encoding="utf-8"))
    mdir.mkdir(parents=True, exist_ok=True)

    df = daily_feat[["trade_date"]].copy()
    df["_y"] = label.values
    df = df.dropna(subset=["_y"])
    tr_mask = df["trade_date"] < TRAIN_CUTOFF
    va_mask = ~tr_mask
    rng = np.random.RandomState(SEED + H)
    tr_idx = df[tr_mask].index.values
    va_idx = df[va_mask].index.values
    if len(tr_idx) > N_TRAIN:
        tr_idx = rng.choice(tr_idx, N_TRAIN, replace=False)
    if len(va_idx) > N_VAL:
        va_idx = rng.choice(va_idx, N_VAL, replace=False)

    def Xy(idx):
        X = (daily_feat.loc[idx, fc].astype("float32")
             .replace([np.inf, -np.inf], np.nan).fillna(0))
        y = label.loc[idx].astype("int")
        return X, y

    Xtr, ytr = Xy(tr_idx)
    Xva, yva = Xy(va_idx)
    dist = ytr.value_counts(normalize=True).round(3).to_dict()
    print(f"[train s{H}] train={len(Xtr):,} val={len(Xva):,} 类分布={dist}", flush=True)
    params = dict(objective="multiclass", num_class=3, metric="multi_logloss",
                  num_leaves=31, learning_rate=0.05, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, min_child_samples=80,
                  seed=SEED + H, verbose=-1, num_threads=0)
    dtr = lgb.Dataset(Xtr, ytr)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=300, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(30, verbose=False),
                                   lgb.log_evaluation(0)])
    mfile.write_text(booster.model_to_string(), encoding="utf-8")
    (mdir / "meta.json").write_text(json.dumps({
        "scale_H": H, "gain_threshold": SCALES[H], "dd_cap": DD_CAP,
        "best_iter": booster.best_iteration, "n_train": len(Xtr), "n_val": len(Xva),
        "train_class_dist": dist, "feature_cols": fc,
        "label_def": f"up: fwd{H}d max_gain>={SCALES[H]} & max_dd>=-{DD_CAP}; down: max_dd<=-{SCALES[H]}",
        "note": "研究模型, 非生产; T-004 多尺度门控启动子",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train s{H}] best_iter={booster.best_iteration} -> {mfile.relative_to(ROOT)}", flush=True)
    return booster


def infer_all_scales() -> pd.DataFrame:
    """load_window 一次, 训 3 尺度, 逐尺度推理 ratio。带 checkpoint。"""
    if SCALE_RATIOS.exists():
        print(f"[infer] checkpoint 命中 {SCALE_RATIOS.name}, 跳过 load_window+训练+推理", flush=True)
        return pd.read_parquet(SCALE_RATIOS)

    meta = json.loads((PROD / "r5_pump_3way_lgbm_v3c" / "feature_meta.json")
                      .read_text(encoding="utf-8"))
    fc = meta["feature_cols"]
    ind_map = meta.get("industry_map", {})

    print(f"[infer] load_window {WIN_START}-{WIN_END} (ST 源头排除, 含 mfk)...", flush=True)
    daily = load_window(WIN_START, WIN_END, with_mfk=True)
    daily["trade_date"] = daily["trade_date"].astype(str)
    if ind_map:
        daily["industry_id"] = (daily["industry"].fillna("unknown")
                                .map(ind_map).fillna(-1).astype(int))
    for c in fc:
        if c not in daily.columns:
            daily[c] = 0.0
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"[infer] daily {len(daily):,} 行 / {daily['ts_code'].nunique()} 股", flush=True)

    # 标签需要 close/high/low; load_window 可能不含, 从 daily_cache 合并
    px = load_daily()
    daily = daily.merge(px, on=["ts_code", "trade_date"], how="left",
                        suffixes=("", "_px"))
    # 价列优先用 daily_cache 的, 保证标签口径一致
    for c in ["close", "high", "low"]:
        if f"{c}_px" in daily.columns:
            daily[c] = daily[f"{c}_px"]
            daily.drop(columns=[f"{c}_px"], inplace=True)
    slim = daily[["ts_code", "trade_date", "close", "high", "low"]].copy()

    out = daily[["ts_code", "trade_date"]].copy()
    for H, g in SCALES.items():
        label = build_label(slim, H, g)
        booster = train_one_scale(daily, fc, label, H)
        print(f"[infer s{H}] 推理 ratio_s{H} ...", flush=True)
        n = len(daily)
        CHUNK = 300_000
        pu = np.empty(n, dtype="float64")
        pd_ = np.empty(n, dtype="float64")
        for s0 in range(0, n, CHUNK):
            sl = daily.iloc[s0:s0 + CHUNK]
            X = sl[fc].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0)
            proba = booster.predict(X, num_iteration=booster.best_iteration)
            pd_[s0:s0 + len(sl)] = proba[:, 1]
            pu[s0:s0 + len(sl)] = proba[:, 2]
        out[f"ratio_s{H}"] = pu / (pd_ + EPS)
        del label
    out.to_parquet(SCALE_RATIOS, index=False)
    print(f"[infer] 落盘 {SCALE_RATIOS.relative_to(ROOT)} ({len(out):,} 行)", flush=True)
    return out


# ───────────────────────── 主流程 ─────────────────────────
def main():
    t0 = time.time()
    print("\n=== T-004 多尺度门控启动子 (最廉价动态窗口) ===\n", flush=True)

    scale_ratios = infer_all_scales()           # ts_code,trade_date,ratio_s3/s5/s10

    daily = load_daily()
    gate = compute_gate_inputs(daily)
    df = scale_ratios.merge(gate, on=["ts_code", "trade_date"], how="left")
    df = df[(df["trade_date"] >= WIN_START) & (df["trade_date"] <= WIN_END)]
    df = df.dropna(subset=["ma5_60_cross_60d"]).reset_index(drop=True)

    regime, p20, p80 = assign_regime(df["ma5_60_cross_60d"])
    df["regime"] = regime
    df["run_len"] = df["run_len"].fillna(0).astype(int)
    df["scale_sel"] = gate_select(df["regime"], df["run_len"])

    # ratio_dyn = 被选尺度的 ratio (硬选择)
    sel = df["scale_sel"].to_numpy()
    ratio_dyn = np.where(sel == 3, df["ratio_s3"],
                 np.where(sel == 5, df["ratio_s5"], df["ratio_s10"]))
    df["ratio_dyn"] = ratio_dyn.astype("float32")

    # ── 落盘 features (列名自检, SIGN-R04) ──
    keep = ["ts_code", "trade_date", "ratio_s3", "ratio_s5", "ratio_s10",
            "regime", "run_len", "scale_sel", "ratio_dyn"]
    out = df[keep].copy()
    forbidden = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                {"max_gain", "maxgain", "future_ret", "fwd_ret", "label", "target"}
    bad = [c for c in out.columns
           if c.lower() in forbidden
           or c.lower().startswith(("fwd_", "future_", "next_"))
           or c.lower().endswith("_forward")]
    assert not bad, f"列名撞 forward-field 黑名单: {bad}"
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"[out] 落盘 {OUT_PARQUET.relative_to(ROOT)} ({len(out):,} 行 x {len(out.columns)} 列)", flush=True)

    # ── 描述性统计 (SIGN-R03: 仅描述) ──
    sel_dist = (df["scale_sel"].value_counts(normalize=True)
                .sort_index().round(4).to_dict())
    sel_by_regime = {}
    for r in ["trend", "mid", "choppy"]:
        sub = df[df["regime"] == r]
        if len(sub):
            sel_by_regime[r] = (sub["scale_sel"].value_counts(normalize=True)
                                .sort_index().round(3).to_dict())
    # ratio_dyn 与各固定尺度的相关 (描述动态窗口"切换"了多少)
    corr = {f"ratio_s{H}": round(float(df["ratio_dyn"].corr(df[f"ratio_s{H}"])), 4)
            for H in SCALE_LIST}
    frac_diff_from_s5 = round(float((df["scale_sel"] != 5).mean()), 4)

    models_info = {}
    for H in SCALE_LIST:
        mp = MODELS / f"pump_scale_{H}" / "meta.json"
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            models_info[f"s{H}"] = {k: m[k] for k in
                                    ("scale_H", "gain_threshold", "best_iter",
                                     "n_train", "train_class_dist") if k in m}

    res = {
        "window": [WIN_START, WIN_END], "seed": SEED,
        "n_rows": int(len(out)), "n_stocks": int(out["ts_code"].nunique()),
        "scales": {f"s{H}": {"H": H, "gain_thr": g} for H, g in SCALES.items()},
        "regime_cut": {"cross_p20": p20, "cross_p80": p80},
        "gate_rule": {"regime_base": REGIME_BASE, "run_long": RUN_LONG,
                      "run_fresh": RUN_FRESH,
                      "logic": "regime 基线尺度; run_len>=run_long 升档/<=run_fresh 降档; 夹 {3,5,10}"},
        "scale_selected_dist": sel_dist,
        "scale_selected_by_regime": sel_by_regime,
        "frac_dynamic_vs_s5": frac_diff_from_s5,
        "ratio_dyn_corr_with_fixed": corr,
        "models": models_info,
        "note": "研究脚手架, 不碰生产; 选尺度分布/相关仅描述非 gate (SIGN-R03); ship 由 T-005 walk-forward 定",
    }
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印
    print("\n=== 门控选尺度分布 (整体) ===", flush=True)
    for k, v in sel_dist.items():
        print(f"  s{int(k)}: {v:.1%}", flush=True)
    print("\n=== 选尺度分布 (分 regime) ===", flush=True)
    for r in ["trend", "mid", "choppy"]:
        if r in sel_by_regime:
            cells = "  ".join(f"s{int(k)}={v:.0%}" for k, v in sel_by_regime[r].items())
            print(f"  {r:8s} {cells}", flush=True)
    print(f"\n动态窗口切换比例 (scale_sel != s5): {frac_diff_from_s5:.1%}", flush=True)
    print(f"ratio_dyn vs 固定尺度相关: {corr}", flush=True)

    # ── 裁决 ──
    conclusion = (f"3 尺度 pump 模型 (s3/s5/s10) 落 research/models/ + regime/run_len 门控产出动态窗口 "
                  f"ratio_dyn ({len(out):,} 行/{out['ts_code'].nunique()} 股); 门控切换比例 "
                  f"{frac_diff_from_s5:.0%} (choppy 偏 s3 / trend 偏 s10); 推理路径样本窗口全跑通, "
                  f"生产模型未触碰。IC/切换分布仅描述非 gate, ship 由 T-005 walk-forward 定。")
    VERDICT.write_text(json.dumps({
        "id": "T-004", "status": "built", "conclusion": conclusion,
        "artifacts": ["research/models/pump_scale_3/", "research/models/pump_scale_5/",
                      "research/models/pump_scale_10/",
                      "research/features/dynamic_ratio.parquet",
                      "research/cache/t004_scale_ratios.parquet",
                      "research/cache/t004_results.json"],
        "scale_selected_dist": {f"s{int(k)}": v for k, v in sel_dist.items()},
        "frac_dynamic_vs_s5": frac_diff_from_s5,
        "note": "依 T-002 window_dynamic_needed 解除条件门; 门控用 T-001/T-002 实证规则非拟合; "
                "中间指标非 gate (SIGN-R03)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] verdict=built -> {VERDICT.relative_to(ROOT)}  耗时 {time.time()-t0:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    main()
