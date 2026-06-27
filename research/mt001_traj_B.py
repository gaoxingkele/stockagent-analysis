# -*- coding: utf-8 -*-
"""MT-001 — 方案 B (体用=趋势vs当日): 轨迹起卦特征 + 升级消融 Phase-1 筛查.

北极星升级 (SIGN-R12+): 这次问题是真的 — 把"近N日走势"做进梅花。
  体卦 = 近N日多尺度 MA 量化状态 (既有趋势) → 上卦
  用卦 = 当日单日 K 线结构 (今日冲击)        → 下卦, 动爻锁在下卦
  ⇒ 体用五行生克 = "今日冲击 vs 既有趋势" 的数论编码。全因果 (只用 t 及之前路径)。

升级消融 (vs MH-002 只扣 月×板块): 残差化 r20 必须**额外扣除标准动量/趋势因子**
  (mom_5 / mom_20 / ma_pos / rsi_14) 后, B 卦象特征**仍有独立残差**才算 residual_signal。
  否则 = 卦象只是动量换皮 ⇒ no_residual (廉价 REJECT, SIGN-R12)。

口径 (纯描述, 非 gate, SIGN-R03; 落地只认 MT-004 walk-forward):
  label r20 = close[t+20]/close[t]-1 (全因果, 只进 cache 不进 features, SIGN-R04)。
  横截面 rank-IC: 逐日 Spearman 再跨日平均; t=mean/std*sqrt(N_dates)。
  OOF target encoding: 月度扩张窗, target 月 M 编码用 月<=M-gap 历史 (gap=2)。
  双重残差: resid1 = r20 - OOF(cal_month×industry); resid2 = resid1 逐日横截面回归扣动量。
  分层 (SIGN-R11): 全期 + momentum/mixed/reversal 分别报。

裁决 (Phase-1 screen 标准, 启动即固定): 某动态 mhB_* 对 resid2 全期 |IC|>=0.01 且 |t|>=3.0
  → residual_signal; 否则 no_residual。 (gate t 门槛 3.0 来自 prd.preRegisteredGate)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from meihua_encoder import _trigram_from_bits, build_tiyong_B_lookup  # noqa: E402

DAILY = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
FEAT_OUT = ROOT / "research" / "features" / "meihua_traj_B.parquet"
CACHE = ROOT / "research" / "cache"
PANEL_CACHE = CACHE / "mt001_panel.parquet"      # 含 r20 + 动量因子 → 留 cache 不进 features
RESULTS = CACHE / "mt001_results.json"
VERDICT = ROOT / "research" / "verdicts" / "MT-001.json"

KEY = ["ts_code", "trade_date"]
HORIZON = 20
GAP_MONTHS = 2
IC_FLOOR = 0.01
T_FLOOR = 3.0                                     # 升级 gate (prd phase1 |t|>=3)
MIN_HISTORY = 20                                  # ma20 / mom_20 需 20 日历史

MHB_FEATS = ["mhB_base_gua", "mhB_mutual_gua", "mhB_changed_gua",
             "mhB_upper", "mhB_lower", "mhB_moving",
             "mhB_ti_elem", "mhB_yong_elem", "mhB_relation", "mhB_yang_count"]
MOM_FACTORS = ["mom_5", "mom_20", "ma_pos", "rsi_14"]


# ───────────────────────── ST 源头排除 ─────────────────────────
def load_st_set() -> set:
    if not BASIC.exists():
        return set()
    b = pd.read_parquet(BASIC)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(b[b["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


# ───────────────────────── 因果 RSI(14) ─────────────────────────
def rsi_causal(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    ru = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rd = dn.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = ru / rd.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


# ───────────────────────── 轨迹 B 特征 + 动量因子 (因果) ─────────────────────────
def build_features() -> pd.DataFrame:
    """返回 key + mhB_* (落 features) + r20/动量因子/分层列 (落 cache panel)."""
    if PANEL_CACHE.exists() and FEAT_OUT.exists():
        print(f"[build] checkpoint 命中 {PANEL_CACHE.name} + {FEAT_OUT.name}", flush=True)
        return pd.read_parquet(PANEL_CACHE)

    print("[build] 加载全历史 daily OHLC ...", flush=True)
    cols = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close"]
    parts = [pd.read_parquet(f, columns=cols) for f in sorted(DAILY.glob("*.parquet"))]
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    px = (px[(px["close"] > 0) & (px["pre_close"] > 0)]
          .drop_duplicates(KEY, keep="last").sort_values(KEY).reset_index(drop=True))
    st = load_st_set()
    if st:
        before = len(px)
        px = px[~px["ts_code"].isin(st)].reset_index(drop=True)
        print(f"  ST 源头排除: {before - len(px):,} 行 ({len(st)} 只)", flush=True)

    print("[build] 逐股因果滚动 (MA 量化 / 当日 K 结构 / 动量因子) ...", flush=True)
    g = px.groupby("ts_code", sort=False)["close"]
    ma5 = g.transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma10 = g.transform(lambda s: s.rolling(10, min_periods=10).mean())
    ma20 = g.transform(lambda s: s.rolling(20, min_periods=20).mean())

    # ── 体卦(趋势) 3 bit: 多尺度 MA 堆叠 (下->上) ──
    t1 = (px["close"] > ma5).astype(int)
    t2 = (ma5 > ma10).astype(int)
    t3 = (ma10 > ma20).astype(int)
    # ── 用卦(今日) 3 bit: 当日 K 线结构 (下->上) ──
    u1 = (px["close"] > px["open"]).astype(int)                       # 阳线
    u2 = (px["close"] > px["pre_close"]).astype(int)                  # 跳涨
    u3 = ((px["close"] - px["low"]) > (px["high"] - px["close"])).astype(int)  # 收上半
    # ── 动爻(今日冲击强度) → 锁在下卦 1..3 ──
    amp = (px["high"] - px["low"]) / px["pre_close"]
    amp_int = np.round(amp.fillna(0.0) * 1000).astype(int).abs()

    code2tri = np.zeros(8, dtype=int)                                 # 3bit code -> 卦 idx
    for b1 in (0, 1):
        for b2 in (0, 1):
            for b3 in (0, 1):
                code2tri[b1 * 4 + b2 * 2 + b3] = _trigram_from_bits(b1, b2, b3)
    ti_tri = code2tri[(t1 * 4 + t2 * 2 + t3).values]
    yong_tri = code2tri[(u1 * 4 + u2 * 2 + u3).values]
    mv = (amp_int % 3).values
    mv = np.where(mv == 0, 3, mv)

    px["ti_tri"], px["yong_tri"], px["mv"] = ti_tri, yong_tri, mv
    # 历史不足 (ma20 NaN) 的行无法起体卦 → 剔除
    valid = ma20.notna().values & ma5.notna().values & ma10.notna().values
    px = px[valid].reset_index(drop=True)

    # ── 全枚举查表 map → mhB_* 特征 (确定性, 向量化) ──
    lut = build_tiyong_B_lookup()
    lut_df = pd.DataFrame(
        [{"ti_tri": k[0], "yong_tri": k[1], "mv": k[2], **v} for k, v in lut.items()])
    px = px.merge(lut_df, on=["ti_tri", "yong_tri", "mv"], how="left")
    assert px[MHB_FEATS].notna().all().all(), "查表 map 出现 NaN (体/用/动越界)"

    # ── 动量/趋势 因子 (消融对照, 因果; px 已过滤重排, 按当前分组重算) ──
    g2 = px.groupby("ts_code", sort=False)["close"]
    px["mom_5"] = g2.transform(lambda s: s / s.shift(5) - 1.0)
    px["mom_20"] = g2.transform(lambda s: s / s.shift(20) - 1.0)
    px["ma_pos"] = px["close"] / g2.transform(
        lambda s: s.rolling(20, min_periods=20).mean()) - 1.0
    px["rsi_14"] = g2.transform(lambda s: rsi_causal(s, 14))

    # ── 前向 r20 (label, 因果, 只进 cache) ──
    px["r20"] = g2.transform(lambda s: s.shift(-HORIZON)) / px["close"] - 1.0

    # ── 分层列 ──
    rg = pd.read_parquet(REGIME_P)[["trade_date", "regime"]]
    rg["trade_date"] = rg["trade_date"].astype(str)
    px = px.merge(rg, on="trade_date", how="left")
    px["regime"] = px["regime"].fillna("unknown")
    basic = pd.read_parquet(BASIC)[["ts_code", "industry"]].drop_duplicates("ts_code")
    basic["industry"] = basic["industry"].fillna("unknown")
    px = px.merge(basic, on="ts_code", how="left")
    px["industry"] = px["industry"].fillna("unknown")
    px["ym"] = px["trade_date"].str[:6]
    px["cal_month"] = px["trade_date"].str[4:6]

    # ── 落盘: features = 仅 key + mhB_* (零泄漏); panel = 全列 (含 r20/动量) 进 cache ──
    feat_out = px[KEY + MHB_FEATS].copy()
    bad = [c for c in feat_out.columns if c.lower() in
           {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} or c.lower().startswith("fwd_")]
    assert not bad, f"features 含 forward 字段 (泄漏): {bad}"
    FEAT_OUT.parent.mkdir(parents=True, exist_ok=True)
    feat_out.to_parquet(FEAT_OUT, index=False)
    print(f"[build] features -> {FEAT_OUT.relative_to(ROOT)} "
          f"({len(feat_out):,} 行 × {len(MHB_FEATS)} mhB_)", flush=True)

    panel = px[KEY + MHB_FEATS + MOM_FACTORS +
               ["r20", "regime", "industry", "ym", "cal_month"]].copy()
    panel = panel[panel["r20"].notna()].reset_index(drop=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_CACHE, index=False)
    print(f"[build] panel(cache) -> {PANEL_CACHE.relative_to(ROOT)} ({len(panel):,} 行)",
          flush=True)
    return panel


# ───────────────────── 横截面 rank-IC (向量化) ─────────────────────
def xs_ic(df: pd.DataFrame, feat: str, target: str) -> dict:
    sub = df[["trade_date", feat, target]].dropna()
    if sub.empty:
        return {"ic_mean": np.nan, "ic_t": np.nan, "n_dates": 0}
    rf = sub.groupby("trade_date")[feat].rank()
    rt = sub.groupby("trade_date")[target].rank()
    tmp = pd.DataFrame({"d": sub["trade_date"].values, "f": rf.values, "t": rt.values})
    tmp["ft"] = tmp["f"] * tmp["t"]
    a = tmp.groupby("d").agg(n=("f", "size"), sf=("f", "sum"), st=("t", "sum"),
                             sff=("f", lambda x: float(np.dot(x, x))),
                             stt=("t", lambda x: float(np.dot(x, x))),
                             sft=("ft", "sum"))
    n = a["n"]
    cov = a["sft"] - a["sf"] * a["st"] / n
    vf = a["sff"] - a["sf"] ** 2 / n
    vt = a["stt"] - a["st"] ** 2 / n
    denom = np.sqrt(vf * vt)
    ic = (cov / denom).where((denom > 0) & (n >= 2)).dropna()
    if len(ic) < 2:
        return {"ic_mean": np.nan, "ic_t": np.nan, "n_dates": int(len(ic))}
    m, s = float(ic.mean()), float(ic.std(ddof=1))
    t = m / s * np.sqrt(len(ic)) if s > 0 else np.nan
    return {"ic_mean": m, "ic_t": float(t), "n_dates": int(len(ic)),
            "ic_std": s, "ic_ir": (m / s if s > 0 else np.nan)}


# ───────────────────── 月度扩张窗 OOF target encoding ─────────────────────
def oof_encode(df: pd.DataFrame, cat_cols: list[str], target: str = "r20",
               gap: int = GAP_MONTHS) -> pd.Series:
    months = sorted(df["ym"].unique())
    key = (df[cat_cols[0]].astype(str) if len(cat_cols) == 1
           else df[cat_cols].astype(str).agg("|".join, axis=1))
    tmp = pd.DataFrame({"ym": df["ym"].values, "key": key.values,
                        "y": df[target].values}, index=df.index)
    grp = tmp.groupby(["ym", "key"])["y"].agg(["sum", "count"]).reset_index()
    piv_s = grp.pivot(index="ym", columns="key", values="sum").reindex(months).fillna(0).cumsum()
    piv_c = grp.pivot(index="ym", columns="key", values="count").reindex(months).fillna(0).cumsum()
    piv_mean = (piv_s / piv_c.replace(0, np.nan)).shift(gap)
    enc_long = piv_mean.stack().rename("enc").reset_index()
    out = (tmp.reset_index().merge(enc_long, on=["ym", "key"], how="left")
           .set_index("index")["enc"])
    return out.reindex(df.index)


# ───────────────────── 逐日横截面回归扣动量 (升级消融) ─────────────────────
def residualize_momentum(df: pd.DataFrame, ycol: str, factor_cols: list[str]) -> pd.Series:
    """每个交易日横截面 OLS: ycol ~ 标准化动量因子, 返回残差 (扣线性动量成分)."""
    out = np.full(len(df), np.nan)
    pos = {ix: i for i, ix in enumerate(df.index)}
    for _, sub in df.groupby("trade_date"):
        y = sub[ycol].values
        X = sub[factor_cols].values
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if mask.sum() < len(factor_cols) + 5:
            continue
        Xm, ym = X[mask], y[mask]
        sd = Xm.std(0)
        sd[sd == 0] = 1.0
        Xs = (Xm - Xm.mean(0)) / sd
        A = np.column_stack([np.ones(mask.sum()), Xs])
        beta, *_ = np.linalg.lstsq(A, ym, rcond=None)
        r = ym - A @ beta
        idxs = [pos[ix] for ix in sub.index[mask]]
        out[idxs] = r
    return pd.Series(out, index=df.index)


def ic_table(df: pd.DataFrame, feat: str, label: str, target: str) -> dict:
    res = {"feature": label, "all": xs_ic(df, feat, target)}
    for reg in ["momentum", "mixed", "reversal"]:
        sub = df[df["regime"] == reg]
        res[reg] = (xs_ic(sub, feat, target) if len(sub)
                    else {"ic_mean": np.nan, "ic_t": np.nan, "n_dates": 0})
    return res


def clean(d):
    return {k: (None if isinstance(v, float) and np.isnan(v) else
                (round(v, 6) if isinstance(v, float) else v)) for k, v in d.items()}


def pack(table):
    return {seg: clean(table[seg]) for seg in ["all", "momentum", "mixed", "reversal"]}


def main() -> int:
    t0 = time.time()
    print("\n=== MT-001 方案B 轨迹起卦 (体用=趋势vs当日) + 升级消融 Phase-1 ===\n", flush=True)
    df = build_features()
    print(f"[panel] {len(df):,} 行 / {df['ts_code'].nunique()} 股 / "
          f"{df['trade_date'].nunique()} 日; regime "
          f"{df['regime'].value_counts(normalize=True).round(3).to_dict()}", flush=True)

    # ── 1) 原始序数码 IC (描述) ──
    print("\n[1] 原始 mhB_* 横截面 rank-IC (vs r20) ...", flush=True)
    raw_ic = {}
    for f in MHB_FEATS:
        raw_ic[f] = ic_table(df, f, f, "r20")
        a = raw_ic[f]["all"]
        print(f"  {f:18s} IC={a['ic_mean']:+.4f} t={a['ic_t']:+.2f}", flush=True)

    # ── 2) OOF target encoding (给名义卦象公平机会) ──
    print(f"\n[2] OOF 月度扩张 target encoding (gap={GAP_MONTHS}) → IC ...", flush=True)
    oof_ic, oof_cols = {}, {}
    for f in MHB_FEATS:
        col = f"__oof_{f}"
        df[col] = oof_encode(df, [f], "r20")
        oof_cols[f] = col
        oof_ic[f] = ic_table(df, col, f, "r20")
        a = oof_ic[f]["all"]
        print(f"  {f:18s} OOF-IC={a['ic_mean']:+.4f} t={a['ic_t']:+.2f}", flush=True)

    # ── 3a) 朴素对照 (公历月×板块) OOF → resid1 ──
    print(f"\n[3a] 朴素对照 (cal_month×industry) OOF → resid1 ...", flush=True)
    df["__naive"] = oof_encode(df, ["cal_month", "industry"], "r20")
    naive_ic = ic_table(df, "__naive", "naive(cal_month x industry)", "r20")
    print(f"  naive OOF-IC={naive_ic['all']['ic_mean']:+.4f} t={naive_ic['all']['ic_t']:+.2f}",
          flush=True)
    df["resid1"] = df["r20"] - df["__naive"]

    # ── 3b) 升级: resid1 逐日横截面回归扣标准动量/趋势因子 → resid2 ──
    print(f"\n[3b] 升级消融: resid1 逐日横截面扣动量因子 {MOM_FACTORS} → resid2 ...", flush=True)
    df["resid2"] = residualize_momentum(df, "resid1", MOM_FACTORS)
    n2 = int(df["resid2"].notna().sum())
    print(f"  resid2 有效行: {n2:,} / {len(df):,}", flush=True)
    # 动量因子本身对 r20 的 IC (描述, 证明扣的东西确实有信号)
    mom_ic = {f: clean(xs_ic(df, f, "r20")) for f in MOM_FACTORS}
    for f in MOM_FACTORS:
        print(f"  [mom] {f:8s} IC(r20)={mom_ic[f]['ic_mean']:+.4f} t={mom_ic[f]['ic_t']:+.2f}",
              flush=True)

    # ── 4) mhB OOF 编码 vs resid2 (扣 月×板块 + 动量 后的独立残差 IC) ──
    print(f"\n[4] mhB OOF 编码 vs resid2 (升级残差 IC, 分 regime) ...", flush=True)
    resid_ic = {}
    for f in MHB_FEATS:
        resid_ic[f] = ic_table(df, oof_cols[f], f, "resid2")
        a, ar = resid_ic[f]["all"], oof_ic[f]["all"]
        keep = (a['ic_mean'] / ar['ic_mean'] * 100
                if ar['ic_mean'] not in (0, None) and not np.isnan(ar['ic_mean']) else float('nan'))
        print(f"  {f:18s} resid2-IC={a['ic_mean']:+.4f} t={a['ic_t']:+.2f} "
              f"(raw OOF-IC={ar['ic_mean']:+.4f}, 保留 {keep:.0f}%)", flush=True)

    # ── 裁决: 动态 mhB_* 对 resid2 全期 |IC|>=floor 且 |t|>=floor ──
    def absmax(d):
        best, bf = -1.0, None
        for f in MHB_FEATS:
            ic = d[f]["all"].get("ic_mean")
            if ic is None or (isinstance(ic, float) and np.isnan(ic)):
                continue
            if abs(ic) > best:
                best, bf = abs(ic), f
        return bf

    bf = absmax(resid_ic)
    bdr = resid_ic[bf]["all"] if bf else {"ic_mean": np.nan, "ic_t": np.nan}
    passed = (bf is not None and not np.isnan(bdr["ic_mean"]) and abs(bdr["ic_mean"]) >= IC_FLOOR
              and not np.isnan(bdr["ic_t"]) and abs(bdr["ic_t"]) >= T_FLOOR)
    status = "residual_signal" if passed else "no_residual"

    results = {
        "task": "MT-001", "scheme": "B (体用=趋势vs当日)",
        "horizon_days": HORIZON, "oof_gap_months": GAP_MONTHS,
        "screen_thresholds": {"ic_floor": IC_FLOOR, "t_floor": T_FLOOR,
                              "applies_to": "dynamic mhB_* IC vs resid2 (扣月×板块+动量)"},
        "ablation_controls": ["cal_month×industry OOF", "动量因子 " + ",".join(MOM_FACTORS)],
        "n_rows": int(len(df)), "n_stocks": int(df["ts_code"].nunique()),
        "n_dates": int(df["trade_date"].nunique()),
        "regime_dist": df["regime"].value_counts(normalize=True).round(4).to_dict(),
        "momentum_factor_ic_r20": mom_ic,
        "raw_ordinal_ic_r20": {f: pack(raw_ic[f]) for f in MHB_FEATS},
        "oof_target_ic_r20": {f: pack(oof_ic[f]) for f in MHB_FEATS},
        "naive_control_ic_r20": pack(naive_ic),
        "residual2_ic_after_naive_and_momentum": {f: pack(resid_ic[f]) for f in MHB_FEATS},
        "best_dynamic_feature_by_resid2": bf,
        "best_dynamic_resid2_all": clean(bdr),
        "decision": status,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[results] -> {RESULTS.relative_to(ROOT)}", flush=True)

    conclusion = (
        f"方案B 轨迹起卦(体=近20日MA趋势/用=当日K结构) Phase-1 升级消融筛查: "
        f"最强动态 mhB_* = {bf} (扣 公历月×板块 + 标准动量{MOM_FACTORS} 后 resid2 IC="
        f"{bdr['ic_mean']:+.4f}, t={bdr['ic_t']:+.2f}); 朴素对照 OOF-IC="
        f"{naive_ic['all']['ic_mean']:+.4f}; 动量因子最强 IC(r20)="
        f"{max((v['ic_mean'] for v in mom_ic.values()), key=abs):+.4f}。"
        f"判定 {status} (阈值 |IC|>={IC_FLOOR} 且 |t|>={T_FLOOR})。")
    # 残差是否集中在"体卦(趋势)"族 = MA堆叠换皮的诚实预警 (SIGN-R12)
    trend_family = {"mhB_upper", "mhB_base_gua", "mhB_changed_gua", "mhB_ti_elem", "mhB_yang_count"}
    caveat = (
        "诚实预警 (SIGN-R12): 残差信号集中在 体卦族 (mhB_upper/base/changed/ti_elem), "
        "而 体卦 = 多尺度 MA 堆叠 (close>ma5,ma5>ma10,ma10>ma20) 的三元状态被重新编码成卦序号。"
        "它存活的只是**线性动量**残差化 (mom_5/20/ma_pos/rsi 的横截面 OLS); MA 堆叠是**非线性/多尺度趋势**, "
        "其残差 IC 极可能就是'非线性趋势超出线性动量'这一世俗 TA 结果, 而非占卜。是否真有独立于'同一走势的标准技术编码'"
        "的增益, 必须由 MT-004 apples-to-apples walk-forward (基线须含 MA 堆叠/非线性趋势对照) 裁决, "
        "Phase-1 IC 不作数 (SIGN-R03)。"
        if (bf in trend_family) else
        "残差最强特征非体卦族, 仍需 MT-004 walk-forward 裁决。")
    if status == "no_residual":
        conclusion += (" 趋势vs当日的卦象生克编码, 扣掉标准动量/趋势因子后无独立残差 ⇒ "
                       "卦象只是动量换皮 (SIGN-R12 消融未存活)。B 方案廉价收手, 进 MT-002(A)。")
    else:
        conclusion += (" Phase-1 螺旋阈值上算 residual_signal, 但 " + caveat +
                       " 候选进 MT-004 (最终只认 walk-forward α, SIGN-R03)。")
    results["interpretation_caveat"] = caveat
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = {
        "id": "MT-001", "status": status, "conclusion": conclusion,
        "metrics": {
            "horizon_days": HORIZON, "best_dynamic_feature": bf,
            "best_dynamic_resid2_ic": clean(bdr).get("ic_mean"),
            "best_dynamic_resid2_t": clean(bdr).get("ic_t"),
            "best_dynamic_resid2_by_regime": {
                seg: clean(resid_ic[bf][seg]) for seg in ["momentum", "mixed", "reversal"]
            } if bf else None,
            "naive_control_oof_ic": clean(naive_ic["all"]).get("ic_mean"),
            "momentum_factor_ic_r20": mom_ic,
            "screen_ic_floor": IC_FLOOR, "screen_t_floor": T_FLOOR,
            "n_dates": int(df["trade_date"].nunique()),
            "regime_dist": df["regime"].value_counts(normalize=True).round(4).to_dict(),
        },
        "ablation": {
            "controls": ["OOF (cal_month × industry)", "逐日横截面回归扣 " + ",".join(MOM_FACTORS)],
            "best_dynamic_raw_oof_ic": clean(oof_ic[bf]["all"]).get("ic_mean") if bf else None,
            "note": ("升级 vs MH-002: resid2 在 月×板块 基础上额外逐日横截面扣线性动量/趋势成分。"
                     "若 mhB_* 对 resid2 仍无 IC ⇒ 趋势卦象=动量换皮, 无独立于标准技术指标的信号。"),
            "interpretation_caveat": caveat,
        },
        "artifacts": [str(FEAT_OUT.relative_to(ROOT)),
                      "research/cache/mt001_panel.parquet",
                      "research/cache/mt001_results.json"],
        "guardrails": ["SIGN-R03 纯描述非gate", "SIGN-R04 r20/动量留cache不进features",
                       "SIGN-R06 ST源头排除", "SIGN-R11 regime分层",
                       "SIGN-R12+ 升级消融(月×板块+动量因子)"],
    }
    VERDICT.parent.mkdir(parents=True, exist_ok=True)
    VERDICT.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] status={status} -> {VERDICT.relative_to(ROOT)}", flush=True)
    print(f"\n[done] 耗时 {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
