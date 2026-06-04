# -*- coding: utf-8 -*-
"""MT-003 — 方案 C (累积卦 + 序列特征): 逐日翻爻路径积分 + 升级消融 Phase-1 筛查.

北极星升级 (SIGN-R12+): 把"走势"做进梅花的最忠实形式 —
  每日收盘当一次"动", 逐日翻对应爻 → 路径积分累积卦 (塌缩成单卦只剩每爻翻动次数的
  奇偶 parity, 有损); 故另抽保留路径/序列信息的标量:
    累积卦 mhC_cum_* (parity 塌缩, 种子=坤; 当日动爻驱动'累积本卦→变卦');
    动爻漂移 mhC_move_drift (动爻位置随时间内→外漂移 = 近半窗均值 − 早半窗均值);
    动爻分散 mhC_move_std; 五行净生克 mhC_wuxing_net / _recent (逐日单日卦体用生克带符号求和)。

  起卦数论: 当日动爻 m_t = abs(round(ret*100)) %6 (余0进位); 逐日翻 base 第 m_t 爻。
  逐日单日卦 (五行净生克用): up=abs(round(ret*100)), lo=round(amp*100), 动爻=m_t。全因果。

升级消融 (同 MT-001/002): 残差化 r20 扣 (公历月×板块 OOF) → resid1, 再逐日横截面回归扣
  标准动量/趋势因子 (mom_5/mom_20/ma_pos/rsi_14) → resid2。mhC_* 对 resid2 仍有独立残差
  才算 residual_signal; 否则 = 累积卦/序列特征只是动量换皮 ⇒ no_residual (SIGN-R12)。

特征类型 (升级筛查的口径修正): 名义卦象 (cum_base/upper/lower/changed/move_last) 用 OOF
  target encoding 给公平机会; 连续序列标量 (cum_yang/move_drift/move_std/wuxing_net/_recent)
  直接用原始值 rank-IC (OOF 对连续值按精确取值编码会退化, 用原始 rank-IC 才公平)。

口径/裁决 同 MT-001/002 (纯描述非gate SIGN-R03; 落地只认 MT-004 walk-forward):
  label r20=close[t+20]/close[t]-1 (全因果, 只进 cache 不进 features, SIGN-R04);
  横截面 rank-IC 逐日 Spearman 跨日平均, t=mean/std*sqrt(N_dates); OOF 月度扩张窗 (gap=2);
  分层 (SIGN-R11) 全期+momentum/mixed/reversal。
  裁决: 某动态 mhC_* 对 resid2 全期 |IC|>=0.01 且 |t|>=3.0 → residual_signal; 否则 no_residual。
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
from meihua_encoder import build_traj_C_lookup, build_wuxing_sign_lookup  # noqa: E402

DAILY = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
REGIME_P = ROOT / "research" / "features" / "regime_timeline.parquet"
FEAT_OUT = ROOT / "research" / "features" / "meihua_traj_C.parquet"
CACHE = ROOT / "research" / "cache"
PANEL_CACHE = CACHE / "mt003_panel.parquet"      # 含 r20 + 动量因子 → 留 cache 不进 features
RESULTS = CACHE / "mt003_results.json"
VERDICT = ROOT / "research" / "verdicts" / "MT-003.json"

KEY = ["ts_code", "trade_date"]
HORIZON = 20
WIN = 20                                          # 近N日轨迹窗 (累积窗)
HALF = WIN // 2                                   # 早半 / 近半
GAP_MONTHS = 2
IC_FLOOR = 0.01
T_FLOOR = 3.0

# 名义卦象 (OOF target encoding) vs 连续序列标量 (原始 rank-IC)
CAT_FEATS = ["mhC_cum_base_gua", "mhC_cum_upper", "mhC_cum_lower",
             "mhC_cum_changed_gua", "mhC_move_last"]
CONT_FEATS = ["mhC_cum_yang", "mhC_move_drift", "mhC_move_std",
              "mhC_wuxing_net", "mhC_wuxing_net_recent"]
MHC_FEATS = CAT_FEATS + CONT_FEATS
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


def build_features() -> pd.DataFrame:
    """返回 key + mhC_* (落 features) + r20/动量因子/分层列 (落 cache panel)."""
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

    print("[build] 逐日动爻 m_t = abs(round(ret*100)) %6 (因果) ...", flush=True)
    g = px.groupby("ts_code", sort=False)["close"]
    ret = g.transform(lambda s: s.pct_change())
    q = np.round(ret * 100.0)                                  # 量化整数涨跌幅
    prev = g.transform(lambda s: s.shift(1))
    amp = (px["high"] - px["low"]) / prev                      # 当日振幅 (相对昨收)
    m = (np.abs(q) % 6)                                        # 0..5
    m = m.where(q.notna())
    m = m.replace(0, 6)                                        # 余0进位 → 1..6 (NaN 保留)
    px["m"] = m

    # ── 累积卦 parity: 每爻 k 在窗内被翻次数的奇偶 (种子=坤 全阴) ──
    print("[build] 累积卦 parity (6 爻 rolling 奇偶) ...", flush=True)
    gm = px.groupby("ts_code", sort=False)
    bits = {}
    for k in range(1, 7):
        ind = (px["m"] == k).astype(float).where(px["m"].notna())
        rs = gm["m"].transform(lambda s, kk=k: (s == kk).astype(float).where(s.notna())
                               .rolling(WIN, min_periods=WIN).sum())
        bits[k] = (rs % 2)                                     # parity bit (NaN 传播)
    for k in range(1, 7):
        px[f"b{k}"] = bits[k]

    # ── 动爻序列特征 (路径/序列信息) ──
    print("[build] 动爻漂移 / 分散 / 五行净生克 (rolling) ...", flush=True)
    sum_full = gm["m"].transform(lambda s: s.rolling(WIN, min_periods=WIN).sum())
    sum_half = gm["m"].transform(lambda s: s.rolling(HALF, min_periods=HALF).sum())
    px["mhC_move_drift"] = (sum_half / HALF) - ((sum_full - sum_half) / HALF)  # 近半 − 早半
    px["mhC_move_std"] = gm["m"].transform(lambda s: s.rolling(WIN, min_periods=WIN).std())
    px["mhC_move_last"] = px["m"]

    # 逐日单日卦 体用生克 带符号: up=abs(q), lo=round(amp*100), 动爻=m_t → 查表 signed
    up_g = (np.abs(q) % 8); up_g = up_g.where(q.notna()).replace(0, 8)
    lo_q = np.round(amp * 100.0)
    lo_g = (np.abs(lo_q) % 8); lo_g = lo_g.where(amp.notna()).replace(0, 8)
    sign_lut = build_wuxing_sign_lookup()
    keyc = (up_g.astype("Int64").astype(str) + "_" + lo_g.astype("Int64").astype(str)
            + "_" + px["m"].astype("Int64").astype(str))
    sign_map = {f"{a}_{b}_{c}": v for (a, b, c), v in sign_lut.items()}
    px["wx_sign"] = keyc.map(sign_map).astype(float)
    px["mhC_wuxing_net"] = gm.apply(lambda d: d["wx_sign"].rolling(WIN, min_periods=WIN).sum()
                                    ).reset_index(level=0, drop=True)
    px["mhC_wuxing_net_recent"] = gm.apply(
        lambda d: d["wx_sign"].rolling(HALF, min_periods=HALF).sum()
    ).reset_index(level=0, drop=True)

    valid = px[[f"b{k}" for k in range(1, 7)]].notna().all(axis=1) & px["m"].notna()
    px = px[valid].reset_index(drop=True)

    # ── 累积卦查表 map → mhC_cum_* ──
    print("[build] 累积卦查表 map (64×6) ...", flush=True)
    for k in range(1, 7):
        px[f"b{k}"] = px[f"b{k}"].astype(np.int64)
    px["mhC_move_last"] = px["mhC_move_last"].astype(np.int64)
    lut = build_traj_C_lookup()
    lut_df = pd.DataFrame([
        {"b1": k[0], "b2": k[1], "b3": k[2], "b4": k[3], "b5": k[4], "b6": k[5],
         "mhC_move_last": k[6], **v} for k, v in lut.items()])
    px = px.merge(lut_df, on=["b1", "b2", "b3", "b4", "b5", "b6", "mhC_move_last"], how="left")
    cum_cols = ["mhC_cum_base_gua", "mhC_cum_upper", "mhC_cum_lower",
                "mhC_cum_changed_gua", "mhC_cum_yang"]
    assert px[cum_cols].notna().all().all(), "累积卦查表 map 出现 NaN (parity/动爻越界)"

    # ── 动量/趋势 因子 (消融对照, 因果) ──
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

    # ── 落盘: features = 仅 key + mhC_* (零泄漏); panel = 全列 进 cache ──
    feat_out = px[KEY + MHC_FEATS].copy()
    bad = [c for c in feat_out.columns if c.lower() in
           {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} or c.lower().startswith("fwd_")]
    assert not bad, f"features 含 forward 字段 (泄漏): {bad}"
    FEAT_OUT.parent.mkdir(parents=True, exist_ok=True)
    feat_out.to_parquet(FEAT_OUT, index=False)
    print(f"[build] features -> {FEAT_OUT.relative_to(ROOT)} "
          f"({len(feat_out):,} 行 × {len(MHC_FEATS)} mhC_)", flush=True)

    panel = px[KEY + MHC_FEATS + MOM_FACTORS +
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
    mn, s = float(ic.mean()), float(ic.std(ddof=1))
    t = mn / s * np.sqrt(len(ic)) if s > 0 else np.nan
    return {"ic_mean": mn, "ic_t": float(t), "n_dates": int(len(ic)),
            "ic_std": s, "ic_ir": (mn / s if s > 0 else np.nan)}


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
    print("\n=== MT-003 方案C 累积卦+序列特征 (逐日翻爻路径积分) + 升级消融 Phase-1 ===\n",
          flush=True)
    df = build_features()
    print(f"[panel] {len(df):,} 行 / {df['ts_code'].nunique()} 股 / "
          f"{df['trade_date'].nunique()} 日; regime "
          f"{df['regime'].value_counts(normalize=True).round(3).to_dict()}", flush=True)

    # ── 1) 原始 mhC_* 横截面 rank-IC (描述, cat+cont 都算) ──
    print("\n[1] 原始 mhC_* 横截面 rank-IC (vs r20) ...", flush=True)
    raw_ic = {}
    for f in MHC_FEATS:
        raw_ic[f] = ic_table(df, f, f, "r20")
        a = raw_ic[f]["all"]
        tag = "cat" if f in CAT_FEATS else "cont"
        print(f"  {f:22s}[{tag}] IC={a['ic_mean']:+.4f} t={a['ic_t']:+.2f}", flush=True)

    # ── 2) OOF target encoding (仅名义卦象; 连续标量不 OOF) ──
    print(f"\n[2] OOF 月度扩张 target encoding (gap={GAP_MONTHS}, 仅 CAT) → IC ...", flush=True)
    oof_ic, oof_cols = {}, {}
    for f in CAT_FEATS:
        col = f"__oof_{f}"
        df[col] = oof_encode(df, [f], "r20")
        oof_cols[f] = col
        oof_ic[f] = ic_table(df, col, f, "r20")
        a = oof_ic[f]["all"]
        print(f"  {f:22s} OOF-IC={a['ic_mean']:+.4f} t={a['ic_t']:+.2f}", flush=True)

    # ── 3a) 朴素对照 (公历月×板块) OOF → resid1 ──
    print(f"\n[3a] 朴素对照 (cal_month×industry) OOF → resid1 ...", flush=True)
    df["__naive"] = oof_encode(df, ["cal_month", "industry"], "r20")
    naive_ic = ic_table(df, "__naive", "naive(cal_month x industry)", "r20")
    print(f"  naive OOF-IC={naive_ic['all']['ic_mean']:+.4f} t={naive_ic['all']['ic_t']:+.2f}",
          flush=True)
    df["resid1"] = df["r20"] - df["__naive"]

    # ── 3b) 升级: resid1 逐日横截面扣标准动量/趋势因子 → resid2 ──
    print(f"\n[3b] 升级消融: resid1 逐日横截面扣动量因子 {MOM_FACTORS} → resid2 ...", flush=True)
    df["resid2"] = residualize_momentum(df, "resid1", MOM_FACTORS)
    n2 = int(df["resid2"].notna().sum())
    print(f"  resid2 有效行: {n2:,} / {len(df):,}", flush=True)
    mom_ic = {f: clean(xs_ic(df, f, "r20")) for f in MOM_FACTORS}
    for f in MOM_FACTORS:
        print(f"  [mom] {f:8s} IC(r20)={mom_ic[f]['ic_mean']:+.4f} t={mom_ic[f]['ic_t']:+.2f}",
              flush=True)

    # ── 4) mhC vs resid2 (扣 月×板块 + 动量 后独立残差 IC); CAT 用 OOF, CONT 用原始 ──
    print(f"\n[4] mhC vs resid2 (升级残差 IC, 分 regime; CAT=OOF / CONT=原始) ...", flush=True)
    resid_ic = {}
    for f in MHC_FEATS:
        src = oof_cols[f] if f in CAT_FEATS else f
        resid_ic[f] = ic_table(df, src, f, "resid2")
        a = resid_ic[f]["all"]
        ref = (oof_ic[f]["all"] if f in CAT_FEATS else raw_ic[f]["all"])
        keep = (a['ic_mean'] / ref['ic_mean'] * 100
                if ref['ic_mean'] not in (0, None) and not np.isnan(ref['ic_mean']) else float('nan'))
        tag = "cat/OOF" if f in CAT_FEATS else "cont/raw"
        print(f"  {f:22s}[{tag:8s}] resid2-IC={a['ic_mean']:+.4f} t={a['ic_t']:+.2f} "
              f"(ref-IC={ref['ic_mean']:+.4f}, 保留 {keep:.0f}%)", flush=True)

    # ── 裁决: 动态 mhC_* 对 resid2 全期 |IC|>=floor 且 |t|>=floor ──
    def absmax(d):
        best, bf = -1.0, None
        for f in MHC_FEATS:
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
        "task": "MT-003", "scheme": "C (累积卦+序列特征: 逐日翻爻路径积分)",
        "window_days": WIN, "horizon_days": HORIZON, "oof_gap_months": GAP_MONTHS,
        "screen_thresholds": {"ic_floor": IC_FLOOR, "t_floor": T_FLOOR,
                              "applies_to": "dynamic mhC_* IC vs resid2 (扣月×板块+动量)"},
        "casting": ("每日动爻 m_t=abs(round(ret*100))%%6 逐日翻 base 第m_t爻 → 累积卦 parity 塌缩 "
                    "(种子=坤); 序列特征=动爻漂移(近半−早半)/动爻std/五行净生克(逐日单日卦体用带符号Σ)"),
        "feature_types": {"categorical_OOF": CAT_FEATS, "continuous_raw_rankIC": CONT_FEATS},
        "ablation_controls": ["cal_month×industry OOF", "动量因子 " + ",".join(MOM_FACTORS)],
        "n_rows": int(len(df)), "n_stocks": int(df["ts_code"].nunique()),
        "n_dates": int(df["trade_date"].nunique()),
        "regime_dist": df["regime"].value_counts(normalize=True).round(4).to_dict(),
        "momentum_factor_ic_r20": mom_ic,
        "raw_ordinal_ic_r20": {f: pack(raw_ic[f]) for f in MHC_FEATS},
        "oof_target_ic_r20": {f: pack(oof_ic[f]) for f in CAT_FEATS},
        "naive_control_ic_r20": pack(naive_ic),
        "residual2_ic_after_naive_and_momentum": {f: pack(resid_ic[f]) for f in MHC_FEATS},
        "best_dynamic_feature_by_resid2": bf,
        "best_dynamic_resid2_all": clean(bdr),
        "decision": status,
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[results] -> {RESULTS.relative_to(ROOT)}", flush=True)

    # 残差集中在哪个特征家族 → 决定 MT-004 的硬约束 (诚实定位 residual 的世俗来源)
    vol_like = {"mhC_move_std", "mhC_move_drift", "mhC_cum_yang"}
    best_is_vol = bf in vol_like
    caveat = (
        "诚实预警 (SIGN-R12): 方案C 累积卦=每爻翻动次数的奇偶(parity), 本质是'近N日量化涨跌幅"
        "分桶'的周期折叠; 序列标量里 mhC_move_std/mhC_move_drift/mhC_wuxing_net 全由逐日涨跌幅/"
        "振幅派生。关键: 动爻 m_t=abs(round(ret*100))%%6 是当日涨跌幅幅度的桶, 故 mhC_move_std="
        "'动爻位置的离散度'≈近N日已实现波动率的换皮代理 (与累积动量正交=正是它存活线性动量残差化"
        "的原因)。本轮 resid2 最强残差集中在 " + str(bf) + " "
        + ("(波动率族)" if best_is_vol else "(五行/累积卦族)") +
        "; 但残差化只扣了 mom_5/mom_20/ma_pos/rsi 四个线性动量/趋势因子, **未含波动率因子**, "
        "也未含 MA堆叠/非线性趋势。故 resid2-IC 极可能是'波动率 + 非线性趋势'这些标准TA量超出线性"
        "动量残差化的世俗结果, 而非占卜。"
        "给 MT-004 的硬约束: apples-to-apples baseline **必须额外含 已实现波动率 (如 std(ret_20) / "
        "ATR) + MA堆叠/非线性趋势对照** (沿用 MT-001 的 MA堆叠约束), 否则会把'波动率/非线性趋势'"
        "误归功于卦象 = 自欺。是否真有独立于'同一走势标准技术编码'的增益, 只能由 MT-004 walk-forward "
        "裁决, Phase-1 IC 不作数 (SIGN-R03)。")
    conclusion = (
        f"方案C 累积卦+序列特征(逐日翻爻路径积分, 窗{WIN}日) Phase-1 升级消融筛查: 最强动态 "
        f"mhC_* = {bf} (扣 公历月×板块 + 标准动量{MOM_FACTORS} 后 resid2 IC={bdr['ic_mean']:+.4f}, "
        f"t={bdr['ic_t']:+.2f}); 朴素对照 OOF-IC={naive_ic['all']['ic_mean']:+.4f}; "
        f"动量因子最强 IC(r20)={max((v['ic_mean'] for v in mom_ic.values()), key=abs):+.4f}。"
        f"判定 {status} (阈值 |IC|>={IC_FLOOR} 且 |t|>={T_FLOOR})。")
    if status == "no_residual":
        conclusion += (" 累积卦/序列特征扣掉标准动量/趋势因子后无独立残差 ⇒ 路径积分梅花也只是"
                       "动量换皮 (SIGN-R12 消融未存活)。C 方案廉价收手, 三组 B/A/C 齐备进 MT-004 对比。")
    else:
        conclusion += (" Phase-1 阈值上算 residual_signal, 但 " + caveat +
                       " 候选进 MT-004 (最终只认 walk-forward α, SIGN-R03)。")
    results["interpretation_caveat"] = caveat
    RESULTS.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = {
        "id": "MT-003", "status": status, "conclusion": conclusion,
        "metrics": {
            "window_days": WIN, "horizon_days": HORIZON, "best_dynamic_feature": bf,
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
            "feature_types": {"categorical_OOF": CAT_FEATS, "continuous_raw_rankIC": CONT_FEATS},
            "note": ("升级消融: resid2 在 月×板块 基础上额外逐日横截面扣线性动量/趋势成分。"
                     "若 mhC_* 对 resid2 仍无 IC ⇒ 累积卦/序列特征=动量换皮, 无独立于标准技术指标的信号。"),
            "interpretation_caveat": caveat,
        },
        "artifacts": [str(FEAT_OUT.relative_to(ROOT)),
                      "research/cache/mt003_panel.parquet",
                      "research/cache/mt003_results.json"],
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
