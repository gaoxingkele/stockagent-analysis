# -*- coding: utf-8 -*-
"""BT-004 — 事件上下文 meta-label 残差测 (单一表示 + Deflated Sharpe/PBO).

预注册 (冻结, prd.preRegisteredGate.bt004_prereg; SIGN-R01 不在结果上调):
  单一表示 ctx = (MA20 在事件前 K 日内已上拐?) × (MA5 上穿 MA20 金叉)。
  K 固定 = 5 (= MA5 短窗), **不搜窗口 / 不搜 MA 组合** (防 López de Prado 组合过拟合)。
  测: ctx 对 T+20 (r20_fresh) 的**残差 RankIC**, 扣 [pump(ratio_s5) + MA20斜率 + pyr_velocity_20_60 + ADX]。
  报: Deflated Sharpe (PSR, 单表示 trials=1) / PBO 说明。
  gate: 残差 |IC| >= 0.01 & t >= 2 & 过 deflation → meta-label / 条件 sizing (非新 α 因子);
        否则 REJECT_reskin。

先验 (DESIGN §5): 大概率被现有 [MA斜率/pump/pyr/ADX] 吃掉 (关系张量 TCN 已 ≈0); 这是唯一干净判它的方式。

数据 (全因果, ST 源头排除 R06):
  - 价表 output/tushare_cache/daily/*.parquet → close/high/low → MA5/MA20/MA20斜率/ADX/金叉/MA20上拐 (全 backward)
  - scored_daily (research/cache/bt003) → ratio_s5(pump) / pyr_velocity_20_60 / r20_fresh (T+20 前向, 只在 cache, R04)
checkpoint (R08): px_feats 落盘 research/cache/bt004/ (gitignored), 已完成跳过。
中间 IC ≠ ship (R03): gate 由残差IC+t+deflation 三闸共同决定; 描述性 REJECT = 合法完成 (R02)。
"""
from __future__ import annotations
import gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "backtest"))

CACHE003 = ROOT / "research" / "cache" / "bt003"
CACHE = ROOT / "research" / "cache" / "bt004"
CACHE.mkdir(parents=True, exist_ok=True)
SCORED = CACHE003 / "scored_daily.parquet"
PXFEAT = CACHE / "px_feats.parquet"
REGIME = ROOT / "research" / "features" / "regime_timeline.parquet"
DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
VERDICT = ROOT / "research" / "verdicts" / "BT-004.json"

# ── 冻结参数 (R01, 单一表示, 不搜) ──
K_UPTURN = 5            # MA20 在事件前 K 日内已上拐 (固定, = MA5 短窗)
SLOPE_WIN = 5          # MA20 斜率控制窗 (固定)
ADX_WIN = 14           # 标准 Wilder ADX
NONOVERLAP = 20        # T+20 重叠 → 非重叠每 20 交易日取样 (独立 IC obs, 防 t 膨胀)
MIN_XS = 50            # 截面最小股数
CONTROLS = ["ratio_s5", "ma20_slope", "pyr_velocity_20_60", "adx"]
SIGNAL = "ctx"
TARGET = "r20_fresh"
PERIODS_PER_YEAR = 252 / NONOVERLAP
_t0 = time.time()


def log(m): print(m, flush=True)


# ───────────────────────── 价格派生特征 (全 backward) ─────────────────────────
def _adx(g: pd.DataFrame, n=ADX_WIN) -> pd.Series:
    """Wilder ADX-n (ewm alpha=1/n 近似 Wilder 平滑). 全 backward."""
    high, low, close = g["high"], g["low"], g["close"]
    pc = close.shift(1)
    tr = pd.concat([(high - low), (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = 1.0 / n
    atr = tr.ewm(alpha=a, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=g.index).ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=g.index).ewm(alpha=a, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def build_px_feats() -> pd.DataFrame:
    if PXFEAT.exists():
        log(f"[px_feats] 复用 checkpoint {PXFEAT.relative_to(ROOT)}")
        d = pd.read_parquet(PXFEAT)
        d["trade_date"] = d["trade_date"].astype(str)
        return d

    log("[px] load daily (close/high/low, ST 源头排除 R06) ...")
    parts = []
    for f in sorted(DAILY_DIR.glob("*.parquet")):
        d = pd.read_parquet(f, columns=["ts_code", "trade_date", "high", "low", "close"])
        parts.append(d)
    px = pd.concat(parts, ignore_index=True)
    px["trade_date"] = px["trade_date"].astype(str)
    basic = pd.read_parquet(BASIC)[["ts_code", "name"]].drop_duplicates("ts_code")
    st = set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])
    px = px[~px["ts_code"].isin(st)].reset_index(drop=True)
    px = px.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    log(f"[px] {len(px):,} 行 / {px['ts_code'].nunique()} 股")

    out = []
    for code, g in px.groupby("ts_code", sort=False):
        if len(g) < 40:
            continue
        g = g.copy()
        ma5 = g["close"].rolling(5).mean()
        ma20 = g["close"].rolling(20).mean()
        # 金叉: MA5 上穿 MA20 (今日 > 且 昨日 <=)
        cross = (ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))
        # MA20 上拐日: MA20 一阶差 由 <=0 转 >0
        d20 = ma20.diff()
        upturn_day = (d20 > 0) & (d20.shift(1) <= 0)
        # 事件前 K 日内已上拐 (含当日): 过去 K 日窗口内存在上拐日
        upturn_in_K = upturn_day.rolling(K_UPTURN, min_periods=1).max().fillna(0) > 0
        g["ctx"] = ((cross) & (upturn_in_K)).astype(float)
        # MA20 斜率控制 (SLOPE_WIN 日归一化)
        g["ma20_slope"] = (ma20 - ma20.shift(SLOPE_WIN)) / ma20.shift(SLOPE_WIN)
        g["adx"] = _adx(g)
        out.append(g[["ts_code", "trade_date", "ctx", "ma20_slope", "adx"]])
    res = pd.concat(out, ignore_index=True)
    res = res.dropna(subset=["ma20_slope", "adx"]).reset_index(drop=True)
    res.to_parquet(PXFEAT, index=False)
    log(f"[px_feats] -> {PXFEAT.relative_to(ROOT)}  {len(res):,} 行  "
        f"ctx 事件率 {res['ctx'].mean():.4%}  (累计 {time.time()-_t0:.0f}s)")
    return res


# ───────────────────────── 残差 RankIC ─────────────────────────
def daily_resid_ic(df: pd.DataFrame) -> pd.DataFrame:
    """每截面: ctx 对 controls 线性正交 (rank 空间) -> 残差; 残差 vs T+20 Spearman IC.
    同时算 raw IC (不扣 controls) 对照。"""
    rows = []
    for d, g in df.groupby("trade_date", sort=True):
        g = g.dropna(subset=[SIGNAL, TARGET] + CONTROLS)
        if len(g) < MIN_XS or g[SIGNAL].sum() < 3:
            continue
        y = g[SIGNAL].values.astype(float)
        tgt = g[TARGET].values.astype(float)
        # controls 转 rank-pct, 加截距, OLS 正交化 ctx
        Xc = np.column_stack([stats.rankdata(g[c].values) / len(g) for c in CONTROLS]
                             + [np.ones(len(g))])
        beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
        resid = y - Xc @ beta
        ic_resid = stats.spearmanr(resid, tgt).correlation
        ic_raw = stats.spearmanr(y, tgt).correlation
        if np.isnan(ic_resid):
            continue
        rows.append({"trade_date": d, "ic_resid": ic_resid,
                     "ic_raw": ic_raw if not np.isnan(ic_raw) else 0.0,
                     "n": len(g), "n_ctx": int(g[SIGNAL].sum())})
    return pd.DataFrame(rows)


def ic_stats(ic_series: pd.Series) -> dict:
    s = ic_series.dropna().values
    n = len(s)
    if n < 2:
        return {"mean_ic": None, "t": None, "n": n}
    mean = float(s.mean())
    se = float(s.std(ddof=1) / np.sqrt(n))
    return {"mean_ic": round(mean, 5), "t": round(mean / se, 3) if se > 0 else None,
            "n": n, "ir": round(mean / s.std(ddof=1), 3) if s.std(ddof=1) > 0 else None}


# ───────────────────────── Deflated Sharpe (PSR) ─────────────────────────
def psr(returns: np.ndarray, sr_benchmark: float = 0.0) -> dict:
    """Probabilistic Sharpe Ratio (López de Prado). 单表示 trials=1 → DSR≈PSR(SR*=0).
    含 skew/kurtosis 偏正态修正。"""
    r = returns[~np.isnan(returns)]
    n = len(r)
    if n < 4 or r.std(ddof=1) == 0:
        return {"sharpe_per_period": None, "psr": None, "n": n}
    sr = r.mean() / r.std(ddof=1)            # per-period Sharpe
    sk = float(stats.skew(r))
    ku = float(stats.kurtosis(r, fisher=False))  # 非超额峰度
    # PSR: P(SR_true > SR*) 在偏正态修正下
    denom = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
    return {"sharpe_per_period": round(float(sr), 4),
            "sharpe_annual": round(float(sr * np.sqrt(PERIODS_PER_YEAR)), 3),
            "psr": round(float(stats.norm.cdf(z)), 4),
            "skew": round(sk, 3), "kurtosis": round(ku, 3), "n": n}


def main():
    log("\n=== BT-004: 事件上下文 meta-label 残差测 (单表示 + Deflated Sharpe) ===\n")
    log(f"[prereg] ctx=(MA20前{K_UPTURN}日内已上拐)×(MA5上穿MA20); 扣{CONTROLS}; 不搜窗口/MA组合 (R01)\n")

    # 1) 价格派生特征
    pxf = build_px_feats()

    # 2) 评分面板 (pump/pyr/T+20), 复用 BT-003 scored_daily
    log("[scored] 复用 bt003/scored_daily (ratio_s5 / pyr_velocity / r20_fresh) ...")
    sc = pd.read_parquet(SCORED, columns=["ts_code", "trade_date", "industry",
                                          "ratio_s5", "pyr_velocity_20_60", "r20_fresh", "month"])
    sc["trade_date"] = sc["trade_date"].astype(str)
    df = sc.merge(pxf, on=["ts_code", "trade_date"], how="inner")
    log(f"[merge] {len(df):,} 行 / {df['trade_date'].nunique()} 日 / {df['month'].nunique()} 月  "
        f"ctx 事件率 {df['ctx'].mean():.4%}  ctx=1 {int(df['ctx'].sum()):,} 例")

    # 3) 残差 RankIC (全期, 逐日)
    log("\n[resid IC] 逐截面 ctx 正交 [pump+MA20斜率+pyr+ADX] -> 残差 vs T+20 ...")
    icd = daily_resid_ic(df)
    icd = icd.sort_values("trade_date").reset_index(drop=True)
    log(f"[resid IC] {len(icd)} 个有效截面 (ctx>=3 例)")

    # 全日 (重叠, t 膨胀, 仅参考) vs 非重叠 (独立, headline gate)
    all_resid = ic_stats(icd["ic_resid"])
    all_raw = ic_stats(icd["ic_raw"])
    dates = sorted(icd["trade_date"].unique())
    no_dates = set(dates[::NONOVERLAP])
    icd_no = icd[icd["trade_date"].isin(no_dates)]
    no_resid = ic_stats(icd_no["ic_resid"])
    no_raw = ic_stats(icd_no["ic_raw"])
    log(f"  全日(重叠,参考): 残差IC {all_resid['mean_ic']} t={all_resid['t']} (n={all_resid['n']}); "
        f"raw IC {all_raw['mean_ic']}")
    log(f"  非重叠(每{NONOVERLAP}日, headline): 残差IC {no_resid['mean_ic']} t={no_resid['t']} (n={no_resid['n']}); "
        f"raw IC {no_raw['mean_ic']} t={no_raw['t']}")

    # 4) regime 分层 (R11)
    log("\n[regime] 残差 IC 分 regime (R11) ...")
    reg_resid = {}
    if REGIME.exists():
        reg = pd.read_parquet(REGIME)
        reg["trade_date"] = reg["trade_date"].astype(str)
        rmap = dict(zip(reg["trade_date"], reg["regime"]))
        icd_r = icd.copy()
        icd_r["regime"] = icd_r["trade_date"].map(rmap)
        for rg, gg in icd_r.dropna(subset=["regime"]).groupby("regime"):
            st = ic_stats(gg["ic_resid"])
            reg_resid[str(rg)] = st
            log(f"  {rg:10s}: 残差IC {st['mean_ic']} t={st['t']} (n={st['n']})")

    # 5) Deflated Sharpe (PSR) — ctx meta-label 多空 spread, 非重叠 20d 周期
    log("\n[DSR] ctx meta-label spread Sharpe + PSR (单表示 trials=1 → DSR≈PSR(SR*=0)) ...")
    spreads = []
    for d in sorted(no_dates):
        g = df[(df["trade_date"] == d)].dropna(subset=[SIGNAL, TARGET])
        if len(g) < MIN_XS:
            continue
        longs = g[g[SIGNAL] == 1][TARGET]
        if len(longs) < 3:
            continue
        # 残差化口径: ctx=1 篮子超额 = ctx篮子均值 - 全截面均值 (扣市场, 与 IC 同向描述)
        spreads.append(float(longs.mean() - g[TARGET].mean()))
    spreads = np.array(spreads)
    dsr = psr(spreads / 100.0, sr_benchmark=0.0)   # r20_fresh 是 % → /100
    log(f"  非重叠周期数 {dsr.get('n')}; per-period Sharpe {dsr.get('sharpe_per_period')} "
        f"年化 {dsr.get('sharpe_annual')}; PSR(SR>0) {dsr.get('psr')} "
        f"(skew {dsr.get('skew')} kurt {dsr.get('kurtosis')})")
    pbo_note = ("PBO (CSCV) 需在多配置/多策略试验集上度量过拟合; 本测严格单一表示 (不搜窗口/MA组合, trials=1), "
                "PBO 退化为 0/不适用。以单表示 Deflated Sharpe=PSR(SR*=0, trials=1) 替代多重检验扣减; "
                f"非重叠周期 n={dsr.get('n')} 偏少, PSR 仅作辅助, gate 主依据残差 IC + 非重叠 t。")

    # 6) Gate 判定 (R01 冻结阈值: |IC|>=0.01 & t>=2 & 过 deflation)
    ic_val = no_resid["mean_ic"]
    t_val = no_resid["t"]
    psr_val = dsr.get("psr")
    pass_ic = ic_val is not None and abs(ic_val) >= 0.01
    pass_t = t_val is not None and abs(t_val) >= 2.0
    pass_defl = psr_val is not None and psr_val >= 0.95
    gate_pass = pass_ic and pass_t and pass_defl
    status = "NEW_INFO_meta_label" if gate_pass else "REJECT_reskin"
    log(f"\n[gate] |残差IC|>=0.01: {pass_ic} ({ic_val}); t>=2: {pass_t} ({t_val}); "
        f"过deflation(PSR>=0.95): {pass_defl} ({psr_val}) => {status}")

    results = {
        "config": {
            "single_representation": "ctx = (MA20 前 K 日内已上拐) × (MA5 上穿 MA20 金叉)",
            "K_upturn": K_UPTURN, "slope_win": SLOPE_WIN, "adx_win": ADX_WIN,
            "controls": CONTROLS, "target": TARGET,
            "nonoverlap_tdays": NONOVERLAP, "no_search": "不搜窗口/不搜 MA 组合 (防钓鱼, R01)",
            "ctx_event_rate": round(float(df["ctx"].mean()), 6),
            "ctx_count": int(df["ctx"].sum()),
            "period": [df["trade_date"].min(), df["trade_date"].max()],
            "n_days": int(df["trade_date"].nunique()), "n_months": int(df["month"].nunique()),
        },
        "resid_ic_all_days_overlap_ref": all_resid,
        "resid_ic_nonoverlap_headline": no_resid,
        "raw_ic_nonoverlap": no_raw,
        "raw_ic_all_days": all_raw,
        "resid_ic_by_regime": reg_resid,
        "deflated_sharpe": dsr,
        "pbo_note": pbo_note,
        "gate": {
            "threshold": "|残差IC(非重叠)|>=0.01 & |t|>=2 & PSR>=0.95 (冻结 R01)",
            "pass_ic": bool(pass_ic), "pass_t": bool(pass_t), "pass_deflation": bool(pass_defl),
            "gate_pass": bool(gate_pass), "status": status,
        },
    }
    (CACHE / "bt004_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    conclusion = (
        f"单一表示 ctx=(MA20前{K_UPTURN}日内已上拐)×(MA5上穿MA20金叉), 严格不搜窗口/MA组合 (R01)。"
        f"事件率 {df['ctx'].mean():.3%} ({int(df['ctx'].sum()):,} 例)。"
        f"扣 [pump(ratio_s5)+MA20斜率+pyr_velocity+ADX] 后残差对 T+20 RankIC: "
        f"非重叠(每{NONOVERLAP}日, headline) **{no_resid['mean_ic']}** t={no_resid['t']} (n={no_resid['n']}); "
        f"全日重叠(参考) {all_resid['mean_ic']} t={all_resid['t']}。"
        f"raw(不扣) 非重叠 IC {no_raw['mean_ic']} → 扣控制后"
        + ("塌缩" if (no_raw['mean_ic'] is not None and no_resid['mean_ic'] is not None
                     and abs(no_resid['mean_ic']) < abs(no_raw['mean_ic'])) else "变化") + "。"
        f"Deflated Sharpe=PSR(SR*=0, 单表示 trials=1): per-period Sharpe {dsr.get('sharpe_per_period')} "
        f"PSR {dsr.get('psr')} (非重叠 n={dsr.get('n')}, 偏少仅辅助)。"
        f"Gate (|IC|>=0.01 & |t|>=2 & PSR>=0.95 三闸): IC {pass_ic} / t {pass_t} / deflation {pass_defl} "
        f"=> **{status}**。"
        + (" 残差信号独立存在, 可作 meta-label/条件 sizing (非新 α)。" if gate_pass else
           " ctx 增益被 [MA斜率/pump/pyr/ADX] 吃掉 (印证 DESIGN §5 先验: 事件序列 ≈ 现有 TA 重打包), "
           "不作 meta-label; 生产线冻结 (R05)。")
    )

    VERDICT.write_text(json.dumps({
        "id": "BT-004", "status": status, "conclusion": conclusion,
        "metrics": {
            "resid_ic_nonoverlap": no_resid["mean_ic"], "resid_t_nonoverlap": no_resid["t"],
            "resid_ic_n_periods": no_resid["n"],
            "resid_ic_all_days": all_resid["mean_ic"], "resid_t_all_days": all_resid["t"],
            "raw_ic_nonoverlap": no_raw["mean_ic"],
            "deflated_sharpe_psr": dsr.get("psr"),
            "spread_sharpe_per_period": dsr.get("sharpe_per_period"),
            "spread_sharpe_annual": dsr.get("sharpe_annual"),
            "ctx_event_rate": round(float(df["ctx"].mean()), 6),
            "ctx_count": int(df["ctx"].sum()),
        },
        "resid_ic_by_regime": reg_resid,
        "gate": results["gate"],
        "pbo_note": pbo_note,
        "caveat": "单一表示严格冻结 (不搜 K/窗口/MA 组合, 防 López de Prado 组合过拟合); "
                  "T+20 重叠 → headline 用非重叠每 20 日取样以防 t 膨胀 (全日 t 仅参考); "
                  "PSR 替代多重检验扣减 (trials=1), PBO 单配置退化不适用; "
                  "r20_fresh 前向只在 research/cache (R04), ST 源头排除 (R06), 生产线未碰 (R05)。",
        "artifacts": [
            "research/backtest/run_bt004.py",
            "research/cache/bt004/px_feats.parquet",
            "research/cache/bt004/bt004_results.json",
        ],
        "note": "预注册单表示残差 gate (R01 阈值冻结); 中间 IC≠ship (R03), 三闸共判; "
                "REJECT_reskin = 合法完成 (R02); regime 分层必带 (R11); checkpoint px_feats (R08); "
                "大缓存 gitignored, 无 features 写入 (R04)。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n[done] verdict={status} -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-_t0)/60:.1f} min")


if __name__ == "__main__":
    main()
