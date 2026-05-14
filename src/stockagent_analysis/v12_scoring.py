"""V12 评分引擎 (V7c LGBM + 4 象限 + 5 铁律).

从 v7c_inference_0508.py 抽出, 提供可复用模块接口给 WEB.

主要类:
  V12Scorer
    - _load_models()                 单例缓存 4 LGBM
    - load_factors_for_date(date)    加载某日全市场因子矩阵 (cached)
    - score_market(date, cb=None)    全市场推理, 含进度回调
    - score_stock(ts_code, date)     单股推理 (走 score_market 的 cache)
    - apply_v7c_rules(df)            5 铁律过滤
    - classify_quadrant(buy, sell)   4 象限标签

进度回调签名:
  cb(phase: str, percent: int, message: str, data: dict | None) -> None
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

ProgressCb = Optional[Callable[[str, int, str, Optional[dict]], None]]

# 锚点 (V15 重训 2026-05-15, 训练区间 20230101-20260213)
# V4 旧锚: r10(-1.44,0.22,2.40) / r20(-7.78,-1.18,8.76)
R10_ANCHOR = (0.02, 0.69, 1.82)
R20_ANCHOR = (-4.45, 1.99, 13.19)
# sell 模型已屏蔽 (用户决策 2026-05-15), 锚点保留兼容旧代码
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)


def _map_anchored(v, p5: float, p50: float, p95: float):
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, 50.0)
    out = np.where(v <= p5, 0, out)
    out = np.where(v >= p95, 100, out)
    mask_lo = (v > p5) & (v <= p50)
    out = np.where(mask_lo, (v - p5) / (p50 - p5) * 50, out)
    mask_hi = (v > p50) & (v < p95)
    out = np.where(mask_hi, 50 + (v - p50) / (p95 - p50) * 50, out)
    return out


def classify_quadrant(buy: float, sell: float) -> str:
    if buy >= 70 and sell <= 30: return "理想多"
    if buy >= 70 and sell >= 70: return "矛盾段"
    if buy <= 30 and sell >= 70: return "主流空"
    if buy <= 30 and sell <= 30: return "沉寂"
    return "中性区"


class V12Scorer:
    """V12 评分引擎 (单例). 模型 + 因子矩阵 lazy load + cache."""

    _instance: Optional["V12Scorer"] = None
    _lock = threading.Lock()

    def __init__(self, project_root: Path):
        self.root = Path(project_root)
        self.prod_dir = self.root / "output" / "production"
        self.ext_dir = self.root / "output" / "factor_lab_3y" / "factor_groups_extension"
        self.regime_path = self.root / "output" / "regimes" / "daily_regime.parquet"
        self.regime_extra_path = self.root / "output" / "regime_extra" / "regime_extra.parquet"
        self._models: dict[str, tuple[lgb.Booster, dict]] = {}
        self._factor_cache: dict[str, pd.DataFrame] = {}  # date -> df

    @classmethod
    def get(cls, project_root: Path) -> "V12Scorer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(project_root)
            return cls._instance

    # ──────── 模型加载 ────────
    def _load_models(self):
        # 用 model_str 而非 model_file 避开 LightGBM 4.x + Python 3.14 + Windows 的 race
        # 2026-05-15: 切到 V15 (r10/r20 重训, IC 提升 2-12x), sell 已屏蔽
        if self._models: return
        # r10/r20: V15; sell_10/sell_20: 保留 V6 (虽然不用但 v12 推理路径仍读)
        for name in ["r10_v15_all", "r20_v15_all", "sell_10_v6", "sell_20_v6"]:
            d = self.prod_dir / name
            booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
            meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
            self._models[name] = (booster, meta)

    def predict_one(self, df: pd.DataFrame, name: str) -> np.ndarray:
        self._load_models()
        booster, meta = self._models[name]
        feat_cols = meta["feature_cols"]
        industry_map = meta.get("industry_map", {})
        df = df.copy()
        df["industry_id"] = df["industry"].fillna("unknown").map(
            lambda x: industry_map.get(str(x), -1)
        )
        for fc in feat_cols:
            if fc not in df.columns:
                df[fc] = np.nan
        return booster.predict(df[feat_cols])

    # ──────── 因子矩阵加载 ────────
    def load_factors_for_date(self, date: str, cb: ProgressCb = None) -> pd.DataFrame:
        """date 格式 YYYYMMDD. 返回当日全市场因子横截面."""
        if date in self._factor_cache:
            return self._factor_cache[date]

        if cb: cb("load_factor_lab", 5, f"加载 factor_lab 截面 {date}...", None)
        fl_parts = []
        for p in sorted(self.ext_dir.glob("*.parquet")):
            d = pd.read_parquet(p)
            d["trade_date"] = d["trade_date"].astype(str)
            d_t = d[d["trade_date"] == date]
            if not d_t.empty:
                fl_parts.append(d_t)
        if not fl_parts:
            raise ValueError(f"factor_lab 没有 {date} 的数据, 先跑 update_factor_lab_from_tushare.py")
        df_fl = pd.concat(fl_parts, ignore_index=True).drop_duplicates(
            subset=["ts_code", "trade_date"], keep="last"
        )
        if cb: cb("load_factor_lab", 15, f"factor_lab 加载完成 {len(df_fl)} 股", {"n": len(df_fl)})

        # 选择 ext 文件名后缀: 取最大日期的 ext_{mmdd}
        # 简化: 用 date 末 4 位作 suffix; 不存在则 fallback 到最新
        suffix = date[-4:]  # eg 0508
        feature_groups = [
            ("amount_features", [
                f"output/amount_features/amount_features.parquet",
                f"output/amount_features/amount_features_ext_{suffix}.parquet",
            ]),
            ("moneyflow_v1", [
                f"output/moneyflow/features.parquet",
                f"output/moneyflow/features_ext_{suffix}.parquet",
            ]),
            ("mfk", [
                f"output/mfk_features/features.parquet",
                f"output/mfk_features/features_ext_{suffix}.parquet",
            ]),
            ("pyramid", [
                f"output/pyramid_v2/features.parquet",
                f"output/pyramid_v2/features_ext_{suffix}.parquet",
            ]),
            ("v7_extras", [
                f"output/v7_extras/features.parquet",
                f"output/v7_extras/features_ext_{suffix}.parquet",
            ]),
            ("cogalpha", [
                f"output/cogalpha_features/features.parquet",
            ]),
        ]

        for i, (name, paths) in enumerate(feature_groups):
            if cb:
                cb("load_features", 20 + i * 8, f"merge {name}...", None)
            d_t = self._slice_target(paths, date)
            if d_t is None or d_t.empty:
                # fallback: 取最近一天
                d_t = self._slice_latest(paths)
                if d_t is None: continue
            d_t = d_t.drop(columns=["trade_date"])
            df_fl = df_fl.merge(d_t, on="ts_code", how="left", suffixes=("", "_x"))

        # 市场级 regime
        if cb: cb("load_regime", 70, "加载 regime...", None)
        rg = pd.read_parquet(self.regime_path)
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg_t = rg[rg["trade_date"] == date]
        if rg_t.empty:
            rg_t = rg.sort_values("trade_date").tail(1).reset_index(drop=True)
        rg_t = rg_t.rename(columns={"ret_5d": "mkt_ret_5d", "ret_20d": "mkt_ret_20d",
                                     "ret_60d": "mkt_ret_60d", "rsi14": "mkt_rsi14",
                                     "vol_ratio": "mkt_vol_ratio"})
        for col in ["regime_id", "mkt_ret_5d", "mkt_ret_20d", "mkt_ret_60d",
                    "mkt_rsi14", "mkt_vol_ratio"]:
            if col not in df_fl.columns and col in rg_t.columns:
                df_fl[col] = rg_t[col].iloc[0]

        rgx = pd.read_parquet(self.regime_extra_path)
        rgx["trade_date"] = rgx["trade_date"].astype(str)
        rgx_t = rgx[rgx["trade_date"] == date]
        if rgx_t.empty:
            rgx_t = rgx.sort_values("trade_date").tail(1).reset_index(drop=True)
        for col in rgx_t.columns:
            if col == "trade_date": continue
            if col not in df_fl.columns:
                df_fl[col] = rgx_t[col].iloc[0]

        self._factor_cache[date] = df_fl
        return df_fl

    def _slice_target(self, paths: list[str], date: str) -> Optional[pd.DataFrame]:
        parts = []
        for p in paths:
            ap = self.root / p
            if not ap.exists(): continue
            d = pd.read_parquet(ap)
            d["trade_date"] = d["trade_date"].astype(str)
            d_t = d[d["trade_date"] == date]
            if not d_t.empty: parts.append(d_t)
        if not parts: return None
        return pd.concat(parts, ignore_index=True).drop_duplicates(
            subset=["ts_code", "trade_date"], keep="last"
        )

    def _slice_latest(self, paths: list[str]) -> Optional[pd.DataFrame]:
        for p in paths:
            ap = self.root / p
            if not ap.exists(): continue
            d = pd.read_parquet(ap)
            d["trade_date"] = d["trade_date"].astype(str)
            d_t = d.sort_values(["ts_code", "trade_date"]).groupby("ts_code").tail(1).reset_index(drop=True)
            return d_t
        return None

    # ──────── 评分 ────────
    def score_market(self, date: str, cb: ProgressCb = None) -> pd.DataFrame:
        """全市场 V12 评分. 返回完整 df 含 buy/sell/quadrant/v7c_recommend."""
        if cb: cb("init", 0, f"启动 V12 全市场推理 {date}", None)
        df = self.load_factors_for_date(date, cb=cb)

        if cb: cb("predict", 75, "LightGBM 4 模型推理 (V15 r10/r20 + V6 sell)...", {"n_stocks": len(df)})
        df = df.copy()
        df["r10_pred"] = self.predict_one(df, "r10_v15_all")
        df["r20_pred"] = self.predict_one(df, "r20_v15_all")
        df["sell_10_v6_prob"] = self.predict_one(df, "sell_10_v6")
        df["sell_20_v6_prob"] = self.predict_one(df, "sell_20_v6")

        if cb: cb("anchor", 88, "锚定 0-100 分 + 双向合成...", None)
        s10 = _map_anchored(df["r10_pred"].values, *R10_ANCHOR)
        s20 = _map_anchored(df["r20_pred"].values, *R20_ANCHOR)
        df["buy_score"] = 0.5 * s10 + 0.5 * s20
        s10s = _map_anchored(df["sell_10_v6_prob"].values, *SELL10_V6)
        s20s = _map_anchored(df["sell_20_v6_prob"].values, *SELL20_V6)
        df["sell_score"] = 0.5 * s10s + 0.5 * s20s

        if cb: cb("zombie_filter", 90, "计算僵尸区过滤 (第 6 铁律)...", None)
        df = self._enrich_zombie(df, date)

        if cb: cb("classify", 92, "4 象限分类 + 6 铁律...", None)
        df["quadrant"] = [classify_quadrant(b, s) for b, s in
                           zip(df["buy_score"], df["sell_score"])]
        df["v7c_recommend"] = self._apply_v7c_rules(df)

        n_main = int(df["v7c_recommend"].sum())
        n_contra = int((df["quadrant"] == "矛盾段").sum())
        if cb: cb("done", 100,
                  f"完成: V7c 主推 {n_main} 股, 矛盾段 {n_contra} 股待 LLM 反挖",
                  {"main": n_main, "contradiction": n_contra, "total": len(df)})
        return df

    def score_stock(self, ts_code: str, date: str) -> dict:
        """单股 V12 评分. 内部走 load_factors_for_date 缓存."""
        df = self.score_market(date)
        row = df[df["ts_code"] == ts_code]
        if row.empty:
            raise ValueError(f"{ts_code} 不在 {date} 全市场截面")
        r = row.iloc[0]
        return {
            "ts_code": ts_code, "trade_date": date,
            "industry": r.get("industry"),
            "buy_score": float(r["buy_score"]),
            "sell_score": float(r["sell_score"]),
            "r10_pred": float(r["r10_pred"]),
            "r20_pred": float(r["r20_pred"]),
            "sell_10_v6_prob": float(r["sell_10_v6_prob"]),
            "sell_20_v6_prob": float(r["sell_20_v6_prob"]),
            "quadrant": r["quadrant"],
            "v7c_recommend": bool(r["v7c_recommend"]),
            "pyr_velocity_20_60": float(r["pyr_velocity_20_60"])
                if "pyr_velocity_20_60" in r and pd.notna(r["pyr_velocity_20_60"]) else None,
            "f1_neg1": float(r["f1_neg1"]) if "f1_neg1" in r and pd.notna(r["f1_neg1"]) else None,
            "f2_pos1": float(r["f2_pos1"]) if "f2_pos1" in r and pd.notna(r["f2_pos1"]) else None,
        }

    def _apply_v7c_rules(self, df: pd.DataFrame) -> pd.Series:
        """V7c 6 条铁律 (除仓位约束).

        2026-05-15 调整: 屏蔽 sell_score ≤ 30 这条 (派发判断翻车率高,
        0508→0514 实测被 sell 标记的"派发"股反而是大涨王: 普冉+33% / 澜起+26% / 长川+15%).

        现行铁律 (5 条, 编号保留):
        1. buy_score ∈ [70, 85]
        2. (已屏蔽) sell_score ≤ 30
        3. pyr_velocity_20_60 < p35
        4. |f1_neg1| < 0.005
        5. |f2_pos1| < 0.005
        6. NOT is_zombie  (横盘 ≥90% + MA60 走平/下)
        """
        if "pyr_velocity_20_60" in df.columns:
            p35 = df["pyr_velocity_20_60"].quantile(0.35)
        else:
            return ((df["buy_score"] >= 70))
        # 屏蔽 sell_score ≤ 30
        base = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                (df["pyr_velocity_20_60"] < p35))
        if "f1_neg1" in df.columns and "f2_pos1" in df.columns:
            base = base & (df["f1_neg1"].abs() < 0.005) & (df["f2_pos1"].abs() < 0.005)
        # 第 6 铁律: zombie 过滤
        if "is_zombie" in df.columns:
            base = base & (~df["is_zombie"].fillna(False).astype(bool))
        return base

    def _enrich_zombie(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """对全市场加 is_zombie 列 (来自 daily cache 计算 MA60 横盘度).

        缓存: 同一日只算一次 (放进 _factor_cache).
        """
        from .zombie_filter import compute_zombie_factors

        daily_cache_dir = self.root / "output" / "tushare_cache" / "daily"
        files = sorted(daily_cache_dir.glob("*.parquet"))
        end_int = int(date)
        # 只读最近 200 天足够算 MA60 + 20 日 lookback
        recent_files = [f for f in files if int(f.stem) <= end_int][-200:]
        parts = [pd.read_parquet(f) for f in recent_files]
        big = pd.concat(parts, ignore_index=True)
        big["trade_date"] = big["trade_date"].astype(str)
        big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        zombie_rows = []
        for ts, g in big.groupby("ts_code"):
            if len(g) < 80:
                zombie_rows.append({"ts_code": ts, "is_zombie": False,
                                     "zombie_days_pct": 0, "ma60_slope_short": 0})
                continue
            z = compute_zombie_factors(g)
            last = z[z["trade_date"] == date]
            if last.empty:
                last = z.tail(1)
            r = last.iloc[0]
            zombie_rows.append({
                "ts_code": ts,
                "is_zombie": bool(r["is_zombie"]),
                "zombie_days_pct": float(r["zombie_days_pct"]) if pd.notna(r["zombie_days_pct"]) else 0,
                "ma60_slope_short": float(r["ma60_slope_short"]) if pd.notna(r["ma60_slope_short"]) else 0,
            })
        zdf = pd.DataFrame(zombie_rows)
        # merge 到 df, 覆盖同名列
        for col in ["is_zombie", "zombie_days_pct", "ma60_slope_short"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df.merge(zdf, on="ts_code", how="left")

    def list_available_dates(self) -> list[str]:
        """从 ext2 parquet 文件的 trade_date 列扫描可用日期."""
        dates: set[str] = set()
        for p in self.ext_dir.glob("group_001_*.parquet"):
            try:
                d = pd.read_parquet(p, columns=["trade_date"])
                dates.update(d["trade_date"].astype(str).unique())
            except Exception:
                pass
        return sorted(dates)
