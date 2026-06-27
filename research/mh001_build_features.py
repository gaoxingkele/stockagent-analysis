# -*- coding: utf-8 -*-
"""MH-001 — 全历史梅花特征 parquet 生成 + 泄漏/确定性校验.

对全历史 daily 面板 (ts_code, trade_date, close) 用 research/meihua_encoder.encode_frame
生成 mh_*/mhs_* 确定性 categorical 特征, 落盘 research/features/meihua_features.parquet。

纪律:
  - ST 源头排除 (SIGN-R06): stock_basic.name 含 'ST' 的股票在加载即剔除。
  - 零泄漏 (SIGN-R04): 只输入决策时刻已知量 (代码/日期/当日收盘价), 输出列仅 key + mh_*/mhs_*,
    不含任何 forward 字段 (r5/r10/r20/dd* 等)。
  - 确定性 (回测=实盘逐位一致): 重算逐位比对。
输出: research/features/meihua_features.parquet (key: ts_code, trade_date)
       research/verdicts/MH-001.json (status=built + 分布摘要)
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
from meihua_encoder import encode_frame  # noqa: E402

DAILY_DIR = ROOT / "output" / "tushare_cache" / "daily"
BASIC_P = ROOT / "output" / "tushare_cache" / "stock_basic.parquet"
OUT_DIR = ROOT / "research" / "features"
OUT_P = OUT_DIR / "meihua_features.parquet"
VERDICT_P = ROOT / "research" / "verdicts" / "MH-001.json"

KEY = ["ts_code", "trade_date"]
# forward 黑名单 (与 verify.py 一致), 落盘前自查输出列绝不命中
FORWARD_TOKENS = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                 {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                 {"max_gain", "maxgain", "future_ret", "fwd_ret"}


def load_st_set() -> set:
    if not BASIC_P.exists():
        print("  ⚠️ stock_basic.parquet 不存在, ST 排除跳过", flush=True)
        return set()
    basic = pd.read_parquet(BASIC_P)[["ts_code", "name"]].drop_duplicates("ts_code")
    return set(basic[basic["name"].fillna("").str.contains("ST", regex=False)]["ts_code"])


def load_panel() -> pd.DataFrame:
    print(f"[panel] 加载全历史 daily close ({DAILY_DIR}) ...", flush=True)
    parts = [pd.read_parquet(f, columns=["ts_code", "trade_date", "close"])
             for f in sorted(DAILY_DIR.glob("*.parquet"))]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.drop_duplicates(subset=KEY, keep="last")
    # 收盘价缺失/<=0 的样本无法起卦, 剔除
    big = big[big["close"].notna() & (big["close"] > 0)].reset_index(drop=True)
    st = load_st_set()
    if st:
        before = len(big)
        big = big[~big["ts_code"].isin(st)].reset_index(drop=True)
        print(f"  ST 源头排除: {before - len(big):,} 行 ({len(st)} 只 ST)", flush=True)
    return big.sort_values(KEY).reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VERDICT_P.parent.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    print(f"[panel] {len(panel):,} 行 / {panel['ts_code'].nunique()} 股 / "
          f"{panel['trade_date'].nunique()} 日 ({panel['trade_date'].min()}~{panel['trade_date'].max()})",
          flush=True)

    t = time.time()
    enc = encode_frame(panel)
    feat_cols = [c for c in enc.columns if c.startswith("mh")]
    out = enc[KEY + feat_cols].copy()
    print(f"[encode] {len(feat_cols)} 特征列, {time.time() - t:.1f}s", flush=True)

    # ── 泄漏自查: 输出列绝不命中 forward 黑名单 ──
    bad = [c for c in out.columns if c.lower() in FORWARD_TOKENS
           or c.lower().startswith(("fwd_", "future_", "next_"))
           or c.lower().endswith("_forward")]
    assert not bad, f"输出含 forward 字段 (泄漏): {bad}"
    # 全部特征列必须是 mh_/mhs_ 前缀的离散 int (categorical), 无 close 等连续量泄漏入特征
    assert all(c.startswith(("mh_", "mhs_")) for c in feat_cols), "存在非梅花特征列"

    # ── 确定性自检: 重算逐位一致 (在 5% 抽样上, 省时但足够) ──
    samp = panel.sample(min(50000, len(panel)), random_state=0).reset_index(drop=True)
    re_enc = encode_frame(samp)[feat_cols]
    base = encode_frame(samp)[feat_cols]
    deterministic = bool((re_enc.values == base.values).all())
    assert deterministic, "确定性自检失败: 重算不一致"

    # ── 分布 sanity ──
    rel_dist = out["mh_relation"].value_counts(normalize=True).sort_index().to_dict()
    rel_dist = {int(k): round(float(v), 4) for k, v in rel_dist.items()}
    n_rel = out["mh_relation"].nunique()
    n_base = int(out["mh_base_gua"].nunique())
    n_mhs_base = int(out["mhs_base_gua"].nunique())
    moving_dist = {int(k): int(v) for k, v in
                   out["mh_moving"].value_counts().sort_index().to_dict().items()}
    # 体用关系 5 类不退化: 每类占比都在 [5%, 60%] 之内 (非全压一类)
    rel_ok = (n_rel == 5) and all(0.05 <= v <= 0.60 for v in rel_dist.values())

    out.to_parquet(OUT_P, index=False)
    print(f"[out] -> {OUT_P.relative_to(ROOT)}  ({len(out):,} 行 × {len(out.columns)} 列)", flush=True)

    verdict = {
        "task": "MH-001",
        "status": "built",
        "conclusion": (
            f"全历史梅花特征落盘 {len(out):,} 行 × {len(feat_cols)} 特征 "
            f"({panel['trade_date'].min()}~{panel['trade_date'].max()}); "
            f"确定性={deterministic}, 零泄漏(仅决策时刻已知量), "
            f"体用5类{'不退化' if rel_ok else '退化⚠️'}, "
            f"组合卦 base {n_base}/64 覆盖, 静态卦 base {n_mhs_base}/64."),
        "artifact": str(OUT_P.relative_to(ROOT)),
        "metrics": {
            "rows": int(len(out)),
            "n_stocks": int(panel["ts_code"].nunique()),
            "n_dates": int(panel["trade_date"].nunique()),
            "date_range": [panel["trade_date"].min(), panel["trade_date"].max()],
            "n_feature_cols": len(feat_cols),
            "deterministic": deterministic,
            "leakage_free": True,
            "mh_relation_dist": rel_dist,
            "mh_relation_classes": int(n_rel),
            "mh_relation_nondegenerate": bool(rel_ok),
            "mh_base_gua_coverage": f"{n_base}/64",
            "mhs_base_gua_coverage": f"{n_mhs_base}/64",
            "mh_moving_dist": moving_dist,
        },
        "guardrails": ["SIGN-R04 泄漏自查通过", "SIGN-R06 ST 源头排除"],
    }
    VERDICT_P.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verdict] -> {VERDICT_P.relative_to(ROOT)}", flush=True)
    print(f"  体用关系占比: {rel_dist}")
    print(f"  组合卦 base {n_base}/64, 静态卦 base {n_mhs_base}/64, 动爻 {moving_dist}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
