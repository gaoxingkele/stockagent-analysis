#!/usr/bin/env python3
"""V12 整合推理 2026-05-08.

V12 = V7c LGBM (Layer 1) + V11 LLM 视觉 (Layer 2, 仅矛盾段反挖)

架构:
  V7c 推荐池 (5 条铁律, ~1% 全市场)        → 直接入场, 主推
  V7c 矛盾段 (buy>=70 + sell>=70)            → V11 LLM 视觉过滤, bull>=0.5 救出
  其他象限 (理想多/沉寂/主流空/中性区)        → 不入选

依赖 v7c_inference_20260508.csv 已生成 (v7c_inference_0508.py 输出).

参数:
  --enable-llm   触发 V11 LLM 视觉过滤 (需改造 TDX→Tushare, 当前 placeholder)
  --top N        矛盾段截取前 N 个 r20_pred 最高的待 LLM 评估 (默认全跑)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGET_DATE = "20260508"
V7C_CSV = ROOT / "output" / "v7c_full_inference" / f"v7c_inference_{TARGET_DATE}.csv"
OUT_DIR = ROOT / "output" / "v12_inference"
OUT_DIR.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable-llm", action="store_true", help="触发 V11 LLM 视觉过滤")
    ap.add_argument("--top", type=int, default=0, help="矛盾段仅评估 r20_pred top N (0=全部)")
    args = ap.parse_args()

    if not V7C_CSV.exists():
        print(f"❌ 缺 V7c csv: {V7C_CSV}\n   先跑: python v7c_inference_0508.py")
        sys.exit(1)

    df = pd.read_csv(V7C_CSV, dtype={"ts_code": str})
    print(f"=== V12 整合推理 ({TARGET_DATE}) ===\n")
    print(f"V7c 全市场: {len(df):,} 股\n")

    # === Layer 1: V7c 主推荐 (5 条铁律) ===
    main_pool = df[df["v7c_recommend"] == True].copy()
    main_pool = main_pool.sort_values("r20_pred", ascending=False).reset_index(drop=True)
    main_pool["v12_source"] = "V7c-main"
    print(f"Layer 1 V7c 主推: {len(main_pool)} 股 (5 铁律)")

    # === Layer 2: V7c 矛盾段 → V11 视觉过滤 ===
    contra = df[df["quadrant"] == "矛盾段"].copy()
    contra = contra.sort_values("r20_pred", ascending=False).reset_index(drop=True)
    if args.top > 0:
        contra = contra.head(args.top)
    print(f"Layer 2 矛盾段候选: {len(contra)} 股 (默认放弃, V11 视觉可救出)\n")

    if args.enable_llm:
        print("⚠ V11 LLM 视觉过滤当前未启用 — 需先改造 analyze_v11_vision_poc.py "
              "数据源 TDX→Tushare. 跳过 LLM, 矛盾段全部不入选.\n")
        rescued = pd.DataFrame()
    else:
        rescued = pd.DataFrame()  # placeholder: bull_prob>=0.5 的子集
        print("[i] LLM 视觉未启用 (--enable-llm 未传), 矛盾段不入选 V12.\n"
              "    通过 V7c 主推 only 模式输出.\n")

    # === V12 最终推荐 = main_pool + rescued (按 r20_pred 排序) ===
    if len(rescued) > 0:
        rescued["v12_source"] = "V11-rescued-contradiction"
        v12 = pd.concat([main_pool, rescued], ignore_index=True)
    else:
        v12 = main_pool
    v12 = v12.sort_values("r20_pred", ascending=False).reset_index(drop=True)
    v12["rank"] = v12.index + 1

    # === 输出 ===
    cols_show = ["rank","ts_code","industry","buy_score","sell_score",
                  "r20_pred","r10_pred","sell_20_v6_prob","quadrant","v12_source"]
    cols_show = [c for c in cols_show if c in v12.columns]

    out_path = OUT_DIR / f"v12_inference_{TARGET_DATE}.csv"
    v12[cols_show + [c for c in v12.columns if c not in cols_show]].to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )

    # 矛盾段单独保留 csv (供后续 LLM 跑)
    contra_path = OUT_DIR / f"v12_contradiction_pending_{TARGET_DATE}.csv"
    contra.to_csv(contra_path, index=False, encoding="utf-8-sig")

    # === 屏幕展示 ===
    print(f"=== V12 最终推荐 (n={len(v12)}, 按 r20_pred 降序) ===")
    print(v12[cols_show].to_string(index=False))

    print(f"\n=== 矛盾段候选 (n={len(contra)}, 待 V11 LLM 视觉过滤) ===")
    show_n = min(20, len(contra))
    print(contra.head(show_n)[
        ["ts_code","industry","buy_score","sell_score","r20_pred","sell_20_v6_prob"]
    ].to_string(index=False))
    if len(contra) > show_n:
        print(f"... +{len(contra)-show_n} 股, 完整见 {contra_path.name}")

    print(f"\n输出:")
    print(f"  {out_path}")
    print(f"  {contra_path}")


if __name__ == "__main__":
    main()
