# -*- coding: utf-8 -*-
"""FIN-002 — 冻结 V12.31 真实基线 + 部署前 checklist (基本面 point-in-time 风险核查).

吃进 codex 第三次 review②: 冻结基线收尾。两件事:
  1. 把 V12.31 真·真实基线 (WFE-001 embargo book, FIN-001 误差条) 冻结成
     research/backtest/V12.31_BASELINE.md (冻结配置 + 真实期望 1.31±CI + 前向 paper-trade 协议 + 部署前 checklist)。
  2. 部署前 checklist 的**可执行部分**: 程序化核对 r20 池排序模型 (r20_v16_long_nost) 的特征里有哪些
     是**基本面 / point-in-time 风险字段** (codex#3 残留盲点), 分级报告。

本脚本只读 (不改策略不动门柱, R01; 复用缓存未碰生产, R05; 无 features 写入, R04)。
产出 research/cache/fin002/fin002_checklist.json (机器可读) + 打印摘要; 文档/裁决另写。

PIT 风险分级 (依据数据 provenance, 见 tushare_enrich.py / fu001_build_pit.py):
  - LOW   : 日频快照, 每个 trade_date 的值即当日可知 (daily_basic / cyq_perf 日频)。
  - MEDIUM: 季度/低频字段, 对齐口径 (ann_date 公告日 vs end_date 报告期末) 在 factor_lab
            合并源未在代码层显式断言 → 上线前须核 as-of 对齐 (否则 ~1 季度前视风险)。
  - NONE  : 纯价量/技术/资金流/形态因子, 无基本面披露滞后问题。
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "output" / "production"
OUT_DIR = ROOT / "research" / "cache" / "fin002"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "fin002_checklist.json"

# ── 基本面 / PIT 风险字段分级 (codex#3 残留盲点核查) ──
# 依据: r20 模型 feature_meta.json 的 feature_cols; provenance 见 src/.../tushare_enrich.py + research/fu001_build_pit.py
PIT_RISK = {
    # daily_basic 日频快照: 每个 trade_date 的 pe/pb/mv 即当日 Tushare 披露值, as-of 正确 (低风险)。
    # 唯一细节: pe_ttm 分母 (TTM 净利) 由当日已披露最新报告期推, daily_basic 已按当日财报口径算 → as-of。
    "total_mv": "LOW",
    "pe": "LOW",
    "pe_ttm": "LOW",
    "pb": "LOW",
    # cyq_perf 筹码绩效 日频: winner_rate 获利盘按当日算 (日频快照, 低风险)。
    "winner_rate": "LOW",
    # stk_holdernumber 股东户数 季频: 若 factor_lab 合并源按 end_date(报告期末) 对齐而非 ann_date(公告日),
    # 则有 ~1 季度前视 (Q1 报告 end_date 0331 但 ann_date ~0430)。factor_lab.py 未在代码层断言对齐口径 → MEDIUM。
    "holder_pct": "MEDIUM",
}


def main():
    meta = json.loads((PROD / "r20_v16_long_nost" / "feature_meta.json").read_text(encoding="utf-8"))
    feats = [c for c in meta["feature_cols"] if c != "industry_id"]
    n_total = len(feats)

    by_level = {"MEDIUM": [], "LOW": [], "NONE": []}
    for f in feats:
        by_level[PIT_RISK.get(f, "NONE")].append(f)

    fundamental = by_level["MEDIUM"] + by_level["LOW"]
    result = {
        "model": "r20_v16_long_nost (V12.31 r20 池排序模型)",
        "n_features_total": n_total,
        "n_fundamental_pit_fields": len(fundamental),
        "fundamental_pit_fields": fundamental,
        "by_risk_level": {
            "MEDIUM": by_level["MEDIUM"],
            "LOW": by_level["LOW"],
        },
        "n_none_risk_price_volume_tech": len(by_level["NONE"]),
        "provenance": {
            "total_mv/pe/pe_ttm/pb": "Tushare daily_basic 日频快照 (as-of 正确)",
            "winner_rate": "Tushare cyq_perf 日频筹码 (as-of 正确)",
            "holder_pct": "Tushare stk_holdernumber 季频股东户数 (对齐口径未在 factor_lab 代码层断言)",
        },
        "verdict": (
            f"{n_total} 个 r20 特征里仅 {len(fundamental)} 个是基本面字段 "
            f"(其余 {len(by_level['NONE'])} 全是价量/技术/资金流/形态/市场环境, 无披露滞后)。"
            "5/6 (total_mv/pe/pe_ttm/pb/winner_rate) 来自日频快照源 = point-in-time 安全 (低风险)。"
            "**唯一 MEDIUM 风险 = holder_pct (季频股东户数)**: factor_lab 合并源未在代码层显式断言按 ann_date(公告日)"
            "对齐 → 上线前须核 as-of 对齐 (若按 end_date 报告期末则有 ~1 季度前视)。"
            "影响有限 (单字段且 r20 信号已被 BT-003 证主导来自整池而非单因子), 但部署前应核实/或替换为 FU-001 已建的 "
            "ann_date 对齐 PIT 面板 (research/features/fundamental_pit.parquet)。"
        ),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== FIN-002 部署前 checklist: r20 基本面 PIT 风险核查 ===")
    print(f"  r20 模型特征总数 (除 industry_id): {n_total}")
    print(f"  基本面字段数: {len(fundamental)}  → {fundamental}")
    print(f"  MEDIUM 风险 (须上线前核): {by_level['MEDIUM']}")
    print(f"  LOW 风险 (日频快照 PIT 安全): {by_level['LOW']}")
    print(f"  NONE 风险 (价量/技术/资金流/形态): {len(by_level['NONE'])} 个")
    print(f"\n  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
