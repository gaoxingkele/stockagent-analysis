# -*- coding: utf-8 -*-
"""OA-SUMMARY — 正交因子 campaign 收尾: 全族裁决汇总 + 多重检验总账 + 元结论。

读 research/verdicts/OA-*.json + research/cache/oa_manifest.json → 汇总表 + findings.md。
不下新结论, 只聚合既有 verdict (SIGN-R02 负结果=合法完成)。
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VDIR = ROOT / "research" / "verdicts"
MANIFEST_P = ROOT / "research" / "cache" / "oa_manifest.json"
OUT_JSON = VDIR / "OA-SUMMARY.json"
OUT_MD = ROOT / "research" / "ORTHOGONAL_ALPHA_FINDINGS.md"


def load(name):
    p = VDIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    man = json.loads(MANIFEST_P.read_text(encoding="utf-8")) if MANIFEST_P.exists() else {"n_trials_total": 0}
    n_trials = man.get("n_trials_total", 0)

    families = {
        "OA-VOL": load("OA-VOL"), "OA-VP": load("OA-VP"), "OA-COM": load("OA-COM"),
    }
    gates = {"OA-VOL-GATE": load("OA-VOL-GATE"), "OA-VP-GATE": load("OA-VP-GATE")}
    h1 = load("OA-1H")

    # 统计
    phase1_proceed = [f for f, v in families.items() if v and v.get("verdict") == "PROCEED"]
    phase1_reject = [f for f, v in families.items() if v and v.get("verdict") == "REJECT"]
    deferred = ["OA-RC (no report_rc permission)", "OA-NEWS (only market-level major_news)"]
    gate_pass = [g for g, v in gates.items() if v and v.get("status") == "PASS"]
    gate_small = [g for g, v in gates.items() if v and v.get("status") == "真小"]
    gate_reject = [g for g, v in gates.items() if v and v.get("status") == "REJECT"]

    summary = {
        "campaign": "orthogonal-alpha (Track C2)",
        "date": "2026-06-18",
        "n_trials_screened": n_trials,
        "n_trials_dsr_with_dof": n_trials * 3,
        "phase1_proceed": phase1_proceed,
        "phase1_reject": phase1_reject,
        "deferred_no_data": deferred,
        "walk_forward_gate": {"pass": gate_pass, "真小": gate_small, "reject": gate_reject},
        "landable_alpha": bool(gate_pass),
        "meta_conclusion": (
            "正交因子 campaign 净结果 = 无可落地新 alpha。两个真异象 (低波动 ivol_60 IC t=-6.69, "
            "隔夜-日内 on_minus_in_20 IC t=8.04) 过 Phase-1 廉价正交闸 (正交于 mom/size/value/quality), "
            "但接 V12.31 池 19月 walk-forward 双双 REJECT (Δα=-0.15/-0.70pp, PBO 0.8 过拟合) — V12.31 "
            "价量均值回归核已捕获 (低波动经 pyr_velocity, 隔夜经 pump/ratio)。商品供应链渠道 Phase-1 "
            "塌缩 (beta 被 size/industry 吸收)。分析师修正 (理论最强) + 新闻政策因数据闸关闭 deferred。"
            "再次印证 0605 元结论: 正交≠有α; 即时可拉信号的可交易部分=V12.31 已吃。第22-23个被否假设。"),
        "what_would_change_it": [
            "report_rc 权限开通 → 分析师修正动量 (codex 排第1真正交 prior, 唯一未测的强候选)",
            "个股级 news/anns_d 权限 或 投 LLM 实体抽取管线 → 新闻/政策横截面因子",
            "微结构/盘口/Level-2 数据 (本 token 不含) → 真正脱离价量空间的正交信息",
        ],
        "artifacts": {
            "screen_tool": "research/oa_screen.py",
            "builds": ["research/oa_vol_factors.py", "research/oa_vp_factors.py",
                       "research/oa_com_factors.py", "research/oa_1h_compare.py"],
            "gate": "research/oa_gate.py",
            "verdicts": sorted(str(p.relative_to(ROOT)) for p in VDIR.glob("OA-*.json")),
            "codex_review": "research/cache/codex_oa_review_out.txt",
        },
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # findings.md
    def fam_row(name, v):
        if not v:
            return f"| {name} | — | deferred/缺 |"
        verd = v.get("verdict", v.get("status", "?"))
        best = v.get("best_factor", "")
        ic = v.get("best_orth_ic", "")
        return f"| {name} | {best} orth_ic={ic} | {verd} |"

    lines = [
        "# 正交因子 campaign 发现 (Track C2, 2026-06-18)",
        "",
        "> 用户开闸广搜正交正-alpha。每族 build→Phase-1 廉价正交IC闸(扣[动量+size+value+roe/margin]残差, "
        "NW-t HAC, ≥40月, 分regime)→ 存活进 V12.31 池 19月 walk-forward GATE + skeptic(bootstrap/PBO/DSR)。",
        "> codex 对抗审计纳入冻结gate (NW-t / DSR含DoF乘子3 / selection-on-selection caveat)。生产线 V12.31 全程冻结。",
        "",
        f"**多重检验基数**: 筛过 {n_trials} 因子变体, DSR n_trials = {n_trials}×3(DoF) = {n_trials*3}。",
        "",
        "## Phase-1 廉价闸结果",
        "",
        "| 族 | 最强因子 | 裁决 |",
        "|---|---|---|",
        fam_row("OA-VOL 波动率", families["OA-VOL"]),
        fam_row("OA-VP 量价", families["OA-VP"]),
        fam_row("OA-COM 商品供应链", families["OA-COM"]),
        "| OA-RC 分析师修正 | — | deferred (无 report_rc 权限) |",
        "| OA-NEWS 新闻政策 | — | deferred (仅市场级 major_news) |",
        "",
        "## walk-forward GATE (存活因子 vs V12.31)",
        "",
        "| 候选 | Δα/月 | gate | 裁决 |",
        "|---|---|---|---|",
    ]
    for g, v in gates.items():
        if v:
            lines.append(f"| {v.get('candidate','')} ({g}) | {v.get('delta_alpha_pp')}pp | "
                         f"regime {v.get('regime_delta')} | **{v.get('status')}** |")
    lines += [
        "",
        "## 元结论",
        "",
        summary["meta_conclusion"],
        "",
        "## 真异象但不可落地 (为什么)",
        "- **低波动 ivol_60** (Ang-Hodrick-Xing-Zhang): orth IC −0.081 t=−6.69 强, 但 V12.31 的 "
        "pyr_velocity 过滤已隐含低波动偏好 → Δα −0.15pp, PBO 0.8 过拟合。",
        "- **隔夜-日内分解 on_minus_in_20** (Lou-Polk-Skouras): orth IC +0.048 t=8.04 (残差后增强), "
        "但隔夜动量已被 V12.31 pump/ratio 启动子捕获 → Δα −0.70pp 全regime负。",
        "- **商品供应链 beta**: raw IC ~0.02-0.03 但被 size/industry 完全吸收 (Phase-1 C_塌缩), "
        "商品 beta 无独立横截面 alpha。",
        "",
        "## 下一步 (条件触发)",
    ]
    for w in summary["what_would_change_it"]:
        lines.append(f"- {w}")
    lines += ["", "## 1H vs 日线对比", ""]
    if h1:
        lines.append(h1.get("conclusion", ""))
    else:
        lines.append("(OA-1H verdict 见 research/verdicts/OA-1H.json)")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[summary] -> {OUT_JSON.relative_to(ROOT)}", flush=True)
    print(f"[findings] -> {OUT_MD.relative_to(ROOT)}", flush=True)
    print(f"\n  n_trials={n_trials} (DSR base {n_trials*3})", flush=True)
    print(f"  Phase-1 PROCEED: {phase1_proceed}", flush=True)
    print(f"  Phase-1 REJECT: {phase1_reject}", flush=True)
    print(f"  deferred: {deferred}", flush=True)
    print(f"  GATE: pass={gate_pass} 真小={gate_small} reject={gate_reject}", flush=True)
    print(f"  可落地新 alpha: {summary['landable_alpha']}", flush=True)
    return 0


if __name__ == "__main__":
    main()
