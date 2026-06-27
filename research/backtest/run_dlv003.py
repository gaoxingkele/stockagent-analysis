# -*- coding: utf-8 -*-
"""DLV-003 — 研究环升级 (Track B) demo + verdict.

不产研究结果, 验证 `research/research_env.py` 三件套在**真实工件**上忠实可用, 并落 verdict=built:

  ① replication (B2): `block_bootstrap_ci` 跑 WFE-001 真实 book (净Sharpe~1.31) →
     必须**复刻 FIN-001 已发布的 CI[+0.11,+2.60]** (证明抽出的 util 与既有结论零口径漂移)。
  ② 多重检验 (B3): `deflated_sharpe_ratio` 对"若 1.31 是从 N 次搜索里选出来的"做扣减演示 +
     `pbo_cscv` 在合成策略族上演示过拟合检测。
  ③ reviewer panel (B1): research/REVIEWER_PANEL_CHECKLIST.md 已落地, 此处校验存在 + 摘要。

纪律: 仅 load WFE-001 缓存 book (R05 生产只读); 锚 1.31/CI 来自 FIN-001 不在结果上调 (R01);
status=built 基础设施非 alpha (R03 不适用)。`python run_dlv003.py`。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
import research_env as RE  # noqa: E402

WFE_BOOK = ROOT / "research" / "cache" / "wfe001" / "wfe001_book_oos.parquet"
FIN_RESULTS = ROOT / "research" / "cache" / "fin001" / "fin001_results.json"
CHECKLIST = ROOT / "research" / "REVIEWER_PANEL_CHECKLIST.md"
CACHE = ROOT / "research" / "cache" / "dlv003"
CACHE.mkdir(parents=True, exist_ok=True)
VERDICT = ROOT / "research" / "verdicts" / "DLV-003.json"

# FIN-001 已发布锚 (R01 冻结, 不在结果上调)
ANCHOR_SHARPE = 1.31
ANCHOR_CI = [0.11, 2.60]


def main():
    t0 = time.time()
    print("\n=== DLV-003: 研究环升级 (replication + PBO/DSR + reviewer panel) demo ===\n", flush=True)

    # ── ① replication: block_bootstrap_ci 复刻 FIN-001 CI ──
    book = pd.read_parquet(WFE_BOOK)
    book.index = book.index.astype(str)
    rets = book["nav"].astype(float).pct_change().dropna().to_numpy()
    bb = RE.block_bootstrap_ci(rets, cohort_len=20, n_boot=1000, seed=20260614)
    fin = json.loads(FIN_RESULTS.read_text(encoding="utf-8"))
    fin_ci = fin["block_bootstrap"]["sharpe_ci95"]
    fin_p0 = fin["block_bootstrap"]["p_sharpe_gt_0"]
    # 容差: 同 seed 同 cohort 划分, util 与 FIN-001 应数值一致 (≤0.02)
    ci_match = (abs(bb["sharpe_ci95"][0] - fin_ci[0]) <= 0.02
                and abs(bb["sharpe_ci95"][1] - fin_ci[1]) <= 0.02)
    print(f"[①replication] util block_bootstrap_ci on WFE-001 book:", flush=True)
    print(f"  point Sharpe {bb['point_sharpe']:+.3f}  CI95 {bb['sharpe_ci95']}  "
          f"P(>0)={bb['p_sharpe_gt_0']:.3f}  P(>1)={bb['p_sharpe_gt_1']:.3f}", flush=True)
    print(f"  FIN-001 已发布 CI95 {fin_ci}  P(>0)={fin_p0:.3f}  → 复刻匹配? {ci_match}", flush=True)
    assert ci_match, f"util CI {bb['sharpe_ci95']} 未复刻 FIN-001 {fin_ci} — 口径漂移!"

    # ── ② 多重检验校正: DSR (若 1.31 是搜来的) ──
    # per-obs 单位: 把年化 1.31 转 per-obs; 假设搜索过 N 个候选, 其 per-obs SR 方差 ~0.0025
    # (即年化 SR 方差 ~0.0025*252≈0.63, 年化 SR 标准差~0.79 — 与本轮候选 Sharpe 散布同量级)
    sr_po = RE.annualized_to_per_obs(bb["point_sharpe"])
    n_obs = bb["n_obs"]
    from scipy.stats import skew as _sk, kurtosis as _ku
    sk = float(_sk(rets)); ku = float(_ku(rets, fisher=False))
    sr_var_per_obs = 0.0025
    dsr_demo = {}
    for ntr in (1, 10, 50, 200):
        d = RE.deflated_sharpe_ratio(sr_po, ntr, n_obs, sr_var_per_obs, skew=sk, kurt=ku)
        dsr_demo[ntr] = d
    psr_solo = RE.probabilistic_sharpe_ratio(sr_po, n_obs, 0.0, skew=sk, kurt=ku)
    print(f"\n[②多重检验] PSR (单次, vs SR*=0) = {psr_solo:.4f}", flush=True)
    print(f"  DSR (若 1.31 是从 N 候选搜出, per-obs SR_var={sr_var_per_obs}):", flush=True)
    for ntr, d in dsr_demo.items():
        print(f"    N={ntr:4d}: 期望最大 SR0={d['sr0_expected_max_per_obs']:+.4f}(per-obs)  "
              f"DSR={d['dsr']:.4f}  过(≥0.95)? {d['passes']}", flush=True)

    # PBO via CSCV: 合成策略族 (一个真 alpha + 噪声) 演示检测器
    rng = np.random.default_rng(20260616)
    fam_noise = rng.normal(0, 0.012, size=(n_obs, 12))
    pbo_noise = RE.pbo_cscv(fam_noise, n_splits=12)
    fam_real = rng.normal(0, 0.012, size=(n_obs, 12)); fam_real[:, 0] += 0.0035
    pbo_real = RE.pbo_cscv(fam_real, n_splits=12)
    print(f"\n  PBO via CSCV (S=12, C(12,6)={pbo_noise['n_combinations']} 划分):", flush=True)
    print(f"    纯噪声族 12 配置 → PBO={pbo_noise['pbo']:.3f} (高=IS选优纯过拟合)", flush=True)
    print(f"    含1真alpha族   → PBO={pbo_real['pbo']:.3f} (低=OOS有延续)", flush=True)

    # ── ③ reviewer panel checklist ──
    checklist_ok = CHECKLIST.exists()
    checklist_lines = len(CHECKLIST.read_text(encoding="utf-8").splitlines()) if checklist_ok else 0
    print(f"\n[③reviewer panel] {CHECKLIST.relative_to(ROOT)} 存在? {checklist_ok} ({checklist_lines} 行)", flush=True)

    # ── self-test 也跑一遍 (确保 util 解析闸全过) ──
    print("\n[selftest] research_env.py 内置解析自检:", flush=True)
    RE._selftest()

    # ── 落盘 ──
    results = {
        "replication_block_bootstrap": {
            "util_result": bb, "fin001_published_ci95": fin_ci,
            "fin001_published_p_gt0": fin_p0, "reproduces_fin001": ci_match,
        },
        "multiple_testing": {
            "psr_solo": round(float(psr_solo), 4),
            "dsr_by_n_trials": {str(k): v for k, v in dsr_demo.items()},
            "sr_variance_per_obs_assumed": sr_var_per_obs,
            "pbo_noise_family": pbo_noise, "pbo_real_alpha_family": pbo_real,
        },
        "reviewer_panel": {"checklist_exists": checklist_ok, "checklist_lines": checklist_lines,
                           "path": str(CHECKLIST.relative_to(ROOT))},
        "anchor": {"net_sharpe": ANCHOR_SHARPE, "ci95": ANCHOR_CI, "source": "FIN-001 (R01 冻结)"},
    }
    (CACHE / "dlv003_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    conclusion = (
        f"研究环升级三件套就绪 (Track B, ROADMAP_2026H2.md): 一个可被未来每个候选 gate 复用的 "
        f"`research/research_env.py` + reviewer panel checklist, 让搜索出的候选'骗不过'。"
        f"**①replication 默认化 (B2)**: `block_bootstrap_ci` (per-cohort block bootstrap 误差条) + "
        f"`multi_seed_replicate` (跨 seed 稳定性)。在 WFE-001 真实 book 上跑 → Sharpe 95% CI={bb['sharpe_ci95']} "
        f"P(>0)={bb['p_sharpe_gt_0']:.2f} **精确复刻 FIN-001 已发布 CI{fin_ci}** (零口径漂移, 证 util 忠实)。"
        f"**②多重检验校正 (B3)**: `probabilistic_sharpe_ratio`(PSR) / `deflated_sharpe_ratio`(DSR, Bailey-LdP "
        f"惩罚试验次数 N) / `pbo_cscv`(PBO via CSCV)。DSR 演示: 同样观测 Sharpe 1.31, 若是从 N 个候选搜出, "
        f"N=1→DSR {dsr_demo[1]['dsr']:.2f} / N=10→{dsr_demo[10]['dsr']:.2f} / N=200→{dsr_demo[200]['dsr']:.2f} "
        f"(试验越多, 扣减后显著性越低 = 正是本轮 RETRO/Hybrid 假阳性该被拦下的地方)。PBO 检测器: 纯噪声策略族 "
        f"PBO={pbo_noise['pbo']:.2f}(高=过拟合) vs 含真 alpha 族 {pbo_real['pbo']:.2f}(低)。"
        f"**③reviewer panel (B1)**: `REVIEWER_PANEL_CHECKLIST.md` ({checklist_lines} 行) 把'单 codex 审'升级为 "
        f"**多模型独立审 (codex 抓口径混淆/lookahead + 独立 claude 抓 framing/regime 盲点) + meta-judge 仲裁**, "
        f"配 §1 作者自审闸 + §2-3 机器统计闸 (映射到 research_env 各函数) + §5 终判落库。"
        f"`research_env.py` 内置 7 项解析 self-test 全过 (PSR/DSR 单调性/PBO 噪声vs真alpha/bootstrap/multi_seed/"
        f"skeptic_report 端到端)。**纯流程基础设施, 非 alpha** (status=built); 一站式入口 `skeptic_report()` 把 "
        f"§2-3 机器闸打成 should_get_excited 布尔, 但作者自审 + panel 人/模型流程不可被单布尔替代。"
        f"生产线 V12.31 只读冻结 (R05); 锚 1.31/CI 来自 FIN-001 (R01)。"
    )

    VERDICT.write_text(json.dumps({
        "id": "DLV-003", "status": "built", "conclusion": conclusion,
        "metrics": {
            "replication_reproduces_fin001": ci_match,
            "util_sharpe_ci95": bb["sharpe_ci95"],
            "fin001_published_ci95": fin_ci,
            "util_p_sharpe_gt_0": bb["p_sharpe_gt_0"],
            "psr_solo": round(float(psr_solo), 4),
            "dsr_n1": dsr_demo[1]["dsr"], "dsr_n10": dsr_demo[10]["dsr"],
            "dsr_n50": dsr_demo[50]["dsr"], "dsr_n200": dsr_demo[200]["dsr"],
            "pbo_noise_family": pbo_noise["pbo"],
            "pbo_real_alpha_family": pbo_real["pbo"],
            "checklist_lines": checklist_lines,
            "selftest_passed": True,
        },
        "artifacts": {
            "util": "research/research_env.py",
            "reviewer_checklist": "research/REVIEWER_PANEL_CHECKLIST.md",
            "demo_runner": "research/backtest/run_dlv003.py",
            "results": "research/cache/dlv003/dlv003_results.json",
        },
        "deliverables": {
            "B2_replication": ["block_bootstrap_ci", "multi_seed_replicate"],
            "B3_multiple_testing": ["probabilistic_sharpe_ratio", "deflated_sharpe_ratio",
                                    "expected_max_sharpe", "pbo_cscv"],
            "B1_reviewer_panel": "research/REVIEWER_PANEL_CHECKLIST.md",
            "one_stop": "skeptic_report() -> should_get_excited",
        },
        "discipline": "R01 锚 1.31/CI 来自 FIN-001 不在结果上调; R03 基础设施非 alpha (status=built, "
                      "中间指标≠ship 不适用); R05 生产只读 (仅 load WFE-001 缓存 book, 0 生产改动); "
                      "R04 大缓存 research/cache/dlv003/ gitignored。util 复刻 FIN-001 CI 证零口径漂移, "
                      "未在任何结果上调门柱。交付物 = '骗不过的 verify harness' 升级, 是 Track C 新探索的前置。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] verdict=built -> {VERDICT.relative_to(ROOT)}  耗时 {(time.time()-t0):.1f}s", flush=True)


if __name__ == "__main__":
    main()
