"""H10 executor: one-time review and non-promotable scientific decision.

On the diagnostic track this stage can only classify the evidence. It must never
emit a promotable decision, and it must keep the blocking reasons at least as
prominent as the findings.
"""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from .abcds_model import TARGET_ID
from .runtime import atomic_json, digest, now

ACCEPTANCE_GAPS = (
    "H01 data admission is FAILED_VALIDITY; no formal promotion is possible on this chain",
    "no prospective evaluation exists, so no unopened evidence was reviewed",
    "no champion was frozen, so there is nothing to promote or to reject on merit",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build(root: Path, directory: Path, *, upstream_directories: dict[str, Path]) -> dict:
    root, directory = Path(root).resolve(), Path(directory).resolve()
    upstream = {name: Path(path) for name, path in upstream_directories.items()}
    for required in ("H02", "H05", "H06", "H07", "H08"):
        if required not in upstream:
            raise ValueError("H10 requires upstream outputs: " + required)
    label_contract = _read_json(upstream["H02"] / "label_contracts.json")
    policies = _read_json(upstream["H05"] / "policy_candidates.json")
    coverage = pd.read_csv(upstream["H05"] / "risk_coverage.csv")
    reliability = pd.read_csv(upstream["H05"] / "selected_reliability.csv")
    ablation = pd.read_csv(upstream["H06"] / "ablation.csv")
    activation = _read_json(upstream["H06"] / "advanced_activation_decisions.json")
    locked = _read_json(upstream["H07"] / "locked_development_report.json")
    paired = _read_json(upstream["H07"] / "paired_intervals.json")
    stress = pd.read_csv(upstream["H07"] / "stress.csv")
    champion = _read_json(upstream["H08"] / "champion_manifest.json")
    power = _read_json(upstream["H08"] / "power_plan.json")
    sealing = _read_json(upstream["H08"] / "sealing_manifest.json")

    primary = stress[(stress.role.eq("candidate")) & (stress.capacity.eq(20))
                     & (stress.cost_multiplier.eq(1.0))]
    stressed = stress[(stress.role.eq("candidate")) & (stress.capacity.eq(20))]
    blocking = [
        {"reason": "H01 FAILED_VALIDITY", "scope": "formal admission of the historical window",
         "evidence": "config/s20_v4_data_scope_decision_20260914.json"},
        {"reason": "G2 not met", "scope": "champion freeze",
         "evidence": f"folds={champion['gate_evidence']['G2_folds_used']}, "
                     f"seeds={champion['gate_evidence']['G2_seeds_used']}"},
        {"reason": "no clean retrospective holdout",
         "scope": "independent confirmation",
         "evidence": sealing["reason"]},
        {"reason": "no prospective evaluation", "scope": "G3 relative promotion",
         "evidence": "H09 is WAITING_MATURITY; no forward signal dates collected"},
    ]
    findings = [
        {"finding": "the five-class head separates safe targets from risk, but not safely",
         "evidence": "H07 paired comparison",
         "numbers": {key: locked.get(key) for key in
                     ("realized_safe_target_rate", "realized_paired_risk_rate",
                      "control_safe_target_rate", "control_paired_risk_rate")}},
        {"finding": "ranking by safe-target probability raises the +15% touch rate and the "
                    "paired risk rate together",
         "evidence": "H04 development frontier",
         "numbers": {"candidate_safe_target_rate": locked.get("realized_safe_target_rate"),
                     "candidate_paired_risk_rate": locked.get("realized_paired_risk_rate")}},
        {"finding": "calibration is well behaved on the full list but optimistic on the "
                    "risk-gated subset",
         "evidence": "H05 selected_reliability.csv",
         "numbers": {"max_gap": float(reliability.gap.max()) if len(reliability) else None,
                     "min_gap": float(reliability.gap.min()) if len(reliability) else None}},
        {"finding": "no advanced family earned activation",
         "evidence": "H06 advanced_activation_decisions.json",
         "numbers": {"activated_families": activation.get("activated_families", [])}},
    ]
    decision = {
        "at": now(),
        "target_id": TARGET_ID,
        "track": "diagnostic",
        "state": "PROMOTION_BLOCKED_BY_VALIDITY",
        "promotable": False,
        "claim_supported": False,
        "claim_statement": ("the diagnostic chain produced reference numbers for the "
                            "five-class ABCDS task; it did not establish a promotable improvement"),
        "blocking_reasons": blocking,
        "findings": findings,
        "reviewed_evidence": {
            "label_version": label_contract["label_version"],
            "class_counts": label_contract["counts"],
            "registered_policies": len(policies.get("registered_policies", [])),
            "ablation_configs": int(len(ablation)),
            "episodes": locked.get("positions"),
            "paired_dates": {key: value.get("paired_dates")
                             for key, value in paired.get("paired", {}).items()},
        },
        "valid_terminal_states": ["REJECTED", "INSUFFICIENT_EVIDENCE",
                                  "PROMOTION_BLOCKED_BY_VALIDITY"],
        "chosen_state_rationale": ("the diagnostic evidence is internally consistent but "
                                   "admission and independence gates are unmet, so the "
                                   "honest terminal state is blocked rather than rejected"),
        "production_change_requires": "a separate authorized task",
        "formal_H10_accepted": False,
        "production_eligible": False,
    }
    consolidated_coverage = coverage.copy()
    consolidated_coverage["source"] = "H05 registered policy grid"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("final_report.md", "risk_coverage.csv", "paired_intervals.json",
                 "promotion_decision.json", "checkpoint.json"):
        if (directory / name).exists():
            raise ValueError("H10 refuses existing outputs: " + name)
    consolidated_coverage.to_csv(directory / "risk_coverage.csv", index=False)
    atomic_json(directory / "paired_intervals.json",
                dict(paired, reviewed_by="H10", promotable=False))
    atomic_json(directory / "promotion_decision.json", decision)
    atomic_json(directory / "checkpoint.json", {
        "at": now(), "protocol_hash": None, "target_id": TARGET_ID,
        "stages_reviewed": sorted(upstream), "terminal_state": decision["state"],
        "formal_accepted": False, "production_eligible": False,
        "artifact_hashes": {name: digest(upstream[stage] / key)
                            for name, (stage, key) in {
                                "labels": ("H02", "labels.parquet"),
                                "split_manifest": ("H03", "split_manifest.json"),
                                "paired_intervals": ("H07", "paired_intervals.json")}.items()},
        "acceptance_gaps": list(ACCEPTANCE_GAPS)})
    report = [
        "# S20 ABCDS 诊断链一次性裁决",
        "",
        f"日期：{now()[:10]}。轨道：diagnostic。结论：**{decision['state']}**，不可晋级。",
        "",
        "## 阻断原因",
        ""] + [f"- {item['reason']}（{item['scope']}）：{item['evidence']}"
               for item in blocking] + [
        "", "## 标签", "",
        f"- {label_contract['label_version']}：{label_contract['counts']}，"
        f"未知 {label_contract['unknown_rows']}",
        f"- 沉默定义隔离：新 S={label_contract['counts']['S']}，"
        f"legacy a-only S={label_contract['legacy_a_only_silence_rows']}",
        "", "## 主要读数", ""] + [
        f"- 候选 vs 控制（配对）：安全达标 {locked.get('realized_safe_target_rate')} vs "
        f"{locked.get('control_safe_target_rate')}；配对风险 "
        f"{locked.get('realized_paired_risk_rate')} vs {locked.get('control_paired_risk_rate')}",
        f"- 入选子集校准偏离：{float(reliability.gap.min()):.4f} ~ "
        f"{float(reliability.gap.max()):.4f}" if len(reliability) else "- 入选子集校准偏离：无",
        f"- 费用压力（容量 20）：{[round(v, 4) for v in stressed.final_nav.tolist()]}",
        f"- 功效规划：达到 {power.get('minimum_practical_effect_pp')}pp 需要约 "
        f"{power.get('required_dates')} 个配对信号日（观测 SD={power.get('observed_sd')}）",
        f"- 高级家族激活：{activation.get('activated_families', [])}",
        "", "## 边界", "",
        "- H01 数据准入仍为 FAILED_VALIDITY，历史窗口不能用于正式验收。",
        "- 外层段已被诊断链消费，不能再当独立留出；唯一干净路径是前瞻采集。",
        "- 未冻结冠军（G2 未达），因此没有可晋级的候选。",
        "- 生产改动需要另立授权任务。",
        ""]
    (directory / "final_report.md").write_text("\n".join(report), encoding="utf-8")
    return {
        "formal_gate_passed": False,
        "formal_H10_accepted": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "decision_state": decision["state"],
        "promotable": False,
        "blocking_reasons": [item["reason"] for item in blocking],
        "findings": [item["finding"] for item in findings],
        "validation_scope": "diagnostic review; no promotion decision is supported",
    }
