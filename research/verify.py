#!/usr/bin/env python3
"""Ralph 纪律检查器 — research/ratio-phase 循环的 verifyCommand.

不是跑测试, 是跑"纪律闸": 全绿才 exit 0。检查四件事:

  1. Leakage guard   — research/features/*.parquet 不得含 forward-field (前视泄漏)
  2. 生产线指纹      — 冻结文件 hash 与基线 manifest 一致 (SIGN-R05)
  3. 工件 + 裁决     — 每个非 skip task 都有 research/verdicts/<id>.json (status + 结论)
  4. Gate 良构       — T-005 若 status=PASS 必须带 4 项指标; REJECT 也是合法完成

退出码: 0 = 全部任务有裁决且无硬违规 (循环可完成); 1 = 未完成或有硬违规。
硬违规 (泄漏 / 指纹不符) 永远 exit 1, 且打印 [VIOLATION], 必须当轮修复。
"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "research"
FEATURES_DIR = RES / "features"
VERDICTS_DIR = RES / "verdicts"
MANIFEST = RES / "baseline_manifest.json"
PRD = ROOT / "plans" / "prd.json"

# ── forward-field 黑名单 (feedback_forward_label_assistant_fields) ──
# factor_lab 里 r5/r10/r20/r30/r40 与 dd5..dd40 是 forward 字段, 不是 backward 因子。
FORWARD_EXACT = {f"r{n}" for n in (1, 3, 5, 10, 20, 30, 40)} | \
                {f"dd{n}" for n in (5, 10, 20, 30, 40)} | \
                {"max_gain", "maxgain", "future_ret", "fwd_ret"}
FORWARD_PATTERNS = [
    re.compile(r"^r\d+_next"), re.compile(r"^fwd_"), re.compile(r"^future_"),
    re.compile(r"^label$"), re.compile(r"^target$"), re.compile(r"_forward$"),
    re.compile(r"^next_"),
]

# ── 生产线冻结文件 (SIGN-R05) ──
PROD_FILES = [
    "src/stockagent_analysis/v12_scoring.py",
]

errors: list[str] = []        # 未完成 (循环继续)
violations: list[str] = []    # 硬违规 (必须当轮修复)


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_leakage() -> None:
    if not FEATURES_DIR.exists():
        return
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import pandas as pd
        pq = None
    bad = []
    for f in sorted(FEATURES_DIR.glob("*.parquet")):
        if pq is not None:
            cols = list(pq.read_schema(f).names)
        else:
            cols = list(pd.read_parquet(f, engine="pyarrow").columns)
        for c in cols:
            cl = c.lower()
            if cl in FORWARD_EXACT or any(p.search(cl) for p in FORWARD_PATTERNS):
                bad.append(f"{f.name}::{c}")
    if bad:
        violations.append(
            "[VIOLATION] leakage guard 命中 forward-field (SIGN-R04): " + ", ".join(bad))


def check_production_fingerprint() -> None:
    cur = {}
    for rel in PROD_FILES:
        p = ROOT / rel
        if p.exists():
            cur[rel] = sha256(p)
    if not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(cur, indent=2), encoding="utf-8")
        print(f"[verify] 基线指纹已捕获 -> {MANIFEST.name} ({len(cur)} 文件冻结)")
        return
    base = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for rel, h in base.items():
        if cur.get(rel) != h:
            violations.append(
                f"[VIOLATION] 生产线文件被改动 (SIGN-R05): {rel} "
                f"— 若是 T-006 opt-in 落地, 更新 manifest; 否则回滚。")


def load_tasks() -> list[dict]:
    if not PRD.exists():
        violations.append(f"[VIOLATION] 缺 {PRD}")
        return []
    d = json.loads(PRD.read_text(encoding="utf-8"))
    return d.get("features", d.get("tasks", []))


def check_verdicts(tasks: list[dict]) -> None:
    for t in tasks:
        if t.get("skip"):
            continue
        tid = t["id"]
        vp = VERDICTS_DIR / f"{tid}.json"
        if not vp.exists():
            errors.append(f"{tid}: 缺 verdict ({vp.relative_to(ROOT)})")
            continue
        try:
            v = json.loads(vp.read_text(encoding="utf-8"))
        except Exception as e:
            violations.append(f"[VIOLATION] {tid} verdict 非法 JSON: {e}")
            continue
        if not str(v.get("status", "")).strip():
            errors.append(f"{tid}: verdict 缺 status")
        if not str(v.get("conclusion", "")).strip():
            errors.append(f"{tid}: verdict 缺 conclusion (一句话结论)")
        if tid == "T-005" and str(v.get("status", "")).upper() == "PASS":
            m = v.get("metrics", {})
            need = {"alpha_delta_pp", "sharpe", "worst_month", "pos_alpha_month_ratio"}
            miss = need - set(m)
            if miss:
                violations.append(
                    f"[VIOLATION] T-005 PASS 但缺指标 {sorted(miss)} — gate 不可凭空 PASS (SIGN-R01/R03)")


def main() -> int:
    print("=== research/verify.py 纪律检查 ===")
    VERDICTS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    check_leakage()
    check_production_fingerprint()
    tasks = load_tasks()
    check_verdicts(tasks)

    if violations:
        print("\n硬违规 (必须当轮修复):")
        for v in violations:
            print("  " + v)
    if errors:
        print("\n未完成:")
        for e in errors:
            print("  - " + e)

    if violations or errors:
        n_done = sum(1 for t in tasks
                     if not t.get("skip") and (VERDICTS_DIR / f"{t['id']}.json").exists())
        n_total = sum(1 for t in tasks if not t.get("skip"))
        print(f"\n[verify] FAIL — 任务裁决 {n_done}/{n_total}, "
              f"违规 {len(violations)}, 未完成 {len(errors)}")
        return 1

    print("\n[verify] ALL CHECKS PASSED — 全部非 skip 任务已有裁决, 无泄漏, 生产线指纹一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
