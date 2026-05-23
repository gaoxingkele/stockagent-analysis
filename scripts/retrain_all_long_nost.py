"""主控: ST 排除后串行重训长 OOS 版模型 (2026-05-21).

模型清单 (_nost 后缀, 保留旧版对照):
  1. r5_v17_long_nost       (日线 5 日)
  2. r10_v16_long_nost      (日线 10 日, 之前没 long 版)
  3. r20_v16_long_nost      (日线 20 日)
  4. r20_1h_v2_long_nost    (1H 20 日)
  5. r1_next_open_v3_long_nost  (T+1 隔夜, ST 偏见最严重的那个)

策略:
  - 串行跑 (避免内存爆), 每个完成即写 checkpoint
  - 任何模型已存在则跳过 (训练脚本自带逻辑)
  - 中断后重启不会重训已完成的

进度: output/retrain_progress.json

用法: python scripts/retrain_all_long_nost.py  (建议 background)
"""
from __future__ import annotations
import json, time, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = ROOT / "output" / "retrain_progress.json"


def load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    return {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}


def save_progress(p: dict):
    p["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    PROGRESS.write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")


def model_done(name: str) -> bool:
    d = ROOT / "output" / "production" / name
    return (d / "classifier.txt").exists() and (d / "meta.json").exists()


def run_script(script_name: str, expected_models: list[str], stage: str, p: dict):
    if all(model_done(m) for m in expected_models):
        print(f"\n[{stage}] 全部已完成, 跳过: {expected_models}", flush=True)
        p["stages"][stage] = {"status": "already_done", "models": expected_models}
        save_progress(p)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"[{stage}] 跑 {script_name}", flush=True)
    print(f"  目标: {expected_models}", flush=True)
    print(f"{'='*60}\n", flush=True)
    p["stages"][stage] = {"status": "running", "models": expected_models,
                            "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_progress(p)

    t0 = time.time()
    proc = subprocess.run([sys.executable, str(ROOT / script_name)],
                            cwd=str(ROOT))
    elapsed = time.time() - t0

    done = {m: model_done(m) for m in expected_models}
    p["stages"][stage].update({
        "status": "done" if all(done.values()) else "partial",
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "models_status": done,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    save_progress(p)

    for m, ok in done.items():
        if ok:
            meta = json.loads((ROOT / "output" / "production" / m / "meta.json").read_text(encoding="utf-8"))
            print(f"  [OK] {m}: IC={meta.get('ic_val','?'):.4f} RankIC={meta.get('rank_ic_val','?'):.4f}",
                   flush=True)
        else:
            print(f"  [FAIL] {m} 未完成", flush=True)
    print(f"  耗时 {elapsed:.0f}s", flush=True)


def main():
    p = load_progress()
    print(f"\n=== ST 排除重训 (长 OOS 版) ===\n", flush=True)
    print(f"进度文件: {PROGRESS}", flush=True)

    t_total = time.time()

    # Stage 1: 日线 (r5/r10/r20 long)
    run_script(
        "train_daily_long_oos.py",
        ["r5_v17_long_nost", "r20_v16_long_nost", "r10_v16_long_nost"],
        "daily_long", p,
    )

    # Stage 2: 1H (r20 + r1)
    run_script(
        "train_long_oos.py",
        ["r20_1h_v2_long_nost", "r1_next_open_v3_long_nost"],
        "hourly_long", p,
    )

    total = time.time() - t_total
    p["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p["total_elapsed_sec"] = round(total, 1)
    save_progress(p)

    print(f"\n=== 全部完成, 总耗时 {total:.0f}s ({total/60:.1f} min) ===\n", flush=True)
    # 汇总
    for stage, info in p["stages"].items():
        print(f"  {stage}: {info.get('status')}, 模型 {info.get('models_status', info.get('models'))}",
               flush=True)


if __name__ == "__main__":
    main()
