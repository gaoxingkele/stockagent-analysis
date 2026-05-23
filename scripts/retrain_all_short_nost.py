"""主控: ST 排除后串行重训短 OOS 版 (生产推理用).

目标:
  - r5_v17_all_nost      (train_v17_r5.py, train_end 20260420)
  - r10_v16_all_nost     (train_v16_full.py)
  - r20_v16_all_nost     (train_v16_full.py)
  - r20_1h_v2_nost       (train_r20_1h_v2.py, train_end 20260228)
  - r1/r4/r8_next_*_v3_nost (train_t1_all.py)

进度: output/retrain_progress_short.json
"""
from __future__ import annotations
import json, time, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = ROOT / "output" / "retrain_progress_short.json"


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


def run_script(script_name: str, expected_models: list, stage: str, p: dict):
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
            ic = meta.get("ic_val", "?")
            rank_ic = meta.get("rank_ic_val", "?")
            ic_s = f"{ic:.4f}" if isinstance(ic, float) else ic
            rk_s = f"{rank_ic:.4f}" if isinstance(rank_ic, float) else rank_ic
            print(f"  [OK] {m}: IC={ic_s} RankIC={rk_s}", flush=True)
        else:
            print(f"  [FAIL] {m} 未完成", flush=True)
    print(f"  耗时 {elapsed:.0f}s", flush=True)


def main():
    p = load_progress()
    print(f"\n=== ST 排除重训 (短 OOS 版, 生产推理用) ===\n", flush=True)
    print(f"进度文件: {PROGRESS}", flush=True)

    t_total = time.time()

    # 日线 R5 (短 OOS)
    run_script("train_v17_r5.py", ["r5_v17_all_nost"], "daily_r5_short", p)

    # 日线 R10/R20 (短 OOS)
    run_script("train_v16_full.py",
                ["r10_v16_all_nost", "r20_v16_all_nost"],
                "daily_r10_r20_short", p)

    # 1H R20 (短 OOS)
    run_script("train_r20_1h_v2.py", ["r20_1h_v2_nost"], "hourly_r20_short", p)

    # T+1 (短 OOS, 3 个模型: r1/r4/r8)
    run_script("train_t1_all.py",
                ["r1_next_open_v3_nost", "r4_next_morn_v3_nost", "r8_next_day_v3_nost"],
                "t1_short", p)

    total = time.time() - t_total
    p["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p["total_elapsed_sec"] = round(total, 1)
    save_progress(p)

    print(f"\n=== 全部完成, 总耗时 {total:.0f}s ({total/60:.1f} min) ===\n", flush=True)
    for stage, info in p["stages"].items():
        print(f"  {stage}: {info.get('status')}, 模型 {info.get('models_status', info.get('models'))}",
               flush=True)


if __name__ == "__main__":
    main()
