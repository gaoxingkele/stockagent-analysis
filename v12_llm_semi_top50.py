"""对半导体板块 Top 50 (按 r20_pred 降序) 跑 V11 LLM 视觉评估."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env.cloubic"); load_dotenv(ROOT / ".env")

from stockagent_analysis.v11_vision import V11VisionFilter

DATE = "20260508"
OUT = ROOT / "output" / "v12_inference" / f"v11_semi_top50_{DATE}.csv"
CHECKPOINT = OUT  # 复用主输出做 checkpoint

def main():
    t0 = time.time()
    cloubic = os.environ.get("CLOUBIC_API_KEY")
    if not cloubic:
        print("CLOUBIC_API_KEY 未配置"); sys.exit(1)

    # 半导体 Top 50
    df = pd.read_csv(ROOT / "output/v7c_full_inference/v7c_inference_20260508.csv",
                      dtype={"ts_code":str})
    semi = df[df["industry"] == "半导体"].sort_values("r20_pred", ascending=False).head(50)
    # 确保 300604 在内
    if "300604.SZ" not in semi["ts_code"].tolist():
        more = df[df["ts_code"] == "300604.SZ"]
        if not more.empty:
            semi = pd.concat([semi, more], ignore_index=True)
    symbols = semi["ts_code"].tolist()

    # 跳过 checkpoint
    done = set()
    if CHECKPOINT.exists():
        done_df = pd.read_csv(CHECKPOINT, dtype={"ts_code":str})
        done = set(done_df["ts_code"].tolist())
    todo = [s for s in symbols if s not in done]
    print(f"半导体 Top 50: 总 {len(symbols)} 只, 已完成 {len(done)}, 待跑 {len(todo)}", flush=True)

    f = V11VisionFilter.get(ROOT, cloubic)
    f._load_daily_cache(DATE)

    write_header = not CHECKPOINT.exists()
    cost_sum = 0.0
    for i, ts in enumerate(todo, 1):
        st = time.time()
        rec = f.filter_one(ts, DATE)
        ti = rec.get("tokens_in", 0); to = rec.get("tokens_out", 0)
        cost_sum += (ti * 3.0 + to * 15.0) / 1e6
        bp = rec.get("bull_prob")
        bp_s = f"{bp:.2f}" if isinstance(bp, float) and bp == bp else "N/A"
        print(f"  [{i:2d}/{len(todo)}] {ts} status={rec['status']:8s} bull={bp_s} ({time.time()-st:.0f}s) ${cost_sum:.3f}", flush=True)
        # 每股 append
        pd.DataFrame([rec]).to_csv(CHECKPOINT, mode="a", header=write_header,
                                    index=False, encoding="utf-8-sig")
        write_header = False

    print(f"\n=== 完成, 耗时 {time.time()-t0:.0f}s, 成本 ${cost_sum:.3f}, 输出 {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
