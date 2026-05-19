"""1H 因子 v3 - 断点续传版 (用户 0518 要求).

vs compute_1h_factors_v3.py: 每 500 股写一个独立 parquet,
失败可续, 内存峰值 ~500 MB 而非 7 GB.

输出:
  output/1h_factors/v3_batches/batch_{idx:04d}.parquet  (每批一个文件)
  output/1h_factors/factors_v3.parquet                  (最后合并)

用法:
  python compute_1h_factors_v3_chunked.py            # 增量跑
  python compute_1h_factors_v3_chunked.py --merge    # 仅合并 (跑完后)
  python compute_1h_factors_v3_chunked.py --force    # 全部重跑
"""
from __future__ import annotations
import sys, time, gc
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "output" / "tushare_cache" / "1h"
OUT = ROOT / "output" / "1h_factors"
BATCH_DIR = OUT / "v3_batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)
OUT_FINAL = OUT / "factors_v3.parquet"

BATCH_SIZE = 500

sys.path.insert(0, str(ROOT))
from compute_1h_factors_v3 import compute_factors_one_stock


def run_batches(force: bool = False):
    files = sorted(SRC.glob("*.parquet"))
    n_total = len(files)
    print(f"加载 1H 数据: {n_total} 个文件", flush=True)
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    t0 = time.time()

    for bi in range(n_batches):
        bp = BATCH_DIR / f"batch_{bi:04d}.parquet"
        if bp.exists() and not force:
            print(f"  [{bi+1}/{n_batches}] ✅ 跳过已存 batch_{bi:04d}.parquet", flush=True)
            continue
        start = bi * BATCH_SIZE
        batch_files = files[start:start + BATCH_SIZE]
        parts = []
        n_fail = 0
        for f in batch_files:
            try:
                df = pd.read_parquet(f)
                df["ts_code"] = f.stem
                feat = compute_factors_one_stock(df)
                if not feat.empty: parts.append(feat)
            except Exception as e:
                n_fail += 1
                if n_fail < 5: print(f"    {f.stem} 失败: {e}", flush=True)
        if parts:
            big = pd.concat(parts, ignore_index=True)
            big.to_parquet(bp, index=False, compression="snappy")
            sz = bp.stat().st_size / 1024 / 1024
            print(f"  [{bi+1}/{n_batches}] {time.time()-t0:.0f}s  ok, "
                  f"{len(big):,} 行  {sz:.0f} MB  fail={n_fail}", flush=True)
        del parts
        gc.collect()


def merge_final():
    bps = sorted(BATCH_DIR.glob("batch_*.parquet"))
    if not bps:
        print("无 batch 文件"); return
    print(f"合并 {len(bps)} 个 batch 文件...", flush=True)
    t0 = time.time()
    parts = []
    for bp in bps:
        parts.append(pd.read_parquet(bp))
        if len(parts) % 5 == 0:
            print(f"  loaded {len(parts)}/{len(bps)}, {time.time()-t0:.0f}s", flush=True)
    big = pd.concat(parts, ignore_index=True)
    print(f"合并完成: {len(big):,} × {len(big.columns)}", flush=True)
    big.to_parquet(OUT_FINAL, index=False, compression="snappy")
    sz = OUT_FINAL.stat().st_size / 1024 / 1024
    print(f"输出: {OUT_FINAL} ({sz:.0f} MB)  耗时 {time.time()-t0:.0f}s", flush=True)


def main():
    args = sys.argv[1:]
    if "--merge" in args:
        merge_final(); return
    force = "--force" in args
    run_batches(force=force)
    print("\n所有批次完成, 自动合并...", flush=True)
    merge_final()


if __name__ == "__main__":
    main()
