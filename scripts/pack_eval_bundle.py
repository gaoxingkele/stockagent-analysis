# -*- coding: utf-8 -*-
"""打包「四引擎评估」最小数据包 (output/ 被 gitignore, 换机器用此包搬运)。

本机(数据源)运行:
    python scripts/pack_eval_bundle.py            # 生成 eval_bundle.tar.gz (默认项目根)
    python scripts/pack_eval_bundle.py -o D:/share # 指定输出目录
    python scripts/pack_eval_bundle.py --check     # 只校验产物是否齐全, 不打包

远程机(纯代码检出)解包 (在项目根):
    tar -xzf eval_bundle.tar.gz                    # 恢复 output/ 下各路径
    python eval_4engine_fast.py                    # 即可跑

包内容 (≈2.5G, 四引擎直接输入, 不含可重建上游):
    output/factor_lab_3y/factor_groups/   OOS 因子面板 (220k 样本)
    output/labels/max_gain_labels.parquet 标签
    output/factor_lab_oos/validity_matrix.json
    output/etf_analysis/stock_to_etfs.json
    output/lgbm_maxgain/                   gain/dd 回归模型
"""
from __future__ import annotations
import argparse, sys, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PATHS = [
    "output/factor_lab_3y/factor_groups",
    "output/labels/max_gain_labels.parquet",
    "output/factor_lab_oos/validity_matrix.json",
    "output/etf_analysis/stock_to_etfs.json",
    "output/lgbm_maxgain",
]


def dir_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human(n: int) -> str:
    for u in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}T"


def check() -> list[Path]:
    missing = []
    total = 0
    print("=== 四引擎数据包清单校验 ===")
    for rel in PATHS:
        p = ROOT / rel
        if p.exists():
            sz = dir_size(p)
            total += sz
            print(f"  [OK] {rel:<45} {human(sz)}")
        else:
            missing.append(p)
            print(f"  [缺失] {rel}")
    print(f"--- 合计 {human(total)} ---")
    if missing:
        print(f"\n[!] {len(missing)} 项缺失, 无法打包。请确认在数据源机器上运行。")
    return missing


def pack(out_dir: Path):
    if check():
        sys.exit(1)
    out = out_dir / "eval_bundle.tar.gz"
    print(f"\n打包 → {out} ...")
    t0 = time.time()
    with tarfile.open(out, "w:gz", compresslevel=1) as tar:
        for rel in PATHS:
            print(f"  + {rel}")
            tar.add(ROOT / rel, arcname=rel)
    print(f"\n完成: {out}  ({human(out.stat().st_size)}, {time.time()-t0:.0f}s)")
    print("远程解包(项目根): tar -xzf eval_bundle.tar.gz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT), help="tar.gz 输出目录")
    ap.add_argument("--check", action="store_true", help="只校验不打包")
    a = ap.parse_args()
    if a.check:
        check()
    else:
        pack(Path(a.out))
