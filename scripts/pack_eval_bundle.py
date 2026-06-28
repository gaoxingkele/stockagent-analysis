# -*- coding: utf-8 -*-
"""打包运行时数据包 (output/ 被 gitignore, 换机器用此包搬运)。

⚠ research/features/ 和 research/cache/ (9.4G) 是已否决实验的死缓存, 运行不需要, 不打包。
   生产唯一需要的 research 文件 fund_portfolio_cache.parquet(192K) 已在 git 里。

本机(数据源)运行:
    python scripts/pack_eval_bundle.py                 # 默认 all (eval+prod 并集, ~5.6G)
    python scripts/pack_eval_bundle.py -p eval         # 仅四引擎评估 (~2.5G)
    python scripts/pack_eval_bundle.py -p prod         # 仅生产看板/web (~5.5G)
    python scripts/pack_eval_bundle.py -p all --check  # 只校验齐全不打包
    python scripts/pack_eval_bundle.py -p eval -o D:/share

远程机(纯代码检出)解包 (在项目根):
    tar -xzf all_bundle.tar.gz       # 恢复 output/ 下各路径
    python eval_4engine_fast.py      # 四引擎评估
    python daily_dashboard.py        # 四池看板 (生产打分)

档位:
    eval — 四引擎评估直接输入 (eval_4engine_fast/position 读)
    prod — daily_dashboard / daily_review / web 看板 的生产打分输入 (V12.31)
    all  — 上面两者并集 (factor_groups 共享, 去重)
"""
from __future__ import annotations
import argparse, sys, tarfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EVAL = [
    "output/factor_lab_3y/factor_groups",
    "output/labels/max_gain_labels.parquet",
    "output/factor_lab_oos/validity_matrix.json",
    "output/etf_analysis/stock_to_etfs.json",
    "output/lgbm_maxgain",
]
PROD = [
    "output/factor_lab_3y/factor_groups",
    "output/factor_lab_3y/factor_groups_extension",
    "output/mfk_features",
    "output/moneyflow",
    "output/pyramid_v2",
    "output/v7_extras",
    "output/amount_features",
    "output/tushare_cache/daily",
    "output/tushare_cache/moneyflow",
    "output/tushare_cache/stock_basic.parquet",
]


def profile_paths(name: str) -> list[str]:
    if name == "eval":
        return EVAL
    if name == "prod":
        return PROD
    seen, out = set(), []  # all = 并集去重, 保序
    for p in EVAL + PROD:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def dir_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human(n: float) -> str:
    for u in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}T"


def check(paths: list[str]) -> list[str]:
    missing, total = [], 0
    print(f"=== 数据包清单校验 ({len(paths)} 项) ===")
    for rel in paths:
        p = ROOT / rel
        if p.exists():
            sz = dir_size(p)
            total += sz
            print(f"  [OK] {rel:<48} {human(sz)}")
        else:
            missing.append(rel)
            print(f"  [缺失] {rel}")
    print(f"--- 合计 {human(total)} ---")
    if missing:
        print(f"\n[!] {len(missing)} 项缺失, 请确认在数据源机器运行。")
    return missing


def pack(profile: str, out_dir: Path):
    paths = profile_paths(profile)
    if check(paths):
        sys.exit(1)
    out = out_dir / f"{profile}_bundle.tar.gz"
    print(f"\n打包 [{profile}] → {out} ...")
    t0 = time.time()
    with tarfile.open(out, "w:gz", compresslevel=1) as tar:
        for rel in paths:
            print(f"  + {rel}")
            tar.add(ROOT / rel, arcname=rel)
    print(f"\n完成: {out}  ({human(out.stat().st_size)}, {time.time()-t0:.0f}s)")
    print(f"远程解包(项目根): tar -xzf {out.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--profile", choices=["eval", "prod", "all"], default="all")
    ap.add_argument("-o", "--out", default=str(ROOT), help="tar.gz 输出目录")
    ap.add_argument("--check", action="store_true", help="只校验不打包")
    a = ap.parse_args()
    if a.check:
        check(profile_paths(a.profile))
    else:
        pack(a.profile, Path(a.out))
