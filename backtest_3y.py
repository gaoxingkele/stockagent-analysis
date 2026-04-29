"""3 年回测 (2023-01 ~ 2026-03), 复用 backtest_new_factors.py 的 framework。

关键差异:
- PERIOD: 起始 2023-01-04, 终止 2026-03-31
- 股票池: 沿用现有 universe.json (5149 只)
- 输出目录: output/backtest_3y_2023_2026/
- 同样支持 checkpoint, 5 worker 并行, 等差数列分组

为节省工程量, 直接复用已有 worker_run_group 等函数, 只改路径常量。
"""
from __future__ import annotations
import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / ".env.cloubic", override=False)

# 强制覆盖原模块的路径常量再 import
import backtest_new_factors as bn
OUTPUT_DIR_3Y = ROOT / "output" / "backtest_3y_2023_2026"
RAW_DATA_DIR_3Y = OUTPUT_DIR_3Y / "raw_data"
GROUP_RESULTS_DIR_3Y = OUTPUT_DIR_3Y / "group_results"
LOG_DIR_3Y = OUTPUT_DIR_3Y / "logs"

for d in (OUTPUT_DIR_3Y, RAW_DATA_DIR_3Y, GROUP_RESULTS_DIR_3Y, LOG_DIR_3Y):
    d.mkdir(parents=True, exist_ok=True)

# 复用旧 universe (股票池一致, 便于和 1 年数据对比)
OLD_UNIVERSE = ROOT / "output" / "backtest_factors_2026_04" / "universe.json"
NEW_UNIVERSE = OUTPUT_DIR_3Y / "universe.json"
if not NEW_UNIVERSE.exists():
    shutil.copy(OLD_UNIVERSE, NEW_UNIVERSE)
    print(f"复用 universe: {OLD_UNIVERSE} → {NEW_UNIVERSE}")

# 监控覆盖 backtest_new_factors 的全局路径
bn.OUTPUT_DIR = OUTPUT_DIR_3Y
bn.RAW_DATA_DIR = RAW_DATA_DIR_3Y
bn.GROUP_RESULTS_DIR = GROUP_RESULTS_DIR_3Y
bn.LOG_DIR = LOG_DIR_3Y
bn.UNIVERSE_FILE = NEW_UNIVERSE
bn.MARKET_FILE = OUTPUT_DIR_3Y / "market_data.json"
bn.PROGRESS_FILE = OUTPUT_DIR_3Y / "progress.json"
bn.REPORT_FILE = OUTPUT_DIR_3Y / "report.md"

# 时间窗口
START_DATE = "20230101"
END_DATE = "20260331"

# 改写 days_ago_str 让它使用 START_DATE
def custom_days_ago_str(n: int = 0) -> str:
    return START_DATE
def custom_today_str() -> str:
    return END_DATE
def custom_safe_cutoff_str() -> str:
    return END_DATE

bn.days_ago_str = custom_days_ago_str
bn.today_str = custom_today_str

print(f"3 年回测: {START_DATE} → {END_DATE}")
print(f"输出目录: {OUTPUT_DIR_3Y}")
print(f"股票池: {NEW_UNIVERSE} (沿用 1 年回测的 universe)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["2", "3"], default="2",
                        help="phase 1 跳过 (复用 universe)")
    parser.add_argument("--group", type=int)
    args = parser.parse_args()

    main_log = bn.setup_logger("main_3y", LOG_DIR_3Y / "master.log")

    if args.phase == "2":
        main_log.info("===== 3y Phase 2: 跑分组 (5 worker × 52 组) =====")
        bn.phase2_run_groups(only_group=args.group)
    elif args.phase == "3":
        main_log.info("===== 3y Phase 3: 聚合报告 =====")
        bn.phase3_aggregate_report()
