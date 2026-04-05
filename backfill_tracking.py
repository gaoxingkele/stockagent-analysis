# -*- coding: utf-8 -*-
"""追踪回填入口 — 每天运行一次，回填历史评分的实际涨跌幅。

用法:
    python backfill_tracking.py                    # 回填未完成记录
    python backfill_tracking.py --import-history   # 先从 output/history/ 导入历史评分再回填
    python backfill_tracking.py --calibrate        # 回填后输出校准报告
    python backfill_tracking.py --all              # 导入 + 回填 + 校准（推荐）
"""
import os
import sys
import argparse

# Windows 控制台 UTF-8
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 默认关闭代理（国内数据源直连）
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from stockagent_analysis.tracking import (
    backfill_returns,
    calibrate,
    print_calibration_report,
    import_from_history,
    TRACKING_DB,
)


def main():
    parser = argparse.ArgumentParser(description="评分追踪回填与校准")
    parser.add_argument("--import-history", action="store_true",
                        help="从 output/history/ 导入历史评分到 tracking.csv")
    parser.add_argument("--calibrate", action="store_true",
                        help="输出校准报告")
    parser.add_argument("--all", action="store_true",
                        help="导入 + 回填 + 校准（推荐）")
    parser.add_argument("--force", action="store_true",
                        help="强制回填已完成的记录")
    parser.add_argument("--min-samples", type=int, default=10,
                        help="校准最小样本数（默认10）")
    args = parser.parse_args()

    do_import = args.import_history or args.all
    do_backfill = True  # 总是回填
    do_calibrate = args.calibrate or args.all

    # Step 1: 导入历史
    if do_import:
        print("=" * 60)
        print("[Step 1] 导入历史评分")
        print("=" * 60)
        import_from_history()

    # Step 2: 回填
    if do_backfill:
        print()
        print("=" * 60)
        print("[Step 2] 回填实际涨跌")
        print("=" * 60)
        result = backfill_returns(force=args.force)
        print(f"  总记录: {result['total']}, 更新: {result['filled']}, 跳过: {result['skipped']}")

    # Step 3: 校准
    if do_calibrate:
        print()
        print("=" * 60)
        print("[Step 3] 校准分析")
        print("=" * 60)
        report = calibrate(min_samples=args.min_samples)
        print_calibration_report(report)

        # 保存校准报告到 JSON
        report_path = TRACKING_DB.parent / "calibration_report.json"
        import json
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[校准] 报告已保存: {report_path}")

    if not do_import and not do_calibrate:
        print(f"\n提示: 运行 python backfill_tracking.py --all 可一键完成导入+回填+校准")


if __name__ == "__main__":
    main()
