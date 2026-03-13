# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path

from .doc_converter import convert_other_domain_doc
from .io_utils import build_run_dir
from .orchestrator import run_analysis


def _cmd_analyze(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"[错误] 指定的run-dir不存在: {run_dir}")
            sys.exit(1)
        print(f"[断点续传] 使用已有run-dir: {run_dir}", flush=True)
    else:
        run_dir = build_run_dir(root, args.symbol)
    result = run_analysis(
        root=root,
        symbol=args.symbol,
        name=args.name,
        run_dir=run_dir,
        llm_provider_override=args.provider,
        multi_eval_providers_override=args.providers,
    )
    if result.get("error"):
        sys.exit(1)
    print(f"run_dir: {run_dir}")
    print(f"final_decision: {result['final_decision']} (score={result['final_score']})")
    if result.get("final_pdf_path"):
        print(f"final_pdf: {result['final_pdf_path']}")


def _cmd_convert_doc(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[2]
    input_path = Path(args.input)
    output_path = convert_other_domain_doc(input_path, root / "docs" / "converted")
    print(f"converted_doc: {output_path}")


def _cmd_intraday_check(args: argparse.Namespace) -> None:
    from .intraday_check import run_intraday_check, print_anomaly_table
    from .config_loader import load_project_config
    root = Path(__file__).resolve().parents[2]
    config = load_project_config(root)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    results = run_intraday_check(symbols=symbols, date_filter=args.date, config=config)
    print_anomaly_table(results)


def _cmd_backtest(args: argparse.Namespace) -> None:
    from .backtest import backfill_returns, compute_summary, evaluate_accuracy
    if args.backfill:
        print("正在回填历史涨跌数据...", flush=True)
        backfill_returns()
    summary = compute_summary()
    if "error" in summary:
        print(f"[回测] {summary['error']}")
        return
    print("\n=== 回测评估报告 ===")
    for k, v in summary.items():
        label = {
            "total_signals": "信号总数",
            "evaluated": "已评估",
            "win_rate": "胜率(%)",
            "direction_accuracy": "方向准确率(%)",
            "stop_loss_hit_rate": "止损命中率(%)",
            "take_profit_hit_rate": "止盈命中率(%)",
            "avg_simulated_return": "模拟平均收益(%)",
            "buy_count": "买入信号数",
            "buy_win_rate": "买入胜率(%)",
            "sell_count": "卖出信号数",
            "sell_win_rate": "卖出胜率(%)",
        }.get(k, k)
        print(f"  {label}: {v}")
    print()
    evaluate_accuracy()


def main() -> None:
    parser = argparse.ArgumentParser(description="中国股市多智能体分析系统")
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze", help="分析单只股票并输出买卖决策")
    p_analyze.add_argument("--symbol", required=True, help="A股代码，如 600519")
    p_analyze.add_argument("--name", required=True, help="股票名称，如 贵州茅台")
    p_analyze.add_argument(
        "--provider",
        choices=["kimi", "grok", "gemini", "deepseek", "glm", "perplexity", "minmax", "chatgpt", "openai", "claude", "doubao", "qwen"],
        default=None,
        help="临时覆盖 LLM 提供商（默认读取 project.json: kimi）",
    )
    p_analyze.add_argument(
        "--providers",
        type=str,
        default=None,
        help="多模型权重/评估时指定，逗号分隔。支持: kimi,grok,gemini,deepseek,glm,perplexity,minmax,chatgpt,openai,claude。如: kimi,deepseek,grok",
    )
    p_analyze.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="指定已有的run目录路径用于断点续传。崩溃或部分provider失败后，可用相同--run-dir重新运行，自动加载已有数据和已完成的provider结果",
    )
    p_analyze.set_defaults(func=_cmd_analyze)

    p_convert = sub.add_parser("convert-doc", help="将其他品类智能体文档转换为中国股市定义文档")
    p_convert.add_argument("--input", required=True, help="输入 Markdown 文档路径")
    p_convert.set_defaults(func=_cmd_convert_doc)

    p_ic = sub.add_parser("intraday-check", help="盘中异常预警：对比前日评分与实时走势")
    p_ic.add_argument("--symbols", type=str, default=None,
                       help="股票代码(逗号分隔)，如 002571,600519。默认读取最近信号的所有股票")
    p_ic.add_argument("--date", type=str, default=None,
                       help="指定信号日期(YYYY-MM-DD)，默认使用最近日期")
    p_ic.set_defaults(func=_cmd_intraday_check)

    p_bt = sub.add_parser("backtest", help="回测评估历史信号")
    p_bt.add_argument("--backfill", action="store_true", help="先回填实际涨跌数据")
    p_bt.add_argument("--days", type=int, default=10, help="评估周期（默认10日）")
    p_bt.set_defaults(func=_cmd_backtest)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)
