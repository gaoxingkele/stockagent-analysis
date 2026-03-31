# -*- coding: utf-8 -*-
"""批量分析多只股票，每只生成PDF，最后终端输出汇总表格。"""
import os
import sys
import json
from pathlib import Path

# Windows 控制台 UTF-8
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 默认关闭代理
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""

ROOT = Path(__file__).resolve().parent


class _Tee:
    """同时输出到终端和文件，带重试机制（解决Windows文件锁定问题）。"""
    def __init__(self, path: Path):
        self.path = path
        self.locked = False

    def write(self, text: str):
        sys.stdout.write(text)
        sys.stdout.flush()
        if not text.endswith("\n"):
            return
        for attempt in range(3):
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(text)
                self.locked = False
                break
            except PermissionError:
                self.locked = True
                import time
                time.sleep(0.5)
            except OSError:
                break

    def flush(self):
        sys.stdout.flush()


_log = _Tee(ROOT / "batch_run.log")


def _log_print(*args, **kwargs):
    """打印到终端+文件。"""
    text = " ".join(str(a) for a in args) + "\n"
    _log.write(text)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CHECKPOINT_FILE = ROOT / "batch_checkpoint.json"

from stockagent_analysis.io_utils import build_run_dir
from stockagent_analysis.orchestrator import run_analysis

# ── 全局状态容器（供信号处理器访问）────────────────────
class _BatchState:
    completed: list = []
    results: list = []
    total: int = 0

_batch = _BatchState()

def _signal_handler(signum, frame):
    if _batch.completed:
        _log_print()
        _log_print(f"[Batch] 收到中断信号，正在保存 checkpoint...")
        _save_checkpoint(_batch.completed, _batch.results)
        _log_print(f"[Batch] checkpoint已保存 ({len(_batch.completed)}/{_batch.total} 完成)")
        _log_print("[Batch] 下次运行会自动从断点继续。")
    sys.exit(0)

import signal
signal.signal(signal.SIGINT, _signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, _signal_handler)

# ── 待分析列表 ──────────────────────────────────────────
STOCKS = [
    ("603806", "福斯特"),
    ("000792", "盐湖股份"),
    ("002460", "赣锋锂业"),
    ("300769", "德方纳米"),
    ("603026", "石大胜华"),
    ("002422", "科伦药业"),
    ("002773", "康弘药业"),
    ("301292", "海科新源"),
    ("603538", "美诺华"),
    ("002466", "天齐锂业"),
    ("603906", "龙蟠科技"),
    ("300390", "天华新能"),
    ("603799", "华友钴业"),
    ("300683", "海特生物"),
    ("600513", "联环药业"),
    ("600773", "西藏城投"),
    ("002192", "融捷股份"),
    ("688799", "华纳药厂"),
    ("002203", "海亮股份"),
    ("002082", "万邦德"),
    ("000155", "川能动力"),
    ("000546", "金圆股份"),
    ("688266", "泽璟制药-U"),
    ("002497", "雅化集团"),
    ("688513", "苑东生物"),
]

PROVIDERS = None  # 使用 project.json 默认 providers


def _decision_label(score: float) -> str:
    if score >= 70:
        return "🔴**买入**"
    if score >= 60:
        return "🔴弱买入"
    if score >= 50:
        return "⚪观望"
    if score >= 40:
        return "🟢弱卖出"
    return "🟢**卖出**"


def _bias_icon(pct: float) -> str:
    if abs(pct) > 8:
        return "🔴"
    if abs(pct) > 5:
        return "⚠️"
    return "✅"


def _safe_float(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _load_result(run_dir: Path) -> dict:
    """从 run_dir 加载 final_decision.json 和 analysis_context.json。"""
    fd_path = run_dir / "final_decision.json"
    ctx_path = run_dir / "data" / "analysis_context.json"
    fd = json.loads(fd_path.read_text(encoding="utf-8")) if fd_path.exists() else {}
    ctx = json.loads(ctx_path.read_text(encoding="utf-8")) if ctx_path.exists() else {}
    # analysis_features 嵌在 final_decision.json 内
    feat = fd.get("analysis_features", {})
    # close price: 优先从日线CSV取最新收盘价，其次 analysis_context snapshot
    close = 0.0
    csv_path = run_dir / "data" / "historical_daily.csv"
    if csv_path.exists():
        import csv as _csv
        with open(csv_path, encoding="utf-8-sig") as f:
            rows = list(_csv.DictReader(f))
            if rows:
                close = _safe_float(rows[-1].get("close"))
    if close <= 0:
        close = _safe_float(ctx.get("snapshot", {}).get("close"))
    return {"fd": fd, "feat": feat, "close": close}


def _print_summary(results: list[dict]) -> None:
    """终端打印5张汇总表格。"""
    # ── Table 1: 综合评分排行 ──
    # 收集所有 provider 名称
    all_providers = []
    for r in results:
        for p in r.get("fd", {}).get("model_totals", {}).keys():
            if p not in all_providers:
                all_providers.append(p)

    print("\n" + "=" * 80)
    print("📊 Table 1: 综合评分排行")
    print("=" * 80)
    header = f"{'#':>2} | {'代码':>6} | {'名称':<8}"
    for p in all_providers:
        header += f" | {p:>8}"
    header += f" | {'均分':>5} | {'MA5乖离':>8} | {'决策':<12}"
    print(header)
    print("-" * len(header))

    buy_cnt = sell_cnt = hold_cnt = 0
    for i, r in enumerate(results, 1):
        fd = r.get("fd", {})
        feat = r.get("feat", {})
        score = _safe_float(fd.get("final_score"))
        mt = fd.get("model_totals", {})
        # MA5 bias
        ma_sys = feat.get("kline_indicators", {}).get("day", {}).get("ma_system", {}).get("ma5", {})
        bias = _safe_float(ma_sys.get("pct_above", 0))
        decision = _decision_label(score)
        if score >= 60:
            buy_cnt += 1
        elif score < 50:
            sell_cnt += 1
        else:
            hold_cnt += 1

        row = f"{i:>2} | {r['symbol']:>6} | {r['name']:<8}"
        for p in all_providers:
            ps = _safe_float(mt.get(p, {}).get("total") if isinstance(mt.get(p), dict) else mt.get(p))
            row += f" | {ps:>8.1f}"
        row += f" | {score:>5.1f} | {_bias_icon(bias)}{bias:>+6.1f}% | {decision}"
        print(row)

    print(f"\n  买入区: {buy_cnt}只 | 观望区: {hold_cnt}只 | 卖出区: {sell_cnt}只")

    # ── Table 2: 狙击点位 ──
    print("\n" + "=" * 80)
    print("🎯 Table 2: 狙击点位")
    print("=" * 80)
    print(f"{'#':>2} | {'代码':>6} | {'名称':<8} | {'现价':>7} | {'🟢理想买点':>10} | {'🟢次选买点':>10} | {'🔴止损':>10} | {'🟡止盈1':>10} | {'🟡止盈2':>10} | {'仓位':>5}")
    print("-" * 110)
    for i, r in enumerate(results, 1):
        fd = r.get("fd", {})
        sp = fd.get("sniper_points", {})
        price = r.get("close", 0.0)
        ideal = _safe_float(sp.get("ideal_buy"))
        sec = _safe_float(sp.get("secondary_buy"))
        sl = _safe_float(sp.get("stop_loss"))
        tp1 = _safe_float(sp.get("take_profit_1"))
        tp2 = _safe_float(sp.get("take_profit_2"))
        pa = fd.get("position_advice", {})
        if isinstance(pa, str):
            pa = {}
        pos = pa.get("position_ratio", pa.get("ratio", ""))

        def _diff(target):
            if price > 0 and target > 0:
                return f"`{(target/price-1)*100:+.1f}%`"
            return ""

        print(f"{i:>2} | {r['symbol']:>6} | {r['name']:<8} | {price:>7.2f} | {ideal:>7.2f} {_diff(ideal):>6} | {sec:>7.2f} {_diff(sec):>6} | {sl:>7.2f} {_diff(sl):>6} | {tp1:>7.2f} {_diff(tp1):>6} | {tp2:>7.2f} {_diff(tp2):>6} | {pos}")

    # ── Table 3: 均线体系 ──
    print("\n" + "=" * 80)
    print("📈 Table 3: 多周期均线支撑/阻力")
    print("=" * 80)
    print(f"{'#':>2} | {'代码':>6} | {'名称':<8} | {'现价':>7} | {'MA5':>7} | {'MA10':>7} | {'MA20':>7} | {'MA60':>7} | {'周MA5':>7} | {'周MA20':>7} | {'评估'}")
    print("-" * 105)
    for i, r in enumerate(results, 1):
        fd = r.get("fd", {})
        feat = r.get("feat", {})
        price = r.get("close", 0.0)
        day_ma = feat.get("kline_indicators", {}).get("day", {}).get("ma_system", {})
        week_ma = feat.get("kline_indicators", {}).get("week", {}).get("ma_system", {})

        def _ma(ma_dict, key):
            return _safe_float(ma_dict.get(key, {}).get("value"))

        ma5 = _ma(day_ma, "ma5")
        ma10 = _ma(day_ma, "ma10")
        ma20 = _ma(day_ma, "ma20")
        ma60 = _ma(day_ma, "ma60")
        wma5 = _ma(week_ma, "ma5")
        wma20 = _ma(week_ma, "ma20")

        bias5 = _safe_float(day_ma.get("ma5", {}).get("pct_above", 0))
        assess = "🔴超买" if bias5 > 8 else ("⚠️偏高" if bias5 > 5 else "✅正常")
        print(f"{i:>2} | {r['symbol']:>6} | {r['name']:<8} | {price:>7.2f} | {ma5:>7.2f} | {ma10:>7.2f} | {ma20:>7.2f} | {ma60:>7.2f} | {wma5:>7.2f} | {wma20:>7.2f} | {assess}")

    # ── Table 4: 操作策略 ──
    print("\n" + "=" * 80)
    print("📋 Table 4: 操作策略建议")
    print("=" * 80)
    print(f"{'#':>2} | {'代码':>6} | {'名称':<8} | {'📭 空仓建议':<30} | {'📬 持仓建议':<30}")
    print("-" * 95)
    for i, r in enumerate(results, 1):
        fd = r.get("fd", {})
        pa = fd.get("position_advice", {})
        no_pos = pa.get("no_position", "—")
        has_pos = pa.get("has_position", "—")
        if isinstance(no_pos, dict):
            no_pos = no_pos.get("summary", str(no_pos))
        if isinstance(has_pos, dict):
            has_pos = has_pos.get("summary", str(has_pos))
        # 截断过长文本
        no_pos = str(no_pos)[:28]
        has_pos = str(has_pos)[:28]
        print(f"{i:>2} | {r['symbol']:>6} | {r['name']:<8} | {no_pos:<30} | {has_pos:<30}")

    # ── Table 5: 乖离预警 ──
    alerts = []
    for r in results:
        feat = r.get("feat", {})
        ma_sys = feat.get("kline_indicators", {}).get("day", {}).get("ma_system", {})
        bias = _safe_float(ma_sys.get("ma5", {}).get("pct_above", 0))
        if abs(bias) > 8:
            alerts.append((r["symbol"], r["name"], bias, _safe_float(ma_sys.get("ma5", {}).get("value"))))
    if alerts:
        print("\n" + "=" * 80)
        print("⚠️ Table 5: 乖离预警区")
        print("=" * 80)
        print(f"{'预警':>4} | {'代码':>6} | {'名称':<8} | {'MA5乖离':>8} | {'现价 vs MA5':<20} | {'风险提示'}")
        print("-" * 80)
        for sym, name, bias, ma5v in alerts:
            level = "🔴高" if abs(bias) > 12 else "⚠️中"
            note = "短线获利回吐压力大" if bias > 0 else "超跌反弹可能"
            print(f"{level:>4} | {sym:>6} | {name:<8} | {bias:>+7.1f}% | MA5={ma5v:.2f} | {note}")
    else:
        print("\n✅ 无乖离预警（所有标的 MA5 乖离率均在 ±8% 以内）")

    print("\n" + "=" * 80)


def _load_checkpoint() -> dict:
    """加载断点续传文件。"""
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
    return {"completed": [], "results": []}


def _save_checkpoint(completed: list[str], results: list[dict]) -> None:
    """保存断点续传文件。"""
    # results 中去掉不可序列化的字段
    safe_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "df"}
        safe_results.append(sr)
    CHECKPOINT_FILE.write_text(
        json.dumps({"completed": completed, "results": safe_results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _is_completed(symbol: str, name: str, checkpoint_completed: list[str]) -> bool:
    """判断股票是否已完成（有final_decision.json或已在checkpoint中成功标记）。"""
    if symbol in checkpoint_completed:
        return True
    # 同时检查output目录中是否已有PDF（崩溃恢复兜底）
    import glob
    pattern = f"output/runs/*_{symbol}/final_decision.json"
    return len(glob.glob(pattern)) > 0


def main():
    from datetime import datetime as _dt

    checkpoint = _load_checkpoint()
    _batch.completed = checkpoint.get("completed", [])
    _batch.results = checkpoint.get("results", [])
    _batch.total = len(STOCKS)

    total = _batch.total
    skipped = 0

    _log_print(f"[Batch] 共 {total} 只, 已完成 {len(_batch.completed)} 只, 断点续传中...\n")

    for idx, (symbol, name) in enumerate(STOCKS, 1):
        if _is_completed(symbol, name, _batch.completed):
            skipped += 1
            _log_print(f"[{idx}/{total}] ⏭️ 跳过已完成: {symbol} {name}")
            # results已在checkpoint加载，只需确保未重复
            if not any(r.get("symbol") == symbol for r in _batch.results):
                run_dir_candidates = sorted(ROOT.glob(f"output/runs/*_{symbol}"))
                if run_dir_candidates:
                    data = _load_result(run_dir_candidates[-1])
                    data["symbol"] = symbol
                    data["name"] = name
                    data["run_dir"] = str(run_dir_candidates[-1])
                    _batch.results.append(data)
            continue

        t0 = _dt.now()
        _log_print()
        _log_print(f"{'=' * 60}")
        _log_print(f"[{idx}/{total}] 开始: {symbol} {name}  ({t0.strftime('%H:%M:%S')})")
        _log_print(f"{'=' * 60}")
        run_dir = build_run_dir(ROOT, symbol)
        try:
            result = run_analysis(
                root=ROOT,
                symbol=symbol,
                name=name,
                run_dir=run_dir,
                llm_provider_override=None,
                multi_eval_providers_override=PROVIDERS,
            )
            data = _load_result(run_dir)
            data["symbol"] = symbol
            data["name"] = name
            data["run_dir"] = str(run_dir)
            data["error"] = result.get("error")
            _batch.results.append(data)
            score = result.get("final_score", "?")
            decision = result.get("final_decision", "?")
            pdf = result.get("final_pdf_path", "")
            elapsed = (_dt.now() - t0).total_seconds()
            _log_print(f"✅ [{idx}/{total}] {symbol} {name}: {decision} (score={score}) [{elapsed:.0f}s]")
            if pdf:
                _log_print(f"   PDF: {pdf}")
            # 仅成功才标记已完成
            _batch.completed.append(symbol)
        except Exception as e:
            elapsed = (_dt.now() - t0).total_seconds()
            _log_print(f"❌ [{idx}/{total}] {symbol} {name}: 失败 [{elapsed:.0f}s] — {e}")
            _batch.results.append({"symbol": symbol, "name": name, "fd": {}, "feat": {}, "error": str(e)})
            # 失败不标记，后续可重试

        # 每只完成后保存checkpoint（崩溃恢复兜底）
        _save_checkpoint(_batch.completed, _batch.results)
        _log_print(f"[Checkpoint] ({len(_batch.completed)}/{total})")

    # 汇总表格（包含所有已完成的，包括重启后加载的）
    _print_summary(_batch.results)

    # 全部完成后清除checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print(f"[Batch] 全部完成，checkpoint已清除。")


if __name__ == "__main__":
    main()
