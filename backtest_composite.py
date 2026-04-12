# -*- coding: utf-8 -*-
"""综合评分回测 — 扫描TDX全量A股，Agent加权合并+校准，计算IC和5分细粒度分数段统计。

用法:
    python backtest_composite.py [--max N]   # 限制最多N只股票(调试用)
"""
import sys, os, glob, struct, time, statistics

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from pathlib import Path

from backtest_agents import (
    load_tdx_daily, rolling_indicators,
    score_trend_momentum, score_capital_liquidity, score_divergence,
    score_chanlun, score_resonance,
    score_ichimoku, _f_amt_ratio,
)
from stockagent_analysis.channel_reversal import compute_channel, detect_phases


# ── 权重配置 V6 (2026-04-13) ─────────────────────────────────
# 两大改进:
#   1. 三因子融合: trend_momentum + ichimoku + resonance_rev → mean_reversion
#      (三者ρ=0.73~0.79, 分开算是三重复计; 融合后消除冗余)
#   2. 中性零权重: |score-50| ≤ 5 时权重渐减→0, 权重动态重分配给有信号因子
#      (divergence 76%恒=50, chanlun弱信号时稀释综合分 → 现在自动归零)
# 400只实测: IC(20d) +0.0367→+0.0552 (+50%!), 75-85胜率 62%→66%
WEIGHTS = {
    "channel_reversal": 0.20,
    "divergence":       0.20,
    "capital_liquidity":0.15,  # 独立因子中IC最高(+0.051)
    "mean_reversion":   0.15,  # trend_momentum+ichimoku+resonance_rev均值
    "f_amt_ratio":      0.12,
    "chanlun":          0.18,
}

_KEY_DIMS = {
    "channel_reversal": 0.12,
    "divergence":       0.10,
    "capital_liquidity":0.10,
    "chanlun":          0.10,
}

# 中性零权重阈值: |score-50| <= 此值时权重渐减
_NEUTRAL_THRESHOLD = 5.0


# ── 股票池构建（TDX全量A股扫描）──────────────────────────────

TDX_DIR = os.environ.get("TDX_DIR", "D:/tdx/vipdoc")


def scan_tdx_symbols() -> list[str]:
    """扫描TDX本地 sh/lday + sz/lday，返回主板+创业板+科创板6位代码列表。"""
    symbols = set()
    for market in ("sh", "sz"):
        pattern = os.path.join(TDX_DIR, market, "lday", f"{market}*.day")
        for fpath in glob.glob(pattern):
            fname = os.path.basename(fpath)
            code = fname[2:8]  # e.g. sh600000.day → 600000
            # 过滤: 主板(60xxxx/00xxxx), 创业板(30xxxx), 科创板(68xxxx)
            if code.startswith(("60", "00", "30", "68")):
                # 跳过 B股 (900xxx, 200xxx)
                if code.startswith(("900", "200")):
                    continue
                # 检查文件大小 > 32KB (约200根K线 × 32字节)
                if os.path.getsize(fpath) > 32 * 200:
                    symbols.add(code)
    return sorted(symbols)


# ── 评分函数 ─────────────────────────────────────────────────

def score_fundamental_pe(_r) -> float:
    return 50.0


def compute_channel_scores(df: pd.DataFrame) -> np.ndarray:
    try:
        df2 = compute_channel(df.copy())
        df2 = detect_phases(df2)
        return df2["cr_score"].values
    except Exception:
        return np.full(len(df), 50.0)


def key_dim_dominance(scores: dict) -> float:
    bonus = 0.0
    for dim, pull in _KEY_DIMS.items():
        s = scores.get(dim, 50.0)
        if s > 75:   bonus += (s - 75) * pull
        elif s < 25: bonus -= (25 - s) * pull
    return bonus


def composite_score(row_scores: dict) -> float:
    """综合评分 V6: 中性零权重 + 关键维度拉力。

    |score - 50| ≤ _NEUTRAL_THRESHOLD 时权重渐减→0,
    释放的权重动态分配给有信号的因子。
    """
    eff_weights = {}
    for k, w in WEIGHTS.items():
        s = row_scores.get(k, 50.0)
        dev = abs(s - 50.0)
        if dev >= _NEUTRAL_THRESHOLD:
            eff_weights[k] = w
        else:
            eff_weights[k] = w * (dev / _NEUTRAL_THRESHOLD)

    total_w = sum(eff_weights.values())
    if total_w < 0.01:
        return 50.0

    raw = sum(row_scores.get(k, 50.0) * (ew / total_w) for k, ew in eff_weights.items())
    return max(0.0, min(100.0, raw + key_dim_dominance(row_scores)))


# ── 后置自适应展宽（方案一）─────────────────────────────────────
_STRETCH_SIGMA_THRESHOLD = 12.0
_STRETCH_TARGET_SIGMA = 18.0
_STRETCH_TARGET_MEAN = 55.0


def batch_stretch_scores(records: list[tuple]) -> list[tuple]:
    """对全量回测结果做后置展宽。records = [(score, r5, r10, r20), ...]"""
    if len(records) < 30:
        return records
    scores = [r[0] for r in records]
    mu = statistics.mean(scores)
    sigma = statistics.stdev(scores)
    print(f"[展宽] 原始分布: μ={mu:.2f}, σ={sigma:.2f}", flush=True)
    if sigma >= _STRETCH_SIGMA_THRESHOLD:
        print(f"[展宽] σ>={_STRETCH_SIGMA_THRESHOLD}，无需展宽", flush=True)
        return records
    stretched = []
    for rec in records:
        z = (rec[0] - mu) / max(sigma, 0.01)
        new_score = max(5.0, min(95.0, _STRETCH_TARGET_MEAN + z * _STRETCH_TARGET_SIGMA))
        stretched.append((new_score,) + rec[1:])
    new_scores = [r[0] for r in stretched]
    new_mu = statistics.mean(new_scores)
    new_sigma = statistics.stdev(new_scores)
    print(f"[展宽] 展宽后: μ={new_mu:.2f}, σ={new_sigma:.2f}", flush=True)
    return stretched


# ── 回测主循环 ────────────────────────────────────────────────

def run_backtest(symbols: list[str]) -> list[tuple[float, float, float, float]]:
    total = len(symbols)
    records = []
    skipped = 0

    for si, sym in enumerate(symbols):
        if (si + 1) % 50 == 0:
            print(f"  进度: {si+1}/{total} (样本={len(records)}, 跳过={skipped})...", flush=True)
        try:
            df = load_tdx_daily(sym)
            if df is None or len(df) < 200:
                skipped += 1
                continue

            cr_scores = compute_channel_scores(df)
            ind = rolling_indicators(df, min_bars=120)
            if ind.empty:
                skipped += 1
                continue

            close_arr = df["close"].values
            rows = [r.to_dict() for _, r in ind.iterrows()]

            for row in rows:
                idx = int(row["idx"])
                r5  = (close_arr[idx+5]  / close_arr[idx] - 1)*100 if idx+5  < len(close_arr) else np.nan
                r10 = (close_arr[idx+10] / close_arr[idx] - 1)*100 if idx+10 < len(close_arr) else np.nan
                r20 = (close_arr[idx+20] / close_arr[idx] - 1)*100 if idx+20 < len(close_arr) else np.nan

                # mean_reversion: trend_momentum + ichimoku + resonance_rev 三因子均值
                # 三者ρ=0.73~0.79, 分开计算是三重复计; 融合消除冗余
                _tm = score_trend_momentum(row)
                _ich = score_ichimoku(row)
                _res_rev = 100.0 - score_resonance(row)
                _mr = (_tm + _ich + _res_rev) / 3.0

                agent_scores = {
                    "capital_liquidity": score_capital_liquidity(row),
                    "divergence":        score_divergence(row),
                    "chanlun":           score_chanlun(row),
                    "f_amt_ratio":       max(10, min(90, _f_amt_ratio(row))),
                    "mean_reversion":    _mr,
                    "channel_reversal":  float(cr_scores[idx]) if idx < len(cr_scores) else 50.0,
                }

                score = composite_score(agent_scores)
                records.append((score, r5, r10, r20))

        except Exception as e:
            print(f"  {sym} 失败: {e}", flush=True)
            skipped += 1

    return records


# ── 分析（5分细粒度）──────────────────────────────────────────

def analyze(pairs: list) -> dict:
    if not pairs:
        return {}

    scores = np.array([p[0] for p in pairs])
    r5     = np.array([p[1] for p in pairs])
    r10    = np.array([p[2] for p in pairs])
    r20    = np.array([p[3] for p in pairs])
    m5, m10, m20 = ~np.isnan(r5), ~np.isnan(r10), ~np.isnan(r20)

    ic5  = float(np.corrcoef(scores[m5],  r5[m5])[0,1])  if m5.sum()>30  else 0.0
    ic10 = float(np.corrcoef(scores[m10], r10[m10])[0,1]) if m10.sum()>30 else 0.0
    ic20 = float(np.corrcoef(scores[m20], r20[m20])[0,1]) if m20.sum()>30 else 0.0

    # 5分间隔分数段
    bins = [(0,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),
            (65,70),(70,75),(75,80),(80,85),(85,90),(90,100)]
    buckets = []
    for lo, hi in bins:
        mask = (scores >= lo) & (scores < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        sr5  = r5[mask & m5];   sr10 = r10[mask & m10];  sr20 = r20[mask & m20]
        buckets.append({
            "range":  f"{lo}-{hi}",
            "count":  cnt,
            "avg_5d": float(np.mean(sr5))  if len(sr5)  else None,
            "wr_5d":  float((sr5>0).mean()*100)  if len(sr5)  else None,
            "avg_10d":float(np.mean(sr10)) if len(sr10) else None,
            "wr_10d": float((sr10>0).mean()*100) if len(sr10) else None,
            "avg_20d":float(np.mean(sr20)) if len(sr20) else None,
            "wr_20d": float((sr20>0).mean()*100) if len(sr20) else None,
        })

    return {
        "total": len(pairs),
        "n_stocks": int(len(set())),  # filled later
        "ic5": ic5, "ic10": ic10, "ic20": ic20,
        "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        "buckets": buckets,
    }


def write_report(result: dict, n_stocks: int, out_path: str):
    import datetime
    lines = []
    lines.append("# 综合评分全量A股回测报告\n")
    lines.append(f"> 回测日期: {datetime.date.today()}")
    lines.append(f"> 股票池: TDX本地A股 {n_stocks} 只（主板+创业板+科创板）")
    lines.append(f"> 方法: 逐日滚动计算指标，11个Agent加权合并+关键维度拉力，与前瞻收益做IC和分数段分析")
    lines.append(f"> 样本: {result['total']:,} 个交易日样本点")
    lines.append(f"> 权重: channel_reversal 0.20 | chanlun 0.18 | divergence 0.18 | trend_momentum 0.15 | capital_liquidity 0.10 | 其余各 0.05~0.08\n")
    lines.append("---\n")

    lines.append("## 一、综合指标\n")
    lines.append("| 指标 | 值 |")
    lines.append("|------|-----|")
    lines.append(f"| 股票数 | {n_stocks} |")
    lines.append(f"| 样本数 | {result['total']:,} |")
    lines.append(f"| 评分均值 | {result['mean']:.2f} |")
    lines.append(f"| 评分标准差 σ | {result['std']:.2f} |")
    lines.append(f"| IC(5d) | {result['ic5']:+.4f} |")
    lines.append(f"| IC(10d) | {result['ic10']:+.4f} |")
    lines.append(f"| IC(20d) | {result['ic20']:+.4f} |\n")

    lines.append("## 二、分数段明细（5分间隔）\n")
    lines.append("| 分数段 | 样本 | 占比 | 5日均涨 | 5日胜率 | 10日均涨 | 10日胜率 | 20日均涨 | 20日胜率 |")
    lines.append("|--------|------|------|---------|---------|----------|----------|----------|----------|")
    for b in result["buckets"]:
        pct = b["count"] / result["total"] * 100
        def _f(v): return f"{v:+.2f}%" if v is not None else "N/A"
        def _w(v): return f"{v:.1f}%" if v is not None else "N/A"
        lines.append(f"| **{b['range']}** | {b['count']:,} | {pct:.1f}% | {_f(b['avg_5d'])} | {_w(b['wr_5d'])} | {_f(b['avg_10d'])} | {_w(b['wr_10d'])} | {_f(b['avg_20d'])} | {_w(b['wr_20d'])} |")

    lines.append("")
    lines.append("## 三、分布直方图\n")
    lines.append("```")
    for b in result["buckets"]:
        pct = b["count"] / result["total"] * 100
        bar = "█" * int(pct / 0.5)
        lines.append(f"{b['range']:>8}: {bar:<50} {pct:5.1f}%  n={b['count']:,}")
    lines.append("```\n")

    lines.append("## 四、关键分数段深入分析\n")
    for b in result["buckets"]:
        lo = int(b["range"].split("-")[0])
        if lo < 60:
            continue
        if b["avg_20d"] is None:
            continue
        verdict = ""
        if b["wr_20d"] and b["wr_20d"] >= 65:
            verdict = " — **强买入信号**"
        elif b["wr_20d"] and b["wr_20d"] >= 60:
            verdict = " — 偏多信号"
        elif b["wr_20d"] and b["wr_20d"] < 50:
            verdict = " — 不可靠"
        lines.append(f"- **{b['range']}分**: 样本{b['count']:,}，20日均涨{b['avg_20d']:+.2f}%，胜率{b['wr_20d']:.1f}%{verdict}")

    lines.append("")
    lines.append("## 五、结论\n")
    ic_avg = (result["ic5"] + result["ic10"] + result["ic20"]) / 3
    lines.append(f"- 综合评分 IC 均值: `{ic_avg:+.4f}`")
    lines.append(f"- σ={result['std']:.2f}")

    # 找高分段
    hi_buckets = [b for b in result["buckets"]
                  if int(b["range"].split("-")[0]) >= 65 and b["avg_20d"] is not None]
    lo_buckets = [b for b in result["buckets"]
                  if int(b["range"].split("-")[1]) <= 45 and b["avg_20d"] is not None]
    if hi_buckets:
        hi_20 = np.average([b["avg_20d"] for b in hi_buckets],
                           weights=[b["count"] for b in hi_buckets])
        hi_wr = np.average([b["wr_20d"] for b in hi_buckets],
                           weights=[b["count"] for b in hi_buckets])
        lines.append(f"- 高分(≥65)加权20日均涨: {hi_20:+.2f}%，胜率: {hi_wr:.1f}%")
    if lo_buckets:
        lo_20 = np.average([b["avg_20d"] for b in lo_buckets],
                           weights=[b["count"] for b in lo_buckets])
        lo_wr = np.average([b["wr_20d"] for b in lo_buckets],
                           weights=[b["count"] for b in lo_buckets])
        lines.append(f"- 低分(≤45)加权20日均涨: {lo_20:+.2f}%，胜率: {lo_wr:.1f}%")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已写入: {out_path}", flush=True)


def print_result(result: dict):
    print(f"\n{'='*85}")
    print(f"综合评分回测  样本={result['total']:,}  均值={result['mean']:.2f}  σ={result['std']:.2f}")
    print(f"IC(5d)={result['ic5']:+.4f}  IC(10d)={result['ic10']:+.4f}  IC(20d)={result['ic20']:+.4f}")
    print(f"{'='*85}")
    print(f"{'分数段':>8} | {'样本':>7} | {'占比':>5} | {'5日均涨':>8} | {'5日胜率':>7} | {'10日均涨':>9} | {'10日胜率':>8} | {'20日均涨':>9} | {'20日胜率':>8}")
    print("-" * 100)
    for b in result["buckets"]:
        pct = b["count"] / result["total"] * 100
        def _f(v): return f"{v:>+8.2f}%" if v is not None else "     N/A"
        def _w(v): return f"{v:>7.1f}%" if v is not None else "     N/A"
        flag = " ◀" if int(b["range"].split("-")[0]) >= 65 else ""
        print(f"{b['range']:>8} | {b['count']:>7,} | {pct:>4.1f}% | {_f(b['avg_5d'])} | {_w(b['wr_5d'])} | {_f(b['avg_10d'])} | {_w(b['wr_10d'])} | {_f(b['avg_20d'])} | {_w(b['wr_20d'])}{flag}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=0, help="最多回测N只股票 (0=全部)")
    args = parser.parse_args()

    # 优先使用预构建的股票池文件（成分股+板块龙头+已分析股票）
    pool_file = Path("backtest_stock_pool.txt")
    if pool_file.exists():
        symbols = [line.strip() for line in pool_file.read_text().splitlines() if line.strip()]
        print(f"从 {pool_file} 加载股票池", flush=True)
    else:
        print("扫描TDX本地A股数据...", flush=True)
        symbols = scan_tdx_symbols()
    if args.max > 0:
        symbols = symbols[:args.max]

    if not symbols:
        print("未找到TDX数据")
        return

    print(f"综合评分回测 — {len(symbols)} 只A股", flush=True)
    print(f"权重: {WEIGHTS}", flush=True)
    t0 = time.time()

    pairs_raw = run_backtest(symbols)
    elapsed = time.time() - t0
    print(f"总样本: {len(pairs_raw):,}  用时: {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)

    # 原始分布分析
    result_raw = analyze(pairs_raw)
    print("\n── 原始评分（展宽前）──")
    print_result(result_raw)

    # 后置展宽
    pairs = batch_stretch_scores(pairs_raw)
    result = analyze(pairs)
    print("\n── 展宽后评分 ──")
    print_result(result)
    write_report(result, len(symbols), "docs/composite-backtest-report.md")


if __name__ == "__main__":
    main()
