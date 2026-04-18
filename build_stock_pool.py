# -*- coding: utf-8 -*-
"""构建主流ETF/指数成分股池,输出到 backtest_stock_pool.txt。

股票池来源:
  - 宽基指数: 沪深300/中证500/中证1000/创业板50/科创50/上证50
  - 行业ETF (见 _INDUSTRY_ETF_MAP) 前若干重仓股
  - 概念ETF (见 _CONCEPT_ETF_MAP) 前若干重仓股

用法:
    python build_stock_pool.py
    python build_stock_pool.py --include-concept  # 含概念ETF
"""
import sys, os, time, argparse
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stockagent_analysis.market_context import _no_proxy_call
from stockagent_analysis.data_backend import _INDUSTRY_ETF_MAP, _ensure_akshare_no_proxy

_ensure_akshare_no_proxy()


# 主流宽基指数
BROAD_INDICES = {
    "000300": "沪深300",
    "000905": "中证500",
    "000852": "中证1000",
    "000016": "上证50",
    "399006": "创业板指",
    "000688": "科创50",
}


def fetch_index_cons(idx_code: str) -> list[str]:
    """获取指数成分股。"""
    try:
        import akshare as ak
        time.sleep(0.3)
        # 优先用中证指数公司接口(更全更稳)
        try:
            df = _no_proxy_call(ak.index_stock_cons_csindex, symbol=idx_code)
            if df is not None and not df.empty:
                code_col = "成分券代码" if "成分券代码" in df.columns else df.columns[0]
                codes = [str(c).zfill(6) for c in df[code_col].tolist()]
                return codes
        except Exception:
            pass
        # 备用: 东财接口
        df = _no_proxy_call(ak.index_stock_cons, symbol=idx_code)
        if df is not None and not df.empty:
            code_col = "品种代码" if "品种代码" in df.columns else df.columns[0]
            codes = [str(c).zfill(6) for c in df[code_col].tolist()]
            return codes
    except Exception as e:
        print(f"  [失败] {idx_code}: {e}", flush=True)
    return []


def fetch_etf_cons(etf_code: str, top_n: int = 15) -> list[str]:
    """获取ETF前N重仓股。"""
    try:
        import akshare as ak
        time.sleep(0.3)
        df = _no_proxy_call(ak.fund_portfolio_hold_em, symbol=etf_code, date="2025")
        if df is not None and not df.empty:
            code_col = "股票代码" if "股票代码" in df.columns else None
            if code_col:
                codes = [str(c).zfill(6) for c in df[code_col].head(top_n).tolist()]
                return [c for c in codes if c.isdigit() and len(c) == 6]
    except Exception as e:
        pass
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-concept", action="store_true", help="含概念ETF重仓股")
    ap.add_argument("--etf-topn", type=int, default=15, help="每个ETF取前N重仓")
    ap.add_argument("--output", type=str, default="backtest_stock_pool.txt")
    args = ap.parse_args()

    pool: set[str] = set()
    source_stats: dict[str, int] = {}

    # 1. 宽基指数成分股
    print("=" * 60)
    print("Step 1: 拉取宽基指数成分股")
    print("=" * 60)
    for code, name in BROAD_INDICES.items():
        codes = fetch_index_cons(code)
        before = len(pool)
        pool.update(codes)
        added = len(pool) - before
        source_stats[f"idx:{name}"] = len(codes)
        print(f"  {name} ({code}): 成分股={len(codes)} 新增={added} 累计={len(pool)}", flush=True)

    # 2. 行业ETF重仓股
    print()
    print("=" * 60)
    print("Step 2: 拉取行业ETF重仓股")
    print("=" * 60)
    industry_total = 0
    for industry, etf_code in _INDUSTRY_ETF_MAP.items():
        codes = fetch_etf_cons(etf_code, top_n=args.etf_topn)
        before = len(pool)
        pool.update(codes)
        added = len(pool) - before
        industry_total += len(codes)
        if added > 0:
            print(f"  {industry} ETF{etf_code}: 重仓={len(codes)} 新增={added}", flush=True)
    source_stats["industry_etf"] = industry_total

    # 3. 概念ETF重仓股(可选)
    if args.include_concept:
        from stockagent_analysis.market_context import _CONCEPT_ETF_MAP
        print()
        print("=" * 60)
        print("Step 3: 拉取概念ETF重仓股")
        print("=" * 60)
        concept_total = 0
        seen_etfs = set()
        for concept, etfs in _CONCEPT_ETF_MAP.items():
            for etf_code in etfs:
                if etf_code in seen_etfs:
                    continue
                seen_etfs.add(etf_code)
                codes = fetch_etf_cons(etf_code, top_n=args.etf_topn)
                before = len(pool)
                pool.update(codes)
                added = len(pool) - before
                concept_total += len(codes)
                if added > 0:
                    print(f"  {concept} ETF{etf_code}: 重仓={len(codes)} 新增={added}", flush=True)
        source_stats["concept_etf"] = concept_total

    # 4. 过滤:只保留主板/创业板/科创板(排除北交所8/4/9开头的权证/优先股)
    valid_prefixes = ("00", "30", "60", "68")  # 深主板/创业/沪主板/科创
    filtered = sorted([c for c in pool if c.startswith(valid_prefixes) and len(c) == 6])

    print()
    print("=" * 60)
    print(f"最终股票池: {len(filtered)} 只 (原始 {len(pool)} 只, 过滤掉 {len(pool)-len(filtered)} 只)")
    print("=" * 60)
    for k, v in source_stats.items():
        print(f"  {k}: {v}")

    # 5. 写出文件
    out_path = Path(args.output)
    out_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    print(f"\n已写入: {out_path.resolve()}")
    print(f"下一步: python backtest_composite_v7.py --reset")


if __name__ == "__main__":
    main()
