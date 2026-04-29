"""ETF 跟踪机制 — 给定股票代码, 查询持有它的 winner ETF 及其表现。

数据组织 (output/etf_analysis/):
  etf_index.json          ← ETF 元数据 + 主题标签 + 表现快照 (此模块产出)
  nav/{ts_code}.json      ← 净值时序 (周期性更新)
  holdings/{ts_code}.json ← 持仓时序 (季度更新)
  stock_to_etfs.json      ← 反向索引: 股票 → 持有它的 ETF 列表 (此模块产出)

核心查询:
  - track_stock(ts_code): 返回这只股票被多少 ETF 持有, 这些 ETF 当前表现
  - top_etfs_by_theme(theme, n): 取某主题表现 top n
  - rebuild_index(): 重建 etf_index.json 和 stock_to_etfs.json (定期跑)
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
from collections import defaultdict


ETF_DIR = Path(__file__).resolve().parent.parent.parent / "output" / "etf_analysis"
INDEX_FILE = ETF_DIR / "etf_index.json"
STOCK2ETF_FILE = ETF_DIR / "stock_to_etfs.json"


# ─── 主题分类 (基于 ETF 名称关键词) ───
THEME_RULES = [
    ("黄金", ["黄金", "金ETF", "上海金"]),
    ("白银", ["白银", "银ETF"]),
    ("通信5G", ["通信", "5G"]),
    ("半导体", ["半导体", "芯片", "中证芯片"]),
    ("人工智能", ["人工智能", "AI", "算力"]),
    ("云计算", ["云计算", "云"]),
    ("光伏新能源", ["光伏", "新能源", "电池"]),
    ("能源煤炭", ["能源", "煤炭", "石油"]),
    ("医药生物", ["医药", "生物医药", "医疗", "创新药"]),
    ("白酒消费", ["白酒", "食品饮料", "消费"]),
    ("银行金融", ["银行", "证券", "保险", "金融"]),
    ("房地产", ["地产", "房"]),
    ("军工", ["军工", "国防"]),
    ("机器人", ["机器人"]),
    ("中证A500", ["A500"]),
    ("中证500", ["中证500"]),
    ("沪深300", ["沪深300", "300"]),
    ("中证1000", ["中证1000", "1000"]),
    ("科创板", ["科创"]),
    ("创业板", ["创业板", "创业"]),
    ("港股", ["港股", "恒生", "H股"]),
    ("美股海外", ["纳斯达克", "标普", "美国", "海外"]),
    ("REITs", ["REITs", "REIT"]),
    ("债券", ["债券", "债", "利率"]),
]


def classify_theme(name: str) -> str:
    """根据 ETF 名称匹配主题。"""
    if not name: return "其他"
    for theme, keys in THEME_RULES:
        for k in keys:
            if k in name:
                return theme
    return "其他"


def to_ts_code(sym):
    if not sym: return None
    s = str(sym).strip()
    if "." in s: return s
    if len(s) == 6:
        if s.startswith("6"): return f"{s}.SH"
        if s.startswith("0") or s.startswith("3"): return f"{s}.SZ"
        if s.startswith("8") or s.startswith("4") or s.startswith("9"): return f"{s}.BJ"
    return None


def rebuild_index():
    """重建 etf_index.json (元数据 + 主题 + 净值快照) 和 stock_to_etfs.json (反向索引)。"""
    nav_dir = ETF_DIR / "nav"
    hold_dir = ETF_DIR / "holdings"
    list_file = ETF_DIR / "etf_list.json"

    if not list_file.exists():
        raise FileNotFoundError(f"先跑 fetch_etf_data.py 生成 {list_file}")

    etf_list = json.loads(list_file.read_text(encoding="utf-8"))
    print(f"[rebuild_index] 处理 {len(etf_list)} 只 ETF...")

    index = []
    stock_to_etfs = defaultdict(list)

    for etf in etf_list:
        code = etf["ts_code"]
        nav_file = nav_dir / f"{code}.json"
        if not nav_file.exists(): continue

        try:
            navs = json.loads(nav_file.read_text(encoding="utf-8"))
        except: navs = []
        valid = [r for r in navs if (r.get("adj_nav") or r.get("unit_nav"))]
        if len(valid) < 50: continue

        # 表现快照
        first_v = valid[0].get("adj_nav") or valid[0].get("unit_nav")
        last_v = valid[-1].get("adj_nav") or valid[-1].get("unit_nav")
        first_d = valid[0]["nav_date"]; last_d = valid[-1]["nav_date"]
        d1 = (int(last_d[:4]) - int(first_d[:4])) + (int(last_d[4:6]) - int(first_d[4:6]))/12
        annual = ((last_v / first_v) ** (1.0 / d1) - 1) if d1 >= 0.5 else None

        # 多区间收益
        def ret_between(d1, d2):
            sv = ev = None
            for r in valid:
                nv = r.get("adj_nav") or r.get("unit_nav")
                if r["nav_date"] >= d1 and sv is None: sv = nv
                if r["nav_date"] <= d2: ev = nv
            return (ev/sv - 1) if (sv and ev and sv > 0) else None

        ret_3y = ret_between("20230101", "20260331")
        ret_924 = ret_between("20240924", "20260331")
        ret_q1 = ret_between("20260101", "20260331")
        ret_30d = ret_between(valid[max(0, len(valid)-30)]["nav_date"], last_d) if len(valid) >= 30 else None

        # 持仓
        hold_file = hold_dir / f"{code}.json"
        latest_holdings = []
        all_held_stocks = set()
        if hold_file.exists():
            try:
                holdings = json.loads(hold_file.read_text(encoding="utf-8"))
                if holdings:
                    latest_date = max(r["end_date"] for r in holdings)
                    latest = sorted([r for r in holdings if r["end_date"] == latest_date],
                                     key=lambda r: -(r.get("stk_mkv_ratio") or 0))[:10]
                    latest_holdings = [{
                        "symbol": r["symbol"],
                        "ts_code": to_ts_code(r["symbol"]),
                        "ratio": r.get("stk_mkv_ratio"),
                    } for r in latest]
                    all_held_stocks = {to_ts_code(r["symbol"]) for r in holdings if r.get("symbol")}
                    all_held_stocks.discard(None)
            except: pass

        theme = classify_theme(etf.get("name", ""))
        info = {
            "ts_code": code,
            "name": etf.get("name", ""),
            "theme": theme,
            "management": etf.get("management"),
            "list_date": etf.get("list_date"),
            "nav_first_date": first_d,
            "nav_last_date": last_d,
            "annual_return_pct": round(annual*100, 2) if annual else None,
            "ret_3y_pct": round(ret_3y*100, 2) if ret_3y else None,
            "ret_924_pct": round(ret_924*100, 2) if ret_924 else None,
            "ret_q1_2026_pct": round(ret_q1*100, 2) if ret_q1 else None,
            "ret_30d_pct": round(ret_30d*100, 2) if ret_30d else None,
            "latest_top10_holdings": latest_holdings,
            "all_held_stocks_count": len(all_held_stocks),
        }
        index.append(info)

        # 反向索引
        for ts in all_held_stocks:
            stock_to_etfs[ts].append({
                "etf": code, "name": etf.get("name", ""),
                "theme": theme,
                "annual_return_pct": info["annual_return_pct"],
                "ret_924_pct": info["ret_924_pct"],
            })

    # 保存
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    STOCK2ETF_FILE.write_text(json.dumps(dict(stock_to_etfs), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[rebuild_index] etf_index: {len(index)}, stock_to_etfs: {len(stock_to_etfs)} 只股票被持有")
    return index, stock_to_etfs


def load_index():
    if not INDEX_FILE.exists():
        rebuild_index()
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def load_stock_to_etfs():
    if not STOCK2ETF_FILE.exists():
        rebuild_index()
    return json.loads(STOCK2ETF_FILE.read_text(encoding="utf-8"))


# ─── 核心查询接口 ───

def track_stock(ts_code: str, min_etf_annual: float = 20.0) -> dict:
    """给定股票代码, 返回持有它的 ETF 信息 + 这些 ETF 的整体表现概况。

    Args:
        ts_code: 股票代码 (如 '600519.SH' 或 '600519')
        min_etf_annual: 只统计年化收益 ≥ X% 的 ETF (过滤垃圾 ETF)

    Returns:
        {
            "stock": ts_code,
            "etf_count": int,           # 总共有多少只 ETF 持有
            "winner_etf_count": int,    # 年化≥min_etf_annual 的 winner ETF 数
            "themes": {主题: 个数},      # 哪些主题的 ETF 持有
            "etfs": [...],              # 详细列表
            "avg_etf_annual": float,    # 持有此股的 ETF 平均年化
            "interpretation": str,      # 主题解读
        }
    """
    if "." not in ts_code and len(ts_code) == 6:
        ts_code = to_ts_code(ts_code)
    s2e = load_stock_to_etfs()
    etfs = s2e.get(ts_code, [])

    if not etfs:
        return {"stock": ts_code, "etf_count": 0, "interpretation": "未被任何 winner ETF 持有"}

    winners = [e for e in etfs if (e.get("annual_return_pct") or 0) >= min_etf_annual]

    # 主题统计
    themes = defaultdict(int)
    for e in winners:
        themes[e.get("theme", "其他")] += 1
    top_themes = sorted(themes.items(), key=lambda x: -x[1])

    # 平均年化
    valids = [e["annual_return_pct"] for e in winners if e.get("annual_return_pct") is not None]
    avg_annual = sum(valids) / len(valids) if valids else None

    # 解读
    interp_parts = []
    if top_themes:
        interp_parts.append(f"主要被「{top_themes[0][0]}」主题 ({top_themes[0][1]} 只 ETF) 持有")
    if avg_annual is not None:
        interp_parts.append(f"持有它的 winner ETF 平均年化 {avg_annual:+.1f}%")
    if len(winners) >= 5:
        interp_parts.append(f"被 {len(winners)} 只年化≥{min_etf_annual:.0f}% 的 ETF 持有, 信号较强")
    elif len(winners) <= 1:
        interp_parts.append("持仓集中度低, 不是核心 ETF 标的")

    return {
        "stock": ts_code,
        "etf_count": len(etfs),
        "winner_etf_count": len(winners),
        "themes": dict(top_themes),
        "etfs": sorted(winners, key=lambda x: -(x.get("annual_return_pct") or 0)),
        "avg_etf_annual": round(avg_annual, 2) if avg_annual else None,
        "interpretation": " | ".join(interp_parts),
    }


def top_etfs_by_theme(theme: str = None, n: int = 10,
                      sort_by: str = "annual_return_pct") -> list:
    """取某主题表现最好的 ETF。

    Args:
        theme: 主题名 (如 '通信5G', '黄金', None=全部)
        n: 取前 n 只
        sort_by: 排序字段, 'annual_return_pct' / 'ret_924_pct' / 'ret_30d_pct'
    """
    idx = load_index()
    if theme:
        idx = [e for e in idx if e.get("theme") == theme]
    idx = [e for e in idx if e.get(sort_by) is not None]
    idx.sort(key=lambda e: -e[sort_by])
    return idx[:n]


def list_themes() -> list:
    """列出所有主题及其 ETF 数量。"""
    idx = load_index()
    themes = defaultdict(list)
    for e in idx:
        themes[e.get("theme", "其他")].append(e)
    out = []
    for t, etfs in themes.items():
        valid = [e["annual_return_pct"] for e in etfs if e.get("annual_return_pct") is not None]
        out.append({
            "theme": t,
            "count": len(etfs),
            "avg_annual": round(sum(valid)/len(valid), 2) if valid else None,
            "best_annual": round(max(valid), 2) if valid else None,
        })
    out.sort(key=lambda x: -(x["avg_annual"] or 0))
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        rebuild_index()
    elif len(sys.argv) > 1 and sys.argv[1] == "themes":
        for t in list_themes():
            print(f"  {t['theme']:15s}  ETF {t['count']:3d}  平均年化 {t['avg_annual'] or '-':>6}%  最强 {t['best_annual'] or '-':>6}%")
    elif len(sys.argv) > 2 and sys.argv[1] == "track":
        result = track_stock(sys.argv[2])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif len(sys.argv) > 2 and sys.argv[1] == "top":
        theme = sys.argv[2] if len(sys.argv) > 2 else None
        for e in top_etfs_by_theme(theme, n=10):
            print(f"  {e['ts_code']:12s} {e['name'][:30]:30s} 年化 {e.get('annual_return_pct', '-'):>+6}% | 924 {e.get('ret_924_pct', '-'):>+6}%")
    else:
        print("用法:")
        print("  python -m stockagent_analysis.etf_tracker rebuild       # 重建索引")
        print("  python -m stockagent_analysis.etf_tracker themes        # 列出所有主题")
        print("  python -m stockagent_analysis.etf_tracker track 600519.SH  # 查股票被哪些 ETF 持有")
        print("  python -m stockagent_analysis.etf_tracker top 通信5G       # 该主题 Top 10 ETF")
