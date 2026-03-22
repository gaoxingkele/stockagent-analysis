# -*- coding: utf-8 -*-
"""汇总25只股票分析结果，输出为PNG图片。"""
import json
import csv
import sys
from pathlib import Path
from datetime import datetime

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

# ── 25只股票 (7 + 18) ──
STOCKS_7 = [
    ("000600", "建投能源"), ("002797", "第一创业"), ("000155", "川能动力"),
    ("002015", "协鑫能科"), ("000537", "绿发电力"), ("002246", "北化股份"),
    ("000422", "湖北宜化"),
]
STOCKS_18 = [
    ("000591", "太阳能"), ("000731", "四川美丰"), ("000007", "全新好"),
    ("001309", "德明利"), ("000601", "韶能股份"), ("000090", "天健集团"),
    ("000912", "泸天化"), ("002470", "金正大"), ("002449", "国星光电"),
    ("001369", "双欣材料"), ("000498", "山东路桥"), ("000949", "新乡化纤"),
    ("000623", "吉林敖东"), ("002859", "洁美科技"), ("000690", "宝新能源"),
    ("000677", "恒天海龙"), ("000534", "万泽股份"), ("000695", "滨海能源"),
]


def _safe_float(v):
    try:
        return float(v) if v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _fetch_realtime_prices(symbols: list[str]) -> dict[str, float]:
    """批量获取腾讯实时行情价格。"""
    import urllib.request
    prices = {}
    # 每次最多查20个
    for i in range(0, len(symbols), 20):
        batch = symbols[i:i+20]
        codes = ",".join(f"sz{s}" if s.startswith(("0", "3")) else f"sh{s}" for s in batch)
        url = f"http://web.sqt.gtimg.cn/q={codes}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode("gbk", errors="replace")
            for line in text.strip().split("\n"):
                if "~" not in line:
                    continue
                parts = line.split("~")
                if len(parts) > 5:
                    sym = parts[2]
                    price = _safe_float(parts[3])
                    if price > 0:
                        prices[sym] = price
        except Exception:
            pass
    return prices


def _find_latest_run(symbol: str) -> Path | None:
    runs = sorted(ROOT.glob(f"output/runs/*_{symbol}"))
    return runs[-1] if runs else None


def _load_stock(symbol: str, name: str, realtime_prices: dict[str, float] | None = None) -> dict:
    run_dir = _find_latest_run(symbol)
    if not run_dir:
        return {"symbol": symbol, "name": name, "score": 0, "decision": "N/A", "bias": 0, "close": 0, "providers": {}}
    fd_path = run_dir / "final_decision.json"
    fd = json.loads(fd_path.read_text(encoding="utf-8")) if fd_path.exists() else {}
    feat = fd.get("analysis_features", {})
    score = _safe_float(fd.get("final_score"))
    # close: 优先用实时行情，其次CSV
    close = 0.0
    if realtime_prices and symbol in realtime_prices:
        close = realtime_prices[symbol]
    if close <= 0:
        csv_path = run_dir / "data" / "historical_daily.csv"
        if csv_path.exists():
            with open(csv_path, encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    close = _safe_float(rows[-1].get("close"))
    # MA5 bias
    ma_sys = feat.get("kline_indicators", {}).get("day", {}).get("ma_system", {})
    bias = _safe_float(ma_sys.get("ma5", {}).get("pct_above", 0))
    ma5 = _safe_float(ma_sys.get("ma5", {}).get("value"))
    # providers
    mt = fd.get("model_totals", {})
    providers = {}
    for p, v in mt.items():
        if isinstance(v, dict):
            providers[p] = _safe_float(v.get("total"))
        else:
            providers[p] = _safe_float(v)
    # decision label
    if score >= 70:
        dec = "买入"
    elif score >= 60:
        dec = "弱买入"
    elif score >= 50:
        dec = "观望"
    elif score >= 40:
        dec = "弱卖出"
    else:
        dec = "卖出"
    # sniper points
    sp = fd.get("sniper_points", {})
    return {
        "symbol": symbol, "name": name, "score": score, "decision": dec,
        "bias": bias, "close": close, "ma5": ma5, "providers": providers,
        "ideal_buy": _safe_float(sp.get("ideal_buy")),
        "stop_loss": _safe_float(sp.get("stop_loss")),
        "tp1": _safe_float(sp.get("take_profit_1")),
    }


def _decision_color(dec: str):
    if dec in ("买入", "弱买入"):
        return "#FF4444"
    if dec == "观望":
        return "#888888"
    return "#22AA22"


def _bias_color(bias: float):
    if abs(bias) > 8:
        return "#FF4444"
    if abs(bias) > 5:
        return "#FF8800"
    return "#22AA22"


def render_image(stocks_data: list[dict], output_path: str):
    """用 matplotlib 绘制汇总表格图片。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.font_manager as fm
    import numpy as np

    # 中文字体
    font_path = None
    for fp in [
        "C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]:
        if Path(fp).exists():
            font_path = fp
            break
    if font_path:
        prop = fm.FontProperties(fname=font_path, size=9)
        prop_title = fm.FontProperties(fname=font_path, size=13, weight="bold")
        prop_header = fm.FontProperties(fname=font_path, size=8.5, weight="bold")
        prop_small = fm.FontProperties(fname=font_path, size=7.5)
    else:
        prop = fm.FontProperties(size=9)
        prop_title = fm.FontProperties(size=13, weight="bold")
        prop_header = fm.FontProperties(size=8.5, weight="bold")
        prop_small = fm.FontProperties(size=7.5)

    # 收集所有 provider 名称
    all_providers = []
    for s in stocks_data:
        for p in s["providers"]:
            if p not in all_providers:
                all_providers.append(p)

    # 按评分排序
    stocks_data.sort(key=lambda x: x["score"], reverse=True)

    n = len(stocks_data)
    # 列: # | 代码 | 名称 | provider1 | ... | 均分 | MA5乖离 | 决策 | 买点 | 止损 | 止盈
    n_prov = len(all_providers)
    n_cols = 6 + n_prov + 3  # #, code, name, providers..., score, bias, decision, buy, sl, tp

    row_h = 0.32
    col_widths = [0.3, 0.6, 0.8] + [0.55] * n_prov + [0.5, 0.65, 0.55, 0.6, 0.55, 0.55]
    total_w = sum(col_widths) + 0.4
    total_h = (n + 3) * row_h + 1.2  # header + separator + data + margins

    fig, ax = plt.subplots(figsize=(total_w, total_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    # Title
    title = f"多智能体股票分析汇总 ({datetime.now().strftime('%Y-%m-%d')})"
    ax.text(total_w / 2, total_h - 0.3, title, ha="center", va="center", fontproperties=prop_title, color="#333333")
    subtitle = f"共{n}只 | 买入区(>=60): 红色 | 观望(50-60): 灰色 | 卖出区(<50): 绿色"
    ax.text(total_w / 2, total_h - 0.65, subtitle, ha="center", va="center", fontproperties=prop_small, color="#666666")

    y_start = total_h - 1.1

    # Header
    headers = ["#", "代码", "名称"] + all_providers + ["均分", "MA5乖离", "决策", "买点", "止损", "止盈"]
    x = 0.2
    for i, h in enumerate(headers):
        w = col_widths[i]
        # header background
        rect = FancyBboxPatch((x, y_start - row_h * 0.5), w - 0.02, row_h * 0.85,
                              boxstyle="round,pad=0.02", facecolor="#4A90D9", edgecolor="none")
        ax.add_patch(rect)
        ax.text(x + w / 2, y_start, h, ha="center", va="center", fontproperties=prop_header, color="white")
        x += w

    # Separator for group 7 vs 18
    group_7_symbols = {s["symbol"] for s in stocks_data[:len(stocks_data)] if s["symbol"] in [x[0] for x in STOCKS_7]}

    y = y_start - row_h
    drawn_sep = False
    for idx, s in enumerate(stocks_data):
        y -= row_h
        # Alternate row bg
        bg_color = "#FFFFFF" if idx % 2 == 0 else "#F0F4F8"

        # Group separator
        is_group7 = s["symbol"] in [x[0] for x in STOCKS_7]

        x = 0.2
        # Background stripe
        rect = FancyBboxPatch((x, y - row_h * 0.35), sum(col_widths) - 0.02, row_h * 0.85,
                              boxstyle="round,pad=0.01", facecolor=bg_color, edgecolor="none", alpha=0.7)
        ax.add_patch(rect)

        row_data = [str(idx + 1), s["symbol"], s["name"]]
        for p in all_providers:
            pv = s["providers"].get(p, 0)
            row_data.append(f"{pv:.1f}" if pv > 0 else "-")
        row_data.append(f"{s['score']:.1f}")
        row_data.append(f"{s['bias']:+.1f}%")
        row_data.append(s["decision"])
        row_data.append(f"{s['ideal_buy']:.2f}" if s['ideal_buy'] > 0 else "-")
        row_data.append(f"{s['stop_loss']:.2f}" if s['stop_loss'] > 0 else "-")
        row_data.append(f"{s['tp1']:.2f}" if s['tp1'] > 0 else "-")

        for i, val in enumerate(row_data):
            w = col_widths[i]
            color = "#333333"
            fp = prop_small

            # Color coding
            col_idx = i
            if col_idx == len(headers) - 4:  # 均分
                sc = s["score"]
                if sc >= 60:
                    color = "#FF4444"
                elif sc < 50:
                    color = "#22AA22"
            elif col_idx == len(headers) - 3:  # MA5乖离
                color = _bias_color(s["bias"])
            elif col_idx == len(headers) - 2:  # 决策
                color = _decision_color(s["decision"])
                fp = prop_header

            ax.text(x + w / 2, y, val, ha="center", va="center", fontproperties=fp, color=color)
            x += w

        # Group tag
        if is_group7:
            ax.text(0.12, y, "◆", ha="center", va="center", fontproperties=prop_small, color="#4A90D9")

    # Legend
    y_legend = y - row_h * 1.2
    ax.text(0.3, y_legend, "◆ = 前7只    按综合评分降序排列    数据来源: grok/doubao/deepseek 多模型加权",
            ha="left", va="center", fontproperties=prop_small, color="#888888")

    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"[OK] 汇总图片已保存: {output_path}")


def main():
    all_stocks = STOCKS_7 + STOCKS_18
    print(f"加载 {len(all_stocks)} 只股票数据...")

    # 先批量获取实时行情
    all_syms = [s[0] for s in all_stocks]
    print("获取实时行情...")
    realtime = _fetch_realtime_prices(all_syms)
    print(f"  获取到 {len(realtime)} 只实时价格")

    data = []
    for sym, name in all_stocks:
        d = _load_stock(sym, name, realtime)
        # 用实时价格重算 MA5 乖离率
        if d["close"] > 0 and d.get("ma5", 0) > 0:
            d["bias"] = round((d["close"] / d["ma5"] - 1) * 100, 1)
        data.append(d)
        src = "实时" if sym in realtime else "CSV"
        print(f"  {sym} {name}: score={d['score']:.1f} close={d['close']:.2f}({src}) decision={d['decision']}")

    out_path = str(ROOT / "output" / f"summary_{datetime.now().strftime('%Y%m%d')}.png")
    render_image(data, out_path)


if __name__ == "__main__":
    main()
