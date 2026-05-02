"""筛选三星概念 A 股: 最近 3 日成交额连续 ≥ 5 亿/日."""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / ".env.cloubic", override=False)

import tushare as ts
pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

STOCKS = [
    # 存储芯片
    ("603986.SH", "兆易创新", "NOR Flash 龙头"),
    ("300223.SZ", "北京君正", "DRAM/SRAM"),
    ("688008.SH", "澜起科技", "内存接口"),
    ("688110.SH", "东芯股份", "NOR/NAND/DRAM"),
    # 代工 / IDM
    ("688981.SH", "中芯国际", "代工龙头"),
    ("688347.SH", "华虹公司", "特色工艺"),
    ("688396.SH", "华润微", "IDM"),
    ("600745.SH", "闻泰科技", "安世 IDM"),
    # OLED / 面板
    ("000725.SZ", "京东方 A", "LCD/OLED"),
    ("000100.SZ", "TCL 科技", "华星光电"),
    ("002387.SZ", "维信诺", "中小 OLED"),
    ("000050.SZ", "深天马 A", "中小 LCD"),
    # 消费电子
    ("002475.SZ", "立讯精密", "苹果链平台"),
    ("601138.SH", "工业富联", "代工"),
    ("002241.SZ", "歌尔股份", "声学/VR"),
    ("300433.SZ", "蓝思科技", "玻璃盖板"),
    # 家电
    ("000333.SZ", "美的集团", "综合家电"),
    ("600690.SH", "海尔智家", "白电"),
    ("000651.SZ", "格力电器", "空调"),
    # 半导体设备 / 材料
    ("002371.SZ", "北方华创", "设备龙头"),
    ("688012.SH", "中微公司", "刻蚀设备"),
    ("688120.SH", "华海清科", "CMP 设备"),
    ("688126.SH", "沪硅产业", "大硅片"),
]

# 找最近 5 个交易日 (容错: 节假日)
end = datetime.now().strftime("%Y%m%d")
start = (datetime.now() - timedelta(days=14)).strftime("%Y%m%d")

# 阈值: 5 亿元 = 500,000 千元 (Tushare amount 单位是千元)
THRESHOLD_AMOUNT = 500_000   # 千元

# 第一步: 用平安银行 (流动性好不会停牌) 探最新交易日
try:
    df_probe = pro.daily(ts_code="000001.SZ", start_date=start, end_date=end)
    LATEST_TRADE_DATE = df_probe.sort_values("trade_date", ascending=False).iloc[0]["trade_date"]
except Exception:
    LATEST_TRADE_DATE = None
print(f"扫描期间 {start} ~ {end}, 全市场最新交易日 = {LATEST_TRADE_DATE}, 阈值 5 亿元/日")
print(f"{'代码':<12} {'名称':<10} {'概念':<14} {'最近 3 日成交额(亿)':<30} {'判定'}")
print("-" * 95)

passed = []

for ts_code, name, concept in STOCKS:
    try:
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or df.empty:
            print(f"{ts_code:<12} {name:<10} {concept:<14} 无数据")
            continue
        # Tushare daily 默认按 trade_date 倒序
        df = df.sort_values("trade_date", ascending=False).head(3)
        if len(df) < 3:
            print(f"{ts_code:<12} {name:<10} {concept:<14} 数据不足({len(df)}日)")
            continue
        amounts_yi = (df["amount"] / 1e5).round(2).tolist()  # 千元 → 亿元 (1 亿 = 1e5 千元)
        dates = df["trade_date"].tolist()

        latest_match = (LATEST_TRADE_DATE is None) or (dates[0] == LATEST_TRADE_DATE)
        all_pass = latest_match and all(a >= 5.0 for a in amounts_yi)
        amount_str = " | ".join(
            f"{d[4:6]}-{d[6:8]} {a:.1f}亿" for d, a in zip(dates, amounts_yi)
        )
        if not latest_match:
            verdict = "[STOP]"   # 最新一天缺数据 = 可能停牌
        elif all_pass:
            verdict = "[PASS]"
        else:
            verdict = "[FAIL]"
        print(f"{ts_code:<12} {name:<10} {concept:<14} {amount_str:<35} {verdict}")
        if all_pass:
            passed.append((ts_code, name, concept, amounts_yi))
    except Exception as e:
        print(f"{ts_code:<12} {name:<10} {concept:<14} ERR: {str(e)[:50]}")

print()
print(f"=== 通过筛选 ({len(passed)} 只) ===")
for ts_code, name, concept, amounts in passed:
    avg = sum(amounts) / len(amounts)
    print(f"  {ts_code} {name:<10} {concept:<14} 3日均成交 {avg:.1f}亿")
