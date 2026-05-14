"""对用户给的 14 只 A 股跑 V11 LLM 视觉评估 + V12 综合排序."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env.cloubic")
load_dotenv(ROOT / ".env")

from stockagent_analysis.v11_vision import V11VisionFilter

CANDIDATES = [
    ("301187.SZ", "欧圣电气"),
    ("300530.SZ", "领湃科技"),
    ("300545.SZ", "联得装备"),
    ("301186.SZ", "超达装备"),
    ("002571.SZ", "德力股份"),
    ("300917.SZ", "特发服务"),
    ("300617.SZ", "安靠智电"),
    ("300525.SZ", "博思软件"),
    ("002922.SZ", "伊戈尔"),
    ("300679.SZ", "电连技术"),
    ("603992.SH", "松霖科技"),
    ("603650.SH", "彤程新材"),
    ("000021.SZ", "深科技"),
    ("300895.SZ", "铜牛信息"),
]

DATE = "20260508"
OUT = ROOT / "output" / "v12_inference" / f"v11_user_list_{DATE}.csv"


def main():
    t0 = time.time()
    cloubic = os.environ.get("CLOUBIC_API_KEY")
    if not cloubic:
        print("CLOUBIC_API_KEY 未配置"); sys.exit(1)
    f = V11VisionFilter.get(ROOT, cloubic)
    symbols = [c for c, _ in CANDIDATES]
    name_map = dict(CANDIDATES)

    def cb(phase, pct, msg, data):
        print(f"  [{pct:3d}%] {phase}: {msg}", flush=True)

    print(f"开始对 {len(symbols)} 只跑 V11 LLM 视觉过滤 ({DATE})...", flush=True)
    results = f.filter_batch(symbols, DATE, cb=cb)

    df = pd.DataFrame(results)
    df["name_h"] = df["ts_code"].map(name_map)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"\n=== 完成, 耗时 {time.time()-t0:.0f}s, 输出 {OUT} ===", flush=True)


if __name__ == "__main__":
    main()
