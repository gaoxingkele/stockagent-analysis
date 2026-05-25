"""修正多源合并: 用 con_code (个股), dc 去重最新日期, 输出标准表."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "concept_local"


def main():
    parts = []

    # THS
    p_ths = OUT / "concept_member_ths.parquet"
    if p_ths.exists():
        df = pd.read_parquet(p_ths)
        # con_code 是个股, ts_code 是概念
        df = df.rename(columns={"con_code": "stock_code", "ts_code": "concept_ts_code",
                                  "con_name": "stock_name"})
        df["source"] = "ths"
        df = df[["stock_code", "stock_name", "concept_ts_code", "concept_name", "source"]]
        df = df.drop_duplicates(subset=["stock_code", "concept_name"])
        print(f"ths: {len(df):,} 关联, {df['stock_code'].nunique():,} 股, "
               f"{df['concept_name'].nunique():,} 概念", flush=True)
        parts.append(df)

    # DC (取最新日期一次去重)
    p_dc = OUT / "concept_member_dc.parquet"
    if p_dc.exists():
        df = pd.read_parquet(p_dc)
        print(f"dc 原始: {len(df):,} 行, 日数 {df['trade_date'].nunique()}", flush=True)
        # 取最新日期数据 (概念成分股一般稳定, 用最新即可)
        latest = df["trade_date"].max()
        df = df[df["trade_date"] == latest].copy()
        df = df.rename(columns={"con_code": "stock_code", "ts_code": "concept_ts_code",
                                  "name": "stock_name"})
        df["source"] = "dc"
        df = df[["stock_code", "stock_name", "concept_ts_code", "concept_name", "source"]]
        df = df.drop_duplicates(subset=["stock_code", "concept_name"])
        print(f"dc (latest {latest}): {len(df):,} 关联, {df['stock_code'].nunique():,} 股, "
               f"{df['concept_name'].nunique():,} 概念", flush=True)
        parts.append(df)

    # 旧的 Tushare concept (concept_detail) 也可加入
    p_old = ROOT / "output" / "tushare_cache" / "concept_detail.parquet"
    if p_old.exists():
        df = pd.read_parquet(p_old)
        df = df.rename(columns={"ts_code": "stock_code", "name": "stock_name"})
        df["concept_ts_code"] = df.get("concept_code", "")
        df["source"] = "concept_old"
        df = df[["stock_code", "stock_name", "concept_ts_code", "concept_name", "source"]]
        df = df.drop_duplicates(subset=["stock_code", "concept_name"])
        print(f"old concept: {len(df):,} 关联, {df['stock_code'].nunique():,} 股, "
               f"{df['concept_name'].nunique():,} 概念", flush=True)
        parts.append(df)

    if not parts:
        print("无数据"); return

    merged = pd.concat(parts, ignore_index=True)
    print(f"\n合并前: {len(merged):,}", flush=True)
    # 关键: 同 stock × concept_name 即使 source 不同也去重
    dedup = merged.drop_duplicates(subset=["stock_code", "concept_name"], keep="first")
    print(f"去重后: {len(dedup):,}", flush=True)

    out_p = OUT / "concept_merged.parquet"
    dedup.to_parquet(out_p, index=False)

    # 板块覆盖统计
    def board(c):
        if c.startswith("688"): return "科创板"
        if c.startswith(("30",)) and len(c.split(".")[0]) == 6: return "创业板"
        if c.startswith("8"): return "北交所"
        return "主板"
    dedup["board"] = dedup["stock_code"].apply(board)
    print(f"\n=== 唯一股票数 / 唯一概念数 ===")
    print(f"  总: {dedup['stock_code'].nunique():,} 股, {dedup['concept_name'].nunique():,} 概念",
           flush=True)
    print(f"\n=== 各板块覆盖 ===")
    for b, g in dedup.groupby("board"):
        print(f"  {b}: {g['stock_code'].nunique():,} 股", flush=True)

    print(f"\n=== 各源贡献 (按股票) ===")
    for s, g in merged.groupby("source"):
        print(f"  {s}: {g['stock_code'].nunique():,} 股", flush=True)

    # 检查用户关注 7 股覆盖
    print(f"\n=== 用户关注 7 股概念覆盖 ===")
    targets = ["688507.SH", "301313.SZ", "688260.SH", "688234.SH",
                 "688525.SH", "688249.SH", "688322.SH"]
    for ts in targets:
        sub = dedup[dedup["stock_code"] == ts]
        if sub.empty:
            print(f"  {ts}: 无概念", flush=True)
        else:
            cpts = sub["concept_name"].tolist()
            print(f"  {ts}: {len(cpts)} 概念", flush=True)
            print(f"    {cpts[:8]}{'...' if len(cpts) > 8 else ''}", flush=True)

    print(f"\n输出: {out_p}")


if __name__ == "__main__":
    main()
