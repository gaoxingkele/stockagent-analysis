"""多源概念板块数据拉取 (同花顺 ths + 东方财富 dc + 通达信 tdx).

输入: Tushare API + 通达信本地

输出:
  output/concept_local/concept_list_<source>.parquet     概念清单
  output/concept_local/concept_member_<source>.parquet   股票-概念关联
  output/concept_local/concept_merged.parquet            合并去重表

每个数据源独立拉, 失败不影响其他源.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

import tushare as ts
TOKEN = os.getenv("TUSHARE_TOKEN")
pro = ts.pro_api(TOKEN) if TOKEN else None

OUT = ROOT / "output" / "concept_local"
OUT.mkdir(parents=True, exist_ok=True)


# =============== 1. Tushare 同花顺 ths_index ===============

def fetch_ths():
    print("\n[1] Tushare ths_index (同花顺概念) ...", flush=True)
    list_p = OUT / "concept_list_ths.parquet"
    member_p = OUT / "concept_member_ths.parquet"
    if list_p.exists() and member_p.exists():
        df_l = pd.read_parquet(list_p)
        df_m = pd.read_parquet(member_p)
        print(f"  缓存: {len(df_l)} 概念, {len(df_m)} 关联", flush=True)
        return df_l, df_m

    try:
        # 拉概念清单 (type=N 概念板块)
        df_list = pro.ths_index(type="N")
        print(f"  ths_index 概念清单: {len(df_list)} 个", flush=True)
        df_list.to_parquet(list_p, index=False)
    except Exception as e:
        print(f"  ths_index 失败 (可能权限不够): {e}", flush=True)
        return None, None

    # 拉每个概念的成分股
    parts = []
    for i, row in df_list.iterrows():
        cid = row["ts_code"]
        cname = row.get("name", "")
        try:
            sub = pro.ths_member(ts_code=cid)
            if sub is not None and not sub.empty:
                sub["concept_code"] = cid
                sub["concept_name"] = cname
                parts.append(sub)
            if (i + 1) % 30 == 0:
                print(f"  进度 {i+1}/{len(df_list)} ({len(parts)} 概念已拉)", flush=True)
            time.sleep(0.4)
        except Exception as e:
            print(f"  概念 {cid} 失败: {e}", flush=True)
            time.sleep(1)

    if not parts:
        print(f"  ths_member 全部失败", flush=True)
        return df_list, None

    df_member = pd.concat(parts, ignore_index=True)
    df_member.to_parquet(member_p, index=False)
    print(f"  → ths member: {len(df_member):,} 关联, "
           f"覆盖 {df_member['con_code'].nunique() if 'con_code' in df_member.columns else df_member['ts_code'].nunique()} 股", flush=True)
    return df_list, df_member


# =============== 2. Tushare 东方财富 dc_index ===============

def fetch_dc():
    print("\n[2] Tushare dc_index (东方财富概念) ...", flush=True)
    list_p = OUT / "concept_list_dc.parquet"
    member_p = OUT / "concept_member_dc.parquet"
    if list_p.exists() and member_p.exists():
        df_l = pd.read_parquet(list_p)
        df_m = pd.read_parquet(member_p)
        print(f"  缓存: {len(df_l)} 概念, {len(df_m)} 关联", flush=True)
        return df_l, df_m

    try:
        df_list = pro.dc_index()
        print(f"  dc_index 清单: {len(df_list)} 个", flush=True)
        df_list.to_parquet(list_p, index=False)
    except Exception as e:
        print(f"  dc_index 失败: {e}", flush=True)
        return None, None

    parts = []
    for i, row in df_list.iterrows():
        cid = row.get("ts_code") or row.get("dc_code")
        cname = row.get("name", "")
        if not cid: continue
        try:
            sub = pro.dc_member(ts_code=cid)
            if sub is not None and not sub.empty:
                sub["concept_code"] = cid
                sub["concept_name"] = cname
                parts.append(sub)
            if (i + 1) % 30 == 0:
                print(f"  进度 {i+1}/{len(df_list)}", flush=True)
            time.sleep(0.4)
        except Exception as e:
            print(f"  概念 {cid} 失败: {e}", flush=True)
            time.sleep(1)

    if not parts:
        return df_list, None
    df_member = pd.concat(parts, ignore_index=True)
    df_member.to_parquet(member_p, index=False)
    print(f"  → dc member: {len(df_member):,} 关联", flush=True)
    return df_list, df_member


# =============== 3. 通达信本地 mootdx ===============

def fetch_tdx_local():
    print("\n[3] mootdx 本地通达信概念 ...", flush=True)
    list_p = OUT / "concept_list_tdx.parquet"
    member_p = OUT / "concept_member_tdx.parquet"
    if list_p.exists() and member_p.exists():
        df_l = pd.read_parquet(list_p)
        df_m = pd.read_parquet(member_p)
        print(f"  缓存: {len(df_l)} 概念, {len(df_m)} 关联", flush=True)
        return df_l, df_m

    try:
        # 用 mootdx 远程接口 (本地未安装 TDX 也能用)
        from mootdx.consts import KIND_FG, KIND_GN, KIND_ZS
        from mootdx.reader import Reader
        from mootdx.quotes import Quotes

        client = Quotes.factory(market="std")

        # 概念板块清单 (KIND_GN=概念, 也可以 KIND_FG=风格)
        all_blocks = []
        for kind in ["概念", "风格", "指数"]:
            try:
                blocks = client.blocks(kind=kind)
                if blocks is not None and not blocks.empty:
                    blocks["block_kind"] = kind
                    all_blocks.append(blocks)
                    print(f"  {kind}: {len(blocks)} 个板块", flush=True)
            except Exception as e:
                print(f"  {kind} 拉取失败: {e}", flush=True)

        if not all_blocks:
            print(f"  mootdx 无返回 (可能需要 pip install mootdx)", flush=True)
            return None, None

        df_list = pd.concat(all_blocks, ignore_index=True)
        df_list.to_parquet(list_p, index=False)
        print(f"  list cols: {df_list.columns.tolist()}", flush=True)

        # 板块成分股 (blocks 表已含 block_code, stock_codes)
        # 简化: 解析 stock_codes 字段
        parts = []
        for _, row in df_list.iterrows():
            block_name = row.get("block_name") or row.get("name", "")
            block_code = row.get("block_code") or row.get("code", "")
            stocks = row.get("stock_codes", "") or row.get("stocks", "")
            if isinstance(stocks, str) and stocks:
                stock_list = stocks.split(",")
                for s in stock_list:
                    s = s.strip()
                    if not s: continue
                    parts.append({
                        "ts_code": _to_ts_code(s),
                        "concept_code": block_code,
                        "concept_name": block_name,
                        "kind": row.get("block_kind", ""),
                    })

        if not parts:
            print(f"  无成分股数据", flush=True)
            return df_list, None
        df_member = pd.DataFrame(parts)
        df_member.to_parquet(member_p, index=False)
        print(f"  → tdx member: {len(df_member):,}", flush=True)
        return df_list, df_member

    except ImportError:
        print(f"  ! mootdx 未安装, 跳过 (pip install mootdx)", flush=True)
        return None, None
    except Exception as e:
        print(f"  mootdx 失败: {e}", flush=True)
        return None, None


def _to_ts_code(s: str) -> str:
    """通达信代码格式 → tushare ts_code."""
    s = s.strip()
    if s.endswith(".SH") or s.endswith(".SZ") or s.endswith(".BJ"):
        return s
    # 默认: 6 开头 SH, 0/3 开头 SZ, 8 开头 BJ
    if s.startswith("6"):
        return f"{s}.SH"
    if s.startswith(("0", "3")):
        return f"{s}.SZ"
    if s.startswith("8") or s.startswith("9"):
        return f"{s}.BJ"
    return s


# =============== 合并去重 ===============

def merge_all(*member_dfs):
    print("\n[4] 合并多源去重 ...", flush=True)
    all_parts = []
    for df, source in member_dfs:
        if df is None or df.empty: continue
        if "con_code" in df.columns and "ts_code" not in df.columns:
            df = df.rename(columns={"con_code": "ts_code"})
        cols = ["ts_code", "concept_code", "concept_name"]
        df_clean = df[[c for c in cols if c in df.columns]].copy()
        df_clean["source"] = source
        all_parts.append(df_clean)

    if not all_parts:
        print(f"  无数据可合并", flush=True); return None
    merged = pd.concat(all_parts, ignore_index=True)
    print(f"  合并前 {len(merged):,} 关联", flush=True)

    # 去重 (同 ts_code + concept_name 视为同一关联)
    merged_dedup = merged.drop_duplicates(subset=["ts_code", "concept_name"], keep="first")
    print(f"  去重后 {len(merged_dedup):,} 关联", flush=True)

    out_p = OUT / "concept_merged.parquet"
    merged_dedup.to_parquet(out_p, index=False)

    # 统计覆盖
    print(f"\n--- 覆盖统计 ---", flush=True)
    print(f"  唯一股票: {merged_dedup['ts_code'].nunique():,}", flush=True)
    print(f"  唯一概念: {merged_dedup['concept_name'].nunique():,}", flush=True)
    print(f"  各源贡献:")
    for s, g in merged.groupby("source"):
        print(f"    {s}: {g['ts_code'].nunique():,} 股, {g['concept_name'].nunique():,} 概念",
               flush=True)

    # 板块覆盖率 (科创/创业/主板)
    board = lambda c: "科创" if c.startswith("688") else ("创业" if c.startswith("30") else "主板")
    merged_dedup["board"] = merged_dedup["ts_code"].apply(board)
    print(f"\n  各板块覆盖:")
    for b, g in merged_dedup.groupby("board"):
        print(f"    {b}: {g['ts_code'].nunique():,} 股", flush=True)

    print(f"\n输出: {out_p}", flush=True)
    return merged_dedup


def main():
    t0 = time.time()
    print(f"\n=== 多源概念板块拉取 ===\n", flush=True)

    ths_list, ths_member = fetch_ths()
    dc_list, dc_member = fetch_dc()
    tdx_list, tdx_member = fetch_tdx_local()

    merge_all(
        (ths_member, "ths"),
        (dc_member, "dc"),
        (tdx_member, "tdx"),
    )

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
