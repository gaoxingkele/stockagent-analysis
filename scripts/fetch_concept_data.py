"""拉 Tushare 概念板块数据 (Phase 2 偏置建模用).

接口选择:
  1. concept + concept_detail (Tushare 原生): 5000 积分以上
  2. ths_index (同花顺概念): 5000 积分
  3. dc_index (东方财富概念): 5000 积分

输出:
  - output/tushare_cache/concept_list.parquet      所有概念清单 (code, name, src)
  - output/tushare_cache/concept_detail.parquet    (concept_code, ts_code) 关联表
  - output/tushare_cache/concept_member_summary.parquet
       每只股的主要概念 (top 3 by 概念在全市场出现频率, 越稀有越特色)

用法: python scripts/fetch_concept_data.py
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
if not TOKEN:
    print("!! TUSHARE_TOKEN 未设置", flush=True); sys.exit(2)
pro = ts.pro_api(TOKEN)

CACHE = ROOT / "output" / "tushare_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def fetch_concept_list() -> pd.DataFrame:
    """获取所有概念清单 (Tushare concept 接口)."""
    out_p = CACHE / "concept_list.parquet"
    if out_p.exists():
        df = pd.read_parquet(out_p)
        print(f"[concept_list] 缓存: {len(df)} 个概念", flush=True)
        return df

    print("[concept_list] 拉取 Tushare concept ...", flush=True)
    try:
        df = pro.concept(src="ts")
        print(f"  → {len(df)} 个概念 (src=ts)", flush=True)
    except Exception as e:
        print(f"  ts src 失败: {e}", flush=True)
        # fallback: 不指定 src
        df = pro.concept()
        print(f"  → {len(df)} 个概念 (默认 src)", flush=True)
    df.to_parquet(out_p, index=False)
    return df


def fetch_concept_detail(concepts: pd.DataFrame) -> pd.DataFrame:
    """逐个概念拉成分股 (concept_detail 接口, 单次只能拉一个概念)."""
    out_p = CACHE / "concept_detail.parquet"
    if out_p.exists():
        df = pd.read_parquet(out_p)
        print(f"[concept_detail] 缓存: {len(df)} 行 ({df['concept_name'].nunique()} 概念 × 股)",
               flush=True)
        return df

    print(f"[concept_detail] 拉 {len(concepts)} 个概念的成分股 ...", flush=True)
    rows = []
    for i, row in concepts.iterrows():
        cid = row.get("code") or row.get("id")  # 不同 src 字段名不同
        cname = row.get("name", "")
        if not cid: continue
        try:
            sub = pro.concept_detail(id=cid, fields="ts_code,name,concept_name")
            sub["concept_code"] = cid
            sub["concept_name"] = cname
            rows.append(sub)
            if (i + 1) % 20 == 0:
                print(f"  进度 {i+1}/{len(concepts)} ({len(rows)} 概念已拉)", flush=True)
            time.sleep(0.6)  # rate limit
        except Exception as e:
            print(f"  概念 {cid}({cname}) 失败: {e}", flush=True)
            time.sleep(2)

    if not rows:
        print("!! 全部失败, 可能权限不够", flush=True); return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(out_p, index=False)
    print(f"  → 总计 {len(df)} 行 ({df['concept_name'].nunique()} 概念 × 股)", flush=True)
    return df


def build_member_summary(detail: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    """对每只股, 选 top_k 最稀有的概念 (在全市场出现频率最低 = 最有特色).

    输出列: ts_code, primary_concept_1, primary_concept_2, primary_concept_3,
            n_concepts (个股关联概念总数)
    """
    out_p = CACHE / "concept_member_summary.parquet"
    if out_p.exists():
        return pd.read_parquet(out_p)

    print(f"[summary] 计算每股 top {top_k} 主要概念 ...", flush=True)
    # 概念稀有度: 该概念关联的股数 (越小越稀有)
    concept_size = detail.groupby("concept_name").size().rename("concept_size")
    detail = detail.merge(concept_size, on="concept_name", how="left")
    # 每股按 concept_size 升序 (稀有优先) 取前 top_k
    detail_sorted = detail.sort_values(["ts_code", "concept_size"])
    rows = []
    for ts_code, g in detail_sorted.groupby("ts_code"):
        names = g["concept_name"].tolist()[:top_k]
        rows.append({
            "ts_code": ts_code,
            **{f"primary_concept_{i+1}": (names[i] if i < len(names) else None)
                for i in range(top_k)},
            "n_concepts": len(g),
        })
    summary = pd.DataFrame(rows)
    summary.to_parquet(out_p, index=False)
    print(f"  → {len(summary)} 只股", flush=True)
    return summary


def main():
    t0 = time.time()
    print(f"\n=== Tushare 概念板块拉取 ===\n", flush=True)
    concepts = fetch_concept_list()
    if concepts.empty:
        print("!! concept_list 为空, 退出"); return
    detail = fetch_concept_detail(concepts)
    if detail.empty:
        print("!! concept_detail 为空, 可能权限不够; 退出"); return
    summary = build_member_summary(detail, top_k=3)
    print(f"\n输出:")
    print(f"  {CACHE / 'concept_list.parquet'}")
    print(f"  {CACHE / 'concept_detail.parquet'}")
    print(f"  {CACHE / 'concept_member_summary.parquet'}")
    print(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
