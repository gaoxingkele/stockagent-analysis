#!/usr/bin/env python
"""Tag the historical announcements that matter for the backtest, concurrently.

Relevant = non-routine announcement type AND the stock is in the stage1 daily Top150
on some signal day t with announcement date in [t-4, t] (calendar days). Those are the
only announcements the event features can ever use, so nothing else is sent to the LLM.
Tags (MiniMax, same prompt as tag_news_llm.py) are cached in
output/news/tags/notices_hist.parquet keyed by text hash; resumable.

    python scripts/tag_notices_history.py --count-only
    python scripts/tag_notices_history.py --workers 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import sys
import threading
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from tag_news_llm import BATCH, MODEL, ask  # noqa: E402

NOTICES = ROOT / "output/news/notices"
OUT = ROOT / "output/news/tags/notices_hist.parquet"
ROUTINE = {
    "保荐/核查意见", "独立董事述职报告", "专项说明/独立意见", "审计报告", "监事会决议公告", "管理办法/制度",
    "召开股东大会通知", "法律意见书", "内部控制报告", "议事规则/实施细则", "股东大会资料", "ESG公告",
    "公司章程修订", "自查报告", "独立董事候选人声明", "独立董事提名人声明", "年度报告摘要", "年度报告全文",
    "一季度报告全文", "三季度报告全文", "半年度报告全文", "半年度报告摘要", "年度财务报告", "股东大会决议公告",
    "董事会决议公告", "募集资金使用情况报告", "调研活动", "社会责任报告", "信息披露制度",
}


def relevant() -> pd.DataFrame:
    s = pd.read_parquet(ROOT / "output/experiments/s20_20r_stage1_forward_outcomes/scored_with_outcomes.parquet",
                        columns=["ts_code", "trade_date", "stage1_score"])
    s["r"] = s.groupby("trade_date").stage1_score.rank(ascending=False)
    pool = s[s.r <= 150][["ts_code", "trade_date"]]
    pool["t"] = pd.to_datetime(pool.trade_date)
    need = pd.concat([pool.assign(d=pool.t - pd.Timedelta(days=k)) for k in range(5)])[["ts_code", "d"]]
    need["notice_date"] = need.d.dt.strftime("%Y%m%d")
    need = need[["ts_code", "notice_date"]].drop_duplicates()
    parts = []
    for f in sorted(NOTICES.glob("*.parquet")):
        n = pd.read_parquet(f)
        if n.empty:
            continue
        n["notice_date"] = f.stem
        n["ts_code"] = n["代码"].astype(str).map(lambda c: c + (".SH" if c.startswith(("6", "9")) else ".SZ"))
        parts.append(n[~n["公告类型"].isin(ROUTINE)])
    allx = pd.concat(parts, ignore_index=True)
    x = allx.merge(need, on=["ts_code", "notice_date"])
    x["text"] = x["公告类型"].astype(str) + "：" + x["公告标题"].astype(str)
    x["h"] = x.text.map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16])
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    x = relevant()
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["h"])
    todo = x.drop_duplicates("h")
    todo = todo[~todo.h.isin(set(done.h))]
    print(f"relevant notices {len(x):,} (unique texts {x.h.nunique():,}) over {x.notice_date.nunique()} days; "
          f"to tag {len(todo):,} -> ~{len(todo) // BATCH + 1} calls", flush=True)
    if a.count_only:
        return 0
    load_dotenv(ROOT / ".env", override=False)
    key = os.environ["MINMAX_API_KEY"]
    lock = threading.Lock()
    chunks = [todo.iloc[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    buf: list[dict] = []

    def work(chunk: pd.DataFrame) -> list[dict]:
        items = [{"id": j, "code": r.ts_code, "text": r.text} for j, r in enumerate(chunk.itertuples())]
        res = {int(z.get("id", -1)): z for z in ask(items, key) if isinstance(z, dict)}
        return [{"h": r.h, "ts_code": r.ts_code, "text": r.text, "sentiment": res.get(j, {}).get("sentiment"),
                 "event_type": res.get(j, {}).get("event_type"), "materiality": res.get(j, {}).get("materiality"),
                 "summary": res.get(j, {}).get("summary"), "model": MODEL} for j, r in enumerate(chunk.itertuples())]

    def flush():
        nonlocal done, buf
        if buf:
            done = pd.concat([done, pd.DataFrame(buf)], ignore_index=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            done.to_parquet(OUT, index=False)
            buf = []

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for i, rows in enumerate(ex.map(work, chunks), 1):
            with lock:
                buf += rows
                if i % 20 == 0:
                    flush()
                    print(f"  {i}/{len(chunks)} calls", flush=True)
    flush()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
