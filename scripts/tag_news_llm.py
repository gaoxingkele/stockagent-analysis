#!/usr/bin/env python
"""Tag archived news / announcements with an LLM (MiniMax, direct API).

Reads the raw archive written by collect_news_daily.py and writes structured tags:
  sentiment    -2 (strongly negative) .. +2 (strongly positive) for the stock's price
  event_type   one of EVENT_TYPES
  materiality  0 (noise) .. 3 (major, likely to move the price)
  summary      <= 30 Chinese characters
Batched (BATCH items per call), cached by text hash, resumable.

    python scripts/tag_news_llm.py --source stock_news --date 20261001
    python scripts/tag_news_llm.py --source notices --date 20260930 --codes-from-pool
Output: output/news/tags/<source>/<date>.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "output/news"
MODEL = "MiniMax-Text-01"
URL = "https://api.minimaxi.com/v1/chat/completions"
BATCH = 20
EVENT_TYPES = ["业绩", "增减持", "回购", "并购重组", "融资", "解禁", "监管处罚", "诉讼", "订单合同", "产品技术",
               "政策行业", "股权激励", "分红", "人事", "经营其他", "调研评论", "无关"]
PROMPT = (
    "你是 A 股事件分析员。下面是若干条与个股相关的新闻或公告，每条带编号、股票代码和文本。"
    "请逐条判断它对该股票未来 1-4 周股价的影响，只依据文本本身，不要猜测文本之外的信息。\n"
    "输出一个 JSON 数组，每个元素为 {\"id\": 编号, \"sentiment\": -2到2的整数, "
    f"\"event_type\": 从 {EVENT_TYPES} 中选一个, \"materiality\": 0到3的整数, \"summary\": 不超过30字}}。"
    "只输出 JSON 数组，不要任何其他文字。\n\n"
)


def text_of(row: pd.Series, source: str) -> str:
    if source == "stock_news":
        return f"{row.get('新闻标题', '')}。{str(row.get('新闻内容', ''))[:200]}"
    if source == "notices":
        return f"{row.get('公告类型', '')}：{row.get('公告标题', '')}"
    return str(row.get("内容") or row.get("摘要") or row.get("标题") or "")[:220]


def ask(items: list[dict], key: str) -> list[dict]:
    body = "\n".join(f"[{it['id']}] {it['code']} {it['text']}" for it in items)
    for attempt in range(4):
        try:
            r = requests.post(URL, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                              json={"model": MODEL, "messages": [{"role": "user", "content": PROMPT + body}],
                                    "max_tokens": 4000, "temperature": 0.1}, timeout=180)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
            m = re.search(r"\[.*\]", txt, flags=re.S)
            return json.loads(m.group(0)) if m else []
        except Exception as exc:  # noqa: BLE001
            print(f"  llm retry {attempt + 1}: {str(exc)[:80]}", flush=True)
            time.sleep(3 * (attempt + 1))
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["stock_news", "notices", "market_flash"], required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--codes-from-pool", action="store_true", help="notices: keep only stage1-pool / list stocks")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    key = os.environ["MINMAX_API_KEY"]
    src = NEWS / a.source / f"{a.date}.parquet"
    df = pd.read_parquet(src)
    if a.source == "notices":
        df["ts_code"] = df["代码"].astype(str).map(lambda c: c + (".SH" if c.startswith("6") else ".SZ"))
        if a.codes_from_pool:
            from collect_news_daily import candidates
            df = df[df.ts_code.isin(set(candidates(400)))]
    if "ts_code" not in df.columns:
        df["ts_code"] = "MARKET"
    df["text"] = [text_of(r, a.source) for _, r in df.iterrows()]
    df["h"] = df.text.map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()[:16])
    df = df.drop_duplicates("h")
    out = NEWS / "tags" / a.source / f"{a.date}.parquet"
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["h"])
    todo = df[~df.h.isin(set(done.h))]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"{a.source} {a.date}: {len(df)} items, {len(todo)} to tag", flush=True)
    rows = []
    for i in range(0, len(todo), BATCH):
        chunk = todo.iloc[i:i + BATCH]
        items = [{"id": j, "code": r.ts_code, "text": r.text} for j, r in enumerate(chunk.itertuples())]
        res = {int(x.get("id", -1)): x for x in ask(items, key) if isinstance(x, dict)}
        for j, r in enumerate(chunk.itertuples()):
            x = res.get(j, {})
            rows.append({"h": r.h, "ts_code": r.ts_code, "text": r.text, "sentiment": x.get("sentiment"),
                         "event_type": x.get("event_type"), "materiality": x.get("materiality"),
                         "summary": x.get("summary"), "model": MODEL})
        print(f"  tagged {min(i + BATCH, len(todo))}/{len(todo)}", flush=True)
    if rows:
        new = pd.DataFrame(rows)
        pd.concat([done, new], ignore_index=True).to_parquet(out.parent.mkdir(parents=True, exist_ok=True) or out, index=False)
    print(f"-> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
