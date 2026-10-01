#!/usr/bin/env python
"""Unattended driver for the two event backtests that wait on slow public-API backfills.

    python scripts/orchestrate_event_backtests.py notices
        every 15 min: tag newly backfilled announcements (resumable), until every business
        day of the evaluation window [20250303, 20260805] has a notices file and nothing is
        left to tag; then run eval_notice_sentiment.py once.
    python scripts/orchestrate_event_backtests.py research
        wait until fetch_research_reports_em.py has written a file for every stock in its
        universe; then run eval_risk_gates_and_revisions.py (revision part reads them).
Progress goes to output/news/orchestrate_<mode>.log.
"""
from __future__ import annotations

import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
WIN_START, WIN_END = "20250303", "20260805"


def log(mode: str, msg: str) -> None:
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with (ROOT / f"output/news/orchestrate_{mode}.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(args: list[str]) -> str:
    r = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return (r.stdout or "") + (r.stderr or "")


def notices_complete() -> bool:
    have = {p.stem for p in (ROOT / "output/news/notices").glob("*.parquet")}
    need = pd.bdate_range(WIN_START, WIN_END).strftime("%Y%m%d")
    missing = [d for d in need if d not in have]
    return not missing


def notices() -> int:
    while True:
        out = run(["scripts/tag_notices_history.py", "--workers", "6"])
        head = out.strip().splitlines()[0] if out.strip() else "?"
        log("notices", f"tagging run: {head}")
        if notices_complete() and "to tag 0 " in head:
            break
        time.sleep(15 * 60)
    log("notices", "backfill + tagging complete; running eval_notice_sentiment.py")
    out = run(["scripts/eval_notice_sentiment.py"])
    log("notices", "eval finished:\n" + out[-3000:])
    return 0


def research() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    from fetch_research_reports_em import OUT, universe
    codes = universe()
    while True:
        have = len(list(OUT.glob("*.parquet")))
        log("research", f"research reports {have}/{len(codes)}")
        if have >= len(codes):
            break
        time.sleep(10 * 60)
    log("research", "running eval_risk_gates_and_revisions.py")
    out = run(["scripts/eval_risk_gates_and_revisions.py"])
    log("research", "eval finished:\n" + out[-3000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(notices() if sys.argv[1] == "notices" else research())
