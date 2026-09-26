"""Leakage and usefulness probe for a semantic-judgment model (Jev/TypeSafe).

The S20 diagnosis is that the available price-volume features carry amplitude
rather than direction. A semantic model could only add direction if it has (a)
text to judge and (b) no prior knowledge of the outcome. This probe measures
(b) directly, before any feature work is funded.

Conditions, all on the same frozen diagnostic rows:

``matching``   the true (instrument, signal date) pair
``shifted``    the same instrument with another signal date; no outcome is knowable
``numeric``    only the five price-volume features plus the signal date, no identity

If ``matching`` separates outcomes while ``shifted`` does not, the model is
recalling the outcome rather than judging the state, and retrospective use is
contaminated. If both sit at chance, prior knowledge is not a usable channel.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
KEY_ENV = "JEV_API_KEY"

QUESTION = (
    "Consider the 20 A-share trading sessions that start on the next trading day "
    "after `as_of`. Admission is at that next day's opening price. Within a "
    "session, the first of these two levels to be reached decides the outcome: "
    "a rise of 15% above the entry price, or a fall of 10% below it. "
    "Will the 15% level be reached first?"
)


def ask(state, *, api_key: str, model: str = MODEL, timeout: int = 90,
        attempts: int = 4) -> dict:
    """One evaluation call with exponential backoff on rate limits."""
    payload = json.dumps({
        "state": state,
        "model": model,
        "questions": {
            "up15_first": {
                "type": "noul",
                "instructions": QUESTION,
                "criteria": {"true": "the +15% level is reached first",
                             "false": "the -10% level is reached first, or neither is reached"},
            }
        },
    }).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return {"ok": True, "noul": float(body["answers"]["up15_first"]["noul"]),
                    "model": body.get("model"), "usage": body.get("usage")}
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}"
            if exc.code in (429, 529):
                time.sleep(2 ** attempt)
                continue
            return {"ok": False, "error": last, "detail": exc.read().decode("utf-8")[:200]}
        except Exception as exc:  # network, timeout, malformed body
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1 + attempt)
    return {"ok": False, "error": last}


def auc(scores, labels) -> float | None:
    """Rank-based AUC; None when a class is missing."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive, negative = scores[labels], scores[~labels]
    if positive.size == 0 or negative.size == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - positive.size * (positive.size + 1) / 2)
                 / (positive.size * negative.size))


def build_items(frame: pd.DataFrame, *, per_class: int, seed: int) -> pd.DataFrame:
    """Stratified sample so AUC is not dominated by the base rate."""
    rng = np.random.default_rng(seed)
    frame = frame[frame.hit15.notna()].copy()
    frame["up15"] = frame.hit15.astype(bool)
    picked = []
    for value in (True, False):
        pool = frame[frame.up15.eq(value)]
        take = min(per_class, len(pool))
        picked.append(pool.sample(n=take, random_state=int(rng.integers(2 ** 31))))
    return pd.concat(picked, ignore_index=True)


def run_probe(items: pd.DataFrame, *, api_key: str, shifted_dates: list[str],
              numeric_columns: tuple[str, ...], pause: float = 0.0,
              progress=None) -> dict:
    rows = []
    for position, item in enumerate(items.itertuples(index=False)):
        # The shifted date must be independent of the label. Assigning it by row
        # position would leak the label whenever the item order is stratified.
        digest = hashlib.sha256(str(item.sample_id).encode("utf-8")).hexdigest()
        shifted = shifted_dates[int(digest[:8], 16) % len(shifted_dates)]
        matching = ask({"instrument": item.code, "as_of": item.signal_iso,
                        "market": "China A-share"},
                       api_key=api_key)
        if pause:
            time.sleep(pause)
        mismatched = ask({"instrument": item.code, "as_of": shifted,
                          "market": "China A-share"},
                         api_key=api_key)
        if pause:
            time.sleep(pause)
        numeric = ask({"as_of": item.signal_iso,
                       "market": "China A-share",
                       "price_volume_state": {
                           name: round(float(getattr(item, name)), 6)
                           for name in numeric_columns}},
                      api_key=api_key)
        if pause:
            time.sleep(pause)
        rows.append({
            "sample_id": item.sample_id, "code": item.code,
            "signal_date": item.signal_date, "up15": bool(item.up15),
            "is_A": bool(item.target == "A"),
            "matching": matching.get("noul"), "shifted": mismatched.get("noul"),
            "numeric": numeric.get("noul"),
            "matching_ok": matching["ok"], "shifted_ok": mismatched["ok"],
            "numeric_ok": numeric["ok"],
        })
        if progress and (position + 1) % 10 == 0:
            progress(position + 1, len(items))
    table = pd.DataFrame(rows)
    summary = {"items": int(len(table)), "model": MODEL, "question": QUESTION}
    for condition in ("matching", "shifted", "numeric"):
        known = table[table[condition].notna()]
        summary[condition] = {
            "scored": int(len(known)),
            "failures": int(len(table) - len(known)),
            "mean_score": float(known[condition].mean()) if len(known) else None,
            "base_rate_up15": float(known.up15.mean()) if len(known) else None,
            "auc_up15": auc(known[condition], known.up15),
            "auc_safe_A": auc(known[condition], known.is_A),
        }
    return {"summary": summary, "rows": rows}
