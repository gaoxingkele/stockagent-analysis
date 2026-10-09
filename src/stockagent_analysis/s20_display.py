"""S20 display score on a 0-100 scale, the same mapping V12 uses for buy_r20_score.

Per trade date, anchors come from that day's full-market stage1 distribution (V12 _live_anchors):
P5 -> 0, P50 -> 50, P95 -> 90, P99.5 -> 100, linear in between (V12 _map_anchored). The transform is
monotone within a day, so it never changes a rank, a pool, a cut or a list: it is presentation only and
the frozen contract is untouched. The score must be computed on the whole market of a day, not on a list.

Because the S20 list is the top ~2% of the market, every listed name maps to 99-100 on that scale. The
pool score separates them: within the day's stage1 top-100 pool, rank 1 -> 100 and rank 100 -> 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockagent_analysis.v12_scoring import _live_anchors, _map_anchored


def s20_score_100(frame: pd.DataFrame, score_col: str = "stage1_probability") -> pd.Series:
    """0-100 score for every row of a full-market frame with trade_date and score_col."""
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, idx in frame.groupby("trade_date").groups.items():
        v = frame.loc[idx, score_col].to_numpy(dtype=float)
        anchors = _live_anchors(v)
        if anchors is None:
            continue
        out.loc[idx] = np.round(_map_anchored(v, *anchors), 1)
    return out


def s20_pool_score(pool_rank: pd.Series, pool_size: int = 100) -> pd.Series:
    """0-100 score inside the stage1 top-`pool_size` pool: rank 1 -> 100, rank pool_size -> 1; NaN outside."""
    r = pd.to_numeric(pool_rank, errors="coerce")
    out = 100 * (1 - (r - 1) / pool_size)
    return out.where(r <= pool_size).round(0)
