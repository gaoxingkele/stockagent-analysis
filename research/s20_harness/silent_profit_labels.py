"""Versioned A-only silence split; frozen ABCD labels remain unchanged."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .take_profit import SILENCE_MAX_GAIN

LABEL_VERSION = "paired_tp15_dd10_abcds_a_only_v1"
CLASSES = ("A", "B", "C", "D", "S")


def split_silent_profit(frame: pd.DataFrame) -> pd.DataFrame:
    """S = legacy A with full-window max gain <8%; C is intentionally unchanged.

    Missing outcomes remain unknown. An A with missing gain cannot be assigned
    to either A or S. This is an outcome label, never an ex-ante universe filter.
    """
    required = {"sample_id", "target", "max_gain"}
    if not required.issubset(frame.columns):
        raise ValueError("sample_id, target and max_gain required")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("sample_id must be present and unique")
    if not frame.target.dropna().isin(list("ABCD")).all():
        raise ValueError("expected legacy ABCD labels")
    out = frame.copy()
    out["legacy_target"] = frame.target
    out["target"] = frame.target.astype("string")
    gain = pd.to_numeric(frame.max_gain, errors="raise")
    known_gain = gain.notna() & np.isfinite(gain)
    a = frame.target.eq("A").fillna(False)
    out.loc[a & known_gain & gain.lt(SILENCE_MAX_GAIN), "target"] = "S"
    out.loc[a & ~known_gain, "target"] = pd.NA
    out["label_version"] = LABEL_VERSION
    return out
