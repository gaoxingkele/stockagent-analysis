#!/usr/bin/env python
"""Model-level ablation: retrain stage1 (S20-20R portable residual) without the volume family.

Identical recipe to train_s20_20r_residual.py (same 50% sample, seeds, nested anchors,
walk-forward folds); the only change is that the 16 stock-level volume / amount
features are added to the excluded set. The R20 anchor inside the recipe is refit
on the same reduced set, so the ablation removes the information everywhere.

    python scripts/ablate_stage1_volume.py
Output: output/experiments/ablation_volume/stage1_no_volume/predictions.parquet
Baseline for comparison: output/experiments/s20_20r_residual_portable/predictions.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

VOLUME_FAMILY = {
    "mfi_14", "vol_ratio_5", "vol_ratio_20", "amount_ratio_20", "vol_price_match", "obv", "obv_diff_20",
    "volume_spike_up", "volume_spike_dn", "corr_close_vol_20", "cord_ret_vol_20", "vma_20", "vstd_20",
    "wvma_20", "ad", "adosc",
}


def main() -> int:
    import train_s20_20r_residual as T
    T.PORTABLE_EXCLUDED_FEATURES = set(T.PORTABLE_EXCLUDED_FEATURES) | VOLUME_FAMILY
    out = ROOT / "output/experiments/ablation_volume/stage1_no_volume"
    sys.argv = [sys.argv[0], "--output-dir", str(out)]
    return T.main()


if __name__ == "__main__":
    raise SystemExit(main())
