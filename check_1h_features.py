"""检查 1H 因子的极端值/inf/nan 分布."""
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_parquet(Path("output") / "1h_factors" / "factors.parquet")
print(f"总行: {len(df):,}, 列: {len(df.columns)}")
print()
print(f"{'feature':<25} {'nan%':>8} {'inf%':>8} {'min':>15} {'max':>15} {'p5':>10} {'p95':>10}")
print("-" * 100)
for c in df.columns:
    if not pd.api.types.is_numeric_dtype(df[c]): continue
    x = df[c].values.astype("float64")
    nan_pct = np.isnan(x).mean() * 100
    inf_pct = np.isinf(x).mean() * 100
    valid = x[np.isfinite(x)]
    if len(valid) == 0:
        print(f"{c:<25} ALL nan/inf")
        continue
    print(f"{c:<25} {nan_pct:>7.2f}% {inf_pct:>7.2f}% {valid.min():>15.4g} {valid.max():>15.4g} "
          f"{np.percentile(valid, 5):>10.4g} {np.percentile(valid, 95):>10.4g}")
