"""Synthetic hardware smoke test only. No stock labels or efficacy claims."""
import json
import time

import lightgbm as lgb
import numpy as np


def main():
    rng = np.random.default_rng(20260913)
    features = rng.normal(size=(5000, 24)).astype("float32")
    labels = rng.integers(0, 4, size=5000)
    start = time.perf_counter()
    model = lgb.LGBMClassifier(n_estimators=20, num_leaves=7, n_jobs=1,
                              random_state=20260913, verbosity=-1, device_type="cpu")
    model.fit(features, labels)
    probability = model.predict_proba(features[:100])
    if probability.shape != (100, 4) or not np.isfinite(probability).all():
        raise ValueError("probe probability schema failed")
    print(json.dumps({"kind": "synthetic_resource_probe_not_scientific_trial", "rows": 5000,
                      "features": 24, "boosting_rounds": 20, "class_count": 4,
                      "cpu_threads": 1, "fit_predict_seconds": time.perf_counter() - start,
                      "production_model_written": False, "stock_performance_metrics": None}))


if __name__ == "__main__":
    main()
