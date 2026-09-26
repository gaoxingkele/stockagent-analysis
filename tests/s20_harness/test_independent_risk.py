import pandas as pd

from research.s20_harness.independent_risk import (
    b10_labels, replay_b10_gate, replay_mass_taus, rolling_calibrate_p_b10, tau_grid,
)
from research.s20_harness.policy_replay import select_pool


def _cal_fixture(late_label="A", late_available="2025-08-01T00:00:00+08:00"):
    samples, raw, labels = [], [], []
    for i in range(12):
        sid = f"cal{i}"
        samples.append(dict(sample_id=sid,
                            prediction_at="2025-03-15T21:00:00+08:00",
                            label_available_at="2025-04-15T15:01:00+08:00"))
        raw.append(dict(sample_id=sid, segment="calibration", p_b10=0.80 if i < 3 else 0.10))
        labels.append(dict(sample_id=sid, target="D" if i < 3 else "A"))
    samples.append(dict(sample_id="late",
                        prediction_at="2025-03-15T21:00:00+08:00",
                        label_available_at=late_available))
    raw.append(dict(sample_id="late", segment="calibration", p_b10=0.95))
    labels.append(dict(sample_id="late", target=late_label))
    for i, p in enumerate((0.40, 0.12)):
        sid = f"sel{i}"
        samples.append(dict(sample_id=sid,
                            prediction_at="2025-07-01T21:00:00+08:00",
                            label_available_at="2025-07-30T15:01:00+08:00"))
        raw.append(dict(sample_id=sid, segment="selection-policy", p_b10=p))
        labels.append(dict(sample_id=sid, target="C"))
    samples.append(dict(sample_id="out0",
                        prediction_at="2025-09-01T21:00:00+08:00",
                        label_available_at="2025-09-30T15:01:00+08:00"))
    raw.append(dict(sample_id="out0", segment="outer-test", p_b10=0.40))
    labels.append(dict(sample_id="out0", target="A"))
    return pd.DataFrame(samples), pd.DataFrame(raw), pd.DataFrame(labels)


def test_b10_label_is_paired_drawdown_not_minus_five_class_name():
    labels = pd.DataFrame({"sample_id": list("ABCD"), "target": list("ABCD")})
    y = b10_labels(labels)
    assert y.target.tolist() == [False, True, False, True]


def test_high_upside_cannot_pass_independent_b10_gate():
    frame = pd.DataFrame({
        "sample_id": ["hot", "ok"],
        "signal_date": ["20250102", "20250102"],
        "p_A": [0.80, 0.40],
        "p_B": [0.05, 0.05],
        "p_C": [0.05, 0.50],
        "p_D": [0.10, 0.05],
        "p_b10": [0.35, 0.08],
    })
    picked = select_pool(frame, n_cap=20, max_risk=0.20, risk_col="p_b10")
    assert picked.loc[picked.selected, "sample_id"].tolist() == ["ok"]
    open_gate = select_pool(frame, n_cap=20, max_risk=1.0, risk_col="p_b10")
    assert set(open_gate.loc[open_gate.selected, "sample_id"]) == {"hot", "ok"}


def test_gate_uses_calibrated_not_raw_p():
    frame = pd.DataFrame({
        "sample_id": ["inflated", "ok"],
        "signal_date": ["20250102", "20250102"],
        "p_A": [0.70, 0.40],
        "p_B": [0.05, 0.05],
        "p_C": [0.10, 0.50],
        "p_D": [0.15, 0.05],
        "p_b10": [0.40, 0.08],
        "p_b10_cal": [0.10, 0.08],
    })
    picked = select_pool(frame, n_cap=20, max_risk=0.20, risk_col="p_b10_cal")
    assert set(picked.loc[picked.selected, "sample_id"]) == {"inflated", "ok"}
    raw_gate = select_pool(frame, n_cap=20, max_risk=0.20, risk_col="p_b10")
    assert raw_gate.loc[raw_gate.selected, "sample_id"].tolist() == ["ok"]


def test_tau_grid_gates_on_calibrated_column():
    grid, pi0 = tau_grid()
    assert pi0["risk_col"] == "p_b10_cal"
    assert all(item["risk_col"] == "p_b10_cal" for item in grid)
    assert 1.0 in {item["max_risk"] for item in grid}


def test_identity_fallback_when_too_few_mature_labels():
    samples = pd.DataFrame({
        "sample_id": ["a", "b"],
        "prediction_at": ["2025-07-01T21:00:00+08:00", "2025-07-01T21:00:00+08:00"],
        "label_available_at": ["2025-07-30T15:01:00+08:00", "2025-07-30T15:01:00+08:00"],
    })
    raw = pd.DataFrame({
        "sample_id": ["a", "b"],
        "segment": ["selection-policy", "selection-policy"],
        "p_b10": [0.31, 0.07],
    })
    labels = pd.DataFrame({"sample_id": ["a", "b"], "target": ["D", "A"]})
    out = rolling_calibrate_p_b10(samples, raw, labels, min_cal_rows=50)
    assert out.set_index("sample_id").p_b10_cal.tolist() == [0.31, 0.07]
    assert (out.calibrator_n == 0).all()


def test_platt_shifts_overconfident_raw_p_toward_realized_rate():
    samples, raw, labels = [], [], []
    for i in range(20):
        sid = f"cal{i}"
        samples.append(dict(sample_id=sid,
                            prediction_at="2025-03-15T21:00:00+08:00",
                            label_available_at="2025-04-15T15:01:00+08:00"))
        raw.append(dict(sample_id=sid, segment="calibration", p_b10=0.85))
        labels.append(dict(sample_id=sid, target="D" if i < 4 else "A"))
    samples.append(dict(sample_id="sel0",
                        prediction_at="2025-07-01T21:00:00+08:00",
                        label_available_at="2025-07-30T15:01:00+08:00"))
    raw.append(dict(sample_id="sel0", segment="selection-policy", p_b10=0.85))
    labels.append(dict(sample_id="sel0", target="A"))
    out = rolling_calibrate_p_b10(pd.DataFrame(samples), pd.DataFrame(raw),
                                  pd.DataFrame(labels), min_cal_rows=10)
    cal = float(out.loc[out.sample_id == "sel0", "p_b10_cal"].iloc[0])
    assert 0.05 < cal < 0.45
    assert cal < 0.70


def test_future_b10_label_cannot_move_todays_calibrated_p():
    base_s, base_r, base_y = _cal_fixture(late_label="A")
    first = rolling_calibrate_p_b10(base_s, base_r, base_y, min_cal_rows=10)
    flipped_y = base_y.copy()
    flipped_y.loc[flipped_y.sample_id == "late", "target"] = "D"
    second = rolling_calibrate_p_b10(base_s, base_r, flipped_y, min_cal_rows=10)
    left = first.set_index("sample_id")
    right = second.set_index("sample_id")
    pd.testing.assert_series_equal(left.loc[["sel0", "sel1"], "p_b10_cal"],
                                   right.loc[["sel0", "sel1"], "p_b10_cal"])
    assert left.loc["out0", "p_b10_cal"] != right.loc["out0", "p_b10_cal"]


def test_same_instant_label_available_at_is_not_used():
    samples = pd.DataFrame({
        "sample_id": ["early", "same", "sel"],
        "prediction_at": [
            "2025-03-15T21:00:00+08:00",
            "2025-07-01T21:00:00+08:00",
            "2025-07-01T21:00:00+08:00",
        ],
        "label_available_at": [
            "2025-04-15T15:01:00+08:00",
            "2025-07-01T21:00:00+08:00",
            "2025-07-30T15:01:00+08:00",
        ],
    })
    raw = pd.DataFrame({
        "sample_id": ["early", "same", "sel"],
        "segment": ["calibration", "calibration", "selection-policy"],
        "p_b10": [0.20, 0.90, 0.20],
    })
    labels = pd.DataFrame({"sample_id": ["early", "same", "sel"], "target": ["A", "D", "A"]})
    one_class = rolling_calibrate_p_b10(samples, raw, labels, min_cal_rows=1)
    assert float(one_class.loc[one_class.sample_id == "sel", "p_b10_cal"].iloc[0]) == 0.20
    assert int(one_class.loc[one_class.sample_id == "sel", "calibrator_n"].iloc[0]) == 0


def test_label_available_at_compared_in_utc():
    samples = pd.DataFrame({
        "sample_id": ["usable", "boundary", "sel"],
        "prediction_at": [
            "2025-03-15T21:00:00+08:00",
            "2025-03-15T21:00:00+08:00",
            "2025-07-01T00:00:00+08:00",
        ],
        "label_available_at": [
            "2025-06-30T15:59:00Z",
            "2025-06-30T16:00:00Z",
            "2025-07-30T15:01:00+08:00",
        ],
    })
    raw = pd.DataFrame({
        "sample_id": ["usable", "boundary", "sel"],
        "segment": ["calibration", "calibration", "selection-policy"],
        "p_b10": [0.15, 0.90, 0.15],
    })
    labels = pd.DataFrame({"sample_id": ["usable", "boundary", "sel"], "target": ["A", "D", "A"]})
    out = rolling_calibrate_p_b10(samples, raw, labels, min_cal_rows=1)
    sel = out.loc[out.sample_id == "sel"].iloc[0]
    assert float(sel.p_b10_cal) == 0.15
    assert int(sel.calibrator_n) == 0


def _daily_book(prefix, dates, n_per_day, *, p_cal, target, p_A=0.50):
    rows, labels = [], []
    for date in dates:
        for i in range(n_per_day):
            sid = f"{prefix}-{date}-{i}"
            rows.append(dict(
                sample_id=sid, signal_date=date,
                p_A=float(p_A) - 0.001 * i, p_B=0.05, p_C=0.20, p_D=0.10,
                p_b10_cal=float(p_cal[i] if hasattr(p_cal, "__getitem__") else p_cal),
            ))
            labels.append(dict(sample_id=sid, target=target[i] if hasattr(target, "__getitem__") else target))
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_replay_mass_taus_use_scores_not_labels_or_outer():
    p_cal = [0.10] * 5 + [0.28] * 10 + [0.41] * 5
    dream, _ = _daily_book("d", ["d1"], 20, p_cal=p_cal, target=["A"] * 20)
    outer, _ = _daily_book("o", ["o1"], 20, p_cal=[0.99] * 20, target=["D"] * 20)
    taus = replay_mass_taus(dream)
    assert taus
    assert 1.0 not in taus
    assert all(0.0 < t < 1.0 for t in taus)
    relabeled, _ = _daily_book("d", ["d1"], 20, p_cal=p_cal, target=["D"] * 20)
    assert replay_mass_taus(relabeled) == taus
    assert replay_mass_taus(outer) != taus


def test_pi0_always_in_calibrated_tau_grid():
    grid, pi0 = tau_grid(extra_taus=(0.28, 0.33))
    assert pi0["max_risk"] == 1.0
    assert pi0["n_cap"] == 20
    assert pi0 in grid
    assert pi0["risk_col"] == "p_b10_cal"
    assert {p["max_risk"] for p in grid} >= {1.0, 0.12, 0.28, 0.33}


def test_coverage_below_floor_is_not_dream_improver():
    p_cal = [0.10] + [0.40] * 19
    targets = ["A"] * 19 + ["D"]
    dream, lab = _daily_book("d", ["d1", "d2"], 20, p_cal=p_cal, target=targets)
    online, olab = _daily_book("o", ["o1", "o2"], 20, p_cal=p_cal, target=targets)
    report = replay_b10_gate(dream, lab, online, olab, extra_taus=())
    tight = next(s for s in report["dream_grid"] if s["policy"]["max_risk"] == 0.12)
    assert tight["coverage"] < 0.3
    assert tight["hold_profit_rate"] == 1.0
    assert tight["hold_drawdown_rate"] == 0.0
    assert report["n_dream_improvers"] == 0
    assert report["champion_policy"] == report["pi0"]
    assert report["recommended_policy"] == report["pi0"]


def test_failed_online_transfer_keeps_pi0():
    dream_p = [0.10] * 18 + [0.16] * 2
    dream_y = ["A"] * 18 + ["D"] * 2
    online_p = [0.10] * 18 + [0.16] * 2
    online_y = ["C"] * 18 + ["A"] * 2
    dream, dlab = _daily_book("d", ["d1", "d2"], 20, p_cal=dream_p, target=dream_y)
    online, olab = _daily_book("o", ["o1", "o2"], 20, p_cal=online_p, target=online_y)
    report = replay_b10_gate(dream, dlab, online, olab, extra_taus=())
    assert report["n_dream_improvers"] >= 1
    assert report["champion_policy"] != report["pi0"]
    assert report["online_transfer_ok"] is False
    assert report["recommended_policy"] == report["pi0"]
    assert report["formal_training_authorized"] is False
    assert report["production_eligible"] is False
