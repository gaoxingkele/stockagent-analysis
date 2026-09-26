import pandas as pd

from research.s20_harness.policy_replay import select_pool
from research.s20_harness.silence import (
    matured_silence_cooldown, replay_screen, rolling_calibrate_binary, screen_grid,
)


def test_current_window_max_gain_cannot_silence_today():
    samples = pd.DataFrame({
        "sample_id": ["today"],
        "entity_id": ["x"],
        "prediction_at": ["2025-07-01T21:00:00+08:00"],
    })
    silent = pd.DataFrame({
        "sample_id": ["today"],
        "silent": [True],
        "label_available_at": ["2025-07-30T15:01:00+08:00"],
    })
    out = matured_silence_cooldown(samples, silent)
    assert out.silent_cooldown.tolist() == [False]


def test_matured_prior_silence_blocks_today_even_if_today_moons():
    samples = pd.DataFrame({
        "sample_id": ["prior", "today"],
        "entity_id": ["x", "x"],
        "prediction_at": ["2025-05-01T21:00:00+08:00", "2025-07-01T21:00:00+08:00"],
    })
    silent = pd.DataFrame({
        "sample_id": ["prior", "today"],
        "silent": [True, False],
        "label_available_at": ["2025-06-01T15:01:00+08:00", "2025-07-30T15:01:00+08:00"],
    })
    out = matured_silence_cooldown(samples, silent).set_index("sample_id")
    assert bool(out.loc["prior", "silent_cooldown"]) is False
    assert bool(out.loc["today", "silent_cooldown"]) is True


def test_unmatured_future_silence_label_does_not_block_today():
    samples = pd.DataFrame({
        "sample_id": ["prior", "today"],
        "entity_id": ["x", "x"],
        "prediction_at": ["2025-06-20T21:00:00+08:00", "2025-07-01T21:00:00+08:00"],
    })
    silent = pd.DataFrame({
        "sample_id": ["prior", "today"],
        "silent": [True, True],
        "label_available_at": ["2025-07-10T15:01:00+08:00", "2025-07-30T15:01:00+08:00"],
    })
    out = matured_silence_cooldown(samples, silent).set_index("sample_id")
    assert bool(out.loc["today", "silent_cooldown"]) is False


def test_future_silence_label_cannot_move_todays_calibrated_p():
    samples = pd.DataFrame({
        "sample_id": [f"c{i}" for i in range(12)] + ["late", "sel"],
        "prediction_at": ["2025-03-15T21:00:00+08:00"] * 13 + ["2025-07-01T21:00:00+08:00"],
        "label_available_at": ["2025-04-15T15:01:00+08:00"] * 12
        + ["2025-08-01T00:00:00+08:00", "2025-07-30T15:01:00+08:00"],
    })
    raw = pd.DataFrame({
        "sample_id": samples.sample_id,
        "segment": ["calibration"] * 13 + ["selection-policy"],
        "p_silent": [0.80] * 3 + [0.10] * 9 + [0.95, 0.40],
    })
    y = pd.DataFrame({
        "sample_id": samples.sample_id,
        "target": [True] * 3 + [False] * 9 + [False, False],
    })
    first = rolling_calibrate_binary(samples, raw, y, raw_col="p_silent", cal_col="p_silent_cal",
                                     min_cal_rows=10)
    y2 = y.copy()
    y2.loc[y2.sample_id == "late", "target"] = True
    second = rolling_calibrate_binary(samples, raw, y2, raw_col="p_silent", cal_col="p_silent_cal",
                                      min_cal_rows=10)
    a = first.set_index("sample_id").loc["sel", "p_silent_cal"]
    b = second.set_index("sample_id").loc["sel", "p_silent_cal"]
    assert a == b


def test_high_upside_cannot_pass_independent_b10_with_silence_cols_present():
    frame = pd.DataFrame({
        "sample_id": ["hot", "ok"],
        "signal_date": ["20250102", "20250102"],
        "p_A": [0.80, 0.40],
        "p_B": [0.05, 0.05],
        "p_C": [0.05, 0.50],
        "p_D": [0.10, 0.05],
        "p_b10_cal": [0.35, 0.08],
        "p_silent_cal": [0.10, 0.10],
        "p_hit15_cal": [0.70, 0.20],
        "silent_cooldown": [False, False],
    })
    picked = select_pool(frame, n_cap=20, max_risk=0.20, risk_col="p_b10_cal",
                         silence_col="p_silent_cal", blocked_col="silent_cooldown")
    assert picked.loc[picked.selected, "sample_id"].tolist() == ["ok"]


def test_pi0_stays_in_screen_grid():
    grid, pi0 = screen_grid(extra_silence_taus=(0.33,))
    assert pi0 in grid
    assert pi0["cooldown"] is False
    assert pi0["max_risk"] == 1.0
    assert pi0["ranking"] == "penalized_utility"


def _book(prefix, dates, n, *, hit, silent, p_hit, p_b10=0.05, p_silent=0.20, blocked=False, dd=False):
    rows, labels = [], []
    for date in dates:
        for i in range(n):
            sid = f"{prefix}-{date}-{i}"
            rows.append(dict(
                sample_id=sid, signal_date=date, entity_id=f"e{i}",
                p_A=0.50 - 0.001 * i, p_B=0.05, p_C=0.20, p_D=0.10,
                p_b10_cal=float(p_b10), p_silent_cal=float(p_silent),
                p_hit15_cal=float(p_hit[i] if hasattr(p_hit, "__getitem__") else p_hit),
                silent_cooldown=bool(blocked[i] if hasattr(blocked, "__getitem__") else blocked),
            ))
            labels.append(dict(
                sample_id=sid,
                target="D" if (dd[i] if hasattr(dd, "__getitem__") else dd) else ("A" if (hit[i] if hasattr(hit, "__getitem__") else hit) else "C"),
                hit15=bool(hit[i] if hasattr(hit, "__getitem__") else hit),
                silent=bool(silent[i] if hasattr(silent, "__getitem__") else silent),
            ))
    return pd.DataFrame(rows), pd.DataFrame(labels)


def test_failed_online_transfer_keeps_pi0_on_specified_rise():
    # Dream: cooldown drops the two non-rises (blocked), remaining all hit +15% with no -10%.
    dream_hit = [True] * 18 + [False] * 2
    dream_block = [False] * 18 + [True] * 2
    dream, dlab = _book("d", ["d1", "d2"], 20, hit=dream_hit, silent=dream_block,
                        p_hit=[0.80] * 18 + [0.10] * 2, blocked=dream_block)
    # Outer: blocking those two removes the only +15% hits, so cooldown does not transfer.
    online_hit = [False] * 18 + [True] * 2
    online_block = [False] * 18 + [True] * 2
    online, olab = _book("o", ["o1", "o2"], 20, hit=online_hit, silent=False,
                         p_hit=[0.20] * 18 + [0.90] * 2, blocked=online_block)
    report = replay_screen(dream, dlab, online, olab, extra_silence_taus=())
    assert report["n_dream_improvers"] >= 1
    assert report["champion_policy"] != report["pi0"]
    assert report["online_transfer_ok"] is False
    assert report["recommended_policy"] == report["pi0"]
    assert report["formal_training_authorized"] is False
    assert report["production_eligible"] is False
