"""H07 executor: locked development evaluation with execution and capacity stress.

The candidate and the matched-coverage control are declared here, before the
evaluation runs. Positions occupy a fixed 1/n_cap notional for the full hold, so
the book can never carry hidden leverage, and selections that do not fit the
capacity cap are rejected and reported rather than silently dropped.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .abcds_model import TARGET_ID
from .policy_support import evaluate_selection, gate_threshold
from .runtime import atomic_json, now
from .splits import assign_segments

FROZEN_AT = "2024-05-01T00:00:00+08:00"
HORIZON_DAYS = 20

# Declared before evaluation. Candidate is the primary regularisation fixed in
# H04; the control is the zero-fit frequency reference at the same gate and cap.
CANDIDATE_CONFIG = "abcds_multinomial_C1.0"
CONTROL_CONFIG = "abcds_frequency_control"
PRIMARY_CAP = 20
CAPACITY_SCENARIOS = (20, 10, 5)
COST_MULTIPLIERS = (1.0, 2.0, 3.0)
BLOCK_SIGNAL_DATES = 5
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260926

ACCEPTANCE_GAPS = (
    "H01 data admission is FAILED_VALIDITY; execution inherits the diagnostic chain",
    "realized net comes from saved labels, not from a re-run order/fill engine",
    "no partial fills, no queue position, no liquidity limit model",
    "capacity is modelled as a concurrency cap only; no cash or margin constraint",
    "single split; paired intervals are diagnostic, not promotion evidence",
)


def trading_calendar(root: Path) -> list[str]:
    directory = root / "output/tushare_cache/daily"
    if not directory.is_dir():
        raise ValueError("daily cache directory is required for the trading calendar")
    days = sorted(path.stem for path in directory.glob("*.parquet")
                  if len(path.stem) == 8 and path.stem.isdigit())
    if not days:
        raise ValueError("empty trading calendar")
    return days


def _plus_days(calendar: list[str], start: str, offset: int) -> str | None:
    try:
        position = calendar.index(start)
    except ValueError:
        return None
    target = position + offset
    return calendar[target] if 0 <= target < len(calendar) else None


def _simulate(episodes: pd.DataFrame, cap: int, calendar: list[str],
              cost_multiplier: float = 1.0) -> tuple[pd.DataFrame, dict]:
    """Concurrency-capped book; returns the accepted episodes and the NAV series."""
    round_trip_cost = 0.001 + 0.0015
    accepted, rejected = [], 0
    open_positions: list[tuple[str, str]] = []   # (exit_date, sample_id)
    ordered = episodes.sort_values(["signal_date", "score"], ascending=[True, False])
    for row in ordered.itertuples(index=False):
        open_positions = [item for item in open_positions if item[0] > row.entry_date]
        if len(open_positions) >= cap:
            rejected += 1
            continue
        open_positions.append((row.exit_date, row.sample_id))
        accepted.append(row)
    frame = pd.DataFrame(accepted, columns=list(episodes.columns))
    if frame.empty:
        return frame, {"nav": pd.DataFrame(columns=["trade_date", "nav"]),
                       "rejected_by_capacity": rejected, "final_nav": 1.0,
                       "mean_net": None, "positions": 0}
    extra_cost = (cost_multiplier - 1.0) * round_trip_cost
    frame = frame.copy()
    frame["stressed_net"] = pd.to_numeric(frame.net, errors="coerce") - extra_cost
    frame["pnl"] = frame["stressed_net"] / cap
    exits = frame.groupby("exit_date")["pnl"].sum()
    index = [day for day in calendar
             if frame.entry_date.min() <= day <= frame.exit_date.max()]
    nav, level = [], 1.0
    for day in index:
        level += float(exits.get(day, 0.0))
        nav.append({"trade_date": day, "nav": level, "open_positions": int(
            ((frame.entry_date <= day) & (frame.exit_date > day)).sum())})
    return frame, {"nav": pd.DataFrame(nav), "rejected_by_capacity": rejected,
                   "final_nav": level, "mean_net": float(frame.stressed_net.mean()),
                   "positions": int(len(frame))}


def _block_bootstrap(dates: np.ndarray, values: np.ndarray, *, seed: int,
                     draws: int) -> dict:
    """Moving-block bootstrap over ordered signal dates."""
    unique = np.array(sorted(set(dates.tolist())))
    if len(unique) < 2:
        return {"estimate": float(np.mean(values)) if len(values) else None,
                "lower": None, "upper": None, "blocks": 0, "draws": 0}
    blocks = [unique[i:i + BLOCK_SIGNAL_DATES] for i in range(0, len(unique), BLOCK_SIGNAL_DATES)]
    grouped = {day: values[dates == day] for day in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        drawn = []
        while len(drawn) < len(unique):
            drawn.extend(list(blocks[rng.integers(len(blocks))]))
        drawn = drawn[:len(unique)]
        pooled = np.concatenate([grouped[day] for day in drawn if len(grouped[day])])
        samples.append(float(np.mean(pooled)) if len(pooled) else np.nan)
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {"estimate": float(np.mean(values)), "lower": None, "upper": None,
                "blocks": len(blocks), "draws": 0}
    return {"estimate": float(np.mean(values)),
            "lower": float(np.quantile(samples, 0.025)),
            "upper": float(np.quantile(samples, 0.975)),
            "blocks": len(blocks), "draws": int(samples.size),
            "block_signal_dates": BLOCK_SIGNAL_DATES}


def build(root: Path, directory: Path, *, upstream_directories: dict[str, Path]) -> dict:
    root, directory = Path(root).resolve(), Path(directory).resolve()
    upstream = {name: Path(path) for name, path in upstream_directories.items()}
    for required in ("H02", "H03", "H04", "H06"):
        if required not in upstream:
            raise ValueError("H07 requires upstream outputs: " + required)
    labels = pd.read_parquet(upstream["H02"] / "labels.parquet")
    split_manifest = json.loads(
        (upstream["H03"] / "split_manifest.json").read_text(encoding="utf-8"))
    boundaries = split_manifest["segments"]
    h04 = pd.read_parquet(upstream["H04"] / "oof_predictions.parquet")
    activation = json.loads(
        (upstream["H06"] / "advanced_activation_decisions.json").read_text(encoding="utf-8"))
    if activation.get("activated_families"):
        raise ValueError("H07 refuses to evaluate against unauthorised advanced families")

    source = root / "output/experiments/s20_safe_v4/sources/v4-paired-campaign"
    samples = pd.read_parquet(source / "samples.parquet")
    resolved = labels[labels.target.notna()].copy()
    samples = samples[samples.sample_id.isin(set(resolved.sample_id))].reset_index(drop=True)
    assignment, split = assign_segments(samples, boundaries)
    if split["split_protocol_sha256"] != split_manifest["split_protocol_sha256"]:
        raise ValueError("H07 rebuilt a different split protocol")
    calibration_ids = assignment.loc[
        assignment.segment.eq("calibration"), "sample_id"].tolist()
    calendar = trading_calendar(root)

    risk_rows = h04[h04.config_id.eq("independent_risk_head")]
    risk_head = pd.Series(pd.to_numeric(risk_rows["risk"], errors="coerce").to_numpy(),
                          index=risk_rows["sample_id"].to_numpy())
    risk_cut = gate_threshold(risk_head.reindex(calibration_ids).dropna().to_numpy(), 0.5)
    identity = samples[["sample_id", "entity_id", "signal_date", "prediction_at"]]
    realized = resolved.set_index("sample_id")
    evaluation_segments = {"selection-policy", "outer-test"}

    book = {}
    for role, config_id in (("candidate", CANDIDATE_CONFIG), ("control", CONTROL_CONFIG)):
        rows = h04[h04.config_id.eq(config_id)]
        if rows.empty:
            raise ValueError("missing H04 predictions for " + config_id)
        score = pd.Series(pd.to_numeric(rows["score"], errors="coerce").to_numpy(),
                          index=rows["sample_id"].to_numpy())
        selected, report = evaluate_selection(
            identity, score, risk_head, calendar,
            policy_id=f"{config_id}::risk_gated_top20", use_gate=True, n_cap=PRIMARY_CAP,
            frozen_at=FROZEN_AT, risk_cut=risk_cut)
        frame = pd.DataFrame({"sample_id": selected})
        identity_by_id = identity.set_index("sample_id")
        frame["entity_id"] = identity_by_id.entity_id.reindex(frame.sample_id).to_numpy()
        frame["signal_date"] = identity.set_index("sample_id").signal_date.reindex(
            frame.sample_id).to_numpy()
        frame["segment"] = assignment.set_index("sample_id").segment.reindex(
            frame.sample_id).to_numpy()
        frame = frame[frame.segment.isin(evaluation_segments)].copy()
        frame["score"] = score.reindex(frame.sample_id).to_numpy()
        frame["risk"] = risk_head.reindex(frame.sample_id).to_numpy()
        frame["target"] = realized.target.reindex(frame.sample_id).to_numpy()
        frame["net"] = pd.to_numeric(realized.net.reindex(frame.sample_id),
                                     errors="coerce").to_numpy()
        frame["entry_date"] = [_plus_days(calendar, day, 1) for day in frame.signal_date]
        frame["exit_date"] = [_plus_days(calendar, day, HORIZON_DAYS) for day in frame.signal_date]
        frame = frame.dropna(subset=["entry_date", "exit_date", "net"])
        frame["role"] = role
        book[role] = frame

    episodes = pd.concat(book.values(), ignore_index=True)
    episodes["holding_trading_days"] = HORIZON_DAYS
    episodes = episodes[["role", "sample_id", "entity_id", "signal_date", "entry_date",
                         "exit_date", "holding_trading_days", "score", "risk", "target",
                         "net"]]
    stress_rows, nav_rows = [], []
    for role, frame in book.items():
        for cap in CAPACITY_SCENARIOS:
            for multiplier in COST_MULTIPLIERS:
                accepted, result = _simulate(frame, cap, calendar, multiplier)
                stress_rows.append(dict(
                    role=role, capacity=cap, cost_multiplier=multiplier,
                    positions=result["positions"],
                    rejected_by_capacity=result["rejected_by_capacity"],
                    final_nav=result["final_nav"], mean_net=result["mean_net"]))
                if cap == PRIMARY_CAP and multiplier == 1.0:
                    series = result["nav"].copy()
                    series["role"] = role
                    nav_rows.append(series)
    nav = pd.concat(nav_rows, ignore_index=True) if nav_rows else pd.DataFrame(
        columns=["trade_date", "nav", "open_positions", "role"])

    paired = {}
    candidate, control = book["candidate"], book["control"]
    metrics = (
        ("realized_net", lambda frame: pd.to_numeric(frame.net, errors="coerce")),
        ("safe_target_rate", lambda frame: frame.target.eq("A")),
        ("paired_risk_rate", lambda frame: frame.target.isin(["B", "D"])),
    )
    for label, measure in metrics:
        left = candidate.assign(value=measure(candidate).to_numpy())
        right = control.assign(value=measure(control).to_numpy())
        per_day_left = left.groupby("signal_date")["value"].mean()
        per_day_right = right.groupby("signal_date")["value"].mean()
        joined = pd.concat([per_day_left.rename("candidate"),
                            per_day_right.rename("control")], axis=1).dropna()
        if joined.empty:
            paired[label] = {"paired_dates": 0,
                             "candidate_mean": None, "control_mean": None,
                             "difference": None}
            continue
        difference = (joined.candidate - joined.control).to_numpy(dtype=float)
        paired[label] = {
            "paired_dates": int(len(joined)),
            "candidate_mean": float(joined.candidate.mean()),
            "control_mean": float(joined.control.mean()),
            "difference": float(difference.mean()),
            "difference_interval": _block_bootstrap(
                joined.index.to_numpy(), difference, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS),
            "method": "paired same-signal-date difference, moving-block bootstrap",
        }

    directory.mkdir(parents=True, exist_ok=True)
    for name in ("locked_development_report.json", "stress.csv", "episodes.parquet",
                 "nav.parquet", "paired_intervals.json"):
        if (directory / name).exists():
            raise ValueError("H07 refuses existing outputs: " + name)
    episodes.to_parquet(directory / "episodes.parquet", index=False)
    nav.to_parquet(directory / "nav.parquet", index=False)
    pd.DataFrame(stress_rows).to_csv(directory / "stress.csv", index=False)
    atomic_json(directory / "paired_intervals.json", {
        "at": now(), "target_id": TARGET_ID, "candidate": CANDIDATE_CONFIG,
        "control": CONTROL_CONFIG, "risk_gate_threshold": risk_cut,
        "capacity_scenarios": list(CAPACITY_SCENARIOS),
        "cost_multipliers": list(COST_MULTIPLIERS),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                      "block_signal_dates": BLOCK_SIGNAL_DATES},
        "paired": paired, "formal_H07_accepted": False,
        "production_eligible": False, "acceptance_gaps": list(ACCEPTANCE_GAPS)})
    primary = [row for row in stress_rows
               if row["capacity"] == PRIMARY_CAP and row["cost_multiplier"] == 1.0]
    atomic_json(directory / "locked_development_report.json", {
        "at": now(), "target_id": TARGET_ID,
        "candidate": CANDIDATE_CONFIG, "control": CONTROL_CONFIG,
        "primary_capacity": PRIMARY_CAP, "positions": int(len(episodes)),
        "realized_safe_target_rate": float(candidate.target.eq("A").mean()) if len(candidate) else None,
        "realized_paired_risk_rate": float(candidate.target.isin(["B", "D"]).mean())
        if len(candidate) else None,
        "control_safe_target_rate": float(control.target.eq("A").mean()) if len(control) else None,
        "control_paired_risk_rate": float(control.target.isin(["B", "D"]).mean())
        if len(control) else None,
        "primary_nav": primary, "paired_intervals": paired,
        "formal_H07_accepted": False, "formal_training_authorized": False,
        "production_eligible": False, "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "note": "diagnostic locked evaluation; not a promotion result"})
    return {
        "formal_gate_passed": False,
        "formal_H07_accepted": False,
        "acceptance_gaps": list(ACCEPTANCE_GAPS),
        "candidate": CANDIDATE_CONFIG, "control": CONTROL_CONFIG,
        "risk_gate_threshold": risk_cut,
        "episodes": int(len(episodes)),
        "primary_nav": primary,
        "paired": paired,
        "validation_scope": "single-split diagnostic execution model; not formal H07",
    }
