#!/usr/bin/env python
"""Score one stock on one signal day with S20-Pure v1 and every layer in the evidence ledger.

  python scripts/score_s20_stock.py 688010.SH [--date YYYYMMDD]   (default: the latest day in the factor store)

Inputs (refresh first: update_daily_caches.py, update_regimes_tushare.py,
build_s20_20r_confirmation_factors.py --output-dir output/experiments/s20_pure_v1_shadow/factor_groups):
  frozen stage1 + funnel (config/s20_pure_v1.json), valve (config/s20_pure_valve_v1.json),
  style gate S0010, HS300 phase S0007, unified v2 reserved model (S0007 down arm),
  pump v3c S0014 (output/pump_history, last available day), risk flags S0016 (15-minute last hour,
  SEMAS F01 and F20 with the pinned evaluator given by --semas).
Signal-day information only; nothing after the signal day is read. Not investment advice.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_s20_pure_v1_shadow import SHADOW, daily_natr, load_factors, valve_table  # noqa: E402
from stockagent_analysis.market_valve import LEVEL_CN  # noqa: E402
from stockagent_analysis.s20_pure import PureConfig, _frozen_models, stage1_probability  # noqa: E402

DEFAULT_SEMAS = Path(os.environ.get("SEMAS_PINNED", "C:/Users/apple/AppData/Local/Temp/claude/C--aicoding-stockagent-analysis/"
                                    "6c024155-1675-4f3b-b739-174ba79b9171/scratchpad/semas_b50aa06"))
F01 = "cs_zscore(div(sub(greater(eps, winsorize(total_mv)), sub(ocfps, turnover_rate)), winsorize(total_mv)))"
F20 = "ts_skew(ts_min_max_scale(return, 3), 5)"


def index_close(code: str) -> pd.Series:
    d = pd.read_parquet(ROOT / "output/tushare_cache/index_daily" / f"{code}.parquet", columns=["trade_date", "close"])
    return d.assign(trade_date=d.trade_date.astype(str)).set_index("trade_date").close.sort_index()


def v2_rank(day: str, code: str) -> tuple[float, int, int]:
    from refit_s20_unified_v2_reserved import BLOCK_START, OUT as V2R, load_unlabelled  # noqa: E402
    from s20_frames import MARKET_FEATURES  # noqa: E402
    from train_s20_unified_v1 import scale_free  # noqa: E402
    from train_s20_unified_v2 import day_ranks  # noqa: E402
    features = json.loads((ROOT / "output/production/s20_pure_v1/features.json").read_text(encoding="utf-8"))
    f = load_unlabelled(SHADOW / "factor_groups", day, features)
    f = f[f.trade_date == day].copy()
    kept = [x for x in scale_free(f, features) if x not in MARKET_FEATURES]
    day_ranks(f, [x for x in kept if x != "industry_id"])
    booster = lgb.Booster(model_file=str(V2R / f"unified_v2_{BLOCK_START}.txt"))
    f["v2"] = booster.predict(f[booster.feature_name()])
    f["rk"] = f.v2.rank(ascending=False, method="first")
    row = f[f.ts_code == code]
    return (float(row.v2.iloc[0]), int(row.rk.iloc[0]), len(f)) if len(row) else (np.nan, -1, len(f))


def last_hour(codes: list[str], day: str) -> pd.Series:
    import tushare as ts
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
    d = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    out = {}
    for c in codes:
        b = pro.stk_mins(ts_code=c, freq="15min", start_date=f"{d} 09:00:00", end_date=f"{d} 15:00:00")
        if b is not None and len(b):
            b = b.set_index(b.trade_time.str[11:16]).close
            if "14:00" in b and "15:00" in b:
                out[c] = b["15:00"] / b["14:00"] - 1
        time.sleep(0.12)
    return pd.Series(out, dtype=float)


def semas_factors(day: str, semas: Path) -> pd.DataFrame:
    sys.path.insert(0, str(semas))
    from china_a_share_alpha.factor import parse_expression  # noqa: E402
    import evaluate_semas_factors as E  # noqa: E402
    E.START, E.END = "20260101", day
    data = E.panel()
    out = pd.DataFrame({"F01": parse_expression(F01).eval(data), "F20": parse_expression(F20).eval(data)}).reset_index()
    out["trade_date"] = out.date.dt.strftime("%Y%m%d")
    return out[out.trade_date == day].rename(columns={"symbol": "ts_code"})[["ts_code", "F01", "F20"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("code")
    ap.add_argument("--date")
    ap.add_argument("--semas", type=Path, default=DEFAULT_SEMAS)
    a = ap.parse_args()
    code = a.code if "." in a.code else (a.code + (".SH" if a.code.startswith(("6", "9")) else ".SZ"))
    cfg = PureConfig()
    _, _, features = _frozen_models()
    f = load_factors(SHADOW / "factor_groups", features, "20260901")
    day = a.date or f.trade_date.max()
    f = f[f.trade_date == day].copy()
    f["stage1_probability"] = stage1_probability(f)
    f = f.merge(daily_natr("20260901"), on=["ts_code", "trade_date"], how="left")
    f["pool_rank"] = f.stage1_probability.rank(ascending=False, method="first")
    pool = f[f.pool_rank <= cfg.pool_size].copy()
    pool["natr_pct_in_pool"] = pool.natr14.rank(pct=True)
    surv = pool[pool.natr_pct_in_pool.isna() | (pool.natr_pct_in_pool <= 0.60)].copy()
    surv["list_rank"] = surv.stage1_probability.rank(ascending=False, method="first")
    me = f[f.ts_code == code]
    if me.empty:
        print(f"{code} not in the factor store on {day} (suspended, ST, or too little history)")
        return 1
    me = me.iloc[0]
    name = me.get("name", "")
    lines = [f"# {code} {name}  signal day {day}  (scored after the close; entry would be the next open)", "",
             "## S20-Pure v1 (frozen contract)"]
    lines.append(f"- stage1 score {me.stage1_probability:.4f}; market rank {int(me.pool_rank)} of {len(f)} "
                 f"(top {100 * me.pool_rank / len(f):.1f}%)")
    in_pool = code in set(pool.ts_code)
    in_surv = code in set(surv.ts_code)
    lr = int(surv.loc[surv.ts_code == code, "list_rank"].iloc[0]) if in_surv else None
    lines.append(f"- top-{cfg.pool_size} pool: {'yes' if in_pool else 'no'}"
                 + (f"; NATR(14) {me.natr14:.4f}, percentile in pool {pool.loc[pool.ts_code == code, 'natr_pct_in_pool'].iloc[0]:.2f} "
                    f"({'kept' if in_surv else 'cut: among the 40% most volatile'})" if in_pool else ""))
    lines.append(f"- final list (top {cfg.top_k} of the survivors): " + (f"YES, rank {lr}" if lr and lr <= cfg.top_k else
                 (f"no, survivor rank {lr} (needs <= {cfg.top_k})" if lr else "no")))
    cut20 = surv[surv.list_rank == cfg.top_k].stage1_probability
    if len(cut20):
        lines.append(f"- score of the 20th name today {cut20.iloc[0]:.4f}; gap {me.stage1_probability - cut20.iloc[0]:+.4f}")

    lines += ["", "## Market layers on the day"]
    hs, cy = index_close("000300.SH"), index_close("399006.SZ")
    r20h, r60h = hs.loc[day] / hs.shift(20).loc[day] - 1, hs.loc[day] / hs.shift(60).loc[day] - 1
    r20c = cy.loc[day] / cy.shift(20).loc[day] - 1
    gate_open = r20c > r20h
    lines.append(f"- S0010 style gate (primary shadow): ChiNext 20d {100 * r20c:+.2f}% vs CSI300 {100 * r20h:+.2f}% -> "
                 + ("OPEN (build new positions)" if gate_open else "CLOSED (no new positions)"))
    phase = "up" if (r20h > 0 and r60h > 0) else ("down" if (r20h < 0 and r60h < 0) else "turn")
    lines.append(f"- S0007 phase: CSI300 20d {100 * r20h:+.2f}%, 60d {100 * r60h:+.2f}% -> {phase} "
                 + {"up": "(use the frozen list)", "down": "(use the v2 list)", "turn": "(no new positions)"}[phase])
    try:
        vt = valve_table().set_index("date")
        lv = vt.loc[day, "level"] if day in vt.index else None
        lines.append(f"- valve: {LEVEL_CN.get(lv, lv) if lv else 'n/a'}" + (" -> action B: red day swaps to the safe list" if lv == "red" else ""))
    except Exception as err:  # noqa: BLE001
        lines.append(f"- valve: n/a ({err})")
    try:
        v, rk, n = v2_rank(day, code)
        lines.append(f"- unified v2 (S0007 down arm): score {v:.4f}, market rank {rk} of {n}")
    except Exception as err:  # noqa: BLE001
        lines.append(f"- unified v2: n/a ({err})")

    lines += ["", "## Stock-level shadows"]
    pf = sorted((ROOT / "output/pump_history").glob("pump_scores_*.parquet"))
    if pf:
        p = pd.concat([pd.read_parquet(x) for x in pf])
        p = p[p.usable & (p.ts_code == code)].sort_values("trade_date")
        if len(p):
            r = p.iloc[-1]
            note = "" if r.trade_date == day else f"  (latest available {r.trade_date}; the V12 feature pipeline runs on the production machine)"
            lines.append(f"- S0014 pump v3c: P(up) {r.pump_score:.3f}, P(down) {r.pump_down_score:.3f}, ratio {r.ratio:.2f}{note}")
    if in_surv:
        flags, detail = 0, []
        try:
            lh = last_hour(list(surv.ts_code), day)
            pct = lh.rank(pct=True).get(code, np.nan)
            hit = bool(pct > 0.8)
            flags += hit
            detail.append(f"last hour {100 * lh.get(code, np.nan):+.2f}% (pct {pct:.2f}{', FLAG' if hit else ''})")
        except Exception as err:  # noqa: BLE001
            detail.append(f"last hour n/a ({err})")
        try:
            sf = semas_factors(day, a.semas)
            sf = sf[sf.ts_code.isin(surv.ts_code)]
            for k, rule in (("F01", lambda q: q > 2 / 3), ("F20", lambda q: q <= 1 / 3)):
                q = sf.set_index("ts_code")[k].rank(pct=True).get(code, np.nan)
                hit = bool(rule(q))
                flags += hit
                detail.append(f"{k} pct {q:.2f}{', FLAG' if hit else ''}")
        except Exception as err:  # noqa: BLE001
            detail.append(f"F01/F20 n/a ({err})")
        lines.append(f"- S0016 risk flags among today's survivors: " + "; ".join(detail)
                     + f" -> {flags} flag(s): " + ("HALF SIZE, buy last" if flags >= 2 else "normal size"))
    else:
        lines.append("- S0016 risk flags: not computed (not a funnel survivor today)")
    lines += ["", "Shadow rules are being verified on the reserved window (first read ~2026-12); only the frozen contract is live."]
    text = "\n".join(lines)
    out = ROOT / "output/experiments/stock_scores" / f"{code}_{day}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
