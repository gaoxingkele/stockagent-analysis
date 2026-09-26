"""Freeze a development-only risk-weight sweep, then evaluate retrospective data."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_s20_v3 import OUT, metric_rows, block_delta_ci, save_json, top_rows


def add_scores(frame):
    scores = {"old_s20":"old_s20", "old_r20":"old_r20"}
    for mode in ["portable_full","lowcorr24"]:
        good = frame[mode+"_up"]
        down = 1 + good - frame[mode]/50
        for weight in [0,.25,.5,.75,1]:
            name = f"{mode}_risk{weight:g}"
            frame[name] = 100*(weight+good-weight*down)/(1+weight)
            scores[name] = name
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze","evaluate"])
    args = parser.parse_args()
    frozen_path = OUT / "risk_weight_selection.json"
    if args.stage == "freeze":
        if frozen_path.exists():
            raise FileExistsError("development selection already frozen")
        data = pd.concat([pd.read_parquet(OUT / f"comparison_wf{i}.parquet") for i in [1,2,3]],ignore_index=True)
        scores = add_scores(data)
        metrics = metric_rows(data,scores,"development_all")
        selected = sorted([r for r in metrics if r["k"]==20 and not r["candidate"].startswith("old")],
                          key=lambda r:(r["utility"],r["immediate_rate"]),reverse=True)[0]
        mode, weight = selected["candidate"].split("_risk")
        report = dict(stage="development_only_secondary_exploration",mode=mode,risk_weight=float(weight),
                      candidate=selected["candidate"],selection_criterion="maximum Top20 immediate-minus-down utility on development",
                      diagnostic_read_for_selection=False, grid=[0,.25,.5,.75,1],
                      metrics=[r for r in metrics if r["k"]==20],
                      warning="secondary development search after primary equal-weight underperformance; not a preregistered primary result")
        save_json(frozen_path,report)
        print(json.dumps(report,indent=2))
        return
    frozen = json.loads(frozen_path.read_text())
    data = pd.read_parquet(OUT / "comparison_diagnostic2026.parquet")
    scores = add_scores(data)
    candidate = frozen["candidate"]
    chosen_scores = {c:scores[c] for c in ["old_s20","old_r20",candidate]}
    rows = metric_rows(data,chosen_scores,"diagnostic2026")
    for month,g in data.groupby(data.trade_date.str[:6]):
        rows.extend(metric_rows(g,chosen_scores,"month_"+month))
    pd.DataFrame(rows).to_csv(OUT / "balanced_metrics.csv",index=False)
    top = top_rows(data,candidate)
    labels = pd.read_parquet(OUT / "labels.parquet",columns=["ts_code","trade_date","reason","max_gain20","window_mae20"])
    top.merge(labels,on=["ts_code","trade_date"]).to_csv(OUT / "balanced_top20.csv",index=False,encoding="utf-8-sig")
    report = dict(candidate=candidate,mode=frozen["mode"],risk_weight=frozen["risk_weight"],
                  status="retrospective_only_no_independent_confirmation",
                  metrics=[r for r in rows if r["period"]=="diagnostic2026" and r["k"]==20],
                  up_delta_ci=block_delta_ci(data,candidate,"old_s20"),
                  down_delta_ci=block_delta_ci(data,candidate,"old_s20","down_risk"))
    save_json(OUT / "balanced_report.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
