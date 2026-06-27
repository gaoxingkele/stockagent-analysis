# -*- coding: utf-8 -*-
"""BRAIN Phase A 正交族侦察 — option波动率 + model16评分, 各挖最佳单信号 (做组合积木).

目标: 找与 analyst 修正动量(0.93)真正交的强单信号。组合(Phase B)能否过 1.25 门槛, 取决于
积木是否 ①各自有 standalone 信号 ②互相低相关。本批侦察 standalone 强度, Phase B 再测相关+组合。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wq_auth import API, load_session, safe_req
from wq_mine2 import BASE, submit, CONCURRENCY

OUT = Path(__file__).resolve().parent / "mineA_results.json"

# (name, expr, decay, neut)
CANDIDATES = [
    # option8 波动率: 低波异象 / 波动风险溢价 / 波动下行
    ("lowvol_60",     "rank(-historical_volatility_60)", 6, "SUBINDUSTRY"),
    ("lowvol_120",    "rank(-historical_volatility_120)", 6, "SUBINDUSTRY"),
    ("volprem_60",    "rank(implied_volatility_call_60 - historical_volatility_60)", 6, "SUBINDUSTRY"),
    ("vol_falling",   "rank(-ts_delta(historical_volatility_60, 20))", 6, "SUBINDUSTRY"),
    # model16 预制 composite 评分 (质量/价值/盈利/成长/现金流/确定性)
    ("m16_quality",   "rank(fscore_bfl_quality)", 4, "SUBINDUSTRY"),
    ("m16_value",     "rank(fscore_bfl_value)", 4, "SUBINDUSTRY"),
    ("m16_profit",    "rank(fscore_bfl_profitability)", 4, "SUBINDUSTRY"),
    ("m16_growth",    "rank(fscore_bfl_growth)", 4, "SUBINDUSTRY"),
    ("m16_total",     "rank(fscore_bfl_total)", 4, "SUBINDUSTRY"),
    ("m16_surface",   "rank(fscore_bfl_surface)", 4, "SUBINDUSTRY"),
    ("m16_earncert",  "rank(earnings_certainty_rank_derivative)", 4, "SUBINDUSTRY"),
    ("m16_cashflow",  "rank(cashflow_efficiency_rank_derivative)", 4, "SUBINDUSTRY"),
    # fundamental6 杠杆 (低杠杆质量)
    ("low_leverage",  "rank(-debt / (assets + 1))", 6, "SUBINDUSTRY"),
]


def run(candidates, out_file, title):
    print(f"=== {title} ({len(candidates)} 候选, 滚动调度) ===\n", flush=True)
    s = load_session()
    queue, active, results = list(candidates), [], []
    while queue or active:
        while queue and len(active) < CONCURRENCY:
            name, expr, decay, neut = queue.pop(0)
            loc = submit(s, expr, decay, neut)
            if loc and loc.startswith("http"):
                active.append({"name": name, "expr": expr, "decay": decay, "neut": neut, "loc": loc})
                print(f"  → {name} ({len(active)}飞/{len(queue)}待)", flush=True)
            else:
                results.append({"name": name, "expr": expr, "is": {}, "n_fail": 9, "status": str(loc)})
                print(f"  ✗ {name} 提交失败 {loc}", flush=True)
        for it in list(active):
            r = safe_req(s, "GET", it["loc"])
            if not r.headers.get("Retry-After"):
                aid = r.json().get("alpha")
                ism = safe_req(s, "GET", f"{API}/alphas/{aid}").json().get("is", {}) if aid else {}
                nf = sum(1 for c in ism.get("checks", []) if c.get("result") == "FAIL")
                results.append({"name": it["name"], "expr": it["expr"], "decay": it["decay"],
                                "neut": it["neut"], "alpha_id": aid, "is": ism, "n_fail": nf})
                m = ism
                print(f"  ✓ {it['name']:14s} Sharpe={m.get('sharpe')} fit={m.get('fitness')} "
                      f"turn={m.get('turnover')} fail={nf}", flush=True)
                active.remove(it)
        if active:
            time.sleep(6)
    ok = sorted([r for r in results if r.get("is")],
                key=lambda r: -(r["is"].get("sharpe") or -9))
    print("\n[排行] Sharpe高→低:", flush=True)
    for r in ok:
        m = r["is"]
        print(f"    {r['name']:14s} Sharpe={str(m.get('sharpe')):>6s} fit={str(m.get('fitness')):>5s} "
              f"turn={str(m.get('turnover')):>6s} fail={r.get('n_fail')}", flush=True)
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[out] -> {out_file.name}", flush=True)
    return results


if __name__ == "__main__":
    run(CANDIDATES, OUT, "Phase A 正交族侦察")
