# -*- coding: utf-8 -*-
"""BRAIN miner v3 (I5 窗口扫尾) — 纯 EPS 修正动量更长窗口, 验证单调趋势能否冲过 1.25 门槛.

v2 发现: rev_mom 窗口 40/60/90/120 → Sharpe 0.47/0.59/0.70/0.93 单调变强, 组合全减分。
本批: 纯 rank(ts_delta(eps_mean, W)) 扫 W=150/180/220/250/300, 看 Sharpe 是否继续涨过门槛,
还是单字段封顶 (回答: 修正动量到底能不能单独成 alpha)。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wq_auth import API, load_session
from wq_mine2 import BASE, submit, CONCURRENCY

OUT = Path(__file__).resolve().parent / "mine3_results.json"
EM = "anl4_afv4_eps_mean"
CANDIDATES = [
    (f"rev_mom_{w}_sub", f"rank(ts_delta({EM}, {w}))", d, "SUBINDUSTRY")
    for w, d in [(150, 15), (180, 18), (220, 20), (250, 20), (300, 20)]
]


def main():
    print("=== miner v3: 修正动量长窗口扫尾 (5 候选) ===\n", flush=True)
    s = load_session()
    queue, active, results = list(CANDIDATES), [], []
    while queue or active:
        while queue and len(active) < CONCURRENCY:
            name, expr, decay, neut = queue.pop(0)
            loc = submit(s, expr, decay, neut)
            if loc and loc.startswith("http"):
                active.append({"name": name, "expr": expr, "loc": loc})
                print(f"  → {name}", flush=True)
            else:
                results.append({"name": name, "is": {}, "n_fail": 9, "status": str(loc)})
        for it in list(active):
            r = s.get(it["loc"])
            if not r.headers.get("Retry-After"):
                aid = r.json().get("alpha")
                ism = s.get(f"{API}/alphas/{aid}").json().get("is", {}) if aid else {}
                nf = sum(1 for c in ism.get("checks", []) if c.get("result") == "FAIL")
                results.append({"name": it["name"], "expr": it["expr"], "alpha_id": aid, "is": ism, "n_fail": nf})
                print(f"  ✓ {it['name']:18s} Sharpe={ism.get('sharpe')} fit={ism.get('fitness')} "
                      f"turn={ism.get('turnover')} fail={nf}", flush=True)
                active.remove(it)
        if active:
            time.sleep(6)
    ok = sorted([r for r in results if r.get("is")], key=lambda r: -(r["is"].get("sharpe") or -9))
    print("\n[窗口扫描] (含 v2: 120→0.93):", flush=True)
    for r in ok:
        m = r["is"]
        print(f"    {r['name']:18s} Sharpe={str(m.get('sharpe')):>6s} fit={str(m.get('fitness')):>5s} fail={r.get('n_fail')}", flush=True)
    surv = [r for r in ok if r.get("n_fail", 9) == 0]
    print(f"\n[结果] {len(surv)}/{len(results)} 过全部 checks", flush=True)
    for r in surv:
        print(f"    ★ {r['name']}: {r['expr']} → alpha/{r.get('alpha_id')}", flush=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[out] -> {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
