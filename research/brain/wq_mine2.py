# -*- coding: utf-8 -*-
"""BRAIN miner v2 (I5 精化批) — analyst4 修正信号组合/中性化变体, 滚动调度冲 1.25 门槛.

v1 教训: ①裸单字段太弱(Sharpe~0.6); ②全提交后轮询 + 并发限~3 → 4/15 退避超时漏跑。
v2 改进:
  - **滚动调度**: 保持 ≤CONCURRENCY 个在飞, 完成即补下一个, 0 漏跑。
  - **精化候选**: 组合 z-score(修正动量+信念+覆盖) / 中性化变体(SUB/IND/SECTOR) / 修正加速度 /
    标准化惊喜(SUE) / winsorize。目标过 BRAIN 全 checks (Sharpe≥1.25 + fitness≥1.0 + turnover≤0.7)。

用法: python research/brain/wq_mine2.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wq_auth import API, load_session, safe_req

OUT = Path(__file__).resolve().parent / "mine2_results.json"
CONCURRENCY = 3

EM = "anl4_afv4_eps_mean"
EH, EL, EN = "anl4_afv4_eps_high", "anl4_afv4_eps_low", "anl4_afv4_eps_number"
EMED, EACT = "anl4_afv4_median_eps", "anl4_af_eps_value"
NISTD = "adj_net_income_stddev"
AEQ, ASQ = "actual_eps_value_quarterly", "actual_sales_value_quarterly"

# (name, expr, decay, neutralization)
CANDIDATES = [
    # 修正动量: 窗口 + 中性化变体
    ("rev_mom_40_sub",   f"rank(ts_delta({EM}, 40))", 8, "SUBINDUSTRY"),
    ("rev_mom_90_sub",   f"rank(ts_delta({EM}, 90))", 12, "SUBINDUSTRY"),
    ("rev_mom_120_sub",  f"rank(ts_delta({EM}, 120))", 15, "SUBINDUSTRY"),
    ("rev_mom_60_ind",   f"rank(ts_delta({EM}, 60))", 10, "INDUSTRY"),
    ("rev_mom_60_sector",f"rank(ts_delta({EM}, 60))", 10, "SECTOR"),
    ("rev_mom_60_z",     f"rank(ts_zscore(ts_delta({EM}, 60), 250))", 10, "SUBINDUSTRY"),
    ("rev_accel",        f"rank(ts_delta(ts_delta({EM}, 20), 20))", 8, "SUBINDUSTRY"),
    ("rev_mom_norm_z",   f"rank(ts_zscore(ts_delta({EM}, 60) / (abs({EM}) + 1), 250))", 10, "SUBINDUSTRY"),
    # 组合 z-score: 修正 + 信念(低离散) + 覆盖
    ("rev+conv",         f"rank(ts_zscore(ts_delta({EM},60),250) + (-ts_zscore({EH}-{EL},250)))", 10, "SUBINDUSTRY"),
    ("rev+cov",          f"rank(ts_zscore(ts_delta({EM},60),250) + ts_zscore(ts_delta({EN},60),250))", 10, "SUBINDUSTRY"),
    ("rev+conv+cov",     f"rank(ts_zscore(ts_delta({EM},60),250) + (-ts_zscore({EH}-{EL},250)) + ts_zscore(ts_delta({EN},60),250))", 10, "SUBINDUSTRY"),
    ("rev+skew",         f"rank(ts_zscore(ts_delta({EM},60),250) + ts_zscore({EMED}-{EM},250))", 10, "SUBINDUSTRY"),
    ("rev+sales",        f"rank(ts_zscore(ts_delta({EM},60),250) + ts_zscore(ts_delta({ASQ},60),250))", 10, "SUBINDUSTRY"),
    ("rev_over_disp_z",  f"rank(ts_zscore(ts_delta({EM},60) / ({EH}-{EL}+1), 250))", 10, "SUBINDUSTRY"),
    # winsorize 组合 (抗离群)
    ("combo_winsor",     f"rank(winsorize(ts_zscore(ts_delta({EM},60),250),std=4) + winsorize(-ts_zscore({EH}-{EL},250),std=4) + winsorize(ts_zscore(ts_delta({EN},60),250),std=4))", 10, "SUBINDUSTRY"),
    # 标准化惊喜 SUE
    ("sue_disp",         f"rank(({EACT}-{EM}) / ({EH}-{EL}+1))", 5, "SUBINDUSTRY"),
    ("sue_std",          f"rank(({AEQ}-{EM}) / ({NISTD}+1))", 5, "SUBINDUSTRY"),
    ("rev+conv_ind",     f"rank(ts_zscore(ts_delta({EM},60),250) + (-ts_zscore({EH}-{EL},250)))", 10, "INDUSTRY"),
]

BASE = {
    "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000",
    "delay": 1, "decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08,
    "pasteurization": "ON", "unitHandling": "VERIFY", "nanHandling": "OFF",
    "language": "FASTEXPR", "visualization": False,
}


def submit(s, expr, decay, neut, retries=8):
    cfg = {**BASE, "decay": decay, "neutralization": neut}
    for i in range(retries):
        r = safe_req(s, "POST", f"{API}/simulations", json={"type": "REGULAR", "settings": cfg, "regular": expr})
        if r.status_code in (200, 201, 202):
            return r.headers.get("Location")
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 6)) + 2 * i)
            continue
        return f"ERR:{r.status_code}:{r.text[:100]}"
    return None


def main():
    print("=== BRAIN miner v2: analyst4 修正精化批 (18 候选, 滚动调度) ===\n", flush=True)
    s = load_session()
    queue = list(CANDIDATES)
    active = []   # [{name,expr,decay,neut,loc}]
    results = []

    def record_done(item, body):
        aid = body.get("alpha")
        if not aid:
            results.append({**item, "status": body.get("status"), "is": {}, "n_fail": 9})
            print(f"  ✗ {item['name']:18s} status={body.get('status')}", flush=True)
            return
        ism = s.get(f"{API}/alphas/{aid}").json().get("is", {})
        checks = ism.get("checks", [])
        nf = sum(1 for c in checks if c.get("result") == "FAIL")
        results.append({**item, "alpha_id": aid, "is": ism, "checks": checks, "n_fail": nf})
        print(f"  ✓ {item['name']:18s} Sharpe={ism.get('sharpe')} fit={ism.get('fitness')} "
              f"turn={ism.get('turnover')} fail={nf}", flush=True)

    while queue or active:
        while queue and len(active) < CONCURRENCY:
            name, expr, decay, neut = queue.pop(0)
            loc = submit(s, expr, decay, neut)
            if loc and loc.startswith("http"):
                active.append({"name": name, "expr": expr, "decay": decay, "neut": neut, "loc": loc})
                print(f"  → 提交 {name} ({len(active)} 在飞, {len(queue)} 待)", flush=True)
            else:
                results.append({"name": name, "expr": expr, "decay": decay, "neut": neut,
                                "status": str(loc), "is": {}, "n_fail": 9})
                print(f"  ✗ {name:18s} 提交失败 {loc}", flush=True)
        for item in list(active):
            r = s.get(item["loc"])
            if not r.headers.get("Retry-After"):
                record_done(item, r.json())
                active.remove(item)
        if active:
            time.sleep(6)

    ok = [r for r in results if r.get("is")]
    ok.sort(key=lambda r: (r.get("n_fail", 9), -(r["is"].get("sharpe") or -9)))
    print("\n[排行] fail少→Sharpe高:", flush=True)
    print(f"    {'name':18s} {'Sh':>6s} {'fit':>5s} {'turn':>6s} {'ret':>7s} {'neut':12s} fail", flush=True)
    for r in ok:
        m = r["is"]
        print(f"    {r['name']:18s} {str(m.get('sharpe')):>6s} {str(m.get('fitness')):>5s} "
              f"{str(m.get('turnover')):>6s} {str(m.get('returns')):>7s} {r.get('neut',''):12s} {r.get('n_fail')}",
              flush=True)
    surv = [r for r in ok if r.get("n_fail", 9) == 0]
    print(f"\n[结果] {len(surv)}/{len(results)} 过全部 checks:", flush=True)
    for r in surv:
        print(f"    ★ {r['name']} ({r['neut']}): {r['expr']}\n      → https://platform.worldquantbrain.com/alpha/{r.get('alpha_id')}", flush=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[out] -> {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
