# -*- coding: utf-8 -*-
"""WorldQuant BRAIN 批量 miner (I5 第三步) — analyst4 分析师修正正交 alpha 验证批.

北极星: 在**未枯竭的美股空间**挖与价量正交的 alpha。第一批选 analyst4 (分析师预估, 1324字段) —
我们整轮 A 股研究指认却拿不到数据的唯一正交前沿 (Tushare report_rc 限频)。分析师预估变化慢 →
天然低换手 (治标准价量 alpha 的 HIGH_TURNOVER 病)。

流程: 15 候选表达式 → 批量 submit (429 退避) → 轮询全部 → 读 IS 指标 + BRAIN 自带 checks
(LOW_SHARPE/LOW_FITNESS/HIGH_TURNOVER/SELF_CORRELATION...) → 按"过 checks 数 + Sharpe + fitness"
排行 → 存 mine_results.json。**这是验证批 (~15 sims), 不是放量搜索。**

用法: python research/brain/wq_mine.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wq_auth import API, load_session

OUT = Path(__file__).resolve().parent / "mine_results.json"

# ── 15 候选: 分析师修正主题 (decay 平滑慢信号, rank 控离群) ──
# 字段来自 analyst4 探测: eps mean/high/low/number/median, af_eps_value(实际), adj_net_income_stddev,
#   actual_eps/sales_value_quarterly。主题: ①revision动量 ②预估离散度(低=高信念) ③覆盖度 ④惊喜。
CANDIDATES = [
    ("eps_rev_mom_20",     "rank(ts_delta(anl4_afv4_eps_mean, 20))", 6),
    ("eps_rev_mom_60",     "rank(ts_delta(anl4_afv4_eps_mean, 60))", 10),
    ("eps_rev_norm60",     "rank(ts_delta(anl4_afv4_eps_mean, 60) / (abs(anl4_afv4_eps_mean) + 1))", 10),
    ("eps_rev_zscore120",  "rank(ts_zscore(anl4_afv4_eps_mean, 120))", 10),
    ("eps_disp_low",       "rank(-(anl4_afv4_eps_high - anl4_afv4_eps_low))", 10),
    ("eps_disp_norm",      "rank(-(anl4_afv4_eps_high - anl4_afv4_eps_low) / (abs(anl4_afv4_eps_mean) + 1))", 10),
    ("netinc_conviction",  "rank(-adj_net_income_stddev)", 10),
    ("eps_coverage_chg",   "rank(ts_delta(anl4_afv4_eps_number, 60))", 10),
    ("eps_surprise",       "rank(anl4_af_eps_value - anl4_afv4_eps_mean)", 5),
    ("eps_skew_med_mean",  "rank(anl4_afv4_median_eps - anl4_afv4_eps_mean)", 10),
    ("eps_actualq_vs_est", "rank(actual_eps_value_quarterly - anl4_afv4_eps_mean)", 5),
    ("eps_mean_growth_yr", "rank(anl4_afv4_eps_mean / (ts_delay(anl4_afv4_eps_mean, 250) + 1) - 1)", 10),
    ("rev_over_disp",      "rank(ts_delta(anl4_afv4_eps_mean, 60) / (anl4_afv4_eps_high - anl4_afv4_eps_low + 1))", 10),
    ("cov_wt_rev",         "rank(ts_delta(anl4_afv4_eps_mean, 60) * log(anl4_afv4_eps_number + 1))", 10),
    ("sales_actualq_mom",  "rank(ts_delta(actual_sales_value_quarterly, 60))", 10),
]

BASE = {
    "instrumentType": "EQUITY", "region": "USA", "universe": "TOP3000",
    "delay": 1, "decay": 0, "neutralization": "SUBINDUSTRY", "truncation": 0.08,
    "pasteurization": "ON", "unitHandling": "VERIFY", "nanHandling": "OFF",
    "language": "FASTEXPR", "visualization": False,
}


def submit(s, expr, decay, retries=6):
    cfg = {**BASE, "decay": decay}
    payload = {"type": "REGULAR", "settings": cfg, "regular": expr}
    for i in range(retries):
        r = s.post(f"{API}/simulations", json=payload)
        if r.status_code in (200, 201, 202):
            return r.headers.get("Location")
        if r.status_code == 429:      # rate / concurrency limit → 退避
            wait = float(r.headers.get("Retry-After", 8)) + 2 * i
            print(f"    [429] 退避 {wait:.0f}s ({expr[:30]}...)", flush=True)
            time.sleep(wait)
            continue
        print(f"    [submit FAIL {r.status_code}] {expr[:40]}: {r.text[:120]}", flush=True)
        return None
    return None


def poll(s, url, t_max=420):
    t0 = time.time()
    while True:
        r = s.get(url)
        if not r.headers.get("Retry-After"):
            return r.json()
        if time.time() - t0 > t_max:
            return {"status": "TIMEOUT"}
        time.sleep(float(r.headers["Retry-After"]))


def main():
    print("=== BRAIN miner: analyst4 分析师修正验证批 (15 候选) ===\n", flush=True)
    s = load_session()

    # ① 批量提交 (收集 progress URL; 并发由平台限, 429 退避)
    print("[1/3] 批量提交 ...", flush=True)
    pending = []
    for name, expr, decay in CANDIDATES:
        loc = submit(s, expr, decay)
        print(f"    {'✓' if loc else '✗'} {name:20s} decay={decay}", flush=True)
        if loc:
            pending.append((name, expr, decay, loc))
        time.sleep(1.5)   # 轻微间隔避免 rate limit

    # ② 轮询全部
    print(f"\n[2/3] 轮询 {len(pending)} 个回测 ...", flush=True)
    results = []
    for name, expr, decay, loc in pending:
        body = poll(s, loc)
        aid = body.get("alpha")
        if not aid:
            print(f"    ✗ {name:20s} status={body.get('status')}", flush=True)
            results.append({"name": name, "expr": expr, "decay": decay,
                            "status": body.get("status"), "is": {}, "checks": []})
            continue
        a = s.get(f"{API}/alphas/{aid}").json()
        ism = a.get("is", {})
        checks = ism.get("checks", [])
        n_pass = sum(1 for c in checks if c.get("result") == "PASS")
        n_fail = sum(1 for c in checks if c.get("result") == "FAIL")
        results.append({"name": name, "expr": expr, "decay": decay, "alpha_id": aid,
                        "status": body.get("status"), "is": ism, "checks": checks,
                        "n_pass": n_pass, "n_fail": n_fail})
        print(f"    ✓ {name:20s} Sharpe={ism.get('sharpe')} fit={ism.get('fitness')} "
              f"turn={ism.get('turnover')} pass={n_pass}/fail={n_fail}", flush=True)

    # ③ 排行 + 存盘
    ok = [r for r in results if r.get("is")]
    ok.sort(key=lambda r: (-r.get("n_fail", 9), -(r["is"].get("sharpe") or -9),
                           -(r["is"].get("fitness") or -9)))
    print("\n[3/3] 排行 (fail少→Sharpe高→fitness高):", flush=True)
    print(f"    {'name':20s} {'Sharpe':>7s} {'fit':>5s} {'turn':>5s} {'ret':>7s} {'dd':>6s} fail", flush=True)
    for r in ok:
        m = r["is"]
        print(f"    {r['name']:20s} {str(m.get('sharpe')):>7s} {str(m.get('fitness')):>5s} "
              f"{str(m.get('turnover')):>5s} {str(m.get('returns')):>7s} {str(m.get('drawdown')):>6s} "
              f"{r.get('n_fail')}", flush=True)
    survivors = [r for r in ok if r.get("n_fail", 9) == 0]
    print(f"\n[结果] {len(survivors)}/{len(results)} 过全部 BRAIN checks (无 FAIL):", flush=True)
    for r in survivors:
        print(f"    ★ {r['name']}: {r['expr']}  → https://platform.worldquantbrain.com/alpha/{r.get('alpha_id')}",
              flush=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n[out] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
