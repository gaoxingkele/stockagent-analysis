# -*- coding: utf-8 -*-
"""WorldQuant BRAIN alpha 模拟器 (I5 第二步) — 提交表达式 → 轮询 → 读 IS 指标.

BRAIN 挖矿基本单元: 把一个 alpha 表达式 (FASTEXPR) 提交回测, 拿 IS (in-sample) 指标
(Sharpe / fitness / turnover / returns / drawdown / margin / 多空数), 可选 OS (out-sample)。

API 流:
  POST /simulations {settings, regular: expr}  → 202 + Location: 进度 URL
  GET  <进度URL> (轮询, Retry-After 头控制节奏)  → 完成后 body.alpha = alpha_id
  GET  /alphas/<id>                              → body.is = IS 指标块

用法:
  python research/brain/wq_simulate.py "rank(-returns)"          # 跑单个 alpha
  python research/brain/wq_simulate.py "rank(-returns)" --region USA --universe TOP3000
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wq_auth import API, load_session

# 默认回测设置 (美股, 与我们 A 股策略正交的未枯竭空间)
DEFAULT_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "INDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "OFF",
    "language": "FASTEXPR",
    "visualization": False,
}


def simulate(expr: str, settings: dict | None = None, s=None,
             poll_max_s: int = 300, verbose: bool = True) -> dict:
    """提交一个 alpha 表达式回测, 返回 {alpha_id, is_metrics, expr, settings}。"""
    s = s or load_session()
    cfg = {**DEFAULT_SETTINGS, **(settings or {})}
    payload = {"type": "REGULAR", "settings": cfg, "regular": expr}

    r = s.post(f"{API}/simulations", json=payload)
    if r.status_code not in (200, 201, 202):
        raise RuntimeError(f"提交失败 HTTP {r.status_code}: {r.text[:300]}")
    progress_url = r.headers.get("Location")
    if not progress_url:
        # 部分情况直接返回结果
        body = r.json()
        progress_url = None
        if verbose:
            print(f"  [submit] 即时返回 (无 Location)")
    if verbose:
        print(f"  [submit] HTTP {r.status_code}, 轮询进度 ...")

    # 轮询进度
    t0 = time.time()
    body = {}
    while progress_url:
        pr = s.get(progress_url)
        retry = pr.headers.get("Retry-After")
        if not retry:                # 完成
            body = pr.json()
            break
        if time.time() - t0 > poll_max_s:
            raise TimeoutError(f"轮询超时 {poll_max_s}s, alpha 仍在跑")
        time.sleep(float(retry))

    status = body.get("status")
    alpha_id = body.get("alpha")
    if not alpha_id:
        raise RuntimeError(f"无 alpha_id (status={status}): {str(body)[:300]}")
    if verbose:
        print(f"  [done] status={status} alpha_id={alpha_id} ({time.time()-t0:.0f}s)")

    ar = s.get(f"{API}/alphas/{alpha_id}")
    ar.raise_for_status()
    a = ar.json()
    return {"alpha_id": alpha_id, "expr": expr, "settings": cfg,
            "is": a.get("is", {}), "status": status, "raw": a}


def fmt_is(m: dict) -> str:
    g = lambda k: m.get(k)
    return (f"Sharpe={g('sharpe')}  fitness={g('fitness')}  turnover={g('turnover')}  "
            f"returns={g('returns')}  drawdown={g('drawdown')}  margin={g('margin')}  "
            f"long/short={g('longCount')}/{g('shortCount')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("expr", help="alpha 表达式 (FASTEXPR), 如 'rank(-returns)'")
    ap.add_argument("--region", default="USA")
    ap.add_argument("--universe", default="TOP3000")
    ap.add_argument("--delay", type=int, default=1)
    ap.add_argument("--neutralization", default="INDUSTRY")
    args = ap.parse_args()

    print(f"=== BRAIN simulate: {args.expr} ===")
    s = load_session()
    res = simulate(args.expr, settings={
        "region": args.region, "universe": args.universe,
        "delay": args.delay, "neutralization": args.neutralization,
    }, s=s)
    print(f"\n[IS 指标] {fmt_is(res['is'])}")
    checks = res["is"].get("checks", [])
    if checks:
        print("[checks]")
        for c in checks:
            print(f"  {c.get('name'):24s} {c.get('result'):6s} "
                  f"value={c.get('value')} limit={c.get('limit')}")
    print(f"\n[alpha] https://platform.worldquantbrain.com/alpha/{res['alpha_id']}")


if __name__ == "__main__":
    main()
