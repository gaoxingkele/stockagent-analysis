# -*- coding: utf-8 -*-
"""今日复盘 — 一条命令: 自动更新数据 + 出四池看板。

流程 (全自动):
  1. 检测最新可用交易日 D (Tushare 盘后数据)
  2. 补缺失 daily + moneyflow 到 D
  3. 增量算 factor_lab 因子 (尾部窗口) → ext_review 文件
  4. 增量算衍生特征 mfk/pyramid/moneyflow/amount/v7 (尾部窗口, 文件名对齐 suffix=D[-4:]) + regime
  5. 四池看板 (daily_dashboard: A系统 / B自选 / C基金 / D ratio>=5)

用法: python daily_review.py            (自动到最新交易日)
      python daily_review.py 20260605   (指定)
      python daily_review.py --no-update (跳过数据更新, 直接看板)
设计: 尾部窗口 TRAIL 天重算 (规避单日compute空 + 自愈数据缺口); 生产线 V12.31 只读。
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
import tushare as ts
pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))

DAILY = ROOT / "output/tushare_cache/daily"
MF_BULK = ROOT / "output/tushare_cache/moneyflow"
TRAIL = 4   # 尾部重算交易日数

def log(s): sys.stdout.buffer.write(("[复盘] " + str(s) + "\n").encode("utf-8"))


# ───────── 1. 检测最新交易日 + 补数据 ─────────
def detect_latest() -> str:
    today = datetime.now()
    for k in range(8):
        d = (today - timedelta(days=k)).strftime("%Y%m%d")
        try:
            df = pro.daily(trade_date=d)
            if df is not None and len(df) > 0:
                return d
        except Exception:
            pass
        time.sleep(0.2)
    return sorted(p.stem for p in DAILY.glob("*.parquet"))[-1]


def fetch_missing(D: str):
    have = set(p.stem for p in DAILY.glob("*.parquet"))
    start = (datetime.strptime(D, "%Y%m%d") - timedelta(days=20)).strftime("%Y%m%d")
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=D, is_open="1")
    days = sorted(cal["cal_date"].astype(str)) if cal is not None and "cal_date" in cal.columns else []
    todo = [d for d in days if d not in have]
    for d in todo:
        df = pro.daily(trade_date=d)
        if df is None or len(df) == 0:
            continue
        df.to_parquet(DAILY / f"{d}.parquet", index=False)
        mf = pro.moneyflow(trade_date=d)
        if mf is not None and len(mf):
            mf.to_parquet(MF_BULK / f"{d}.parquet", index=False)
        log(f"拉取 {d}: daily {len(df)} + moneyflow {0 if mf is None else len(mf)}")
        time.sleep(0.4)
    return todo


# ───────── 2. factor_lab 增量 (尾部窗口) ─────────
def update_factor_lab(D: str, win_start: str):
    from factor_lab import compute_factors
    PARQUET = ROOT / "output/factor_lab_3y/factor_groups"
    EXT = ROOT / "output/factor_lab_3y/factor_groups_extension"; EXT.mkdir(exist_ok=True)
    files = sorted(p for p in DAILY.glob("*.parquet") if "20240101" <= p.stem <= D)
    big = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    by_code = {ts: g.sort_values("trade_date").reset_index(drop=True)
               for ts, g in big.groupby("ts_code")}
    t0 = time.time(); total = 0
    for i, gp in enumerate(sorted(PARQUET.glob("group_*.parquet")), 1):
        base = pd.read_parquet(gp); base["trade_date"] = base["trade_date"].astype(str)
        last = base.sort_values(["ts_code", "trade_date"]).groupby("ts_code").tail(1)
        tpl = {r["ts_code"]: r.to_dict() for _, r in last.iterrows()}
        cols = list(next(iter(tpl.values())).keys())
        rows = []
        for ts in tpl:
            d = by_code.get(ts)
            if d is None or len(d) < 60:
                continue
            try:
                f = compute_factors(d)
            except Exception:
                continue
            f["ts_code"] = ts
            new = f[(f["trade_date"] >= win_start) & (f["trade_date"] <= D)].copy()
            if new.empty:
                continue
            for c in cols:
                if c not in new.columns:
                    new[c] = tpl[ts].get(c, pd.NA)
            rows.append(new)
        if rows:
            out = pd.concat(rows, ignore_index=True)
            out.to_parquet(EXT / f"{gp.stem}_ext_review.parquet", index=False)
            total += len(out)
    log(f"factor_lab 增量 {win_start}~{D}: {total:,} 行 ({time.time()-t0:.0f}s)")


# ───────── 3. 衍生特征增量 (尾部窗口, 文件名 suffix=D[-4:]) ─────────
def update_features(D: str, win_start: str):
    suf = D[-4:]
    import extract_mfk_features as _mfk, extract_pyramid_multiwindow as _pyr, \
           extract_v7_extras as _v7, extract_amount_features as _am
    for m in (_mfk, _pyr, _v7, _am):
        m.END = D
    from extract_amount_features import compute_amount_features
    from extract_mfk_features import compute_mfk
    from extract_pyramid_multiwindow import compute_pyramid_v2
    from extract_v7_extras import compute_v7
    from stockagent_analysis.moneyflow.features import compute_features as compute_mf_v1
    MF_CACHE = ROOT / "output/moneyflow/cache"

    # merge moneyflow bulk → 单股 cache
    files = sorted(MF_BULK.glob("*.parquet"))
    bulk = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    bulk["trade_date"] = bulk["trade_date"].astype(str)
    for tscode, g in bulk.groupby("ts_code"):
        cp = MF_CACHE / f"{tscode}.parquet"
        if not cp.exists():
            continue
        cur = pd.read_parquet(cp); cur["trade_date"] = cur["trade_date"].astype(str)
        new = g[g["trade_date"] > cur["trade_date"].max()]
        if new.empty:
            continue
        for c in cur.columns:
            if c not in new.columns:
                new[c] = pd.NA
        pd.concat([cur, new[cur.columns]], ignore_index=True).sort_values(
            "trade_date").reset_index(drop=True).to_parquet(cp, index=False)

    # 全市场 daily by code
    dfiles = sorted(p for p in DAILY.glob("*.parquet") if "20240101" <= p.stem <= D)
    big = pd.concat([pd.read_parquet(f) for f in dfiles], ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    by_code = {ts: g.sort_values("trade_date").reset_index(drop=True) for ts, g in big.groupby("ts_code")}

    amts, mfs, mfks, pyrs, v7s = [], [], [], [], []
    for ts, daily in by_code.items():
        if len(daily) < 60:
            continue
        ohlc = list(zip(daily["trade_date"], daily["open"], daily["high"], daily["low"], daily["close"], daily["vol"]))
        cl = list(zip(daily["trade_date"], daily["close"]))
        try:
            a = compute_amount_features(ohlc, win_start, D)
            if not a.empty:
                a["ts_code"] = ts; amts.append(a)
        except Exception:
            pass
        mp = MF_CACHE / f"{ts}.parquet"
        if not mp.exists():
            continue
        raw = pd.read_parquet(mp); raw["trade_date"] = raw["trade_date"].astype(str)
        for fn, lst, args in [(compute_mf_v1, mfs, (raw,)), (compute_mfk, mfks, (raw, cl)),
                              (compute_pyramid_v2, pyrs, (raw,)), (compute_v7, v7s, (raw, cl))]:
            try:
                r = fn(*args)
                if r is not None and not r.empty:
                    r = r[(r["trade_date"] >= win_start) & (r["trade_date"] <= D)]
                    if not r.empty:
                        lst.append(r)
            except Exception:
                pass

    def write(parts, d, nm):
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(d / f"{nm}_ext_{suf}.parquet", index=False)
            return True
        return False
    ok = [write(amts, ROOT / "output/amount_features", "amount_features"),
          write(mfs, ROOT / "output/moneyflow", "features"),
          write(mfks, ROOT / "output/mfk_features", "features"),
          write(pyrs, ROOT / "output/pyramid_v2", "features"),
          write(v7s, ROOT / "output/v7_extras", "features")]
    log(f"衍生特征增量 ext_{suf}: 写出 {sum(ok)}/5 (amount/mf/mfk/pyr/v7)")
    # mfk/pyramid/v7 若空 → 用最近可用 ext 兜底 (规避单日 compute 空)
    for sub, nm, wrote in [("mfk_features", "features", ok[2]), ("pyramid_v2", "features", ok[3]), ("v7_extras", "features", ok[4])]:
        dst = ROOT / f"output/{sub}/{nm}_ext_{suf}.parquet"
        if not wrote and not dst.exists():
            prev = sorted((ROOT / f"output/{sub}").glob(f"{nm}_ext_*.parquet"))
            if prev:
                import shutil; shutil.copy(prev[-1], dst)
                log(f"  {sub}: 单日空 → 兜底 {prev[-1].name}")

    # regime
    import update_features_to_0605 as uf
    uf.NEW_END = D
    try:
        uf.update_regimes_from_tushare()
        log("regime + regime_extra 更新")
    except Exception as e:
        log(f"regime 更新跳过: {str(e)[:60]}")


def main():
    args = sys.argv[1:]
    no_update = "--no-update" in args
    date_arg = next((a for a in args if a.isdigit() and len(a) == 8), None)

    D = date_arg or detect_latest()
    log(f"目标交易日 D = {D}")
    if not no_update:
        cd = sorted(p.stem for p in DAILY.glob("*.parquet"))
        win_start = cd[max(0, cd.index(D) - TRAIL)] if D in cd else \
                    (datetime.strptime(D, "%Y%m%d") - timedelta(days=10)).strftime("%Y%m%d")
        fetch_missing(D)
        cd = sorted(p.stem for p in DAILY.glob("*.parquet"))
        win_start = cd[max(0, cd.index(D) - TRAIL)]
        log(f"尾部重算窗口: {win_start} ~ {D}")
        update_factor_lab(D, win_start)
        update_features(D, win_start)
    else:
        log("--no-update: 跳过数据更新")

    log("出四池看板 ...\n")
    import importlib, daily_dashboard
    sys.argv = ["daily_dashboard.py", D]
    importlib.reload(daily_dashboard)
    daily_dashboard.main()


if __name__ == "__main__":
    main()
