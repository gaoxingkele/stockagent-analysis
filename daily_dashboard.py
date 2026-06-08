# -*- coding: utf-8 -*-
"""每日四池看板 (收盘后跑) — 系统抄底 vs 自选/基金/动量追高 一图对照。

池A 系统推荐: V12.31 (v7c 6铁律 → ratio=P_up/(P_down+ε) 排序 → 行业 cap 4 → Top N)
池B 自选 / 池C 基金重仓: 给系统打分看排名/是否入池 (WATCHLISTS 可编辑)
池D ratio>=5.0: 全市场高决断度 (=追高动量股, 与系统DNA对立, 对照用)

用法: python daily_dashboard.py [YYYYMMDD]   (默认=daily cache 最新交易日)
前置: 该日 factor_lab + 衍生特征已就绪 (见 update_factor_lab / update_features 脚本)。
生产线 V12.31 全程只读, 不改。
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.v12_scoring import V12Scorer

# ───────── 可编辑配置 ─────────
TOP_N_A = 20          # 池A 取前 N
IND_CAP = 4           # 池A 单行业上限
RATIO_D = 5.0         # 池D 阈值
WATCHLISTS = {        # 自定义池 (代码可加减/新增池)
    "B 自选": ["002571.SZ", "600388.SH", "300648.SZ"],
    "C 基金重仓": [
        "000973.SZ", "002384.SZ", "002463.SZ", "300207.SZ", "300308.SZ", "300438.SZ",
        "300502.SZ", "301200.SZ", "301377.SZ", "600105.SH", "600183.SH", "600487.SH",
        "600498.SH", "600522.SH", "601869.SH", "688048.SH", "688183.SH", "688195.SH",
        "688498.SH", "688630.SH", "688777.SH",
    ],
}
COLS = ["ts_code", "name", "industry", "buy_r20_score", "pump_score",
        "pump_down_score", "ratio", "past_r5", "v7c_recommend"]


def w(s=""):
    sys.stdout.buffer.write((str(s) + "\n").encode("utf-8"))


def latest_date() -> str:
    return sorted(p.stem for p in (ROOT / "output/tushare_cache/daily").glob("*.parquet"))[-1]


def fmt_table(d: pd.DataFrame, with_rank=False):
    w(f"  {'#' if with_rank else ' ':3s}{'代码':11s}{'名称':9s}{'行业':9s}"
      f"{'r20':>5s}{'pump↑':>7s}{'pump↓':>7s}{'↑/↓':>6s}{'past5':>8s}{' 主推'}")
    for i, (_, p) in enumerate(d.iterrows(), 1):
        rec = "★" if p.get("v7c_recommend") else ""
        rk = f"{i:>2d}." if with_rank else "   "
        w(f"  {rk}{p['ts_code']:11s}{str(p.get('name', ''))[:8]:9s}{str(p.get('industry', ''))[:8]:9s}"
          f"{p.get('buy_r20_score', 0):>5.0f} {p['pump_score']:.3f}  {p['pump_down_score']:.3f}  "
          f"{p['ratio']:>4.1f}x {p.get('past_r5', 0) * 100:>+6.1f}%  {rec}")


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else latest_date()
    out = ROOT / "output/daily_pick" / f"dashboard_{date}"
    out.mkdir(parents=True, exist_ok=True)

    w(f"\n############ 每日四池看板 {date} ############")
    s = V12Scorer.get(ROOT)
    df = s.score_market(date, cb=None)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "name", "industry"]].drop_duplicates("ts_code")
    for c in ["name", "industry"]:
        if c not in df.columns:
            df = df.merge(basic[["ts_code", c]], on="ts_code", how="left")
    df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)

    # ── 池A 系统推荐 ──
    main_pool = df[df["v7c_recommend"] == True].sort_values("ratio", ascending=False)
    picks, ind = [], {}
    for _, r in main_pool.iterrows():
        k = str(r.get("industry") or "?")
        if ind.get(k, 0) >= IND_CAP:
            continue
        ind[k] = ind.get(k, 0) + 1
        picks.append(r)
        if len(picks) >= TOP_N_A:
            break
    A = pd.DataFrame(picks)
    w(f"\n===== 池A 系统推荐 (V12.31: v7c主推{len(main_pool)}股 → ratio排序 → 行业cap{IND_CAP} → Top{TOP_N_A}) =====")
    fmt_table(A, with_rank=True)
    A[COLS].to_csv(out / "poolA_system.csv", index=False, encoding="utf-8-sig")

    # ── 池B/C 自定义 ──
    for name, codes in WATCHLISTS.items():
        sub = df[df["ts_code"].isin(codes)].sort_values("ratio", ascending=False)
        nrec = int(sub["v7c_recommend"].sum()) if len(sub) else 0
        w(f"\n===== 池{name} ({len(sub)}/{len(codes)}股, 入主推★ {nrec}) =====")
        fmt_table(sub)
        miss = set(codes) - set(sub["ts_code"])
        if miss:
            w(f"  (缺数据/未评分: {sorted(miss)})")
        sub[COLS].to_csv(out / f"pool_{name.split()[0]}.csv", index=False, encoding="utf-8-sig")

    # ── 池D ratio>=阈值 ──
    D = df.dropna(subset=["ratio"])
    D = D[D["ratio"] >= RATIO_D].sort_values("ratio", ascending=False)
    pos = (D["past_r5"] > 0).mean() * 100 if len(D) else 0
    w(f"\n===== 池D ratio>={RATIO_D} 全市场 ({len(D)}股 | 入主推★ {int(D['v7c_recommend'].sum())} | "
      f"r20中位{D['buy_r20_score'].median():.0f} | past5为正{pos:.0f}% 中位{D['past_r5'].median()*100:+.1f}% = 追高动量) =====")
    w("  行业分布(前6): " + " / ".join(f"{k}:{v}" for k, v in D["industry"].fillna("?").value_counts().head(6).items()))
    w("  ratio 前15:")
    fmt_table(D.head(15), with_rank=True)
    D[COLS].to_csv(out / "poolD_ratio_ge5.csv", index=False, encoding="utf-8-sig")

    w(f"\n[saved] {out}/ (poolA_system / pool_B / pool_C / poolD_ratio_ge5 .csv)")
    w("提示: 池A=抄底回调(系统), 池C≈池D=动量追高(基金/高ratio), 两套逻辑对立; 谁对取决当下 regime。")


if __name__ == "__main__":
    main()
