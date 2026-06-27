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
CROWD_MIN = 2         # 池C 动态: 被 >=N 只高收益基金持有 (好基金优选池, ~70-80只)
FUND_PORT = ROOT / "research/cache/fund_portfolio_cache.parquet"  # top70高收益基金持仓
# 池C fallback (基金持仓缓存缺失时用的固定21只)
C_FALLBACK = [
    "000973.SZ", "002384.SZ", "002463.SZ", "300207.SZ", "300308.SZ", "300438.SZ",
    "300502.SZ", "301200.SZ", "301377.SZ", "600105.SH", "600183.SH", "600487.SH",
    "600498.SH", "600522.SH", "601869.SH", "688048.SH", "688183.SH", "688195.SH",
    "688498.SH", "688630.SH", "688777.SH",
]


def build_c_pool(min_funds=CROWD_MIN):
    """池C 动态: 从 top70 高收益基金持仓取最新季度被 >=min_funds 只持有的股 (好基金优选池)。"""
    if not FUND_PORT.exists():
        return C_FALLBACK
    fp = pd.read_parquet(FUND_PORT)
    fp["end_date"] = fp["end_date"].astype(str); fp["symbol"] = fp["symbol"].astype(str)
    q = fp["end_date"].max()
    L = fp[fp["end_date"] == q]
    cnt = L.groupby("symbol")["fund"].nunique()
    codes = sorted(cnt[cnt >= min_funds].index.tolist())
    return codes if codes else C_FALLBACK


# ── 池B 自选: JSON 持久化 (CLI/web 单一真相, 加删自选不改代码) ──
import json as _json
WATCHLIST_B_FILE = ROOT / "config" / "watchlist_b.json"
_B_DEFAULT = ["002571.SZ", "600388.SH", "300648.SZ",
              "688783.SH", "300706.SZ", "000962.SZ", "300054.SZ", "002842.SZ", "688662.SH", "000733.SZ",
              "301027.SZ", "603992.SH"]


def _norm_code(code: str) -> str:
    """规范化为 6位.SZ/SH (容错: 纯6位补后缀, 小写转大写)."""
    c = str(code).strip().upper()
    if c.endswith(".SZ") or c.endswith(".SH"):
        return c
    if c.isdigit() and len(c) == 6:
        return c + (".SH" if c[0] == "6" else ".SZ")
    return c


def load_watchlist_b() -> list[str]:
    """读池B自选 (JSON 优先, 缺则 seed 默认并落盘). 去重保序."""
    if WATCHLIST_B_FILE.exists():
        try:
            codes = _json.loads(WATCHLIST_B_FILE.read_text(encoding="utf-8"))
            if isinstance(codes, list):
                return list(dict.fromkeys(_norm_code(c) for c in codes if c))
        except Exception:
            pass
    save_watchlist_b(_B_DEFAULT)   # seed
    return list(_B_DEFAULT)


def save_watchlist_b(codes: list[str]) -> list[str]:
    """落盘池B (去重保序, 规范化)."""
    clean = list(dict.fromkeys(_norm_code(c) for c in codes if c))
    WATCHLIST_B_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_B_FILE.write_text(_json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return clean


def add_to_b(code: str) -> list[str]:
    """加一只到池B, 返回新清单 (已存在则原样)."""
    return save_watchlist_b(load_watchlist_b() + [code])


def remove_from_b(code: str) -> list[str]:
    """从池B删一只, 返回新清单."""
    c = _norm_code(code)
    return save_watchlist_b([x for x in load_watchlist_b() if x != c])


WATCHLISTS = {        # 自定义池 (代码可加减/新增池)
    "B 自选": load_watchlist_b(),    # JSON 持久化 (config/watchlist_b.json), 见 add_to_b/remove_from_b
    "C 基金重仓": build_c_pool(),   # 动态: 好基金优选池 (被>=2只高收益基金持有)
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


POOL_META = {
    "A": {"key": "A", "name": "系统推荐", "dna": "抄底回调 (系统自营)",
          "desc": "V12.31 v7c主推 → ratio排序 → 行业cap → TopN"},
    "B": {"key": "B", "name": "自选", "dna": "用户关注", "desc": "自选清单打分排名"},
    "C": {"key": "C", "name": "基金重仓", "dna": "机构抱团 (动量追高)",
          "desc": "好基金优选池 (被≥2只高收益基金持有)"},
    "D": {"key": "D", "name": "追高动量", "dna": "高决断度 (与系统对立, 对照)",
          "desc": f"全市场 ratio≥{RATIO_D}"},
}


def _items(df: pd.DataFrame) -> list[dict]:
    """DataFrame → 前端/API 用的 records (含四池统一字段)."""
    if df is None or len(df) == 0:
        return []
    out = []
    for _, p in df.iterrows():
        out.append({
            "ts_code": p["ts_code"], "name": p.get("name") if pd.notna(p.get("name")) else None,
            "industry": p.get("industry") if pd.notna(p.get("industry")) else None,
            "r20": float(p.get("buy_r20_score", 0) or 0),
            "pump_up": float(p.get("pump_score", 0) or 0),
            "pump_down": float(p.get("pump_down_score", 0) or 0),
            "ratio": float(p["ratio"]) if pd.notna(p.get("ratio")) else None,
            "past5": float(p.get("past_r5", 0) or 0),
            "v7c": bool(p.get("v7c_recommend", False)),
            "n_funds": int(p["n_funds"]) if pd.notna(p.get("n_funds")) else None,
        })
    return out


def build_pools(date: str, cb=None, write_csv: bool = True) -> dict:
    """四池统一构建: score_market → A系统/B自选/C基金重仓/D追高 → 写CSV + 返回结构化数据.

    CLI 与 web 共用此函数 (单一真相, 避免逻辑漂移)。返回 {date, pools:{A,B,C,D}, ...}。
    """
    out = ROOT / "output/daily_pick" / f"dashboard_{date}"
    if write_csv:
        out.mkdir(parents=True, exist_ok=True)
    s = V12Scorer.get(ROOT)
    df = s.score_market(date, cb=cb)
    basic = pd.read_parquet(ROOT / "output/tushare_cache/stock_basic.parquet")[
        ["ts_code", "name", "industry"]].drop_duplicates("ts_code")
    for c in ["name", "industry"]:
        if c not in df.columns:
            df = df.merge(basic[["ts_code", c]], on="ts_code", how="left")
    df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)

    # 池A 系统推荐
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

    # 池B/C 自定义 watchlist
    c_codes = WATCHLISTS["C 基金重仓"]
    B = df[df["ts_code"].isin(WATCHLISTS["B 自选"])].sort_values("ratio", ascending=False)
    Cdf = df[df["ts_code"].isin(c_codes)].sort_values("ratio", ascending=False)
    # 池C 附加 持有基金数 (从 fund_portfolio_cache)
    Cdf = _attach_fund_count(Cdf)

    # 池D ratio>=阈值
    D = df.dropna(subset=["ratio"])
    D = D[D["ratio"] >= RATIO_D].sort_values("ratio", ascending=False)

    if write_csv:
        # 全量评分快照 (供次日 r20/ratio 涨跌对比, 含全市场所有评分股)
        snap = [c for c in ["ts_code", "name", "buy_r20_score", "pump_score",
                            "pump_down_score", "ratio", "past_r5", "v7c_recommend"] if c in df.columns]
        df[snap].to_parquet(ROOT / "output/daily_pick" / f"scores_{date}.parquet", index=False)
        A[COLS].to_csv(out / "poolA_system.csv", index=False, encoding="utf-8-sig")
        B[COLS].to_csv(out / "pool_B.csv", index=False, encoding="utf-8-sig")
        ccols = COLS + (["n_funds"] if "n_funds" in Cdf.columns else [])
        Cdf[ccols].to_csv(out / "pool_C.csv", index=False, encoding="utf-8-sig")
        D[COLS].to_csv(out / "poolD_ratio_ge5.csv", index=False, encoding="utf-8-sig")

    d_pos = float((D["past_r5"] > 0).mean() * 100) if len(D) else 0.0
    return {
        "date": date,
        "pools": {
            "A": {**POOL_META["A"], "n_pool": int(len(main_pool)), "items": _items(A)},
            "B": {**POOL_META["B"], "n_codes": len(WATCHLISTS["B 自选"]), "items": _items(B)},
            "C": {**POOL_META["C"], "n_codes": len(c_codes), "items": _items(Cdf)},
            "D": {**POOL_META["D"], "n_total": int(len(D)),
                  "past5_pos_pct": round(d_pos, 0),
                  "items": _items(D.head(30))},
        },
    }


def _attach_fund_count(cdf: pd.DataFrame) -> pd.DataFrame:
    fp = ROOT / "research/cache/fund_portfolio_cache.parquet"
    if not fp.exists() or len(cdf) == 0:
        return cdf
    f = pd.read_parquet(fp); f["end_date"] = f["end_date"].astype(str); f["symbol"] = f["symbol"].astype(str)
    q = f["end_date"].max()
    cnt = f[f["end_date"] == q].groupby("symbol")["fund"].nunique()
    cdf = cdf.copy()
    cdf["n_funds"] = cdf["ts_code"].map(cnt).fillna(0).astype(int)
    return cdf


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else latest_date()
    w(f"\n############ 每日四池看板 {date} ############")
    res = build_pools(date)
    for k in ["A", "B", "C", "D"]:
        p = res["pools"][k]
        w(f"\n===== 池{k} {p['name']} ({p['dna']}; {len(p['items'])}股) — {p['desc']} =====")
        df = pd.DataFrame([{
            "ts_code": it["ts_code"], "name": it["name"], "industry": it["industry"],
            "buy_r20_score": it["r20"], "pump_score": it["pump_up"],
            "pump_down_score": it["pump_down"], "ratio": it["ratio"],
            "past_r5": it["past5"], "v7c_recommend": it["v7c"],
        } for it in p["items"]])
        fmt_table(df, with_rank=(k in ("A", "D")))
    w(f"\n[saved] output/daily_pick/dashboard_{date}/")
    w("提示: 池A=抄底回调(系统), 池C≈池D=动量追高(基金/高ratio), 两套逻辑对立; 谁对取决当下 regime。")


if __name__ == "__main__":
    main()
