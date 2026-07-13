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


# ── 池E 外部信号: 调用外部策略脚本 (stock_benchmark) 取清单 ──
# 每日固定调用: python <脚本> --date latest --top-n N --format json → unique_stocks[].ts_code
# 成功 → 用结果并缓存到 config/pool_e.json (审计+离线回退); 失败 → 回退读缓存文件。
import os, subprocess

POOL_E_FILE = ROOT / "config" / "pool_e.json"
# 可用环境变量覆盖 (换机器时改路径, 不动代码)
POOL_E_SCRIPT = os.environ.get(
    "POOL_E_SCRIPT", r"D:\aicoding\stock_benchmark\scripts\export_strategy_list.py")
POOL_E_TOPN = os.environ.get("POOL_E_TOPN", "20")


def _fetch_pool_e_external() -> list[str] | None:
    """调用外部策略脚本 (--format json) 取 ts_code 列表. 任何失败返回 None."""
    if not Path(POOL_E_SCRIPT).exists():
        return None
    cmd = [sys.executable, POOL_E_SCRIPT,
           "--date", "latest", "--top-n", str(POOL_E_TOPN), "--format", "json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", timeout=180)
        if r.returncode != 0 or not (r.stdout or "").strip():
            return None
        data = _json.loads(r.stdout)
        codes = [u.get("ts_code") for u in data.get("unique_stocks", [])]
        codes = list(dict.fromkeys(_norm_code(c) for c in codes if c))
        return codes or None
    except Exception:
        return None


def load_pool_e() -> list[str]:
    """池E 外部信号: 优先调用外部脚本, 成功则缓存到 config/pool_e.json; 失败回退缓存文件.

    换到没有 stock_benchmark 的机器上时, 自动用最近一次缓存的 config/pool_e.json。
    """
    codes = _fetch_pool_e_external()
    if codes:
        try:
            POOL_E_FILE.parent.mkdir(parents=True, exist_ok=True)
            POOL_E_FILE.write_text(_json.dumps(codes, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        except Exception:
            pass
        return codes
    # 回退: 读缓存文件 (外部脚本不可用时)
    if POOL_E_FILE.exists():
        try:
            j = _json.loads(POOL_E_FILE.read_text(encoding="utf-8"))
            if isinstance(j, list):
                return list(dict.fromkeys(_norm_code(c) for c in j if c))
        except Exception:
            pass
    return []


# ── SEMAS 最优组合推荐 (stock_benchmark 另一个项目的输出) ──
# 流程: 先更新 stock_benchmark 日线数据 → 生成 SEMAS 清单 → 读取 CSV → 计算 r20/ratio
SEMAS_DIR = Path(r"D:\aicoding\stock_benchmark\experiments\semas_best_combo_stock_list")
SEMAS_UPDATE_SCRIPT = Path(r"D:\aicoding\stock_benchmark\scripts\update_lingxi_v2_cn_daily_latest.py")
SEMAS_GENERATE_SCRIPT = Path(r"D:\aicoding\stock_benchmark\scripts\generate_semas_best_combo_stock_list.py")


def update_and_generate_semas() -> bool:
    """更新 stock_benchmark 日线数据到最新, 再生成 SEMAS 清单. 返回是否成功."""
    if not SEMAS_UPDATE_SCRIPT.exists() or not SEMAS_GENERATE_SCRIPT.exists():
        return False
    try:
        # 1. 更新日线数据
        subprocess.run(
            [sys.executable, str(SEMAS_UPDATE_SCRIPT), "--sleep", "0.05"],
            cwd=str(SEMAS_UPDATE_SCRIPT.parents[1]),
            capture_output=True, text=True, encoding="utf-8", timeout=600,
        )
        # 2. 生成 SEMAS 清单
        r = subprocess.run(
            [sys.executable, str(SEMAS_GENERATE_SCRIPT)],
            cwd=str(SEMAS_GENERATE_SCRIPT.parents[1]),
            capture_output=True, text=True, encoding="utf-8", timeout=300,
        )
        return r.returncode == 0
    except Exception:
        return False


def load_semas_list() -> pd.DataFrame:
    """读取 SEMAS 最优组合统一清单 (最新 CSV). 返回 DataFrame, 失败返回空."""
    if not SEMAS_DIR.exists():
        return pd.DataFrame()
    csvs = sorted(SEMAS_DIR.glob("semas_best_combo_unified_stock_list_*.csv"))
    csvs = [p for p in csvs if not p.name.endswith("_meta.csv")]
    if not csvs:
        return pd.DataFrame()
    latest = csvs[-1]
    try:
        # 校验 meta 一致性
        date_part = latest.stem.replace("semas_best_combo_unified_stock_list_", "")
        meta_path = SEMAS_DIR / f"semas_best_combo_unified_stock_list_{date_part}_meta.json"
        if meta_path.exists():
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("signal_date") != meta.get("data_max_date"):
                return pd.DataFrame()  # 数据不一致, 不可用
        df = pd.read_csv(latest, dtype={"ts_code": str, "symbol": str})
        df = df.sort_values("merged_rank")
        df["ts_code"] = df["ts_code"].apply(_norm_code)
        return df
    except Exception:
        return pd.DataFrame()


WATCHLISTS = {        # 自定义池 (代码可加减/新增池)
    "B 自选": load_watchlist_b(),    # JSON 持久化 (config/watchlist_b.json), 见 add_to_b/remove_from_b
    "C 基金重仓": build_c_pool(),   # 动态: 好基金优选池 (被>=2只高收益基金持有)
    "E 外部": load_pool_e(),        # 外部程序每日覆写 config/pool_e.json
}
COLS = ["ts_code", "name", "industry", "buy_r20_score", "pump_score",
        "pump_down_score", "ratio", "past_r5", "v7c_recommend"]

# 大盘指数基准 (每个池子底部附一行做比较)
_INDEX_CODES = [
    ("000300.SH", "沪深300"),
    ("000905.SH", "中证500"),
    ("399006.SZ", "创业板指"),
]


def _fetch_index_benchmark(date: str) -> list[dict]:
    """取三大指数近 N 日涨幅, 作为每个池子的基准参照行."""
    import tushare as ts
    try:
        pro = ts.pro_api()
    except Exception:
        return []
    rows = []
    for ts_code, name in _INDEX_CODES:
        try:
            df = pro.index_daily(ts_code=ts_code, start_date="20260101", end_date=date)
            if df is None or len(df) == 0:
                continue
            df = df.sort_values("trade_date").reset_index(drop=True)
            close_now = df["close"].iloc[-1]
            # past_r5
            r5 = 0.0
            if len(df) >= 6:
                r5 = (close_now / df["close"].iloc[-6] - 1) * 100
            # past_r20
            r20 = 0.0
            if len(df) >= 21:
                r20 = (close_now / df["close"].iloc[-21] - 1) * 100
            rows.append({
                "ts_code": ts_code, "name": name, "industry": "指数",
                "buy_r20_score": None, "pump_score": None, "pump_down_score": None,
                "ratio": None, "past_r5": round(r5, 1), "v7c_recommend": False,
                "_past_r20": round(r20, 1), "_close": round(close_now, 2),
            })
        except Exception:
            continue
    return rows


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
          "desc": "V12.31 v7c主推 → r20排名(3分内ratio优先) → 行业cap → TopN"},
    "B": {"key": "B", "name": "自选", "dna": "用户关注", "desc": "自选清单打分排名"},
    "C": {"key": "C", "name": "基金重仓", "dna": "机构抱团 (动量追高)",
          "desc": "好基金优选池 (被≥2只高收益基金持有)"},
    "D": {"key": "D", "name": "追高动量", "dna": "高决断度 (与系统对立, 对照)",
          "desc": f"全市场 ratio≥{RATIO_D}"},
    "E": {"key": "E", "name": "外部信号", "dna": "外部程序 (每日固定调用)",
          "desc": "外部清单打分排名 (config/pool_e.json)"},
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

    # 池A 系统推荐: v7c主推 → r20排名(3分一档) + ratio优先 → 行业cap → TopN
    main_pool = df[df["v7c_recommend"] == True].copy()
    main_pool["_r20_bucket"] = (main_pool["buy_r20_score"] // 3).astype(int)
    main_pool = main_pool.sort_values(
        ["_r20_bucket", "ratio"], ascending=[False, False]).drop(columns=["_r20_bucket"])
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

    # 池B/C/E 自定义 watchlist — 调用时实时取 (web 长驻进程不 reload 模块,
    # 若读模块级 WATCHLISTS 会冻结在进程启动那一刻; 这里实时加载保证每日刷新)
    b_codes = load_watchlist_b()
    c_codes = build_c_pool()
    e_codes = load_pool_e()
    B = df[df["ts_code"].isin(b_codes)].sort_values("ratio", ascending=False)
    Cdf = df[df["ts_code"].isin(c_codes)].sort_values("ratio", ascending=False)
    # 池C 附加 持有基金数 (从 fund_portfolio_cache)
    Cdf = _attach_fund_count(Cdf)

    # 池D ratio>=阈值
    D = df.dropna(subset=["ratio"])
    D = D[D["ratio"] >= RATIO_D].sort_values("ratio", ascending=False)

    # 池E 外部信号 (外部脚本每日调用的 watchlist, 打分排名, 同池B)
    E = df[df["ts_code"].isin(e_codes)].sort_values("ratio", ascending=False)

    # SEMAS 最优组合 (stock_benchmark 外部推荐)
    semas_df = load_semas_list()
    semas_scored = pd.DataFrame()
    if len(semas_df) > 0:
        semas_codes = semas_df["ts_code"].tolist()
        semas_scored = df[df["ts_code"].isin(semas_codes)].copy()
        # 合并 SEMAS 原始字段 (merged_rank, hit_count, recommended_horizons 等)
        merge_cols = ["ts_code"] + [c for c in ["merged_rank", "stock_name", "recommended_horizons",
                                     "hit_count", "avg_rank", "rank_score"] if c in semas_df.columns]
        semas_scored = semas_scored.merge(semas_df[merge_cols], on="ts_code", how="left", suffixes=("", "_semas"))
        semas_scored = semas_scored.sort_values("merged_rank")

    # 大盘指数基准
    idx_rows = _fetch_index_benchmark(date)

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
        E[COLS].to_csv(out / "pool_E.csv", index=False, encoding="utf-8-sig")
        # SEMAS 推荐 (带 r20/ratio 评分)
        if len(semas_scored) > 0:
            semas_cols = [c for c in COLS + ["merged_rank", "stock_name", "recommended_horizons",
                                  "hit_count", "avg_rank", "rank_score"] if c in semas_scored.columns]
            semas_scored[semas_cols].to_csv(out / "semas_stocks.csv", index=False, encoding="utf-8-sig")
        # 指数基准单独写 (各池通用)
        if idx_rows:
            idx_df = pd.DataFrame(idx_rows)
            idx_df[COLS].to_csv(out / "benchmark_indices.csv", index=False, encoding="utf-8-sig")

    d_pos = float((D["past_r5"] > 0).mean() * 100) if len(D) else 0.0
    # SEMAS 结构化数据
    semas_items = []
    if len(semas_scored) > 0:
        for _, r in semas_scored.iterrows():
            semas_items.append({
                "ts_code": r["ts_code"], "name": r.get("name") or r.get("stock_name"),
                "industry": r.get("industry"), "merged_rank": int(r.get("merged_rank", 0)),
                "hit_count": int(r.get("hit_count", 0)),
                "recommended_horizons": r.get("recommended_horizons", ""),
                "avg_rank": round(float(r.get("avg_rank", 0)), 1),
                "r20": float(r.get("buy_r20_score", 0) or 0),
                "pump_up": float(r.get("pump_score", 0) or 0),
                "pump_down": float(r.get("pump_down_score", 0) or 0),
                "ratio": float(r.get("ratio", 0) or 0),
                "past5": float(r.get("past_r5", 0) or 0),
                "v7c": bool(r.get("v7c_recommend", False)),
            })
    return {
        "date": date,
        "benchmark": idx_rows,
        "semas": {"count": len(semas_items), "items": semas_items},
        "pools": {
            "A": {**POOL_META["A"], "n_pool": int(len(main_pool)), "items": _items(A)},
            "B": {**POOL_META["B"], "n_codes": len(b_codes), "items": _items(B)},
            "C": {**POOL_META["C"], "n_codes": len(c_codes), "items": _items(Cdf)},
            "D": {**POOL_META["D"], "n_total": int(len(D)),
                  "past5_pos_pct": round(d_pos, 0),
                  "items": _items(D.head(30))},
            "E": {**POOL_META["E"], "n_codes": len(e_codes), "items": _items(E)},
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
    w(f"\n############ 每日五池看板 {date} ############")
    res = build_pools(date)
    # 指数基准
    benchmark = res.get("benchmark", [])
    if benchmark:
        w(f"\n{'='*60}")
        w(f"  📊 大盘指数基准 (5日/20日涨幅)")
        w(f"{'='*60}")
        for idx in benchmark:
            r20 = idx.get("_past_r20", 0)
            w(f"  {idx['ts_code']}  {idx['name']:<8}  收盘 {idx['_close']:>10.2f}  5日 {idx['past_r5']:>+6.1f}%  20日 {r20:>+6.1f}%")
    for k in ["A", "B", "C", "D", "E"]:
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
