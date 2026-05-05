#!/usr/bin/env python3
"""V10 视觉 PoC: 历史 K 线图 + Sonnet 4.6 双图视觉分析.

严格无未来信息:
  样本日 D 的 K 线图仅含 D 及以前数据 (close <= D)
  日线: 最近 60 个交易日 (~3 月)
  周线: 最近 60 个交易周 (~14 月)

50 样本 = 25 理想多 + 25 矛盾段, 跨 8 月分散.

Sonnet 4.6 视觉双图调用:
  输入: 日线 PNG + 周线 PNG + V7c 上下文
  prompt: 4 象限场景化 (理想多确认 / 矛盾段反挖)
  输出: confirm/reject + conviction + key_pattern

输出:
  output/v10/charts/{ts_code}_{trade_date}_{day|week}.png (历史 K 线图保留)
  output/v10/poc_results.csv (50 样本评估)
"""
from __future__ import annotations
import os, struct, json, time, random, base64
from pathlib import Path
from datetime import datetime, date as dt_date
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.cloubic")
load_dotenv(".env")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from stockagent_analysis.chart_generator import generate_kline_chart, load_image_base64

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
OUT_DIR = ROOT / "output" / "v10"
CHARTS_DIR = OUT_DIR / "charts"
OUT_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)
TDX = os.getenv("TDX_DIR", "D:/tdx")

TEST_START = "20250501"
TEST_END   = "20260126"
R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)

N_PER_QUADRANT = 25
CLOUBIC_KEY = os.environ["CLOUBIC_API_KEY"]
CLOUBIC_BASE = "https://api.cloubic.com/v1"


def read_tdx(market, code, end_date_inclusive):
    """读 TDX 日线, 严格过滤到 end_date_inclusive (含).

    端到端确保无未来信息 — 调用方传入样本日 D, 这里只返回 date <= D 的行.
    """
    p = Path(TDX) / "vipdoc" / market / "lday" / f"{market}{code}.day"
    if not p.is_file(): return None
    data = p.read_bytes()
    n = len(data) // 32
    if n == 0: return None
    rows = []
    end_int = int(end_date_inclusive)
    for i in range(n):
        f = struct.unpack_from("<8I", data, i*32)
        di = f[0]
        if di > end_int:
            continue  # ⭐ 关键: 严格过滤未来数据
        try: d = dt_date(di//10000, (di%10000)//100, di%100)
        except: continue
        rows.append({
            "ts": d.strftime("%Y-%m-%d"),
            "open": f[1]/100.0, "high": f[2]/100.0,
            "low": f[3]/100.0, "close": f[4]/100.0,
            "volume": float(f[6]),
        })
    return rows


def code_market(ts):
    if "." in ts: c, ex = ts.split("."); return c, ex.lower()
    if ts.startswith(("8","4","9")): return ts, "bj"
    if ts.startswith(("5","6")): return ts, "sh"
    return ts, "sz"


def daily_to_weekly(rows):
    """日线聚合到周线."""
    if not rows: return []
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["ts"])
    df = df.set_index("dt")
    weekly = df.resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    weekly["ts"] = weekly.index.strftime("%Y-%m-%d")
    return weekly[["ts","open","high","low","close","volume"]].to_dict("records")


def gen_history_charts(ts_code, trade_date, name="?"):
    """生成历史日线 + 周线 PNG (无未来信息).
    返回 (daily_path, weekly_path) 或 (None, None)
    """
    code, market = code_market(ts_code)
    rows = read_tdx(market, code, trade_date)
    if not rows or len(rows) < 30:
        return None, None

    # 日线最近 60 日
    daily_60 = rows[-60:]
    df_day = pd.DataFrame(daily_60)
    daily_path = CHARTS_DIR / f"{ts_code}_{trade_date}_day.png"
    if not daily_path.exists():
        png = generate_kline_chart(df_day, "day", ts_code, name,
                                     save_path=daily_path)
        if png is None:
            return None, None

    # 周线最近 60 周 (~ 60 × 5 = 300 个交易日)
    weekly_all = daily_to_weekly(rows)
    if len(weekly_all) < 20:
        return daily_path, None
    weekly_60 = weekly_all[-60:]
    df_week = pd.DataFrame(weekly_60)
    weekly_path = CHARTS_DIR / f"{ts_code}_{trade_date}_week.png"
    if not weekly_path.exists():
        png = generate_kline_chart(df_week, "week", ts_code, name,
                                     save_path=weekly_path)
        if png is None:
            return daily_path, None

    return daily_path, weekly_path


def _map_anchored(v, p5, p50, p95):
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, 50.0)
    out = np.where(v <= p5, 0, out)
    out = np.where(v >= p95, 100, out)
    mask_lo = (v > p5) & (v <= p50)
    out = np.where(mask_lo, (v - p5) / (p50 - p5) * 50, out)
    mask_hi = (v > p50) & (v < p95)
    out = np.where(mask_hi, 50 + (v - p50) / (p95 - p50) * 50, out)
    return out


def predict_model(df, name):
    d = PROD_DIR / name
    booster = lgb.Booster(model_file=str(d / "classifier.txt"))
    meta = json.loads((d / "feature_meta.json").read_text(encoding="utf-8"))
    feat_cols = meta["feature_cols"]
    industry_map = meta.get("industry_map", {})
    df = df.copy()
    df["industry_id"] = df["industry"].fillna("unknown").map(
        lambda x: industry_map.get(str(x), -1)
    )
    return booster.predict(df[feat_cols])


def load_oos():
    PARQUET_DIR = ROOT / "output" / "factor_lab_3y" / "factor_groups"
    LABELS_10 = ROOT / "output" / "cogalpha_features" / "labels_10d.parquet"
    LABELS_20 = ROOT / "output" / "labels" / "max_gain_labels.parquet"
    parts = []
    for p in sorted(PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p)
        df["trade_date"] = df["trade_date"].astype(str)
        df = df[(df["trade_date"] >= TEST_START) & (df["trade_date"] <= TEST_END)]
        if not df.empty: parts.append(df)
    full = pd.concat(parts, ignore_index=True)
    l10 = pd.read_parquet(LABELS_10, columns=["ts_code","trade_date","r10"])
    l10["trade_date"] = l10["trade_date"].astype(str)
    full = full.merge(l10, on=["ts_code","trade_date"], how="left")
    l20 = pd.read_parquet(LABELS_20, columns=["ts_code","trade_date","max_gain_20","max_dd_20"])
    l20["trade_date"] = l20["trade_date"].astype(str)
    full = full.merge(l20, on=["ts_code","trade_date"], how="left")
    for path in [
        ROOT / "output" / "amount_features" / "amount_features.parquet",
        ROOT / "output" / "regime_extra" / "regime_extra.parquet",
        ROOT / "output" / "moneyflow" / "features.parquet",
        ROOT / "output" / "cogalpha_features" / "features.parquet",
        ROOT / "output" / "mfk_features" / "features.parquet",
        ROOT / "output" / "pyramid_v2" / "features.parquet",
        ROOT / "output" / "v7_extras" / "features.parquet",
    ]:
        if not path.exists(): continue
        d = pd.read_parquet(path)
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
        if "ts_code" in d.columns:
            full = full.merge(d, on=["ts_code","trade_date"], how="left")
        else:
            full = full.merge(d, on="trade_date", how="left")
    rg_path = ROOT / "output" / "regimes" / "daily_regime.parquet"
    if rg_path.exists():
        rg = pd.read_parquet(rg_path,
              columns=["trade_date","regime_id","ret_5d","ret_20d","ret_60d","rsi14","vol_ratio"])
        rg["trade_date"] = rg["trade_date"].astype(str)
        rg = rg.rename(columns={"ret_5d":"mkt_ret_5d","ret_20d":"mkt_ret_20d",
                                  "ret_60d":"mkt_ret_60d","rsi14":"mkt_rsi14","vol_ratio":"mkt_vol_ratio"})
        full = full.merge(rg, on="trade_date", how="left")
    return full


SYS_PROMPT_IDEAL = """你是 A 股 K 线视觉技术分析师 (重要: 严格基于图像本身判断, 不依赖外部信息).

背景: V7c 量化系统已推荐这只股票 (高 buy + 低 sell, 4 象限"理想多" 段, OOS 实测 r20=+6.31%).
你的任务: 看日线 + 周线 K 线综合图, 判断 V7c 的"温和上涨"判断是否成立.

判断角度:
1. 日线/周线趋势是否一致看多?
2. 是否处于"健康上涨" (温和阳线 + 量能配合 + 均线支撑) 而非"派发拉高" (放量长上影/超买)?
3. MACD/RSI/KDJ 是否在合理区间, 还是已极度超买?
4. 周线视角下, 当前位置是趋势中段还是末端?

输出 JSON (仅 JSON, 无其他文字):
{"verdict": "confirm" 或 "reject",
"conviction": 0-100 (置信度, 越高越确定 V7c 推荐成立),
"key_pattern": "30字内核心形态描述",
"risk_warning": "30字内最大风险点"}"""


SYS_PROMPT_CONTRADICTION = """你是 A 股 K 线视觉解谜专家 (重要: 严格基于图像本身判断).

背景: V7c 系统识别这只股为"矛盾段" (高 buy + 高 sell, OOS r20=+3.07% 但 dd<-15%=29% 风险高).
你的任务: 看日线 + 周线 K 线综合图, 判断这是"洗盘后突破"还是"顶部派发".

判断角度:
1. 周线视角: 长期趋势是上升通道中的洗盘, 还是顶部 M 头?
2. 日线视角: 异常波动是缩量洗盘 (蓄势)还是放量派发 (出货)?
3. 量价是否健康配合 (上涨放量, 调整缩量)?
4. 关键支撑位是否有效?

输出 JSON (仅 JSON, 无其他文字):
{"verdict": "真涨" 或 "假涨" 或 "不确定",
"conviction": 0-100,
"key_pattern": "30字内核心形态描述",
"risk_warning": "30字内最大风险点"}"""


def call_vision_dual(client, daily_b64, weekly_b64, sys_prompt, max_retry=2):
    """Sonnet 4.6 视觉双图调用."""
    user_content = [
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{daily_b64}"}},
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{weekly_b64}"}},
        {"type": "text",
         "text": "第一张是日线 (60 日), 第二张是周线 (60 周). 请综合判断, 输出 JSON 决策."}
    ]
    for r in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="claude-sonnet-4-6",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=400, temperature=0.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if r < max_retry - 1:
                time.sleep(2); continue
            return None, None


def main():
    t0 = time.time()
    print(f"[{int(time.time()-t0)}s] 加载 OOS + V7c 评分...", flush=True)
    df = load_oos()
    df["r10_pred"] = predict_model(df, "r10_v4_all")
    df["r20_pred"] = predict_model(df, "r20_v4_all")
    df["sell_10_v6_prob"] = predict_model(df, "sell_10_v6")
    df["sell_20_v6_prob"] = predict_model(df, "sell_20_v6")
    s10 = _map_anchored(df["r10_pred"].values, *R10_ANCHOR)
    s20 = _map_anchored(df["r20_pred"].values, *R20_ANCHOR)
    df["buy_score"] = 0.5 * s10 + 0.5 * s20
    s10s = _map_anchored(df["sell_10_v6_prob"].values, *SELL10_V6)
    s20s = _map_anchored(df["sell_20_v6_prob"].values, *SELL20_V6)
    df["sell_score"] = 0.5 * s10s + 0.5 * s20s

    p35_v20 = df["pyr_velocity_20_60"].quantile(0.35)
    pool_a = df[((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30) &
                  (df["pyr_velocity_20_60"] < p35_v20) &
                  (df["f1_neg1"].abs() < 0.005) &
                  (df["f2_pos1"].abs() < 0.005))].dropna(subset=["r20"]).reset_index(drop=True)
    pool_b = df[((df["buy_score"] >= 70) &
                  (df["sell_score"] >= 70))].dropna(subset=["r20"]).reset_index(drop=True)
    print(f"  Task A 理想多 池: n={len(pool_a):,}, r20 mean={pool_a['r20'].mean():.2f}%", flush=True)
    print(f"  Task B 矛盾段 池: n={len(pool_b):,}, r20 mean={pool_b['r20'].mean():.2f}%", flush=True)

    random.seed(42)
    sample_a = pool_a.sample(min(N_PER_QUADRANT, len(pool_a)), random_state=42).reset_index(drop=True)
    sample_b = pool_b.sample(min(N_PER_QUADRANT, len(pool_b)), random_state=42).reset_index(drop=True)
    print(f"  抽样 A: {len(sample_a)}, B: {len(sample_b)}", flush=True)

    client = OpenAI(api_key=CLOUBIC_KEY, base_url=CLOUBIC_BASE)
    total_in = total_out = 0
    cost_per_in = 3.0  # Sonnet 4.6
    cost_per_out = 15.0
    cost = 0.0
    results = []

    for task_label, samples, sys_prompt in [
        ("理想多", sample_a, SYS_PROMPT_IDEAL),
        ("矛盾段", sample_b, SYS_PROMPT_CONTRADICTION),
    ]:
        print(f"\n[{int(time.time()-t0)}s] === Task {task_label} ===", flush=True)
        for i, row in samples.iterrows():
            sample_t0 = time.time()
            ts_code = row["ts_code"]; trade_date = row["trade_date"]
            name = row.get("name", "?")
            # 1. 生成 K 线 PNG (无未来信息)
            daily_p, weekly_p = gen_history_charts(ts_code, trade_date, name)
            if not (daily_p and weekly_p):
                print(f"  [{i+1}/{len(samples)}] {ts_code} {trade_date} 图生成失败", flush=True)
                results.append({**row.to_dict(), "task": task_label,
                                  "verdict": "no_image", "conviction": 0,
                                  "key_pattern": "图生成失败", "risk_warning": ""})
                continue

            # 2. 双图视觉调用
            daily_b64 = load_image_base64(daily_p)
            weekly_b64 = load_image_base64(weekly_p)
            text, usage = call_vision_dual(client, daily_b64, weekly_b64, sys_prompt)
            if usage:
                total_in += usage.prompt_tokens
                total_out += usage.completion_tokens
                cost += (usage.prompt_tokens * cost_per_in
                         + usage.completion_tokens * cost_per_out) / 1e6
            try:
                j = json.loads(text) if text else {}
                results.append({**row.to_dict(), "task": task_label,
                                  "verdict": str(j.get("verdict", "unknown")),
                                  "conviction": int(j.get("conviction", 50)),
                                  "key_pattern": str(j.get("key_pattern", ""))[:60],
                                  "risk_warning": str(j.get("risk_warning", ""))[:60]})
            except:
                results.append({**row.to_dict(), "task": task_label,
                                  "verdict": "parse_error", "conviction": 0,
                                  "key_pattern": text[:60] if text else "",
                                  "risk_warning": ""})
            sample_dur = time.time() - sample_t0
            print(f"  [{i+1}/{len(samples)}] {ts_code} {trade_date} → {results[-1]['verdict']:8s} conv={results[-1]['conviction']:3d} ({sample_dur:.0f}s) ${cost:.3f}",
                  flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "poc_results.csv", index=False, encoding="utf-8-sig")

    # ── 评估 ──
    print(f"\n[{int(time.time()-t0)}s] === 评估 ===", flush=True)

    print(f"\n# Task A 理想多 (V7c 已推荐):", flush=True)
    a = res_df[res_df["task"] == "理想多"]
    print(a["verdict"].value_counts().to_string(), flush=True)
    for v in ["confirm","reject","no_image","parse_error","unknown"]:
        sub = a[a["verdict"] == v]
        if len(sub) >= 3:
            r20 = sub["r20"].dropna()
            print(f"  {v:8s} (n={len(sub):2d}): r20 mean={r20.mean():+.2f}% win={(r20>0).mean()*100:.0f}%",
                  flush=True)
    confirm = a[a["verdict"] == "confirm"]
    reject = a[a["verdict"] == "reject"]
    if len(confirm) >= 5 and len(reject) >= 5:
        t, p = stats.ttest_ind(confirm["r20"].dropna(), reject["r20"].dropna())
        print(f"  confirm vs reject: diff={confirm['r20'].mean()-reject['r20'].mean():+.2f}pp, p={p:.4f}",
              flush=True)
    # conviction 分位
    if len(a) > 5:
        hi = a[a["conviction"] >= 70]
        lo = a[a["conviction"] < 50]
        if len(hi) >= 3 and len(lo) >= 3:
            print(f"  high conv≥70 (n={len(hi)}): r20={hi['r20'].mean():+.2f}%, low<50 (n={len(lo)}): r20={lo['r20'].mean():+.2f}%",
                  flush=True)

    print(f"\n# Task B 矛盾段 (V7c 难判, LLM 反挖):", flush=True)
    b = res_df[res_df["task"] == "矛盾段"]
    print(b["verdict"].value_counts().to_string(), flush=True)
    for v in ["真涨","假涨","不确定","no_image","parse_error","unknown"]:
        sub = b[b["verdict"] == v]
        if len(sub) >= 3:
            r20 = sub["r20"].dropna()
            dd15 = (sub["max_dd_20"] <= -15).mean() * 100
            print(f"  {v:8s} (n={len(sub):2d}): r20={r20.mean():+.2f}% win={(r20>0).mean()*100:.0f}% dd<-15%={dd15:.0f}%",
                  flush=True)
    real_up = b[b["verdict"] == "真涨"]
    fake_up = b[b["verdict"] == "假涨"]
    if len(real_up) >= 5 and len(fake_up) >= 5:
        t, p = stats.ttest_ind(real_up["r20"].dropna(), fake_up["r20"].dropna())
        print(f"  真涨 vs 假涨: diff={real_up['r20'].mean()-fake_up['r20'].mean():+.2f}pp, p={p:.4f}",
              flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s, 总 token in={total_in:,} out={total_out:,}, 成本 ${cost:.4f}",
          flush=True)
    print(f"PNG 保存至 {CHARTS_DIR}/ ({len(list(CHARTS_DIR.glob('*.png')))} 张)", flush=True)


if __name__ == "__main__":
    main()
