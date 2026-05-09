#!/usr/bin/env python3
"""V11 视觉 PoC: 5 大改进 (基于 claude-trading-skills + 缠论 + Elliott).

5 大改进:
  1. 多场景概率分布 (Bull/Base/Bear/Alt) 替代二元 confirm/reject
  2. 三时间框架 (月 120 + 周 60 + 日 60)
  3. 盲测: prompt 不给 V7c 上下文
  4. 显式缠论 + Elliott + Dow 视角注入
  5. Pydantic 鲁棒解析

200 样本 = 50 each from 4 V7c 象限 (理想多/矛盾段/主流空/沉寂)
评估: 校准度 + 集中度 + 4 象限交叉

输入: 已预生成 K 线 PNG (output/v11/charts/), 严格无未来信息
输出: output/v11/poc_results.csv
"""
from __future__ import annotations
import os, struct, json, time, random, base64, re
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
from stockagent_analysis.chart_generator import generate_kline_chart

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
OUT_DIR = ROOT / "output" / "v11"
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

N_PER_QUADRANT = 50  # 50 each × 4 = 200 总样本
CLOUBIC_KEY = os.environ["CLOUBIC_API_KEY"]
CLOUBIC_BASE = "https://api.cloubic.com/v1"


# ==================== V11 改进后 prompt (盲测 + 5 域 + 多场景) ====================
SYS_PROMPT_V11 = """你是 A 股专业 K 线视觉技术分析师, 严格基于图像本身做客观分析 (不依赖外部信息).

你将看到同一只股票的 3 张 K 线综合图 (按顺序: 月线 / 周线 / 日线), 每张图含蜡烛+MA5/10/20/60/120/250+布林带+趋势线+成交量+MACD+RSI+KDJ.

## 分析框架 (必须覆盖 5 大域)

### 1. 趋势分析 (Dow Theory)
- 主要趋势 (月线): 上升/下降/震荡, 强度
- 次要趋势 (周线): 是否与主要趋势同向?
- 短期趋势 (日线): 当前位置
- HH/HL pattern (Higher High/Higher Low) 是否完整?

### 2. 支撑/阻力 (3+ touches 才算 strong)
- 月线/周线 关键长期支撑阻力位
- 日线 短期支撑阻力位
- 角色翻转 (broken support → resistance)

### 3. 移动均线
- 价格 vs MA20/60/250 关系 (above/below/between)
- 金叉/死叉 (Golden Cross / Death Cross)
- MA 排列 (bullish 多头排列 / bearish 空头排列)

### 4. 成交量
- 量价配合 (rising price + rising volume = healthy)
- 量价背离 (rising price + falling volume = weak)
- 异常放量/缩量

### 5. 形态 + 缠论 + Elliott Wave
- 反转形态: hammer/engulfing/双底/双顶/M 头/W 底
- 延续形态: bull flag/bear flag/triangle
- **缠论视角**: 当前是顶分型/底分型? 在中枢震荡? 突破中枢? 接近哪类买卖点 (一类背驰/二类回撤/三类突破)?
- **Elliott Wave**: 周线视角下当前在主升浪 (1/3/5 推动) 还是修正浪 (a/b/c)? 关键 Fibonacci 位 (38.2/50/61.8)

## 输出要求

输出 JSON 对象 (不要 markdown 包裹):
{
  "trend_main": "uptrend/downtrend/sideways",
  "trend_strength": "strong/moderate/weak",
  "ma_alignment": "bullish/bearish/neutral",
  "volume_confirms_price": true/false,
  "key_pattern": "30字内核心形态描述, 含缠论术语",
  "elliott_phase": "30字内 Elliott Wave 当前位置",
  "scenarios": [
    {"name": "Bull", "prob": 0.0-1.0, "target_pct_20d": +/- 数值, "invalidation": "30字内场景失效条件"},
    {"name": "Base", "prob": 0.0-1.0, "target_pct_20d": +/- 数值, "invalidation": "30字内"},
    {"name": "Bear", "prob": 0.0-1.0, "target_pct_20d": +/- 数值, "invalidation": "30字内"}
  ]
}

scenarios 概率求和必须等于 1.0. target_pct_20d 是该场景下 20 个交易日预期涨跌幅 (% as number).

注意: 你不知道 V7c 量化系统对这只股的看法, 完全独立判断."""


# ==================== TDX OHLC 读取 (严格无未来信息) ====================
def read_tdx(market, code, end_date_inclusive):
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
        if di > end_int: continue  # 严格过滤未来
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


def daily_to_periodic(rows, freq):
    """日线 → 周线 / 月线 (无未来信息)."""
    if not rows: return []
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["ts"])
    df = df.set_index("dt")
    if freq == "W":
        agg = df.resample("W-FRI").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    elif freq == "M":
        agg = df.resample("ME").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    else:
        return rows
    agg["ts"] = agg.index.strftime("%Y-%m-%d")
    return agg[["ts","open","high","low","close","volume"]].to_dict("records")


def gen_3tf_charts(ts_code, trade_date, name="?"):
    """生成月/周/日 3 张图. 复用已 saved 跳过."""
    code, market = code_market(ts_code)
    rows = read_tdx(market, code, trade_date)
    if not rows or len(rows) < 60:
        return None, None, None

    paths = {}
    for tf, freq, take_n in [("month","M",120), ("week","W",60), ("day",None,60)]:
        target_path = CHARTS_DIR / f"{ts_code}_{trade_date}_{tf}.png"
        if target_path.exists():
            paths[tf] = target_path; continue
        if freq:
            data = daily_to_periodic(rows, freq)
        else:
            data = rows[-take_n:]
        if not data or len(data) < 10:
            paths[tf] = None; continue
        data = data[-take_n:]
        df = pd.DataFrame(data)
        png = generate_kline_chart(df, tf, ts_code, name, save_path=target_path)
        paths[tf] = target_path if png else None

    return paths.get("month"), paths.get("week"), paths.get("day")


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ==================== 鲁棒 JSON 解析 ====================
def parse_json_robust(text):
    if not text: return None
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m: text = m.group(1)
    else:
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if m: text = m.group(0)
    try: return json.loads(text)
    except: return None


def call_vision_3tf(client, month_b64, week_b64, day_b64, max_retry=2):
    """三图视觉调用."""
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{month_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{week_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{day_b64}"}},
        {"type": "text", "text": "上面三张图按顺序是: 月线 (10 年)、周线 (60 周)、日线 (60 日). 请按 5 域框架做分析, 输出 JSON."}
    ]
    for r in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="claude-sonnet-4-6",
                messages=[
                    {"role": "system", "content": SYS_PROMPT_V11},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=600, temperature=0.0,
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if r < max_retry - 1:
                time.sleep(2); continue
            return None, None


# ==================== V7c 评分 (用于抽样, 不给 LLM) ====================
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
    booster = lgb.Booster(model_str=(d / "classifier.txt").read_text(encoding="utf-8"))
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

    # 4 象限定义
    p35_v20 = df["pyr_velocity_20_60"].quantile(0.35)
    pool_ideal = df[((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                      (df["sell_score"] <= 30) &
                      (df["pyr_velocity_20_60"] < p35_v20) &
                      (df["f1_neg1"].abs() < 0.005) &
                      (df["f2_pos1"].abs() < 0.005))].dropna(subset=["r20"]).reset_index(drop=True)
    pool_contradiction = df[((df["buy_score"] >= 70) & (df["sell_score"] >= 70))
                              ].dropna(subset=["r20"]).reset_index(drop=True)
    pool_main_short = df[((df["buy_score"] <= 30) & (df["sell_score"] >= 70))
                            ].dropna(subset=["r20"]).reset_index(drop=True)
    pool_quiet = df[((df["buy_score"] <= 30) & (df["sell_score"] <= 30))
                       ].dropna(subset=["r20"]).reset_index(drop=True)

    print(f"  V7c 4 象限池: 理想多={len(pool_ideal):,}, 矛盾段={len(pool_contradiction):,}, "
          f"主流空={len(pool_main_short):,}, 沉寂={len(pool_quiet):,}", flush=True)

    random.seed(42)
    samples = []
    for label, pool in [
        ("理想多", pool_ideal), ("矛盾段", pool_contradiction),
        ("主流空", pool_main_short), ("沉寂", pool_quiet)
    ]:
        if len(pool) == 0: continue
        n_take = min(N_PER_QUADRANT, len(pool))
        sub = pool.sample(n=n_take, random_state=42).copy()
        sub["quadrant"] = label
        samples.append(sub)
    samples = pd.concat(samples, ignore_index=True)
    print(f"  抽样: {len(samples)}", flush=True)

    client = OpenAI(api_key=CLOUBIC_KEY, base_url=CLOUBIC_BASE)
    total_in = total_out = 0
    cost = 0.0
    results = []

    for i, row in samples.iterrows():
        sample_t0 = time.time()
        ts_code = row["ts_code"]; trade_date = str(row["trade_date"])
        name = row.get("name", "?")
        # 1. 生成 3 时间框架图 (无未来信息)
        m_p, w_p, d_p = gen_3tf_charts(ts_code, trade_date, name)
        if not (m_p and w_p and d_p):
            print(f"  [{i+1}/{len(samples)}] {ts_code} {trade_date} 图生成失败", flush=True)
            results.append({**row.to_dict(), "v11_status": "no_image",
                              "bull_prob": np.nan, "base_prob": np.nan,
                              "bear_prob": np.nan, "key_pattern": "",
                              "elliott_phase": "", "trend_strength": ""})
            continue

        # 2. 视觉调用
        m_b64 = encode_image_b64(m_p)
        w_b64 = encode_image_b64(w_p)
        d_b64 = encode_image_b64(d_p)
        text, usage = call_vision_3tf(client, m_b64, w_b64, d_b64)
        if usage:
            total_in += usage.prompt_tokens
            total_out += usage.completion_tokens
            cost += (usage.prompt_tokens * 3.0 + usage.completion_tokens * 15.0) / 1e6

        # 3. 解析 JSON
        j = parse_json_robust(text)
        if j is None:
            results.append({**row.to_dict(), "v11_status": "parse_error",
                              "bull_prob": np.nan, "base_prob": np.nan,
                              "bear_prob": np.nan,
                              "key_pattern": (text or "")[:80],
                              "elliott_phase": "", "trend_strength": ""})
        else:
            # 提取场景概率
            scenarios = j.get("scenarios", [])
            bull_p = base_p = bear_p = np.nan
            bull_t = base_t = bear_t = np.nan
            for s in scenarios:
                name_s = str(s.get("name","")).lower()
                p = float(s.get("prob", 0))
                t = float(s.get("target_pct_20d", 0))
                if "bull" in name_s: bull_p, bull_t = p, t
                elif "base" in name_s: base_p, base_t = p, t
                elif "bear" in name_s: bear_p, bear_t = p, t
            results.append({**row.to_dict(), "v11_status": "ok",
                              "bull_prob": bull_p, "base_prob": base_p, "bear_prob": bear_p,
                              "bull_target": bull_t, "base_target": base_t, "bear_target": bear_t,
                              "trend_main": str(j.get("trend_main",""))[:30],
                              "trend_strength": str(j.get("trend_strength",""))[:20],
                              "ma_alignment": str(j.get("ma_alignment",""))[:20],
                              "volume_confirms": bool(j.get("volume_confirms_price", False)),
                              "key_pattern": str(j.get("key_pattern",""))[:100],
                              "elliott_phase": str(j.get("elliott_phase",""))[:80]})
        sample_dur = time.time() - sample_t0
        last = results[-1]
        bp = last.get("bull_prob", np.nan)
        bp_str = f"{bp:.2f}" if not pd.isna(bp) else "N/A"
        print(f"  [{i+1:3d}/{len(samples)}] {ts_code} {trade_date} ({row['quadrant']}) → "
              f"status={last.get('v11_status','?'):10s} bull={bp_str} ({sample_dur:.0f}s) ${cost:.3f}",
              flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "poc_results.csv", index=False, encoding="utf-8-sig")

    # ── 评估 ──
    print(f"\n[{int(time.time()-t0)}s] === 评估 ===", flush=True)
    ok = res_df[res_df["v11_status"] == "ok"].copy()
    print(f"成功解析: {len(ok)}/{len(res_df)}", flush=True)
    if len(ok) < 20:
        print("⚠ 样本太少, 评估有限", flush=True)
        return

    # 1. 校准度 (按 bull_prob bin 看实际胜率)
    print(f"\n# 1. 校准度 (bull_prob bin → 实际胜率)", flush=True)
    ok["bull_bin"] = pd.cut(ok["bull_prob"], bins=[0, 0.2, 0.35, 0.5, 0.7, 1.0],
                             labels=["0-20","20-35","35-50","50-70","70-100"])
    for label in ["0-20","20-35","35-50","50-70","70-100"]:
        sub = ok[ok["bull_bin"] == label]
        if len(sub) >= 5:
            r20 = sub["r20"].dropna()
            print(f"  bull_prob {label}% (n={len(sub):3d}): r20={r20.mean():+.2f}% 胜率={(r20>0).mean()*100:.0f}% target_avg={sub['bull_target'].mean():+.1f}%",
                  flush=True)

    # 2. 集中度 (高 bull_prob vs 低)
    print(f"\n# 2. 集中度 (high vs low bull_prob)", flush=True)
    hi = ok[ok["bull_prob"] >= 0.5]
    lo = ok[ok["bull_prob"] < 0.3]
    if len(hi) >= 10 and len(lo) >= 10:
        t, p = stats.ttest_ind(hi["r20"].dropna(), lo["r20"].dropna())
        print(f"  bull>=0.5 (n={len(hi)}): r20={hi['r20'].mean():+.2f}% 胜率={(hi['r20']>0).mean()*100:.0f}%", flush=True)
        print(f"  bull<0.3  (n={len(lo)}): r20={lo['r20'].mean():+.2f}% 胜率={(lo['r20']>0).mean()*100:.0f}%", flush=True)
        print(f"  diff={hi['r20'].mean()-lo['r20'].mean():+.2f}pp, p={p:.4f}", flush=True)

    # 3. V7c 4 象限交叉 (V7c × LLM bull_prob)
    print(f"\n# 3. V7c 4 象限 × LLM bull_prob 交叉", flush=True)
    for q in ["理想多","矛盾段","主流空","沉寂"]:
        q_sub = ok[ok["quadrant"] == q]
        if len(q_sub) < 10: continue
        q_hi = q_sub[q_sub["bull_prob"] >= 0.5]
        q_lo = q_sub[q_sub["bull_prob"] < 0.3]
        print(f"  [{q}] n={len(q_sub):3d}, r20 mean={q_sub['r20'].mean():+.2f}%", flush=True)
        if len(q_hi) >= 5:
            print(f"    + LLM bull>=0.5 (n={len(q_hi):2d}): r20={q_hi['r20'].mean():+.2f}%", flush=True)
        if len(q_lo) >= 5:
            print(f"    + LLM bull<0.3  (n={len(q_lo):2d}): r20={q_lo['r20'].mean():+.2f}%", flush=True)

    # 4. trend_strength × r20
    print(f"\n# 4. LLM trend_strength × r20", flush=True)
    for ts in ["strong","moderate","weak"]:
        sub = ok[ok["trend_strength"] == ts]
        if len(sub) >= 5:
            r20 = sub["r20"].dropna()
            print(f"  {ts:8s} (n={len(sub):3d}): r20={r20.mean():+.2f}% 胜率={(r20>0).mean()*100:.0f}%", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s, token in={total_in:,} out={total_out:,}, 成本 ${cost:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
