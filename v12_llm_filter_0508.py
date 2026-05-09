#!/usr/bin/env python3
"""V12 矛盾段 LLM 视觉过滤 (2026-05-08).

改造 V11 vision PoC: 数据源 TDX → Tushare cache, 用于矛盾段 216 股反挖.

输入: output/v12_inference/v12_contradiction_pending_20260508.csv
输出:
  - output/v12_inference/v11_filter_results_20260508.csv (含 bull/base/bear prob, 每股一行, checkpoint)
  - output/v12_inference/v12_inference_final_20260508.csv (V7c 主推 + V11 救出, 按 r20_pred 排)
  - output/v12/charts/{ts_code}_{trade_date}_{tf}.png (复用 PNG)

成本估算: 216 股 × ~$0.0085 ≈ $1.84
模型: claude-sonnet-4-6 (cloubic API)
checkpoint: 每股写入一次, 中途崩溃可续
"""
from __future__ import annotations
import os, json, time, base64, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.cloubic")
load_dotenv(".env")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from stockagent_analysis.chart_generator import generate_kline_chart

TARGET_DATE = "20260508"
DAILY_CACHE = ROOT / "output" / "tushare_cache" / "daily"
CONTRA_CSV  = ROOT / "output" / "v12_inference" / f"v12_contradiction_pending_{TARGET_DATE}.csv"
V7C_CSV     = ROOT / "output" / "v7c_full_inference" / f"v7c_inference_{TARGET_DATE}.csv"

OUT_DIR     = ROOT / "output" / "v12_inference"
CHARTS_DIR  = ROOT / "output" / "v12" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT  = OUT_DIR / f"v11_filter_results_{TARGET_DATE}.csv"
FINAL_OUT   = OUT_DIR / f"v12_inference_final_{TARGET_DATE}.csv"

CLOUBIC_KEY = os.environ.get("CLOUBIC_API_KEY")
CLOUBIC_BASE = "https://api.cloubic.com/v1"
if not CLOUBIC_KEY:
    print("缺 CLOUBIC_API_KEY (检查 .env / .env.cloubic)"); sys.exit(1)

# ──── V11 prompt (盲测, 不给 V7c 上下文) ────
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


# ──── 数据源: Tushare daily cache (替代 read_tdx) ────
_daily_cache = None

def load_daily_cache():
    """一次性加载全市场 daily, 切到 dict by ts_code (耗时 ~30s)."""
    global _daily_cache
    if _daily_cache is not None: return _daily_cache
    files = sorted(DAILY_CACHE.glob("*.parquet"))
    end_int = int(TARGET_DATE)
    parts = [pd.read_parquet(f) for f in files if int(f.stem) <= end_int]
    big = pd.concat(parts, ignore_index=True)
    big["trade_date"] = big["trade_date"].astype(str)
    big = big.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    _daily_cache = {ts: g.reset_index(drop=True) for ts, g in big.groupby("ts_code")}
    print(f"  加载 daily cache: {len(_daily_cache)} 股")
    return _daily_cache


def get_ohlc_rows(ts_code, end_date_inclusive):
    """返回 list of {ts, open, high, low, close, volume}, 严格 ≤ end_date."""
    cache = load_daily_cache()
    df = cache.get(ts_code)
    if df is None: return None
    df = df[df["trade_date"] <= end_date_inclusive]
    if df.empty: return None
    rows = []
    for _, r in df.iterrows():
        td = r["trade_date"]
        # 转 yyyy-mm-dd
        ts = f"{td[:4]}-{td[4:6]}-{td[6:8]}"
        rows.append({
            "ts": ts,
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]),   "close": float(r["close"]),
            "volume": float(r.get("vol", r.get("volume", 0))),
        })
    return rows


def daily_to_periodic(rows, freq):
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
    rows = get_ohlc_rows(ts_code, trade_date)
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


def encode_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def call_vision_3tf(client, m_b64, w_b64, d_b64, max_retry=2):
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{m_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{w_b64}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{d_b64}"}},
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
            print(f"    LLM err (retry {r+1}/{max_retry}): {str(e)[:120]}")
            if r < max_retry - 1:
                time.sleep(3); continue
            return None, None


def extract_scenarios(j):
    bull_p = base_p = bear_p = np.nan
    bull_t = base_t = bear_t = np.nan
    for s in j.get("scenarios", []):
        n = str(s.get("name","")).lower()
        try: p = float(s.get("prob", 0))
        except: p = np.nan
        try: t = float(s.get("target_pct_20d", 0))
        except: t = np.nan
        if "bull" in n: bull_p, bull_t = p, t
        elif "base" in n: base_p, base_t = p, t
        elif "bear" in n: bear_p, bear_t = p, t
    return bull_p, base_p, bear_p, bull_t, base_t, bear_t


def main():
    t0 = time.time()
    if not CONTRA_CSV.exists():
        print(f"❌ 缺 {CONTRA_CSV}"); sys.exit(1)

    contra = pd.read_csv(CONTRA_CSV, dtype={"ts_code":str})
    contra["trade_date"] = contra["trade_date"].astype(str)
    print(f"=== V12 矛盾段 LLM 视觉过滤 ({TARGET_DATE}) ===")
    print(f"待评估: {len(contra)} 股")

    # checkpoint: 跳过已处理
    done_codes = set()
    if CHECKPOINT.exists():
        done = pd.read_csv(CHECKPOINT, dtype={"ts_code":str})
        done_codes = set(done["ts_code"].tolist())
        print(f"checkpoint: 已完成 {len(done_codes)} 股, 跳过")

    todo = contra[~contra["ts_code"].isin(done_codes)].reset_index(drop=True)
    if len(todo) == 0:
        print("全部已完成, 直接合并输出")
    else:
        print(f"待跑: {len(todo)} 股, 预估 {len(todo)*5/60:.1f} 分钟, 成本 ~${len(todo)*0.0085:.2f}\n")

    # 预热 daily cache
    print("加载 daily cache ...")
    load_daily_cache()
    print(f"cache 加载耗时 {time.time()-t0:.0f}s\n")

    client = OpenAI(api_key=CLOUBIC_KEY, base_url=CLOUBIC_BASE)

    total_in = total_out = 0
    cost_sum = 0.0
    n_ok = n_fail_img = n_fail_llm = n_fail_parse = 0

    write_header = not CHECKPOINT.exists()

    for idx, row in todo.iterrows():
        sample_t0 = time.time()
        ts_code = row["ts_code"]; trade_date = str(row["trade_date"])

        # 1. 生成 3 图
        m_p, w_p, d_p = gen_3tf_charts(ts_code, trade_date)
        if not (m_p and w_p and d_p):
            print(f"  [{idx+1:3d}/{len(todo)}] {ts_code} 图生成失败")
            n_fail_img += 1
            rec = {**row.to_dict(), "v11_status": "no_image",
                   "bull_prob": np.nan, "base_prob": np.nan, "bear_prob": np.nan}
        else:
            # 2. LLM 视觉
            m_b64 = encode_b64(m_p); w_b64 = encode_b64(w_p); d_b64 = encode_b64(d_p)
            text, usage = call_vision_3tf(client, m_b64, w_b64, d_b64)
            if usage:
                total_in += usage.prompt_tokens
                total_out += usage.completion_tokens
                cost_sum += (usage.prompt_tokens * 3.0 + usage.completion_tokens * 15.0) / 1e6

            if text is None:
                n_fail_llm += 1
                rec = {**row.to_dict(), "v11_status": "llm_error",
                       "bull_prob": np.nan, "base_prob": np.nan, "bear_prob": np.nan}
            else:
                j = parse_json_robust(text)
                if j is None:
                    n_fail_parse += 1
                    rec = {**row.to_dict(), "v11_status": "parse_error",
                           "bull_prob": np.nan, "base_prob": np.nan, "bear_prob": np.nan,
                           "raw_text": (text or "")[:200]}
                else:
                    bp, bsp, brp, bt, bst, brt = extract_scenarios(j)
                    n_ok += 1
                    rec = {**row.to_dict(), "v11_status": "ok",
                           "bull_prob": bp, "base_prob": bsp, "bear_prob": brp,
                           "bull_target": bt, "base_target": bst, "bear_target": brt,
                           "trend_main": str(j.get("trend_main",""))[:30],
                           "trend_strength": str(j.get("trend_strength",""))[:20],
                           "ma_alignment": str(j.get("ma_alignment",""))[:20],
                           "key_pattern": str(j.get("key_pattern",""))[:120],
                           "elliott_phase": str(j.get("elliott_phase",""))[:80]}

        # 写入 checkpoint (每股一次)
        df_one = pd.DataFrame([rec])
        df_one.to_csv(CHECKPOINT, mode="a", header=write_header, index=False, encoding="utf-8-sig")
        write_header = False

        bp_str = f"{rec.get('bull_prob', float('nan')):.2f}" if not pd.isna(rec.get("bull_prob")) else "N/A"
        dur = time.time() - sample_t0
        print(f"  [{idx+1:3d}/{len(todo)}] {ts_code} {row['industry'][:6]:6s} "
              f"r20={row['r20_pred']:+5.2f}% → bull={bp_str} "
              f"({rec.get('v11_status','?'):6s}, {dur:.0f}s, ${cost_sum:.3f})")

    print(f"\n=== LLM 视觉评估完成 ===")
    print(f"  ok={n_ok}, no_image={n_fail_img}, llm_err={n_fail_llm}, parse_err={n_fail_parse}")
    print(f"  token in={total_in:,} out={total_out:,}, 累计成本 ${cost_sum:.4f}")
    print(f"  耗时: {time.time()-t0:.0f}s")

    # ──── 合并到 V12 final ────
    print(f"\n=== 合并 V7c 主推 + V11 救出 → V12 final ===")
    res = pd.read_csv(CHECKPOINT, dtype={"ts_code":str})
    rescued = res[(res["v11_status"] == "ok") & (res["bull_prob"] >= 0.5)].copy()
    print(f"  V11 救出: {len(rescued)} 股 (bull≥0.5, total ok={n_ok if n_ok else len(res[res['v11_status']=='ok'])})")

    v7c = pd.read_csv(V7C_CSV, dtype={"ts_code":str})
    main_pool = v7c[v7c["v7c_recommend"] == True].copy()
    main_pool["v12_source"] = "V7c-main"
    if len(rescued) > 0:
        rescued["v12_source"] = "V11-rescued-contradiction"
        common_cols = [c for c in main_pool.columns if c in rescued.columns]
        v12 = pd.concat([main_pool[common_cols], rescued[common_cols]], ignore_index=True)
    else:
        v12 = main_pool

    v12 = v12.sort_values("r20_pred", ascending=False).reset_index(drop=True)
    v12["rank"] = v12.index + 1
    show_cols = ["rank","ts_code","industry","buy_score","sell_score","r20_pred",
                  "sell_20_v6_prob","quadrant","v12_source"]
    show_cols = [c for c in show_cols if c in v12.columns]
    v12.to_csv(FINAL_OUT, index=False, encoding="utf-8-sig")

    print(f"\n=== V12 最终推荐 (n={len(v12)}, 按 r20_pred 降序) ===")
    print(v12[show_cols].to_string(index=False))

    if len(rescued) > 0:
        print(f"\n=== V11 救出明细 ===")
        rcols = ["ts_code","industry","buy_score","sell_score","r20_pred",
                  "bull_prob","base_prob","bear_prob","trend_strength","key_pattern"]
        rcols = [c for c in rcols if c in rescued.columns]
        print(rescued.sort_values("r20_pred", ascending=False)[rcols].to_string(index=False))

    print(f"\n输出: {FINAL_OUT}")


if __name__ == "__main__":
    main()
