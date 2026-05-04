#!/usr/bin/env python3
"""P4 LLM 反馈学习 PoC — 方向 A.

测试假设: LLM 看 V7c 推荐池里的股票特征做综合 yes/no 判断,
是否能比 V7c 单独 alpha 更高?

成本控制:
  - 200 样本 (跨 8 月分散), DeepSeek-chat ($0.14/M in + $0.28/M out)
  - 预估 ~$0.04 总成本

输入 LLM 的特征 (~700 token):
  ts_code, name, industry
  buy_score, sell_score (V7c)
  r10_pred, r20_pred, sell_10_prob, sell_20_prob
  PE/PB/MV/换手
  mfk_pyramid_top_heavy, pyr_velocity_20_60
  f1_neg1, f2_pos1
  mkt_ret_5d/20d/60d, regime_id

LLM 输出 (~150 token JSON):
  decision: "buy" | "skip"
  confidence: 0-100
  reason: 50 字内

输出: output/v8/llm_poc_results.csv (200 样本, 含 LLM decision + 实际 r20)
评估: yes 子集 vs no 子集 alpha 差异
"""
from __future__ import annotations
import os, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
OUT_DIR = ROOT / "output" / "v8"
OUT_DIR.mkdir(exist_ok=True)
TEST_START = "20250501"
TEST_END   = "20260126"

R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)

N_SAMPLES = 200
SYS_PROMPT = """你是一名 A 股量化分析师, 评估一只股票当前是否值得买入 (持有 20 日).

输入是该股的多维量化数据:
- V7c 系统评分 (buy_score 70-85 是机器学习推荐区间, sell_score 越低越安全)
- ML 预测 (r10/r20 预期收益, sell_10/sell_20 跌破阈值概率)
- 基本面 (PE/PB/市值)
- 资金流 (mfk_pyramid_top_heavy 机构占比, pyr_velocity_20_60 机构趋势变化)
- 市场状态 (regime_id, mkt_ret_60d 60 日大盘表现)

任务: 综合判断, 输出 JSON 格式 (无其他文字):
{"decision": "buy" 或 "skip", "confidence": 0-100, "reason": "30字内中文理由"}

判断原则:
- buy: 多维信号一致看多, 风险可控, 不在过激段
- skip: 信号矛盾、估值过高、所处行业景气下行、回撤风险明显
"""


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
        ROOT / "output" / "moneyflow_v2" / "features.parquet",
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


def build_prompt(row) -> str:
    """构造单股决策 prompt."""
    return f"""股票: {row['ts_code']} {row.get('name', '?')} ({row.get('industry', '?')})
日期: {row['trade_date']}

【V7c 系统评分】
buy_score: {row['buy_score']:.0f}/100 (70-85 最优区间)
sell_score: {row['sell_score']:.0f}/100 (≤30 安全)

【ML 预测】
r10 预测: {row.get('r10_pred', 0):+.2f}%, r20 预测: {row.get('r20_pred', 0):+.2f}%
sell_10 概率: {row.get('sell_10_v6_prob', 0)*100:.0f}%, sell_20 概率: {row.get('sell_20_v6_prob', 0)*100:.0f}%

【基本面】
PE: {row.get('pe_ttm', 0):.1f}, 总市值: {row.get('total_mv', 0)/1e4:.1f} 亿

【资金流】
机构占比 (mfk_pyramid_top_heavy): {row.get('mfk_pyramid_top_heavy', 0):.2f} (<0.45 较好)
机构趋势 (pyr_velocity_20_60): {row.get('pyr_velocity_20_60', 0):+.3f} (<0 表示机构温和退出)
f1_neg1 (大跌+主力流入): {row.get('f1_neg1', 0):+.4f} (绝对值 < 0.005 静默)
f2_pos1 (大涨+主力流出): {row.get('f2_pos1', 0):+.4f} (绝对值 < 0.005 静默)

【市场状态】
regime_id: {int(row.get('regime_id', 0))}
60 日大盘涨幅: {row.get('mkt_ret_60d', 0):+.1f}%

请输出 JSON 决策 (仅 JSON, 无其他文字):
"""


def call_llm(client, prompt, model="deepseek-chat", max_retry=2):
    for retry in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200, temperature=0.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if retry < max_retry - 1:
                time.sleep(1)
                continue
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

    # V7c 推荐池
    p35_v20 = df["pyr_velocity_20_60"].quantile(0.35)
    pool_mask = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30) &
                  (df["pyr_velocity_20_60"] < p35_v20) &
                  (df["f1_neg1"].abs() < 0.005) &
                  (df["f2_pos1"].abs() < 0.005))
    pool = df[pool_mask].dropna(subset=["r20"]).reset_index(drop=True)
    print(f"  V7c 推荐池: n={len(pool):,}", flush=True)

    # 200 样本: 跨 8 月分层抽样
    pool["yyyymm"] = pool["trade_date"].astype(str).str[:6]
    samples = []
    n_per_month = N_SAMPLES // pool["yyyymm"].nunique()
    random.seed(42)
    for m, g in pool.groupby("yyyymm"):
        n_take = min(n_per_month, len(g))
        idx = random.sample(range(len(g)), n_take)
        samples.append(g.iloc[idx])
    samples = pd.concat(samples).reset_index(drop=True)
    if len(samples) > N_SAMPLES:
        samples = samples.sample(N_SAMPLES, random_state=42).reset_index(drop=True)
    print(f"  抽样: {len(samples)} 个 (跨 {samples['yyyymm'].nunique()} 月)", flush=True)

    # DeepSeek
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                     base_url="https://api.deepseek.com/v1")

    print(f"\n[{int(time.time()-t0)}s] 开始 LLM 调用 ({len(samples)} 个)...", flush=True)
    results = []
    total_in = total_out = 0
    for i, row in samples.iterrows():
        prompt = build_prompt(row)
        resp_text, usage = call_llm(client, prompt)
        if resp_text is None:
            results.append({**row.to_dict(), "llm_decision": "error",
                              "llm_confidence": 0, "llm_reason": ""})
            continue
        try:
            j = json.loads(resp_text)
            results.append({**row.to_dict(),
                              "llm_decision": j.get("decision", "unknown"),
                              "llm_confidence": int(j.get("confidence", 50)),
                              "llm_reason": str(j.get("reason", "")).strip()[:60]})
            if usage:
                total_in += usage.prompt_tokens
                total_out += usage.completion_tokens
        except:
            results.append({**row.to_dict(), "llm_decision": "parse_error",
                              "llm_confidence": 0, "llm_reason": resp_text[:60]})
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(samples)}] {int(time.time()-t0)}s, "
                  f"tokens: in={total_in:,} out={total_out:,}, "
                  f"成本估: ${total_in*0.14/1e6 + total_out*0.28/1e6:.4f}", flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "llm_poc_results.csv", index=False, encoding="utf-8-sig")

    # ── 评估 ──
    print(f"\n[{int(time.time()-t0)}s] === LLM 决策分布 ===", flush=True)
    print(res_df["llm_decision"].value_counts().to_string(), flush=True)

    print(f"\n=== buy vs skip 子集对比 ===", flush=True)
    for dec in ["buy", "skip"]:
        sub = res_df[res_df["llm_decision"] == dec]
        if len(sub) < 10: continue
        r20 = sub["r20"].dropna()
        if len(r20) > 0:
            print(f"  {dec:5s}: n={len(sub):3d}, r20 mean={r20.mean():+.2f}%, "
                  f"win={(r20>0).mean()*100:.1f}%, std={r20.std():.2f}",
                  flush=True)

    print(f"\n=== 配对 t-test (LLM yes vs no 子集) ===", flush=True)
    yes = res_df[res_df["llm_decision"] == "buy"]["r20"].dropna()
    no = res_df[res_df["llm_decision"] == "skip"]["r20"].dropna()
    if len(yes) > 10 and len(no) > 10:
        t, p = stats.ttest_ind(yes, no)
        diff = yes.mean() - no.mean()
        print(f"  yes n={len(yes)}, no n={len(no)}", flush=True)
        print(f"  diff = {diff:+.2f}pp, t={t:.3f}, p={p:.4f}", flush=True)
        if p < 0.05 and diff > 0:
            print(f"  ✅ LLM 二次过滤显著有效 (yes 子集 r20 显著 > no)", flush=True)
        elif p < 0.05 and diff < 0:
            print(f"  ⚠️ LLM 反向 (no 子集反而更好)", flush=True)
        else:
            print(f"  ❌ LLM 二次过滤无显著差异", flush=True)

    print(f"\n=== 高置信度 buy 段 (confidence >= 70) ===", flush=True)
    high_conf = res_df[(res_df["llm_decision"] == "buy") & (res_df["llm_confidence"] >= 70)]
    if len(high_conf) >= 10:
        r20 = high_conf["r20"].dropna()
        print(f"  n={len(high_conf)}, r20 mean={r20.mean():+.2f}%, win={(r20>0).mean()*100:.1f}%",
              flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s, "
          f"总 token: in={total_in:,} out={total_out:,}, "
          f"实际成本: ${total_in*0.14/1e6 + total_out*0.28/1e6:.4f}", flush=True)


if __name__ == "__main__":
    main()
