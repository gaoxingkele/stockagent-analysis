#!/usr/bin/env python3
"""V9 LLM 双任务 PoC.

Task A: 理想多确定性增强
  输入: V7c 推荐池 (BUY 70-85 + SELL≤30 + pyr_v20_60<p35 + 双静默)
  LLM: 给 conviction 0-100 评分
  评估: high conviction (>=70) vs low (<70) 子集 r20

Task B: 矛盾段反挖
  输入: V7c 矛盾段 (BUY≥70 + SELL≥70)
  LLM: 识别 "真涨" / "假涨" / "不确定"
  评估: LLM "真涨" 子集 r20 vs 矛盾段平均

LLM 输入 (避免 V8 失败重演):
  - V7c 评分 (量化结论)
  - AKShare 8 指标 (LGBM 没用 ⭐)
  - 行业 (半导体 +12.4pp, 出版 -2.9pp 等)
  - regime (3 是 alpha 主战场)
  - mfk 关键状态 (显式暴露 LGBM 内部信号)

成本: 60 样本 × DeepSeek = ~$0.05
"""
from __future__ import annotations
import os, json, time, random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.cloubic")
load_dotenv(".env")

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
V9_DIR = ROOT / "output" / "v9_data"
OUT_DIR = ROOT / "output" / "v9"
OUT_DIR.mkdir(exist_ok=True)
TEST_START = "20250501"
TEST_END   = "20260126"

R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)

N_PER_TASK = 30  # 每个任务 30 样本 PoC
CLOUBIC_KEY = os.environ["CLOUBIC_API_KEY"]
CLOUBIC_BASE = "https://api.cloubic.com/v1"

# 行业先验 (来自 V4 supplements)
TOP_INDUSTRIES = {"半导体","元件","化学制品","染料涂料","塑胶","能源金属","半导体设备"}
WEAK_INDUSTRIES = {"出版","食品加工","路桥","证券","化妆品"}


SYS_TASK_A = """你是 A 股量化系统的高级审核员.
背景: V7c 量化系统已筛出"理想多"段股票 (高 buy + 低 sell, OOS 实测 r20=+6.31%).
你的任务: 在已通过的股票中识别"high conviction" (top 30% 确定性最高) 和 "low conviction" (bot 30%).
不要否决 V7c, 仅评估它的确定性强度.

判断要点:
- 信号一致性 (V7c+AKShare+行业 是否一致)
- 拥挤交易风险 (高关注度 + 高综合分 是否过热)
- 行业地位 (热门赛道 vs 冷门防御)
- 资金动作 (机构进场 vs 散户接盘)

输出 JSON (无其他文字):
{"conviction": 0-100 数值, "level": "high"/"medium"/"low", "key_reason": "30字内"}"""


SYS_TASK_B = """你是 A 股矛盾信号解谜专家.
背景: V7c 系统识别"矛盾段" (高 buy + 高 sell), OOS 实测 r20=+3.07%, 但 dd<-15%=29% 风险高.
你的任务: 在矛盾段中识别少数"真涨股" (信号矛盾但实际后续上涨).

判断要点:
- 矛盾本质: 异常事件 (大涨派发/大跌吸筹) 是否是 "洗盘后突破" 或 "顶部派发"?
- 资金侧支撑: 机构是否仍在加仓?
- 行业景气度
- 市场状态 (regime)

输出 JSON (无其他文字):
{"verdict": "真涨"/"假涨"/"不确定", "confidence": 0-100, "key_reason": "30字内"}"""


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


def build_summary(row):
    """构造给 LLM 的多维度数据 (LGBM 没用的部分 + 关键内部信号)."""
    industry = row.get("industry", "?")
    industry_tag = ""
    if industry in TOP_INDUSTRIES: industry_tag = "(实测 alpha +6~+12pp 强势行业)"
    elif industry in WEAK_INDUSTRIES: industry_tag = "(实测 alpha -1~-3pp 弱势行业)"

    regime = int(row.get("regime_id", 0))
    regime_tag = "(alpha 主战场)" if regime == 3 else ("(弱 alpha 段)" if regime == 0 else "")

    # AKShare 8 指标 (LGBM 没用 ⭐)
    ak_block = ""
    if pd.notna(row.get("ef_score")):
        ak_block = f"""【AKShare 数据 (LGBM 模型外, 来自东财+雪球)】
- 东财综合评分: {row.get('ef_score', 0):.0f}/100
- 关注指数: {row.get('ef_focus', 0):.0f}/100
- 机构参与度: {row.get('ef_inst_pct', 0)*100:.0f}%
- 主力成本偏离: {row.get('ef_main_cost_dev', 0):+.1f}%
- 排名变化: {row.get('ef_rank_chg', 0):+.0f}
- 雪球关注分位: {row.get('xq_follow_pct', 0)*100:.0f}%
- 雪球讨论分位: {row.get('xq_tweet_pct', 0)*100:.0f}%"""

    return f"""股票: {row['ts_code']} ({industry}{industry_tag})
日期: {row['trade_date']}, regime: {regime}{regime_tag}

【V7c 量化结论】
buy_score: {row['buy_score']:.0f}/100, sell_score: {row['sell_score']:.0f}/100
r10 预测: {row.get('r10_pred', 0):+.2f}%, r20 预测: {row.get('r20_pred', 0):+.2f}%
sell_10/20 概率: {row.get('sell_10_v6_prob', 0)*100:.0f}%/{row.get('sell_20_v6_prob', 0)*100:.0f}%

【关键 mfk 因子 (LGBM 内部信号显式暴露)】
机构占比: {(row.get('mfk_pyramid_top_heavy') if pd.notna(row.get('mfk_pyramid_top_heavy')) else 0):.2f}
机构趋势 (20d-60d): {(row.get('pyr_velocity_20_60') if pd.notna(row.get('pyr_velocity_20_60')) else 0):+.3f} (<0=机构温和退出)
mfk MA 金叉态: {int(row.get('mfk_main_cross_state') if pd.notna(row.get('mfk_main_cross_state')) else 0)} (+1=金叉, -1=死叉)
主力 5 日净流入: {(row.get('main_net_5d') if pd.notna(row.get('main_net_5d')) else 0):+.2f} 亿
f1/f2 异常事件: {(row.get('f1_neg1') if pd.notna(row.get('f1_neg1')) else 0):+.4f}/{(row.get('f2_pos1') if pd.notna(row.get('f2_pos1')) else 0):+.4f}
{ak_block}

【市场状态】
60 日大盘: {row.get('mkt_ret_60d', 0):+.1f}%, RSI: {row.get('mkt_rsi14', 50):.0f}
"""


def call_llm(client, sys_prompt, user_msg, model="deepseek-v3.2", max_retry=2):
    for r in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg}
                ],
                max_tokens=200, temperature=0.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if r < max_retry - 1:
                time.sleep(1); continue
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

    # 加载 AKShare (时间错位仅作 LLM 输入辅助, 不严格)
    today = datetime.now().strftime("%Y%m%d")
    ak_files = sorted(V9_DIR.glob("akshare_snapshot_*.parquet"))
    if ak_files:
        ak = pd.read_parquet(ak_files[-1])
        if ak["ef_inst_pct"].max() < 0.05:
            ak["ef_inst_pct"] = ak["ef_inst_pct"] * 100
        df = df.merge(ak, on="ts_code", how="left")
        print(f"  AKShare 8 指标合并 (时间错位, 仅作 LLM 输入辅助)", flush=True)

    # ── Task A 候选: 理想多 (V7c 推荐池) ──
    p35_v20 = df["pyr_velocity_20_60"].quantile(0.35)
    pool_a = df[((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30) &
                  (df["pyr_velocity_20_60"] < p35_v20) &
                  (df["f1_neg1"].abs() < 0.005) &
                  (df["f2_pos1"].abs() < 0.005))].dropna(subset=["r20"]).reset_index(drop=True)
    print(f"  Task A (理想多): n={len(pool_a):,}, r20 mean={pool_a['r20'].mean():.2f}%", flush=True)

    # ── Task B 候选: 矛盾段 ──
    pool_b = df[((df["buy_score"] >= 70) & (df["sell_score"] >= 70))
               ].dropna(subset=["r20"]).reset_index(drop=True)
    print(f"  Task B (矛盾段): n={len(pool_b):,}, r20 mean={pool_b['r20'].mean():.2f}%, "
          f"dd<-15% rate={(pool_b['max_dd_20']<=-15).mean()*100:.1f}%", flush=True)

    # 抽样
    random.seed(42)
    sample_a = pool_a.sample(min(N_PER_TASK, len(pool_a)), random_state=42).reset_index(drop=True)
    sample_b = pool_b.sample(min(N_PER_TASK, len(pool_b)), random_state=42).reset_index(drop=True)
    print(f"  抽样 Task A: {len(sample_a)}, Task B: {len(sample_b)}", flush=True)

    client = OpenAI(api_key=CLOUBIC_KEY, base_url=CLOUBIC_BASE)
    total_in = total_out = 0
    cost = 0.0

    # ── Task A: 理想多 conviction ──
    print(f"\n[{int(time.time()-t0)}s] === Task A: 理想多 conviction 评分 ===", flush=True)
    a_results = []
    for i, row in sample_a.iterrows():
        summary = build_summary(row)
        text, usage = call_llm(client, SYS_TASK_A, summary)
        if usage:
            total_in += usage.prompt_tokens; total_out += usage.completion_tokens
            cost += usage.prompt_tokens * 0.14/1e6 + usage.completion_tokens * 0.28/1e6
        try:
            j = json.loads(text) if text else {}
            a_results.append({**row.to_dict(),
                                "conviction": int(j.get("conviction", 50)),
                                "level": str(j.get("level", "medium")),
                                "key_reason": str(j.get("key_reason", ""))[:50]})
        except:
            a_results.append({**row.to_dict(), "conviction": 50,
                                "level": "medium", "key_reason": "parse error"})
        if (i+1) % 10 == 0:
            print(f"  Task A [{i+1}/{len(sample_a)}] 累计 ${cost:.4f}", flush=True)

    a_df = pd.DataFrame(a_results)
    a_df.to_csv(OUT_DIR / "task_a_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== Task A 结果分布 ===", flush=True)
    print(a_df["level"].value_counts().to_string(), flush=True)
    for level in ["high","medium","low"]:
        sub = a_df[a_df["level"] == level]
        if len(sub) > 0:
            r20 = sub["r20"].dropna()
            print(f"  {level} (n={len(sub)}): r20 mean={r20.mean():+.2f}%, win={(r20>0).mean()*100:.1f}%",
                  flush=True)
    high_conv = a_df[a_df["conviction"] >= 70]
    low_conv = a_df[a_df["conviction"] < 50]
    if len(high_conv) >= 5 and len(low_conv) >= 5:
        t, p = stats.ttest_ind(high_conv["r20"].dropna(), low_conv["r20"].dropna())
        print(f"  conviction>=70 (n={len(high_conv)}) r20={high_conv['r20'].mean():+.2f}% vs "
              f"<50 (n={len(low_conv)}) r20={low_conv['r20'].mean():+.2f}%, "
              f"diff={high_conv['r20'].mean()-low_conv['r20'].mean():+.2f}, p={p:.4f}", flush=True)

    # ── Task B: 矛盾段反挖 ──
    print(f"\n[{int(time.time()-t0)}s] === Task B: 矛盾段反挖真涨股 ===", flush=True)
    b_results = []
    for i, row in sample_b.iterrows():
        summary = build_summary(row)
        text, usage = call_llm(client, SYS_TASK_B, summary)
        if usage:
            total_in += usage.prompt_tokens; total_out += usage.completion_tokens
            cost += usage.prompt_tokens * 0.14/1e6 + usage.completion_tokens * 0.28/1e6
        try:
            j = json.loads(text) if text else {}
            b_results.append({**row.to_dict(),
                                "verdict": str(j.get("verdict", "不确定")),
                                "confidence": int(j.get("confidence", 50)),
                                "key_reason": str(j.get("key_reason", ""))[:50]})
        except:
            b_results.append({**row.to_dict(), "verdict": "不确定", "confidence": 0,
                                "key_reason": "parse error"})
        if (i+1) % 10 == 0:
            print(f"  Task B [{i+1}/{len(sample_b)}] 累计 ${cost:.4f}", flush=True)

    b_df = pd.DataFrame(b_results)
    b_df.to_csv(OUT_DIR / "task_b_results.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== Task B 结果分布 ===", flush=True)
    print(b_df["verdict"].value_counts().to_string(), flush=True)
    for v in ["真涨","假涨","不确定"]:
        sub = b_df[b_df["verdict"] == v]
        if len(sub) > 0:
            r20 = sub["r20"].dropna()
            dd15 = (sub["max_dd_20"] <= -15).mean() * 100
            print(f"  {v} (n={len(sub)}): r20={r20.mean():+.2f}% win={(r20>0).mean()*100:.1f}% dd<-15%={dd15:.1f}%",
                  flush=True)
    real_up = b_df[b_df["verdict"] == "真涨"]
    fake_up = b_df[b_df["verdict"] == "假涨"]
    if len(real_up) >= 3 and len(fake_up) >= 3:
        t, p = stats.ttest_ind(real_up["r20"].dropna(), fake_up["r20"].dropna())
        print(f"  '真涨' vs '假涨' diff={real_up['r20'].mean()-fake_up['r20'].mean():+.2f}, p={p:.4f}",
              flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s, "
          f"总 token in={total_in:,} out={total_out:,}, 实际成本 ${cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
