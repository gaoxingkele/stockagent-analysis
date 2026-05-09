#!/usr/bin/env python3
"""V8 三层决策架构 PoC (50 样本).

Layer 1: V7c 量化筛选 (本地, 免费)
Layer 2: 7 票 (1 V7c LGBM + 6 LLM Agent) + dot plot + 视觉解读
Layer 3: Opus 4.7 仲裁 (最终决策)

7 投票席:
  1. V7c LGBM 量化票 (本地, 免费)
  2. TrendMomentumAgent (DeepSeek V3.2) — 趋势动量
  3. CapitalLiquidityAgent (DeepSeek V3.2) — 主力散户分歧
  4. SentimentFlowAgent (DeepSeek V3.2) — 情绪/sell 风险
  5. FundamentalAgent (Sonnet 4.6) — 基本面 PE/MV
  6. RiskGuardAgent (Sonnet 4.6) — pyramid/异常事件
  7. ContrastAgent (DeepSeek V3.2 反向) — 故意唱反调避免群体思维

输出 output/v8/poc_3layer.csv (50 样本含每层结果)
预算 ~$21 (50 样本)
"""
from __future__ import annotations
import os, json, time, random, base64
from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env.cloubic")
load_dotenv(".env")

ROOT = Path(__file__).resolve().parent
PROD_DIR = ROOT / "output" / "production"
OUT_DIR = ROOT / "output" / "v8"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
TEST_START = "20250501"
TEST_END   = "20260126"

R10_ANCHOR = (-1.44, 0.22, 2.40)
R20_ANCHOR = (-7.78, -1.18, 8.76)
SELL10_V6 = (0.18, 0.48, 0.78)
SELL20_V6 = (0.05, 0.43, 0.87)

N_SAMPLES = 50
CLOUBIC_KEY = os.environ["CLOUBIC_API_KEY"]
CLOUBIC_BASE = "https://api.cloubic.com/v1"

# ==================== Agent 定义 ====================
AGENTS = [
    # name, team_color, model, system_prompt
    ("TrendMomentum", "tech", "deepseek-v3.2",
     """你是技术面趋势动量分析师. 看 MA / MACD / 金叉死叉. 输出 JSON:
{"r20_pred": -15~+25 数值预测, "decision": "buy" 或 "skip", "confidence": 0-100, "reason": "30字内"}"""),
    ("CapitalLiquidity", "capital", "deepseek-v3.2",
     """你是资金分层分析师. 看主力/散户/超大单/中户分歧, 关注 mfk_pyramid_top_heavy 机构占比. 输出 JSON:
{"r20_pred": -15~+25, "decision": "buy"/"skip", "confidence": 0-100, "reason": "30字内"}"""),
    ("SentimentFlow", "capital", "deepseek-v3.2",
     """你是情绪资金流分析师. 关注 sell_score 风险,北向资金,regime 状态. 输出 JSON:
{"r20_pred": -15~+25, "decision": "buy"/"skip", "confidence": 0-100, "reason": "30字内"}"""),
    ("Fundamental", "fundamental", "claude-sonnet-4-6",
     """你是基本面分析师. 看 PE/PB/市值/行业, 价值锚定. 输出 JSON:
{"r20_pred": -15~+25, "decision": "buy"/"skip", "confidence": 0-100, "reason": "30字内"}"""),
    ("RiskGuard", "fundamental", "claude-sonnet-4-6",
     """你是风险守门员. 关注 max_dd 概率/异常事件 (f1/f2)/机构占比集中度. 输出 JSON, 倾向保守:
{"r20_pred": -15~+25, "decision": "buy"/"skip", "confidence": 0-100, "reason": "30字内"}"""),
    ("Contrast", "tech", "deepseek-v3.2",
     """你是反方思维分析师, 故意从悲观 / 反向思考找弱点. 输出 JSON:
{"r20_pred": -15~+25, "decision": "buy"/"skip", "confidence": 0-100, "reason": "找出 30字内反方理由"}"""),
]


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


def build_stock_summary(row) -> str:
    """单股数据摘要 (~500 token)."""
    return f"""股票: {row['ts_code']} {row.get('name', '?')} ({row.get('industry', '?')})
日期: {row['trade_date']}

V7c 量化系统:
  buy_score: {row['buy_score']:.0f}/100
  sell_score: {row['sell_score']:.0f}/100
  r10 预测: {row.get('r10_pred', 0):+.2f}%, r20 预测: {row.get('r20_pred', 0):+.2f}%
  sell_10/20 概率: {row.get('sell_10_v6_prob', 0)*100:.0f}% / {row.get('sell_20_v6_prob', 0)*100:.0f}%

资金流:
  机构占比 (mfk_pyramid_top_heavy): {row.get('mfk_pyramid_top_heavy', 0):.2f}
  机构趋势 (pyr_velocity_20_60): {row.get('pyr_velocity_20_60', 0):+.3f}
  主力 5 日净流入: {row.get('main_net_5d', 0):+.2f} 亿
  f1/f2 (大跌1%吸筹/大涨1%派发): {row.get('f1_neg1', 0):+.4f} / {row.get('f2_pos1', 0):+.4f}

基本面:
  PE: {row.get('pe_ttm', 0):.1f}, 总市值: {row.get('total_mv', 0)/1e4:.1f} 亿

市场:
  regime: {int(row.get('regime_id', 0))}, 60 日大盘: {row.get('mkt_ret_60d', 0):+.1f}%, RSI: {row.get('mkt_rsi14', 50):.0f}
"""


def call_agent(client, agent_name, model, sys_prompt, summary, max_retry=2):
    for r in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": summary}
                ],
                max_tokens=200, temperature=0.3,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if r < max_retry - 1:
                time.sleep(1); continue
            return None, None


def make_dot_plot(votes, ts_code, trade_date, save_path):
    """FOMC 风格点阵图.
    votes: [{name, team, r20_pred, decision, confidence}, ...]
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    team_color = {"quant": "#000000", "tech": "#1976D2",
                   "capital": "#388E3C", "fundamental": "#FFA000"}
    decision_marker = {"buy": "o", "skip": "x"}

    y_jitter = {}
    for v in votes:
        x = v["r20_pred"]
        team = v.get("team", "tech")
        # y 抖动避免点重叠
        bin_x = round(x / 2) * 2
        y_jitter.setdefault(bin_x, 0)
        y = y_jitter[bin_x]
        y_jitter[bin_x] += 1
        size = 80 + (v["confidence"] / 100) * 200
        marker = decision_marker.get(v["decision"], "o")
        ax.scatter(x, y, s=size, c=team_color[team],
                    marker=marker, alpha=0.8, edgecolors="black", linewidths=1.0)
        ax.annotate(v["name"], (x, y), xytext=(5, 5),
                     textcoords="offset points", fontsize=7)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(np.median([v["r20_pred"] for v in votes]),
                color="red", linestyle=":", alpha=0.6, label=f"median={np.median([v['r20_pred'] for v in votes]):.1f}%")
    ax.set_xlim(-15, 25)
    ax.set_xlabel("Predicted r20 (%)", fontsize=10)
    ax.set_ylabel("(jitter only, no meaning)", fontsize=8)
    ax.set_title(f"FOMC Dot Plot: {ts_code} on {trade_date}\n"
                  f"7 expert votes (size=confidence, color=team, o=buy x=skip)",
                  fontsize=10)
    # legend
    legend_handles = []
    for team, c in team_color.items():
        legend_handles.append(plt.scatter([], [], c=c, s=80, label=team))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision_dotplot(client, image_path, max_retry=2):
    """Sonnet 4.6 视觉解读 dot plot."""
    img_b64 = encode_image_b64(image_path)
    sys_prompt = """你是 FOMC 投票分析专家. 看专家点阵图分析:
1. 共识强度 (聚集 vs 分散)
2. 团队对立 (技术/资金/基本面是否一致)
3. 离群点是否值得关注
4. 综合 7 票, 你的最终决策 buy/skip
输出 JSON:
{"consensus_level": 0-100, "team_aligned": true/false, "outlier_count": 0-7,
"vision_decision": "buy"/"skip", "vision_confidence": 0-100, "reason": "50字内分析"}"""

    for r in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="claude-sonnet-4-6",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": "分析这张专家点阵图."}
                    ]}
                ],
                max_tokens=300, temperature=0.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content, resp.usage
        except Exception as e:
            if r < max_retry - 1:
                time.sleep(2); continue
            return None, None


def call_layer3_arbiter(client, summary, votes_text, vision_text, max_retry=2):
    """Opus 4.7 最终仲裁."""
    sys_prompt = """你是最终量化决策仲裁人. 整合下面三层信息做 BUY/SKIP 决策:
- Layer 1 V7c 量化数据 (LGBM 评分 + 资金流 + 基本面)
- Layer 2 7 专家投票
- Layer 2.5 视觉 LLM 对 dot plot 的解读

注意:
- V7c 已硬筛, 这只股票本身已通过量化推荐池
- 警惕单一信号过强 (如 buy_score>85 实测反转)
- 警惕信号矛盾 (双高象限实测 dd<-15%=29%)
- 警惕共识过度 (FOMC 一致看多反而是顶部信号)

输出 JSON:
{"final_decision": "buy"/"skip",
"final_confidence": 0-100,
"position_pct": 1-5 仓位建议百分比,
"key_risk": "30字内最大风险点",
"reasoning": "50字内理由"}"""

    user_msg = f"""【Layer 1 V7c 数据】
{summary}

【Layer 2 7 票投票结果】
{votes_text}

【Layer 2.5 视觉 LLM dot plot 解读】
{vision_text}

请仲裁最终决策."""

    for r in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model="claude-opus-4-7",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg}
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
    print(f"[{int(time.time()-t0)}s] 加载 OOS + V7c...", flush=True)
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
    pool_mask = ((df["buy_score"] >= 70) & (df["buy_score"] <= 85) &
                  (df["sell_score"] <= 30) &
                  (df["pyr_velocity_20_60"] < p35_v20) &
                  (df["f1_neg1"].abs() < 0.005) &
                  (df["f2_pos1"].abs() < 0.005))
    pool = df[pool_mask].dropna(subset=["r20"]).reset_index(drop=True)
    print(f"  V7c 推荐池: n={len(pool):,}", flush=True)

    pool["yyyymm"] = pool["trade_date"].astype(str).str[:6]
    samples = []
    n_per_month = N_SAMPLES // pool["yyyymm"].nunique()
    random.seed(42)
    for m, g in pool.groupby("yyyymm"):
        n_take = min(max(1, n_per_month), len(g))
        idx = random.sample(range(len(g)), n_take)
        samples.append(g.iloc[idx])
    samples = pd.concat(samples).reset_index(drop=True)
    if len(samples) > N_SAMPLES:
        samples = samples.sample(N_SAMPLES, random_state=42).reset_index(drop=True)
    print(f"  抽样: {len(samples)}", flush=True)

    client = OpenAI(api_key=CLOUBIC_KEY, base_url=CLOUBIC_BASE)

    results = []
    total_in = total_out = 0
    cost_per_token_in = {"deepseek-v3.2": 0.14, "claude-sonnet-4-6": 3.0, "claude-opus-4-7": 15.0}
    cost_per_token_out = {"deepseek-v3.2": 0.28, "claude-sonnet-4-6": 15.0, "claude-opus-4-7": 75.0}
    total_cost_usd = 0.0

    for idx, row in samples.iterrows():
        sample_t0 = time.time()
        summary = build_stock_summary(row)

        # ── Layer 2: 7 票 (1 V7c + 6 LLM) ──
        votes = [{
            "name": "V7c-LGBM", "team": "quant",
            "r20_pred": float(row["r20_pred"]),
            "decision": "buy" if row["buy_score"] >= 70 and row["sell_score"] <= 30 else "skip",
            "confidence": int(min(100, abs(row["r20_pred"]) * 10 + 50)),
            "reason": f"buy={row['buy_score']:.0f}/sell={row['sell_score']:.0f}",
        }]

        for ag_name, ag_team, ag_model, ag_prompt in AGENTS:
            text, usage = call_agent(client, ag_name, ag_model, ag_prompt, summary)
            if text is None:
                votes.append({"name": ag_name, "team": ag_team,
                                "r20_pred": 0, "decision": "skip",
                                "confidence": 0, "reason": "API 失败"})
                continue
            try:
                j = json.loads(text)
                votes.append({
                    "name": ag_name, "team": ag_team,
                    "r20_pred": float(j.get("r20_pred", 0)),
                    "decision": str(j.get("decision", "skip")).lower(),
                    "confidence": int(j.get("confidence", 50)),
                    "reason": str(j.get("reason", ""))[:50],
                })
            except:
                votes.append({"name": ag_name, "team": ag_team,
                                "r20_pred": 0, "decision": "skip",
                                "confidence": 0, "reason": "parse error"})
            if usage:
                total_in += usage.prompt_tokens
                total_out += usage.completion_tokens
                total_cost_usd += (usage.prompt_tokens * cost_per_token_in[ag_model]
                                    + usage.completion_tokens * cost_per_token_out[ag_model]) / 1e6

        # ── Layer 2.5: dot plot + 视觉解读 ──
        plot_path = PLOTS_DIR / f"{row['ts_code']}_{row['trade_date']}.png"
        make_dot_plot(votes, row["ts_code"], row["trade_date"], plot_path)
        vision_text, vision_usage = call_vision_dotplot(client, plot_path)
        try:
            vision_j = json.loads(vision_text) if vision_text else {}
        except:
            vision_j = {}
        if vision_usage:
            total_in += vision_usage.prompt_tokens
            total_out += vision_usage.completion_tokens
            total_cost_usd += (vision_usage.prompt_tokens * cost_per_token_in["claude-sonnet-4-6"]
                                + vision_usage.completion_tokens * cost_per_token_out["claude-sonnet-4-6"]) / 1e6

        # ── Layer 3: Opus 4.7 仲裁 ──
        votes_text = "\n".join([f"  {v['name']:18s} ({v['team']:11s}): {v['decision']:4s} r20={v['r20_pred']:+.1f}% conf={v['confidence']} — {v['reason']}"
                                 for v in votes])
        vision_summary = json.dumps(vision_j, ensure_ascii=False)
        l3_text, l3_usage = call_layer3_arbiter(client, summary, votes_text, vision_summary)
        try:
            l3_j = json.loads(l3_text) if l3_text else {}
        except:
            l3_j = {}
        if l3_usage:
            total_in += l3_usage.prompt_tokens
            total_out += l3_usage.completion_tokens
            total_cost_usd += (l3_usage.prompt_tokens * cost_per_token_in["claude-opus-4-7"]
                                + l3_usage.completion_tokens * cost_per_token_out["claude-opus-4-7"]) / 1e6

        n_buy = sum(1 for v in votes if v["decision"] == "buy")
        n_skip = sum(1 for v in votes if v["decision"] == "skip")

        result = {
            **row.to_dict(),
            "n_buy": n_buy, "n_skip": n_skip,
            "consensus": vision_j.get("consensus_level", 0),
            "team_aligned": vision_j.get("team_aligned", False),
            "vision_decision": vision_j.get("vision_decision", "skip"),
            "vision_confidence": vision_j.get("vision_confidence", 0),
            "final_decision": l3_j.get("final_decision", "skip"),
            "final_confidence": l3_j.get("final_confidence", 0),
            "position_pct": l3_j.get("position_pct", 0),
            "key_risk": str(l3_j.get("key_risk", ""))[:60],
            "votes": json.dumps([{k: v[k] for k in ["name","decision","r20_pred","confidence"]}
                                  for v in votes], ensure_ascii=False),
        }
        results.append(result)

        sample_dur = time.time() - sample_t0
        print(f"  [{idx+1}/{len(samples)}] {row['ts_code']} {row['trade_date']} "
              f"n_buy={n_buy}/7 vision={result['vision_decision'][:4]} "
              f"final={result['final_decision'][:4]} ({sample_dur:.0f}s) "
              f"累计 ${total_cost_usd:.3f}", flush=True)

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "poc_3layer.csv", index=False, encoding="utf-8-sig")

    # ── 评估 ──
    print(f"\n[{int(time.time()-t0)}s] === 4 方对比 ===", flush=True)

    # baseline V7c (整个 N_SAMPLES, 都通过 V7c, r20 平均)
    print(f"V7c baseline: n={len(res_df)}, r20={res_df['r20'].mean():+.2f}%", flush=True)

    # 加 7 票多数 (n_buy >= 4)
    sub_majority = res_df[res_df["n_buy"] >= 4]
    print(f"V7c + 7 票多数 (n_buy≥4): n={len(sub_majority)}, "
          f"r20={sub_majority['r20'].mean():+.2f}%", flush=True)

    # 加 7 票一致 (n_buy >= 6)
    sub_strong = res_df[res_df["n_buy"] >= 6]
    print(f"V7c + 7 票强共识 (n_buy≥6): n={len(sub_strong)}, "
          f"r20={sub_strong['r20'].mean():+.2f}%", flush=True)

    # 加 vision buy
    sub_vision = res_df[res_df["vision_decision"] == "buy"]
    print(f"V7c + Vision buy: n={len(sub_vision)}, "
          f"r20={sub_vision['r20'].mean():+.2f}%", flush=True)

    # 加 Layer 3 buy
    sub_l3 = res_df[res_df["final_decision"] == "buy"]
    print(f"V7c + Layer3 (Opus) buy: n={len(sub_l3)}, "
          f"r20={sub_l3['r20'].mean():+.2f}%", flush=True)

    # 加 Layer 3 buy + 高 confidence
    sub_l3_hc = res_df[(res_df["final_decision"] == "buy") & (res_df["final_confidence"] >= 70)]
    print(f"V7c + Layer3 buy + conf>=70: n={len(sub_l3_hc)}, "
          f"r20={sub_l3_hc['r20'].mean():+.2f}%", flush=True)

    # t-test
    print(f"\n=== Layer3 buy vs skip t-test ===", flush=True)
    yes = res_df[res_df["final_decision"] == "buy"]["r20"].dropna()
    no = res_df[res_df["final_decision"] == "skip"]["r20"].dropna()
    if len(yes) >= 5 and len(no) >= 5:
        t, p = stats.ttest_ind(yes, no)
        print(f"  yes n={len(yes)}, no n={len(no)}, "
              f"diff={yes.mean()-no.mean():+.2f}pp, t={t:.3f}, p={p:.4f}", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s, 总 token in={total_in:,} out={total_out:,}, "
          f"实际成本 ${total_cost_usd:.3f}", flush=True)


if __name__ == "__main__":
    main()
