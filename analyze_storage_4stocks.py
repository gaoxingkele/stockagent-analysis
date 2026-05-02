"""对 4 只存储芯片股跑 sparse_layered + quant_score 综合评分."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / ".env.cloubic", override=False)

from stockagent_analysis.tushare_enrich import enrich_with_tushare, compute_quant_score
from stockagent_analysis.sparse_layered_score import (
    extract_features_from_enrich,
    derive_context_from_enrich,
    derive_mf_state,
    compute_sparse_layered_score,
)

STOCKS = [
    ("603986.SH", "兆易创新", "NOR Flash 龙头"),
    ("300223.SZ", "北京君正", "DRAM/SRAM"),
    ("688008.SH", "澜起科技", "内存接口 (DDR5)"),
    ("688110.SH", "东芯股份", "NOR/NAND/DRAM"),
]

REGIME = {"trend": "slow_bull", "dispersion": "high_industry"}

results = []

for ts_code, name, concept in STOCKS:
    print(f"\n{'='*80}")
    print(f"### {ts_code} {name} ({concept})")
    print('='*80)
    try:
        enrich = enrich_with_tushare(ts_code)
        if not enrich:
            print("  enrich 失败")
            continue

        tsf = enrich.get("tushare_factors") or {}
        # 显示基础信息
        close = tsf.get("close_qfq")
        pct = tsf.get("pct_chg")
        mv = tsf.get("total_mv")
        pe = tsf.get("pe_ttm")
        industry = enrich.get("industry") or "?"
        print(f"  最新数据: 收盘 {close} ({pct:+.2f}%) | 市值 {mv/10000:.0f}亿 | PE {pe} | 行业 {industry}")

        # 1) sparse_layered
        features = extract_features_from_enrich(enrich)
        context = derive_context_from_enrich(enrich, industry=industry)
        mf_state = derive_mf_state(enrich)
        sl = compute_sparse_layered_score(
            features=features, context=context,
            regime=REGIME, mf_state=mf_state,
        )

        # 2) quant_score (旧)
        qs = compute_quant_score(enrich)

        print(f"\n  上下文: mv={context.get('mv_seg')} pe={context.get('pe_seg')} "
              f"行业={context.get('industry')}")
        print(f"  资金流: {mf_state}")

        print(f"\n  ## sparse_layered: {sl['layered_score']:.1f} "
              f"(Δ={sl['sum_delta']:+.1f}) | active={sl['n_active']} silent={sl['n_silent']} "
              f"| K={sl['conflict_K']:.3f} conf={sl['confidence']}")

        if sl['active_factors']:
            print(f"\n  激活因子 (top 8):")
            print(f"  {'因子':<20} {'Q桶':<5} {'胜率':<8} {'Q3基线':<8} {'方向':<6} {'delta':<8}")
            for f in sl['active_factors'][:8]:
                d = "↑看多" if f['sign'] > 0 else "↓看空"
                print(f"  {f['name']:<20} {f['q_bucket']:<5} "
                      f"{f['w_eff']*100:>5.1f}%   {f['q3_eff']*100:>5.1f}%   "
                      f"{d:<6} {f['delta']:>+6.2f}")

        if sl.get('gates_applied'):
            print(f"\n  资金流门控: {', '.join(sl['gates_applied'][:5])}")

        print(f"\n  ## quant_score (4 维 + 6 分层): {qs['quant_score']:.1f} "
              f"(Δ={qs['total_delta']:+.1f})")
        if qs['adjustments']:
            for a in qs['adjustments'][:8]:
                print(f"    {a['factor']:<22} {a['delta']:+.1f}  {a['reason'][:60]}")

        # 综合
        score_sl = sl['layered_score']
        score_q = qs['quant_score']
        # 简化版 final (no LLM, 仅 quant + sparse)
        # final = 0.6 × sparse + 0.4 × quant (无专家分时调权)
        composite = 0.6 * score_sl + 0.4 * score_q
        print(f"\n  >> 综合参考分 (60% sparse + 40% quant): {composite:.1f}")

        results.append({
            "ts_code": ts_code, "name": name, "concept": concept,
            "sparse_score": score_sl, "sparse_active": sl['n_active'],
            "sparse_K": sl['conflict_K'], "sparse_conf": sl['confidence'],
            "quant_score": score_q,
            "composite": composite,
            "context": context,
        })

    except Exception as e:
        import traceback
        print(f"  ERR: {e}")
        traceback.print_exc()

# 汇总
print("\n" + "="*80)
print("### 汇总对比")
print("="*80)
print(f"{'股票':<26} {'sparse':<8} {'active':<8} {'K':<6} {'conf':<6} {'quant':<8} {'综合':<8}")
print("-" * 75)
for r in sorted(results, key=lambda x: x['composite'], reverse=True):
    print(f"{r['ts_code']} {r['name']:<10} "
          f"{r['sparse_score']:>6.1f}   {r['sparse_active']:<8} "
          f"{r['sparse_K']:.2f}   {r['sparse_conf']:<6} "
          f"{r['quant_score']:>6.1f}   {r['composite']:>6.1f}")
