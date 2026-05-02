"""3 年 vs 1 年 validity_matrix 对比 + 4 只存储芯片股新数据评分."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT / ".env.cloubic", override=False)
sys.stdout.reconfigure(encoding='utf-8')

from stockagent_analysis.tushare_enrich import enrich_with_tushare
from stockagent_analysis.sparse_layered_score import (
    extract_features_from_enrich, derive_context_from_enrich,
    derive_mf_state, compute_sparse_layered_score, load_validity_matrix,
)

REGIME = {'trend': 'slow_bull', 'dispersion': 'high_industry'}

# ──── 1) Matrix 元数据对比 ────
m_1y = load_validity_matrix(ROOT / "output" / "factor_lab" / "validity_matrix.json")
m_3y = load_validity_matrix(ROOT / "output" / "factor_lab_3y" / "validity_matrix.json")

print("=" * 80)
print("validity_matrix 1 年 vs 3 年 对比")
print("=" * 80)
for label, m in [("1 年版", m_1y), ("3 年版", m_3y)]:
    meta = m['meta']
    print(f"\n{label}:")
    print(f"  期间: {meta['data_period'][0]} ~ {meta['data_period'][1]}")
    print(f"  样本: {meta['n_samples']:,} | 股票: {meta['n_stocks']} | 行业: {meta['n_industries']}")
    print(f"  全市场基准 D+20 胜率: {meta['base_win_rate']*100:.2f}%")
    # 激活段统计
    n_active_global = sum(1 for f in m['factors'].values()
                          if f.get('global', {}) and f['global'].get('active'))
    n_active_mv = sum(1 for f in m['factors'].values()
                      for s in f.get('mv', {}).values() if s.get('active'))
    n_active_pe = sum(1 for f in m['factors'].values()
                      for s in f.get('pe', {}).values() if s.get('active'))
    n_active_ind = sum(1 for f in m['factors'].values()
                       for s in f.get('industry', {}).values() if s.get('active'))
    print(f"  激活段: global={n_active_global}, mv={n_active_mv}, pe={n_active_pe}, industry={n_active_ind}")

# ──── 2) 关键指标对比表 ────
print("\n" + "=" * 80)
print("关键差异 (3 年版相对 1 年版)")
print("=" * 80)
print(f"  样本量: 102 万 → 380 万 (×3.7)")
print(f"  时间跨度: 1 年 → 3 年 (含 2023 熊市 + 2024 震荡 + 2025-2026 牛)")
print(f"  全市场基准胜率: 55.0% → 49.0% (-6pp)  ← 3 年含熊市拉低均胜率")
print(f"  行业激活段: 1044 → 636 (-39%)  ← 3 年更严, 因为基准低 → 5pp 超额相对要求更高")

# ──── 3) 4 只股票用 3 年版重新评分 ────
print("\n" + "=" * 80)
print("4 只存储芯片股: 1 年版 vs 3 年版 评分对比")
print("=" * 80)

STOCKS = [
    ("603986.SH", "兆易创新"),
    ("300223.SZ", "北京君正"),
    ("688008.SH", "澜起科技"),
    ("688110.SH", "东芯股份"),
]

results = []
for ts_code, name in STOCKS:
    try:
        e = enrich_with_tushare(ts_code)
        if not e:
            continue
        f = extract_features_from_enrich(e)
        c = derive_context_from_enrich(e, industry=e.get('industry'))
        mf = derive_mf_state(e)

        r_1y = compute_sparse_layered_score(features=f, context=c, matrix=m_1y, regime=REGIME, mf_state=mf)
        r_3y = compute_sparse_layered_score(features=f, context=c, matrix=m_3y, regime=REGIME, mf_state=mf)
        results.append((ts_code, name, r_1y, r_3y))
    except Exception as e:
        print(f"  {ts_code} {name}: ERR {e}")

print(f"\n{'股票':<24} {'1y分':<6} {'1y激活':<8} {'1y K':<7} {'1y conf':<8} | {'3y分':<6} {'3y激活':<8} {'3y K':<7} {'3y conf':<8}")
print("-" * 100)
for ts_code, name, r1, r3 in results:
    print(f"{ts_code} {name:<10} "
          f"{r1['layered_score']:<6.1f} {r1['n_active']:<8} "
          f"{r1['conflict_K']:<6.2f}  {r1['confidence']:<8} | "
          f"{r3['layered_score']:<6.1f} {r3['n_active']:<8} "
          f"{r3['conflict_K']:<6.2f}  {r3['confidence']:<8}")

# ──── 4) 重要因子的胜率变化 ────
print("\n" + "=" * 80)
print("关键因子: 1 年版 vs 3 年版 (大盘段 1000亿+ 的胜率对比)")
print("=" * 80)
print(f"\n{'因子':<22} {'1y best_win':<12} {'3y best_win':<12} {'1y W3':<10} {'3y W3':<10} {'变化':<8}")
print("-" * 80)
focus_factors = ['ma_ratio_60', 'rsi_24', 'macd_hist', 'mfi_14', 'trix', 'channel_pos_60',
                  'sump_20', 'ht_trendmode', 'atr_pct', 'ma20_ma60', 'sumn_20', 'roc_20']
for fc in focus_factors:
    s1y = m_1y['factors'].get(fc, {}).get('mv', {}).get('1000亿+')
    s3y = m_3y['factors'].get(fc, {}).get('mv', {}).get('1000亿+')
    if s1y and s3y:
        bw1 = s1y.get('best_win', 0)
        bw3 = s3y.get('best_win', 0)
        q3_1 = s1y.get('q3_win', 0)
        q3_3 = s3y.get('q3_win', 0)
        a1 = '✓' if s1y.get('active') else '×'
        a3 = '✓' if s3y.get('active') else '×'
        diff = (bw3 - bw1) * 100
        print(f"{fc:<22} {bw1*100:>5.1f}% {a1}    {bw3*100:>5.1f}% {a3}    "
              f"{q3_1*100:>5.1f}%    {q3_3*100:>5.1f}%    {diff:+.1f}pp")

# ──── 5) 跨周期"鲁棒"因子 vs "失效"因子 ────
print("\n" + "=" * 80)
print("跨周期一致性分析")
print("=" * 80)

robust = []   # 1y 激活 AND 3y 也激活
disappeared = []   # 1y 激活但 3y 不激活
new_emerged = []   # 1y 不激活但 3y 激活

for fc in m_1y['factors']:
    if fc not in m_3y['factors']:
        continue
    e1 = m_1y['factors'][fc]
    e3 = m_3y['factors'][fc]
    # 检查 mv + industry 段
    for dim in ('mv', 'pe', 'industry'):
        for seg in e1.get(dim, {}):
            if seg in e3.get(dim, {}):
                a1 = e1[dim][seg].get('active', False)
                a3 = e3[dim][seg].get('active', False)
                if a1 and a3:
                    robust.append((fc, dim, seg))
                elif a1 and not a3:
                    disappeared.append((fc, dim, seg))
                elif not a1 and a3:
                    new_emerged.append((fc, dim, seg))

print(f"\n  跨周期稳健 (1y✓ AND 3y✓): {len(robust)} 段")
print(f"  1 年牛市期独有 (1y✓ but 3y×): {len(disappeared)} 段")
print(f"  3 年长期新涌现 (1y× but 3y✓): {len(new_emerged)} 段")

# 显示 Top 10 跨周期稳健的"段"
robust_with_win = []
for fc, dim, seg in robust:
    s = m_3y['factors'][fc][dim][seg]
    robust_with_win.append((fc, dim, seg, s['best_win']))
robust_with_win.sort(key=lambda r: r[3], reverse=True)

print(f"\n## Top 10 跨周期稳健段 (按 3 年 best_win)")
for fc, dim, seg, bw in robust_with_win[:10]:
    print(f"  {fc:<20} ({dim}/{seg}) → 3y best_win = {bw*100:.1f}%")
