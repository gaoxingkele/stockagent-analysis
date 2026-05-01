"""测试 sparse_layered 的 A/B/C 三个新功能."""
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
    derive_mf_state, compute_sparse_layered_score,
    explain_layered_score, render_for_llm_prompt, compare_stocks,
)

REGIME = {'trend': 'slow_bull', 'dispersion': 'high_industry'}

def get_result(ts_code):
    e = enrich_with_tushare(ts_code)
    f = extract_features_from_enrich(e)
    c = derive_context_from_enrich(e, industry=e.get('industry'))
    mf = derive_mf_state(e)
    return compute_sparse_layered_score(features=f, context=c, regime=REGIME, mf_state=mf)

# A: explain
print("\n" + "#" * 78)
print("# A. explain_layered_score (东芯股份 688110)")
print("#" * 78)
r1 = get_result('688110.SH')
print(explain_layered_score(r1, verbose=True))

# B: LLM 注入
print("\n" + "#" * 78)
print("# B. render_for_llm_prompt (东芯股份)")
print("#" * 78)
print(render_for_llm_prompt(r1, max_active=10))

# C: compare
print("\n" + "#" * 78)
print("# C. compare_stocks (东芯 vs 兆易)")
print("#" * 78)
r2 = get_result('603986.SH')
print(compare_stocks(r1, '东芯股份(688110)', r2, '兆易创新(603986)'))
