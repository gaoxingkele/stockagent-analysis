"""sparse_layered_score 单元测试 — 用真实 validity_matrix 跑 6 个场景."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stockagent_analysis.sparse_layered_score import (
    compute_sparse_layered_score,
    load_validity_matrix,
    bucket_mv,
    bucket_pe,
)


def show_result(name: str, result: dict):
    print(f"\n=== {name} ===")
    print(f"  layered_score = {result['layered_score']:.1f} "
          f"(delta={result['sum_delta']:+.1f})")
    print(f"  active={result['n_active']}, silent={result['n_silent']}, "
          f"K={result['conflict_K']:.3f}, conf={result['confidence']}")
    if result["active_factors"]:
        print(f"  激活因子:")
        for f in result["active_factors"][:8]:
            print(f"    {f['name']:<22} {f['delta']:+.2f} ({f['reason']})")
    if result["gates_applied"]:
        print(f"  门控: {result['gates_applied']}")


def main():
    matrix = load_validity_matrix()
    print(f"validity_matrix 加载: {matrix['meta']['n_factors']} 因子, "
          f"激活规则 {matrix['meta'].get('activation_rule', 'v1')}")

    # === 场景 1: 小盘股跌透了 ===
    show_result("场景 1: 小盘股跌透 (期望 高分: 反弹机会)",
        compute_sparse_layered_score(
            features={
                "ma_ratio_60": -0.06,
                "ma_ratio_120": -0.05,
                "channel_pos_60": 0.10,
                "sump_20": 8.0,
                "ma20_ma60": -0.03,
                "rsi_24": 35,
            },
            context={"mv_seg": "20-50亿", "pe_seg": "15-30",
                     "industry": "电气设备", "etf_held": False},
            matrix=matrix,
        ))

    # === 场景 2: 小盘股高位涨多 ===
    show_result("场景 2: 小盘股高位涨多 (期望 低分: 反转风险)",
        compute_sparse_layered_score(
            features={
                "ma_ratio_60": +0.20,
                "ma_ratio_120": +0.18,
                "channel_pos_60": 0.92,
                "sump_20": 35.0,
                "ma20_ma60": +0.12,
                "trix": 0.8,
            },
            context={"mv_seg": "20-50亿", "pe_seg": "100+",
                     "industry": "软件服务", "etf_held": False},
            matrix=matrix,
        ))

    # === 场景 3: 大盘股动量启动 ===
    show_result("场景 3: 大盘股动量启动 (期望 高分: 慢牛跟随)",
        compute_sparse_layered_score(
            features={
                "macd_hist": 0.5,
                "rsi_24": 65,
                "mfi_14": 75,
                "ma_ratio_60": +0.05,
                "trix": 0.4,
                "ppo": 1.2,
            },
            context={"mv_seg": "1000亿+", "pe_seg": "15-30",
                     "industry": "化工原料", "etf_held": True},
            matrix=matrix,
            regime={"trend": "slow_bull", "dispersion": "high_industry"},
            mf_state="main_inflow_3d",
        ))

    # === 场景 4: 大盘股弱势 ===
    show_result("场景 4: 大盘股弱势 (期望 低分)",
        compute_sparse_layered_score(
            features={
                "macd_hist": -0.6,
                "rsi_24": 30,
                "mfi_14": 25,
                "ma_ratio_60": -0.08,
                "trix": -0.3,
            },
            context={"mv_seg": "1000亿+", "pe_seg": "15-30",
                     "industry": "化工原料", "etf_held": True},
            matrix=matrix,
            regime={"trend": "slow_bull"},
            mf_state="main_outflow_3d",
        ))

    # === 场景 5: 多因子冲突 (期望 K 大, confidence low) ===
    show_result("场景 5: 多因子冲突 (期望 K 高 confidence low)",
        compute_sparse_layered_score(
            features={
                "ma_ratio_60": -0.06,    # 小盘 Q1 → 看多 +
                "channel_pos_60": 0.10,  # 小盘 Q1 → 看多 +
                "trix": +0.7,            # 小盘 Q5 → 看空 -
                "sump_20": 35.0,         # 小盘 Q5 → 看空 -
            },
            context={"mv_seg": "20-50亿", "pe_seg": "15-30",
                     "industry": "电气设备", "etf_held": False},
            matrix=matrix,
        ))

    # === 场景 6: 半导体行业 + 低波动率 (期望 极高分) ===
    show_result("场景 6: 半导体 + 低波动率 (期望 强买信号 65%+)",
        compute_sparse_layered_score(
            features={
                "atr_pct": 0.018,       # 半导体 Q1 (低波动率)
                "natr_14": 1.2,         # 同上
                "boll_width": 0.04,     # 半导体 Q1
                "sump_20": 9.0,         # 半导体 Q1 (跌透)
                "ma_ratio_60": -0.07,   # 半导体 Q1
            },
            context={"mv_seg": "100-300亿", "pe_seg": "15-30",
                     "industry": "半导体", "etf_held": True},
            matrix=matrix,
            regime={"trend": "slow_bull"},
            mf_state="main_inflow_3d",
        ))


if __name__ == "__main__":
    main()
