# -*- coding: utf-8 -*-
"""DLV-003 — 研究环升级 util (Track B): "骗不过的" 候选筛查工具箱.

非研究结果, 是**一次性流程投资**: 让未来每个搜索出来的候选在"激动"之前必须过三关。
本轮失败模式 = 假阳性 (RETRO +0.0287 → CL 两轮才证伪; Hybrid +54% 是 stratification artifact)。
强 gate 一轮就能抓。三件套 (对应 ROADMAP_2026H2.md Track B):

  B2. Replication 默认化  → `block_bootstrap_ci` (per-cohort block bootstrap 误差条)
                            `multi_seed_replicate` (多 seed 聚合, 抓种子敏感性)
  B3. 多重检验校正        → `probabilistic_sharpe_ratio` (PSR, Bailey & López de Prado 2012)
                            `deflated_sharpe_ratio`      (DSR, 惩罚试验次数 N)
                            `pbo_cscv`                   (PBO via CSCV, Bailey et al. 2017)
  B1. Reviewer panel      → research/REVIEWER_PANEL_CHECKLIST.md (codex+claude+meta-judge 流程)

一站式入口: `skeptic_report(...)` — 给搜索出的候选打包跑三关, 出一个 should_get_excited 判定。

★ 单位约定 (极易错): PSR/DSR 的所有 Sharpe 都是**每观测单位 (per-observation)**, 不是年化。
  用 `annualized_to_per_obs(sr_ann, ppy)` 转换。block_bootstrap_ci 报的是年化 Sharpe (与 FIN-001 同口径)。

纪律: 本 util 不读/不写生产文件 (R05); 不下结论改门柱 (R01); 默认 seed 固定可复现。
`python research_env.py` 跑内置 self-test (解析型断言, 无需缓存)。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

try:
    from scipy.stats import norm as _norm
    _PHI = _norm.cdf
    _PHI_INV = _norm.ppf
except Exception:  # pragma: no cover — scipy 缺失时的纯 numpy 退路
    def _PHI(x):
        x = np.asarray(x, float)
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

    def _PHI_INV(p):
        # Acklam 有理逼近 (绝对误差 < 1.15e-9)
        p = np.asarray(p, float)
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425

        def one(pp):
            if pp <= 0:
                return -np.inf
            if pp >= 1:
                return np.inf
            if pp < plow:
                q = math.sqrt(-2 * math.log(pp))
                return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                       ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            if pp > phigh:
                q = math.sqrt(-2 * math.log(1 - pp))
                return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                       ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            q = pp - 0.5
            r = q * q
            return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                   (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        return np.vectorize(one)(p)

EULER_GAMMA = 0.5772156649015329
DEFAULT_PPY = 252
DEFAULT_SEED = 20260616


# ════════════════════════════════════════════════════════════════════════════
# 基础度量
# ════════════════════════════════════════════════════════════════════════════
def sharpe(rets: np.ndarray, ppy: int = DEFAULT_PPY) -> float:
    """年化 Sharpe (ddof=1)。少于 3 个有限值返回 nan。"""
    r = np.asarray(rets, float)
    r = r[np.isfinite(r)]
    if len(r) < 3:
        return float("nan")
    vol = float(np.std(r, ddof=1))
    return float(np.mean(r) / (vol + 1e-12) * math.sqrt(ppy))


def ann_return(rets: np.ndarray, ppy: int = DEFAULT_PPY) -> float:
    r = np.asarray(rets, float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    return float(np.prod(1.0 + r) ** (ppy / len(r)) - 1.0)


def annualized_to_per_obs(sr_ann: float, ppy: int = DEFAULT_PPY) -> float:
    """年化 Sharpe → 每观测单位 Sharpe (PSR/DSR 要用 per-obs)。"""
    return float(sr_ann) / math.sqrt(ppy)


# ════════════════════════════════════════════════════════════════════════════
# B2. Replication 默认化
# ════════════════════════════════════════════════════════════════════════════
def block_bootstrap_ci(returns: Sequence[float], cohort_len: int = 20,
                       n_boot: int = 1000, seed: int = DEFAULT_SEED,
                       ppy: int = DEFAULT_PPY) -> dict:
    """Per-cohort block bootstrap 的年化 Sharpe / 年化收益 95% CI.

    cohort = 连续 `cohort_len` 个观测 (= 再平衡/持有期自然依赖块, 保块内自相关)。
    重采样 cohort (有放回) → 拼接 → 重算指标。与 FIN-001 同口径 (此处抽出为可复用函数)。
    返回: sharpe_ci95 / sharpe_median / sharpe_se / p_sharpe_gt_0 / p_sharpe_gt_1 / ann_return_ci95。
    """
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < cohort_len * 2:
        raise ValueError(f"样本 {n} 太短 (需 ≥ {cohort_len*2})")
    cid = np.arange(n) // cohort_len
    n_co = int(cid.max()) + 1
    blocks = [r[cid == c] for c in range(n_co)]
    rng = np.random.default_rng(seed)
    bs_s, bs_a = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, n_co, size=n_co)
        concat = np.concatenate([blocks[c] for c in pick])
        s = sharpe(concat, ppy)
        if np.isfinite(s):
            bs_s.append(s)
            bs_a.append(ann_return(concat, ppy))
    bs_s = np.asarray(bs_s); bs_a = np.asarray(bs_a)
    return {
        "point_sharpe": round(sharpe(r, ppy), 4),
        "n_obs": int(n), "n_cohorts": int(n_co), "n_boot": int(len(bs_s)),
        "sharpe_ci95": [round(float(np.percentile(bs_s, 2.5)), 4),
                        round(float(np.percentile(bs_s, 97.5)), 4)],
        "sharpe_median": round(float(np.median(bs_s)), 4),
        "sharpe_se": round(float(np.std(bs_s, ddof=1)), 4),
        "p_sharpe_gt_0": round(float((bs_s > 0).mean()), 4),
        "p_sharpe_gt_1": round(float((bs_s > 1.0).mean()), 4),
        "ann_return_ci95": [round(float(np.percentile(bs_a, 2.5)), 4),
                            round(float(np.percentile(bs_a, 97.5)), 4)],
    }


def multi_seed_replicate(metric_fn: Callable[[int], float],
                         seeds: Sequence[int]) -> dict:
    """跨 seed 复跑同一候选, 聚合稳定性。

    metric_fn(seed) -> 标量指标 (如 OOS Sharpe / IC)。任何依赖随机性 (子采样/初始化/
    bootstrap split) 的候选都该过这关: 若指标随 seed 大幅摆动 = 不可信。
    返回 mean/std/min/max/cv (变异系数) + sign_stable (全同号)。
    """
    vals = []
    for s in seeds:
        v = metric_fn(int(s))
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if len(vals) < 2:
        raise ValueError("至少需 2 个有效 seed 结果")
    arr = np.asarray(vals, float)
    mean = float(arr.mean()); std = float(arr.std(ddof=1))
    return {
        "n_seeds": len(arr), "mean": round(mean, 6), "std": round(std, 6),
        "min": round(float(arr.min()), 6), "max": round(float(arr.max()), 6),
        "cv": round(abs(std / mean), 4) if mean != 0 else float("inf"),
        "sign_stable": bool((arr > 0).all() or (arr < 0).all()),
        "values": [round(v, 6) for v in vals],
    }


# ════════════════════════════════════════════════════════════════════════════
# B3. 多重检验校正
# ════════════════════════════════════════════════════════════════════════════
def probabilistic_sharpe_ratio(sr_per_obs: float, n_obs: int,
                               sr_benchmark: float = 0.0,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR (Bailey & López de Prado 2012): P(真 SR > 基准 SR) 的概率.

    sr_per_obs / sr_benchmark = **每观测单位** Sharpe (非年化!)。
    skew=γ3, kurt=γ4 (非超额, 正态=3) 用样本估计修正非正态。
    """
    sr, sr0, n = float(sr_per_obs), float(sr_benchmark), int(n_obs)
    if n < 2:
        return float("nan")
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    z = (sr - sr0) * math.sqrt(n - 1) / denom
    return float(_PHI(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """N 次独立试验下的期望最大 Sharpe (Bailey & López de Prado 2014, per-obs 单位).

    SR0 = sqrt(V)·[(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]
    V = 各试验 SR 估计的方差 (per-obs); γ = Euler-Mascheroni。
    这是"纯靠多试一次能蒙到多高 Sharpe"的零假设基准。
    """
    N = max(2, int(n_trials))
    V = max(0.0, float(sr_variance))
    g = EULER_GAMMA
    z1 = float(_PHI_INV(1.0 - 1.0 / N))
    z2 = float(_PHI_INV(1.0 - 1.0 / (N * math.e)))
    return math.sqrt(V) * ((1.0 - g) * z1 + g * z2)


def deflated_sharpe_ratio(sr_per_obs: float, n_trials: int, n_obs: int,
                          sr_variance: float, skew: float = 0.0,
                          kurt: float = 3.0) -> dict:
    """DSR (Bailey & López de Prado 2014): 扣掉"试了 N 次"的选择偏差后, SR 仍显著为正的概率.

    = PSR(观测SR; 基准 = N 次试验的期望最大 SR)。所有 SR 为 **per-obs** 单位。
    sr_variance = 搜索过的 N 个候选 SR (per-obs) 的样本方差。
    返回 dsr (概率) + sr0_expected_max (蒙到的基准) + passes (dsr ≥ 0.95)。
    """
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    dsr = probabilistic_sharpe_ratio(sr_per_obs, n_obs, sr_benchmark=sr0,
                                     skew=skew, kurt=kurt)
    return {
        "dsr": round(float(dsr), 4),
        "sr0_expected_max_per_obs": round(float(sr0), 6),
        "observed_sr_per_obs": round(float(sr_per_obs), 6),
        "n_trials": int(n_trials), "n_obs": int(n_obs),
        "passes": bool(dsr >= 0.95),
    }


def pbo_cscv(returns_matrix: np.ndarray, n_splits: int = 16,
             ppy: int = DEFAULT_PPY) -> dict:
    """PBO via CSCV (Bailey, Borwein, López de Prado, Zhu 2017).

    returns_matrix: shape (T 观测, N 配置) — N 个被搜索的策略/参数配置的逐期收益矩阵。
    把 T 切成 n_splits 个偶数块, 枚举 C(S, S/2) 种 IS/OOS 划分:
      在 IS 选 Sharpe 最高配置 n* → 看 n* 在 OOS 的相对排名 ω → logit λ=ln(ω/(1−ω))。
    PBO = P(λ ≤ 0) = "IS 最优配置在 OOS 跌到中位数以下"的频率 = 过拟合概率。
    PBO 越接近 0.5+ 越说明 IS 选优纯属过拟合 (OOS 无延续性)。
    """
    from itertools import combinations
    M = np.asarray(returns_matrix, float)
    if M.ndim != 2 or M.shape[1] < 2:
        raise ValueError("returns_matrix 需 (T, N≥2)")
    T, N = M.shape
    S = int(n_splits)
    if S % 2 != 0:
        S -= 1
    if S < 4 or T < S * 2:
        raise ValueError(f"n_splits={S} 与样本 T={T} 不匹配 (需 S 偶且 T≥2S)")
    bounds = np.linspace(0, T, S + 1).astype(int)
    parts = [np.arange(bounds[i], bounds[i + 1]) for i in range(S)]
    all_idx = list(range(S))
    lambdas, n_overfit, n_total = [], 0, 0
    for is_sel in combinations(all_idx, S // 2):
        oos_sel = [i for i in all_idx if i not in is_sel]
        is_idx = np.concatenate([parts[i] for i in is_sel])
        oos_idx = np.concatenate([parts[i] for i in oos_sel])
        is_sr = np.array([sharpe(M[is_idx, j], ppy) for j in range(N)])
        oos_sr = np.array([sharpe(M[oos_idx, j], ppy) for j in range(N)])
        if not np.isfinite(is_sr).any():
            continue
        nstar = int(np.nanargmax(is_sr))
        # n* 在 OOS 的相对排名 (1=最好). rank/(N+1) 避免 0/1
        order = np.argsort(np.argsort(np.nan_to_num(oos_sr, nan=-np.inf)))
        rank = (order[nstar] + 1) / (N + 1)        # ∈ (0,1)
        omega = min(max(rank, 1e-6), 1 - 1e-6)
        lam = math.log(omega / (1 - omega))
        lambdas.append(lam)
        n_total += 1
        if lam <= 0:
            n_overfit += 1
    lambdas = np.asarray(lambdas)
    return {
        "pbo": round(float(n_overfit / n_total), 4) if n_total else float("nan"),
        "n_combinations": int(n_total), "n_splits": int(S),
        "n_configs": int(N), "n_obs": int(T),
        "lambda_median": round(float(np.median(lambdas)), 4) if n_total else float("nan"),
        "lambda_mean": round(float(np.mean(lambdas)), 4) if n_total else float("nan"),
    }


# ════════════════════════════════════════════════════════════════════════════
# 一站式入口
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class SkepticReport:
    """给一个搜索出来的候选打包跑三关, 出 should_get_excited 判定。"""
    candidate_name: str
    bootstrap: dict
    deflated_sharpe: dict | None = None
    pbo: dict | None = None
    multi_seed: dict | None = None
    gates: dict = field(default_factory=dict)
    should_get_excited: bool = False

    def to_dict(self) -> dict:
        return {
            "candidate_name": self.candidate_name,
            "bootstrap": self.bootstrap,
            "deflated_sharpe": self.deflated_sharpe,
            "pbo": self.pbo,
            "multi_seed": self.multi_seed,
            "gates": self.gates,
            "should_get_excited": self.should_get_excited,
        }


def skeptic_report(candidate_name: str, candidate_returns: Sequence[float],
                   n_trials: int = 1, trial_sharpes_per_obs: Sequence[float] | None = None,
                   returns_matrix: np.ndarray | None = None,
                   multi_seed: dict | None = None,
                   cohort_len: int = 20, n_splits: int = 16,
                   ppy: int = DEFAULT_PPY, seed: int = DEFAULT_SEED) -> SkepticReport:
    """组合三关。candidate_returns = 候选的逐期 (日) 收益。

    - bootstrap: per-cohort CI (B2) — 永远跑。gate: P(Sharpe>0) ≥ 0.95 且 CI 下界 > 0。
    - deflated_sharpe: 若 n_trials>1 且给了 trial_sharpes_per_obs (搜索过的所有候选 SR) → DSR (B3)。
                       gate: dsr ≥ 0.95。
    - pbo: 若给了 returns_matrix (T×N 配置矩阵) → PBO via CSCV (B3)。gate: pbo ≤ 0.5。
    - multi_seed: 调用方可先跑 multi_seed_replicate 传进来。gate: sign_stable 且 cv ≤ 1。
    should_get_excited = 所有**适用**的 gate 全过 (任一不过即 False)。
    """
    r = np.asarray(candidate_returns, float)
    boot = block_bootstrap_ci(r, cohort_len=cohort_len, ppy=ppy, seed=seed)
    gates = {"bootstrap_p_gt0": bool(boot["p_sharpe_gt_0"] >= 0.95
                                     and boot["sharpe_ci95"][0] > 0)}
    rep = SkepticReport(candidate_name=candidate_name, bootstrap=boot)

    if n_trials > 1 and trial_sharpes_per_obs is not None:
        sr_po = annualized_to_per_obs(boot["point_sharpe"], ppy)
        var = float(np.var(np.asarray(trial_sharpes_per_obs, float), ddof=1))
        rr = r[np.isfinite(r)]
        from scipy.stats import skew as _sk, kurtosis as _ku
        sk = float(_sk(rr)); ku = float(_ku(rr, fisher=False))
        rep.deflated_sharpe = deflated_sharpe_ratio(
            sr_po, n_trials, len(rr), var, skew=sk, kurt=ku)
        gates["deflated_sharpe"] = bool(rep.deflated_sharpe["passes"])

    if returns_matrix is not None:
        rep.pbo = pbo_cscv(returns_matrix, n_splits=n_splits, ppy=ppy)
        gates["pbo_not_overfit"] = bool(rep.pbo["pbo"] <= 0.5)

    if multi_seed is not None:
        rep.multi_seed = multi_seed
        gates["multi_seed_stable"] = bool(multi_seed.get("sign_stable")
                                          and multi_seed.get("cv", 9e9) <= 1.0)

    rep.gates = gates
    rep.should_get_excited = all(gates.values()) if gates else False
    return rep


# ════════════════════════════════════════════════════════════════════════════
# self-test (解析型, 无需缓存)
# ════════════════════════════════════════════════════════════════════════════
def _selftest() -> None:
    print("=== research_env.py self-test ===", flush=True)
    rng = np.random.default_rng(0)

    # 1) PSR: 强正 Sharpe 高 PSR, 零 Sharpe ≈ 0.5
    p_strong = probabilistic_sharpe_ratio(0.20, 252, 0.0)
    p_zero = probabilistic_sharpe_ratio(0.0, 252, 0.0)
    assert p_strong > 0.99, p_strong
    assert abs(p_zero - 0.5) < 1e-6, p_zero
    print(f"  [PSR] sr=0.20 → {p_strong:.4f}  sr=0 → {p_zero:.4f}  OK", flush=True)

    # 2) expected_max_sharpe 随 N 单调增
    e10 = expected_max_sharpe(10, 0.01)
    e1000 = expected_max_sharpe(1000, 0.01)
    assert e1000 > e10 > 0, (e10, e1000)
    print(f"  [E[max SR]] N=10 → {e10:.4f}  N=1000 → {e1000:.4f} (单调增) OK", flush=True)

    # 3) DSR: 同一观测 SR, 试验越多 DSR 越低
    d_few = deflated_sharpe_ratio(0.12, 5, 500, 0.01)["dsr"]
    d_many = deflated_sharpe_ratio(0.12, 500, 500, 0.01)["dsr"]
    assert d_few > d_many, (d_few, d_many)
    print(f"  [DSR] obs_sr=0.12 N=5 → {d_few:.4f}  N=500 → {d_many:.4f} (惩罚试验数) OK", flush=True)

    # 4) PBO: 纯噪声配置 (无真 alpha) → PBO ≈ 0.5; 有一个真 alpha 配置 → PBO 低
    noise = rng.normal(0, 0.01, size=(400, 12))
    pbo_noise = pbo_cscv(noise, n_splits=10)["pbo"]
    real = rng.normal(0, 0.01, size=(400, 12))
    real[:, 0] += 0.004                                    # 配置 0 有真 drift
    pbo_real = pbo_cscv(real, n_splits=10)["pbo"]
    assert pbo_noise > pbo_real, (pbo_noise, pbo_real)
    assert pbo_real < 0.3, pbo_real
    print(f"  [PBO] 纯噪声 → {pbo_noise:.3f} (高=过拟合)  含真alpha → {pbo_real:.3f} (低) OK", flush=True)

    # 5) block_bootstrap_ci: 强正漂移序列 P(Sharpe>0) 高 (弱漂移在 21 cohort 下 CI 必宽)
    pos = rng.normal(0.0020, 0.01, size=420)
    bb = block_bootstrap_ci(pos, cohort_len=20, n_boot=500)
    assert bb["p_sharpe_gt_0"] > 0.9, bb
    print(f"  [bootstrap] 正漂移 P(Sharpe>0)={bb['p_sharpe_gt_0']:.3f} CI{bb['sharpe_ci95']} OK", flush=True)

    # 6) multi_seed_replicate: 稳定 vs 不稳定
    ms = multi_seed_replicate(lambda s: 1.5 + 0.01 * (s % 3), seeds=range(5))
    assert ms["sign_stable"] and ms["cv"] < 0.1, ms
    print(f"  [multi_seed] mean={ms['mean']} cv={ms['cv']} sign_stable={ms['sign_stable']} OK", flush=True)

    # 7) skeptic_report 端到端: 纯噪声候选不该 excited
    noise_ret = rng.normal(0, 0.012, size=420)
    rep = skeptic_report("noise_candidate", noise_ret,
                         n_trials=20,
                         trial_sharpes_per_obs=rng.normal(0, 0.05, 20),
                         returns_matrix=rng.normal(0, 0.012, (420, 10)),
                         n_splits=10)
    assert rep.should_get_excited is False, rep.gates
    print(f"  [skeptic_report] 噪声候选 should_get_excited={rep.should_get_excited} gates={rep.gates} OK", flush=True)

    print("\n[selftest] ALL PASSED — 三件套 (replication + PBO/DSR + skeptic) 解析自检全过", flush=True)


if __name__ == "__main__":
    _selftest()
