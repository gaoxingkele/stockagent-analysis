# Execution Plan — Movement Onset Paper Research Track

**配套文档**：`research/paper_plan.md`（理论 / 方法 / outline）
**本文档**：数据、标注、算法、改进的执行细节
**起算日期**：2026-06-01（M1 起点）
**目标投稿**：KDD'27 ADS Track (deadline ~ 2027-02) / CIKM'26 (deadline ~ 2026-05 已过, 推 2027)
**约束**：生产线 V12.31 不动，每日推理继续；研究兼职，每周 15-20h

---

## 0. 文档导读

| 节 | 内容 | 优先级 |
|---|---|---|
| 1 | 数据准备计划（4 个 dataset + 质量控制） | P0 |
| 2 | 数据标注计划（8 种 label + onset 弱监督） | P0 |
| 3 | 算法实验计划（6 个 Phase / 25 个核心实验） | P0 |
| 4 | 持续改进循环（8 个 Cycle） | P1 |
| 5 | 时间表与 Gate 协同 | P0 |
| 6 | 实验基础设施 | P0 |
| 7 | 验收标准 | P0 |

---

## 1. 数据准备计划

### 1.1 总体策略

三个核心 dataset + 一个 backup，分私有 / 公开 / 跨市场三层：

| Dataset | 角色 | 时间 | 规模 | 复现性 |
|---|---|---|---|---|
| **D1 A股-Tushare** | 主实验数据集 | 2018-2026 | ~5200 股 × 2000 天 | 私有（牌照） |
| **D2 A股-Qlib-Alpha158** | 公开复现版 | 2010-2024 | ~4000 股 × 3500 天 | 完全公开 |
| **D3 NASDAQ-100** | 跨市场泛化 | 2015-2024 | 100 股（含历史成分） × 2500 天 | yfinance 公开 |
| **D4 KOSPI/NIFTY** (backup) | NASDAQ 失效时启用 | 2018-2024 | ~200 股 | 韩国/印度公开数据 |

### 1.2 D1: A股-Tushare 准备

**复用资产**：生产线 `update_factor_lab_from_tushare.py` + `factor_lab_3y/` + `tushare_cache/`

**清单**：
| 字段 | 来源 | 时间粒度 | 已有? |
|---|---|---|---|
| OHLCV | Tushare daily | day | ✓ |
| 复权因子 | Tushare adj_factor | day | ✓ |
| 153 技术因子 | factor_lab | day | ✓ |
| 8 概念热度因子 | concept DB | day, 6.15M 行 | ✓ |
| 行业分类 | stock_basic | static | ✓ |
| 流通市值 | daily_basic | day | ✓ |
| PE/PB/ROE | daily_basic | day | ✓ |
| ST 标记 | namechange | day | ✓ |
| 退市股 | 全市场（含历史已退） | day | ✓ |
| 龙虎榜 | top_list | day, 部分 | ✓ |
| 涨跌停 | limit_list | day | 待补 |
| 资金流 | moneyflow | day | ✓ |

**Deliverables**：
- `research/data/ashares_tushare_v1.parquet` — 整合所有字段的单文件 panel data
- `research/data/symbol_universe.yaml` — 股票池 (含历史退市)
- `research/data/trading_calendar.csv` — 标准交易日历
- `research/data/data_card_d1.md` — 数据卡片（字段说明 + 覆盖 + 已知问题）

**工作量**：1 周（数据基本已有，做整合 + 标准化 + 文档化）

### 1.3 D2: A股-Qlib-Alpha158 准备

**目的**：让审稿人完全可复现实验，不依赖 Tushare 付费。

**步骤**：
1. 安装 Qlib，下载官方 cn_data
2. 提取 Alpha158 因子集（Qlib 内置 158 个）
3. 因子映射表：你 153 因子 vs Alpha158 因子的对应关系
4. 时间对齐：Qlib 截止 2024，与 D1 部分重叠便于交叉验证

**因子映射示例**（节选）：
| 你的因子 | Alpha158 对应 | 备注 |
|---|---|---|
| past_r5 | RESI5/CLOSE | 残差收益 |
| volume_ratio_20 | VSTD20 | 成交量标准差 |
| ha_pattern_* | ✗ | 你独有，paper 中标注 |
| concept_heat | ✗ | 你独有，公开版本不用 |
| ... | | |

**Deliverables**：
- `research/data/ashares_qlib_v1.parquet`
- `research/data/factor_mapping.csv` — 完整映射表
- `research/data/data_card_d2.md`
- 复现脚本：`research/scripts/build_d2.py`（任何人 git clone 后跑得通）

**工作量**：2 周

### 1.4 D3: NASDAQ-100 准备

**目的**：跨市场验证（Gate 4）。

**步骤**：
1. 用 `yfinance` 抓 NASDAQ-100 当前 + 历史成分（2015-至今所有曾入选过的股票，避免幸存者偏差）
2. 对齐 Alpha158 因子（Qlib us_data 也提供 158 因子的美股版）
3. 加入"卖空约束代理"特征：
   - Days-to-cover (借贷天数)
   - Short interest ratio (做空比例)
   - Borrow fee (借贷费率) — 来自 IBKR API 或 Finra
   - 这些用于 §1.5 hard-to-borrow 子集分析

**Deliverables**：
- `research/data/nasdaq100_v1.parquet`
- `research/data/nasdaq100_constituents_history.csv` — 历年成分变动
- `research/data/short_constraint_features.parquet` — 卖空约束代理

**工作量**：2 周

### 1.5 D4: KOSPI/NIFTY backup（条件触发）

**触发条件**：Gate 4 失败（NASDAQ 上 PWC 不显著优于 baseline）

**理由**：韩国 / 印度市场卖空约束接近 A 股，更易验证 paper motivation。

**Deliverables**：触发时启动，预算 1 周。

### 1.6 数据质量控制（所有 dataset 通用）

| 检查 | 方法 | Pass 标准 |
|---|---|---|
| Look-ahead bias | 自动测试：t 时刻的特征只能用 ≤ t-1 的数据 | 100% pass |
| Survivorship bias | 必须包含历史退市股 | 退市股占比 > 10% |
| Corporate action 调整 | 拆股 / 分红前后价格连续性 | 价格跳跃 > 30% 必须有公司事件解释 |
| Missing value | 比例统计 + 填充策略 | 主因子 missing < 5%，填 forward fill |
| Outlier | Winsorize ±3σ | 应用于训练，不用于评估 |
| Trade date 一致性 | 与官方交易日历对比 | 0 差异 |
| 因子尺度 | 横截面 zscore 归一化（panel）/ 时序 zscore（individual） | 训练前确定 |

**自动化测试**：`research/tests/data_quality.py`，每次数据更新跑一遍，CI 阻塞合并。

### 1.7 时序切分协议（所有实验通用）

严格 walk-forward，禁止任何"未来数据"泄漏：

```
训练窗：滚动 36 月（最少 24 月）
验证窗：紧接训练窗的 3 月
测试窗：紧接验证窗的 1 月
推进步长：1 月
```

D1：2018-01 起，36+3+1 滚动到 2026-06，共 ~60 个测试月
D2：2010-01 起，~120 个测试月
D3：2015-01 起，~75 个测试月

每个测试月独立训练，**不允许跨月 fine-tune**（防止信息泄漏）。

---

## 2. 数据标注计划

### 2.1 Label 矩阵概览

| Label | 角色 | 复杂度 | 已有实现? | 优先级 |
|---|---|---|---|---|
| L1 Fixed-Horizon (FH) | 最弱 baseline | 简单 | ✓ 生产线已有 | P0 |
| L2 Triple-Barrier (TB, LdP) | 教科书 baseline | 中 | 待写 | P0 |
| L3 Optimal Trend (OTL, IEEE 2023) | 同领域 SOTA | 高 | 待复现 | P0 |
| L4 Continuous Trend (CTL, Entropy 2020) | 补充 | 中 | 待写 | P1 |
| L5 Denoised Label (DL, arXiv 2021) | 现代角度 | 高 | 待复现 | P1 |
| L6 Fixed-Window PWC (v3c) | 你的 first-order | 中 | ✓ 生产线已有 | P0 |
| L7 Adaptive-BOCPD PWC | 中间方法 | 中 | 待写 | P0 |
| L8 Adaptive-HSSM PWC | **主方法** | 高 | 待写 | P0 |

### 2.2 Label 详细规范

#### L1: Fixed-Horizon (FH)

```python
y_FH(i, t, h) = sign(P[i, t+h] / P[i, t] - 1)
# 三分类版本：
# +1 if r > threshold_up, -1 if r < threshold_down, 0 otherwise
```

参数网格：h ∈ {5, 10, 20}, threshold_up ∈ {3%, 5%}, threshold_down ∈ -threshold_up

#### L2: Triple-Barrier (TB, LdP 2018)

```python
# 对每个 (i, t) 模拟未来 H 天的路径：
#   碰到 upper barrier (+u) → label = +1
#   碰到 lower barrier (-d) → label = -1
#   均未碰到（H 天到期） → label = 0
```

参数网格：(u, d, H) ∈ {(0.05, 0.05, 10), (0.08, 0.05, 20), (0.10, 0.08, 20)}

实现：调 `mlfinpy.labeling.triple_barrier` 或自己实现（~200 行）。

#### L3: Optimal Trend Labeling (OTL)

复现 IEEE 2023 论文：
- 用动态规划求 piecewise-linear 拟合的最优分段
- 每段标 trend direction
- 参数：min_segment_length, regularization_lambda

复现来源：作者 GitHub（如有）或自己实现（~500 行）。

#### L4: Continuous Trend Labeling (CTL)

Entropy 2020：
```
找局部 trough/peak，若 |close - last_extreme| > w * close，标为新 trend
```

参数：w ∈ {0.03, 0.05, 0.08}

#### L5: Denoised Labels (DL)

arXiv 2112.10139：
- 用 self-supervised denoising autoencoder 预训练
- 编码器输出作为"去噪后 label"

复现作者 PyTorch 代码 + 适配你的因子集。

#### L6: Fixed-Window PWC (v3c, your baseline)

```python
y_PWC(i, t) = {
    +1 if r5_forward > 10% and drawdown_5 < 5% and past_r5 <= 8%,
    -1 if r5_forward < -10% and rebound_5 < 5% and past_r5 <= 8%,
     0 otherwise
}
```

已有完整生产实现，移植即可。

#### L7: Adaptive-BOCPD PWC

```python
# Step 1: 对每只股票 i 在线跑 BOCPD（Adams & MacKay 2007）
# Step 2: τ_cp(t) = 最近 change-point 时刻
# Step 3: 累积收益 from τ_cp(t) to t-1
# Step 4: 同 PWC 规则，但 past 窗口为 t - τ_cp(t)
```

BOCPD 实现：~300 行 Python（hazard prior + run length posterior + multivariate extension）。

#### L8: Adaptive-HSSM PWC（主方法）

```python
# Step 1: 训练 HSSM (见 §3 Phase 3)
# Step 2: 推断 P(z^O_{i,t} | x_{i,1:t}, market context)
# Step 3: filter(i,t) = 1 if P(z^O = onset_running | x, context) < θ
```

HSSM 实现：见 §3。

### 2.3 Onset State 弱监督锚点

为加速 HSSM 训练收敛、避免离散状态崩溃，提供启发式的 onset state 弱监督：

```python
def heuristic_onset_label(i, t):
    past_r5 = (P[t-1] / P[t-6] - 1)
    future_r5 = (P[t+5] / P[t] - 1)
    future_dd5 = max_drawdown(P[t:t+5])
    
    if past_r5 <= 0.08 and future_r5 >= 0.10 and future_dd5 <= 0.05:
        return 'bullish_onset'
    elif past_r5 <= 0.08 and future_r5 <= -0.10 and future_rb5 <= 0.05:
        return 'bearish_onset'
    elif abs(future_r5) < 0.03:
        return 'rest'
    elif future_r5 > 0.03 and past_r5 > 0.08:
        return 'trend' (continuation)
    else:
        return 'unknown' (skip in supervision)
```

**注意**：这是**弱**监督，仅用作 HSSM 的 anchor loss 项；最终 onset state 由 HSSM ELBO 学出来。

### 2.4 标注质量验证

#### 各 label 之间一致性（Cohen's kappa）

```
       FH    TB    OTL   CTL   DL    PWC
FH    1.00  0.62  0.45  0.51  0.58  0.72
TB    0.62  1.00  0.39  0.44  0.55  0.68
OTL   ...
```

预期：PWC 和 FH 高度一致（同源），与 OTL/CTL 中度一致（同思路）。

#### 类别分布

每个 label 在每个 dataset 上的类别比例：

```
L1 FH (h=5, threshold=3%):
  +1: 32%, 0: 35%, -1: 33%   (大致平衡)

L6 PWC (v3c):
  +1: 8%, 0: 80%, -1: 12%    (稀有事件)
```

PWC 类别极不平衡，训练时用 class-weight 或 focal loss。

#### Onset Duration 经验分布（Proposition 1 实证）

对每只股票，标出所有"启动期"（onset_running 状态连续时间段），统计 duration：

```
A 股 5200 股 × 8 年：
  Mean = 4.2 days, Std = 3.8 days, CV = 0.90
  
  KS test against constant: p < 0.0001 (拒绝常数假设)
  KS test against geometric: p < 0.05 (拒绝几何分布, 支持 HSMM)
```

**这张图就是 paper 的 first table**——非平稳的实证证据。

#### Contamination Ratio Matrix（验证 §2.5）

```
              t+1    t+3    t+5    t+10
FH error%    40.4   50.5   56.7   62.3
TB error%    35.2   45.8   51.2   58.9
PWC v3c %    27.8   23.5   28.0   45.6
OTL error%   28.5   32.1   38.4   52.7
DL error%    33.1   38.2   42.5   55.8
PWC adapt%   18.2   16.4   19.8   31.2  (期望值)
```

这是 T6 的核心数据。

### 2.5 Label 生成时间表

| Phase | 周 | Label | 数据集 |
|---|---|---|---|
| M1.1 | 1 | L1, L6 (已有) | D1 |
| M1.2 | 2 | L2 (TB 实现) | D1 |
| M2.1 | 5 | L3 (OTL) | D1 |
| M2.2 | 6 | L7 (BOCPD) | D1, D2 |
| M2.3 | 7 | L4, L5 | D1 |
| M3.1 | 9 | 所有 L1-L7 在 D2 | D2 |
| M3.2 | 11 | 所有 L1-L7 在 D3 | D3 |
| M4.1 | 13 | L8 (HSSM) 在 D1 | D1 |
| M4.2 | 14 | L8 在 D2, D3 | D2, D3 |

---

## 3. 算法实验计划

### 3.1 实验编号系统

```
B = Baseline 复现
E = PWC 主实验（with/without filter）
A = Ablation
C = Cross-market
S = Statistical 严格性
R = Real portfolio backtest
```

### 3.2 Phase 1: Baseline 复现（M1.2 - M2.4）

| ID | Backbone | Label | Dataset | 工作量 | 验收标准 |
|---|---|---|---|---|---|
| **B1.1** | LGBM | FH | D1 | 0.5 周 | RankIC > 0.05，移植自生产线 |
| **B1.2** | LGBM | TB | D1 | 0.5 周 | RankIC 与 B1.1 同量级 |
| **B1.3** | LGBM | OTL | D1 | 1 周 | RankIC 与论文报告偏差 < 20% |
| **B1.4** | LGBM | DL | D1 | 1 周 | RankIC 与论文偏差 < 20% |
| **B2.1** | **HIST** | FH | D2 | 2 周 | 在 D2 上复现论文 RankIC ± 20% |
| **B2.2** | HIST | FH | D1 | 0.5 周 | D1 上跑通即可（不必复现论文数字） |
| **B3.1** | **MASTER** | FH | D2 | 2 周 | 在 D2 上复现论文数字 |
| **B3.2** | MASTER | FH | D1 | 0.5 周 | D1 跑通 |
| **B4** | StockMixer | FH | D2, D1 | 2 周 | 跑通 |
| **B5** | FactorVAE | FH | D2, D1 | 1.5 周 | 跑通（可选，可省）|

**M1-M2 总工作量**：12 周（兼职 ≈ 8 周日历周）

**Hard check**：B2.1 / B3.1 在 D2 上必须复现论文报告的 ≥80% 性能，否则说明实现错误，**停下来 debug**。

### 3.3 Phase 2: PWC 核心实验（M2.4 - M3.2）

#### Gate 1 关键实验（M2.4，决定 paper 主线）

| ID | Backbone | Label | Filter | Dataset | 关键 |
|---|---|---|---|---|---|
| **E1.1** | LGBM | FH | none | D1 | baseline |
| **E1.2** | LGBM | FH | Fixed-PWC | D1 | **v3c 复现** |
| **E1.3** | LGBM | TB | none | D1 | LdP baseline |
| **E1.4** | LGBM | TB | Fixed-PWC | D1 | **Gate 1: TB+PWC 是否正交** |

**Gate 1 判定**：
- E1.4 α > E1.3 α + 0.2pp（显著） → PASS，paper 主线确立
- 否则 → FAIL，paper 退化为 "PWC 替代 TB"，CCF 等级降一档

#### Gate 2 关键实验（M3.1）

| ID | Backbone | Label | Filter | Dataset |
|---|---|---|---|---|
| **E2.1-4** | HIST | {FH, TB} × {none, Fixed-PWC} | D1 |
| **E2.5-8** | MASTER | {FH, TB} × {none, Fixed-PWC} | D1 |
| **E2.9-12** | StockMixer | {FH, TB} × {none, Fixed-PWC} | D1 |

12 个 cell。

**Gate 2 判定**：
- ≥75%（9/12）cell 上 Fixed-PWC 带正向 → PASS, model-agnostic 立住
- < 75% → FAIL，paper 改写 "Tree-based model-specific"

### 3.4 Phase 3: Adaptive PWC 与 HSSM（M3.2 - M4.4）

#### E3: BOCPD-Adaptive PWC

| ID | Backbone | Label | Filter | Dataset |
|---|---|---|---|---|
| **E3.1-4** | LGBM | {FH, TB} × {Adaptive-BOCPD-PWC} | D1, D2 |
| **E3.5-12** | {HIST, MASTER, StockMixer} | {FH, TB} × {Adaptive-BOCPD-PWC} | D1 |

#### E4: HSSM-Adaptive PWC（主方法）

| ID | Backbone | Label | Filter | Dataset |
|---|---|---|---|---|
| **E4.1** | LGBM | FH | Adaptive-HSSM-PWC (单层 SNLDS) | D1 |
| **E4.2** | LGBM | FH | Adaptive-HSSM-PWC (双层) | D1 |
| **E4.3-6** | {HIST, MASTER, StockMixer, FactorVAE} | FH | Adaptive-HSSM-PWC (双层) | D1 |
| **E4.7-10** | 同上 | FH | Adaptive-HSSM-PWC (双层) | D2 |

#### Gate 3 关键实验（M4.4）

**Gate 3 判定**：
- E4.2 显著优于 E3.1（HSSM 双层 > BOCPD）
- 且 E4.2 显著优于 E4.1（双层 > 单层，macro layer 必要）
- 都 PASS → 主方法成立，paper 故事最强
- E4.2 ≤ E3.1 → 降级到 BOCPD 作主方法，HSSM 进 supplementary

### 3.5 Phase 4: Cross-Market（M4.4 - M5.2）

| ID | Backbone | Label | Filter | Dataset |
|---|---|---|---|---|
| **C1.1-16** | 所有 4 backbone × {FH, TB} × {none, Fixed-PWC, BOCPD, HSSM} | D3 (NASDAQ) |
| **C2.1** | 子集分析：NASDAQ hard-to-borrow stocks | E4.2 复制 | D3 |

#### Gate 4 关键实验（M5.2）

**Gate 4 判定**：
- C1.x 中 HSSM-PWC 在 NASDAQ 上至少在 50% cell 上正向 → PASS（跨市场有效）
- C2.1 在 hard-to-borrow 子集上 HSSM-PWC 收益更强 → STRONG PASS（验证 short-sale constraint motivation）
- 否则 → FAIL，启动 D4 (KOSPI/NIFTY) 实验

### 3.6 Phase 5: Ablation（M4.4 - M5.4 并行）

| ID | 实验 | 目的 |
|---|---|---|
| **A1** | τ ∈ {0.04, 0.06, 0.08, 0.10, 0.12, 0.15, ∞} | T3 敏感性 |
| **A2** | past-only / past+future / future-only | 验证单向是否够 |
| **A3** | HSSM 状态数 ∈ {3, 5, 7} | onset state 粒度 |
| **A4** | Duration ∈ {Geometric, NegBinom, Poisson, learned} | duration head 选择 |
| **A5** | Filter threshold θ ∈ {0.3, 0.5, 0.7} | filter 决策点 |
| **A6** | Macro layer 来源 ∈ {none, market_context_sys, learned} | 验证生产资产对接 |
| **A7** | HSSM 中层条件 ∈ {独立, +concept_heat, +龙虎榜} | concept 接入价值 |
| **A8** | Encoder ∈ {Bi-LSTM, Transformer, Mamba} | encoder 选择 |

### 3.7 Phase 6: 统计严格性（M5.2 - M5.4）

| ID | 实验 |
|---|---|
| **S1** | T1 每个 cell 配 DM 检验 p 值 |
| **S2** | 1000-resample bootstrap CI |
| **S3** | BH-FDR 多重比较校正 |
| **S4** | 不同 OOS 窗口（19 月 / 24 月 / 36 月）稳健性 |
| **S5** | Random seed × 5 跑，方差报告 |
| **S6** | Hyperparameter sensitivity（学习率、batch、dropout） |

### 3.8 Phase 7: 实战回测（M5.4 - M6.2）

| ID | 实验 |
|---|---|
| **R1** | Walk-forward 24+ 月在主方法上（A 股） |
| **R2** | Dual-track portfolio 构造（移植生产线） |
| **R3** | 与 V12.31 生产线对比（同期 OOS） |
| **R4** | 灾难月分析（202602 / 202403 / 等） |
| **R5** | NASDAQ 上的 portfolio simulation |
| **R6** | Transaction cost sensitivity（35bps / 50bps / 100bps） |

### 3.9 实验总数与工作量

```
Phase 1 (Baseline):       ~10 cell × 平均 1 周 = 10 周
Phase 2 (Gate 1+2):       ~16 cell × 0.3 周 = 5 周
Phase 3 (Adaptive+HSSM):  ~20 cell × 0.5 周 = 10 周
Phase 4 (Cross-market):   ~17 cell × 0.3 周 = 5 周
Phase 5 (Ablation):       8 ablation × 1 周 = 8 周
Phase 6 (Statistical):    6 task × 0.5 周 = 3 周
Phase 7 (Backtest):       6 task × 1 周 = 6 周

总计：47 周（兼职日历）/ 24 周（全职）
```

兼职预算太紧，**必须并行化**：Phase 5 / Phase 6 可在 Phase 3-4 后半段并行启动。

---

## 4. 持续改进实验计划（Paper 投稿后 / 投稿过程中）

### 4.1 改进循环总览

每个 cycle 周期 2-4 周，可并行。

| Cycle | 主题 | 触发 | 工作量 |
|---|---|---|---|
| **C1** | 数据扩充 | 投稿前 | 2 周 |
| **C2** | 特征工程优化 | M2-M3 |2 周 |
| **C3** | HSSM 架构调优 | M3-M5 | 4 周 |
| **C4** | 外部信号融合 | M4-M5 | 3 周 |
| **C5** | 跨市场迁移学习 | M5-M6 | 3 周 |
| **C6** | 在线学习 | 投稿后 | 4 周 |
| **C7** | 因果推断分析 | 投稿后 | 3 周 |
| **C8** | 可解释性研究 | 投稿后 | 3 周 |

### 4.2 Cycle 1: 数据扩充

**目的**：把训练数据从 2018-2026 扩展到 2010-2026（16 年），覆盖完整市场周期。

**关键问题**：onset duration 分布是否在不同年代稳定？
- 若稳定 → HSSM 框架的通用性立住
- 若漂移 → paper 需补 regime-aware adaptation 章节

**Deliverable**：长时间数据下的 Proposition 1 重新验证。

### 4.3 Cycle 2: 特征工程优化

**目的**：从 153 因子中找出对 HSSM 收敛最关键的 top-K。

**方法**：
- ablation：去掉每个因子组，看 HSSM ELBO 影响
- 找出"启动子的 prototypical features"（如：5 日收益偏度、10 日成交量比、20 日波动率压缩）

**Deliverable**：附录的 feature importance 表 + paper 中 "key onset signals" 一节。

### 4.4 Cycle 3: HSSM 架构调优

**目的**：让 HSSM 训练更稳、性能更好。

**实验**：
- 状态数 K ∈ {3, 5, 7, 10}
- Duration head：Negative Binomial / Mixture of Geometrics / learned bins
- Bi-LSTM vs Transformer vs Mamba encoder
- KL annealing schedule
- Macro layer 监督强度（弱 anchor loss 系数）

**Deliverable**：Ablation A3-A8 的完整数据 + best config 报告。

### 4.5 Cycle 4: 外部信号融合

**目的**：把生产线 V12.31 的外部信号（概念热度、龙虎榜、市场环境感知）系统接入 HSSM。

**Three-way fusion**：
```
HSSM 高层 z^M  ← 市场环境感知系统输出 (regime classification)
HSSM 中层 z^O  ← 概念热度 + 龙虎榜 (onset reinforcement)
HSSM 观测 x    ← 153 因子
```

**Deliverable**：A6, A7 ablation 数据 + 完整融合架构图。

### 4.6 Cycle 5: 跨市场迁移学习

**目的**：A 股训练的 HSSM 能否 fine-tune 到 NASDAQ？

**实验**：
- Cold-start NASDAQ（从零训练）
- Warm-start NASDAQ（A 股 pretrained）
- Domain adaptation 技巧（GRL、MMD、CORAL）

**Deliverable**：迁移性 ablation + paper supplementary 一节。

### 4.7 Cycle 6: 在线学习（投稿后）

**目的**：HSSM 增量更新，适配生产线实时 onset detection。

**挑战**：
- 在线 EM / 在线 variational inference
- Catastrophic forgetting 防护
- 状态空间漂移检测

**Deliverable**：可部署的在线 HSSM；下一篇 paper 的种子。

### 4.8 Cycle 7: 因果推断分析（投稿后）

**目的**：用 do-calculus 量化"PWC filter 移除 X% 样本" 对下游 α 的因果效应。

**方法**：
- 把 filter 看作 intervention
- 用 backdoor adjustment 或 instrumental variable
- 估计 ATE / CATE

**Deliverable**：因果分析章节（适合投 cause-and-effect 期刊）。

### 4.9 Cycle 8: 可解释性研究（投稿后）

**目的**：找出"启动子的 prototypical features"。

**方法**：
- HSSM 在 onset detection 时的 attention 分析
- Saliency map（如 Integrated Gradients）
- Counterfactual 分析：改哪个因子能让一只 onset 股变成 rest

**Deliverable**：可视化 + paper 的 case study 章节 + 生产线增强（让交易员看得懂模型在看什么）。

---

## 5. 时间表与 Gate 协同

### 5.1 主时间线（6 个月兼职 / 3 个月全职）

```
M1 (Jun)   数据 D1 + Label L1/L2/L6 + B1.1-2 + 读 P0 文献
M2 (Jul)   数据 D2 + Label L3 + L7(BOCPD) + B2.1, B3.1
           ╠ Gate 1: TB+PWC 正交性 (M2 末)
M3 (Aug)   数据 D3 + Label L4/L5 + B4, B5 + Phase 2 完整 + Cycle 2 启动
           ╠ Gate 2: HIST+PWC model-agnostic (M3 末)
M4 (Sep)   Label L8 (HSSM) + Phase 3 + Cycle 3 启动
           ╠ Gate 3: HSSM 双层显著优于单层 (M4 末)
M5 (Oct)   Phase 4 (cross-market) + Phase 5 (ablation) 并行 + Cycle 4 启动
           ╠ Gate 4: NASDAQ 泛化 (M5 中)
M6 (Nov)   Phase 6 (statistical) + Phase 7 (backtest) + 写作 + Cycle 5 启动
           ╠ Gate 5: 完整 T1 矩阵 (M6 中)
M7 (Dec)   写作 + 内部 review + 润色 + supplementary
M8 (Jan)   submission deadline (KDD'27 ~ Feb 上旬)
```

### 5.2 每个 Gate 的决策树

```
Gate 1 FAIL → paper 改 "PWC 替代 TB"，目标 CIKM
         PASS → 继续 Gate 2

Gate 2 FAIL → paper 改 "Tree-based PWC"，目标国内 CCF B
         PASS → 继续 Gate 3

Gate 3 FAIL → paper 主方法用 BOCPD，HSSM 进 supplementary
              目标 CIKM-A 边界
         PASS → 主线确立，继续 Gate 4

Gate 4 FAIL → 启动 D4 (KOSPI/NIFTY)，写跨市场 nuance
              或退到 "A 股特异性" paper
         PASS → KDD ADS 可行

Gate 5 FAIL → 推迟一个 cycle，补实验
         PASS → submit
```

### 5.3 并行化策略

不能完全串行，必须 overlap：

```
[数据 D1] ─→ [Label L1/L6] ─→ [B1.1-2] ─→ [E1.x: Gate 1]
                                                ↓
[数据 D2] ─→ [Label L2/L3] ─→ [B2.1/B3.1] ─→ [E2.x: Gate 2]
                                                ↓
[数据 D3] ─→ [Label L7/L8] ─→ [E3.x/E4.x: Gate 3+4]
                                                ↓
                                          [C1.x cross-market]
                                                ↓
                            ┌────────────────────┴───────────────────┐
                       [Ablation A1-A8]                      [Statistical S1-S6]
                                                                    ↓
                                                            [Backtest R1-R6]
                                                                    ↓
                                                                [写作]
```

### 5.4 应急方案

| 情境 | 应急 |
|---|---|
| HIST/MASTER 在 D2 上无法复现论文数字 | 改用 Qlib 集成版本（牺牲一些性能但保证可复现） |
| HSSM 训练完全不收敛 | 退到 BOCPD 单层，HSSM 进 supplementary |
| 时间不够，砍 cell | 优先保 T1 / T6 / T7；T3 / T5 简化；T4 推 supplementary |
| 投稿前发现重大 bug | 重跑 Gate 5 涉及的实验，推迟 1 个 cycle |

---

## 6. 实验基础设施

### 6.1 工具栈

| 工具 | 用途 |
|---|---|
| **Hydra** | 配置管理（每个实验一个 yaml） |
| **MLflow** 或 **Weights & Biases** | 实验 tracking |
| **DVC** | 数据版本管理（D1/D2/D3 大文件） |
| **pytest** | 数据质量 + 因子 + label 单元测试 |
| **Jupyter + nbconvert** | 自动化报告生成 |
| **Qlib** | 公开数据 + Alpha158 因子 |
| **LightGBM** + **PyTorch** | 模型训练 |
| **scipy.stats** | DM 检验 + bootstrap + FDR |
| **plotly / matplotlib** | 论文图表 |

### 6.2 Git 仓库结构

```
stockagent-research/
├─ configs/                     Hydra yaml
│   ├─ data/
│   │   ├─ d1_ashares_tushare.yaml
│   │   ├─ d2_ashares_qlib.yaml
│   │   └─ d3_nasdaq100.yaml
│   ├─ label/
│   │   ├─ fh.yaml, tb.yaml, ..., hssm_pwc.yaml
│   ├─ model/
│   │   ├─ lgbm.yaml, hist.yaml, master.yaml, ...
│   ├─ experiment/
│   │   ├─ e1_1.yaml ... e4_10.yaml
│   │   ├─ ablation_a1.yaml ...
│   └─ default.yaml
├─ src/                         (见 paper_plan.md §9.1)
├─ experiments/
│   ├─ runs/                    每个 run 一个目录, MLflow + 自动报告
│   └─ aggregate/               跨 run 聚合分析
├─ data/                        DVC tracked
├─ paper/                       LaTeX
├─ tests/                       pytest
├─ notebooks/                   分析草稿
├─ Makefile                     make exp_e1_1 / make gate_1 等
└─ README.md
```

### 6.3 实验运行约定

每个实验 run：
1. 用 `python -m src.train experiment=e1_1` 启动（Hydra）
2. 自动记录到 MLflow，含 git SHA、config hash、env 信息
3. 输出落 `experiments/runs/<run_id>/`
4. 自动生成 `report.md`（metrics + 关键图 + config summary）
5. CI 触发 `pytest tests/run/<run_id>.py` 跑回归测试

### 6.4 可复现性 SLA

每个 paper 主表 cell 必须：
- ✓ 5 个不同 seed × 平均（不允许单一 seed）
- ✓ Git SHA 锁定
- ✓ Environment lock（`environment.yml` / `requirements.lock`）
- ✓ Config 完全 declarative（无 hardcode）
- ✓ 任何人 git clone + 一条命令 reproduce

### 6.5 文档化

每周生成进度报告 `progress/YYYY-WW.md`：
- 本周完成的实验
- 关键 metric 变化
- 遇到的问题与解决
- 下周计划

---

## 7. 验收标准

### 7.1 数据准备验收

- [ ] D1 整合，含 1.7 节所有质控通过
- [ ] D2 复现版本，任何人 `python scripts/build_d2.py` 一条命令完成
- [ ] D3 含 NASDAQ-100 历史成分 + 卖空约束代理
- [ ] 每个 dataset 配 data card

### 7.2 标注验收

- [ ] L1-L7 在 D1 上全部生成完成
- [ ] L8 (HSSM) 在 D1, D2, D3 上推断完成
- [ ] Label 之间的 kappa 一致性表
- [ ] Onset duration 经验分布图（Proposition 1 证据）
- [ ] Contamination ratio 矩阵（§2.5 / T6）

### 7.3 算法实验验收

- [ ] 5 Gate 全部 PASS（或明确 FAIL fallback 后的 paper 重定位）
- [ ] T1 主表 ≥ 75% cell 显著正向
- [ ] T2-T7 全部完成
- [ ] 所有 cell 配 5-seed 平均 + std
- [ ] DM / Bootstrap / BH-FDR 完整

### 7.4 持续改进验收

- [ ] Cycle 1-5 在投稿前完成
- [ ] Cycle 6-8 在投稿后 6 个月内完成
- [ ] 每个 cycle 产出独立 progress report + 是否进 paper 的决策

### 7.5 写作验收

- [ ] 完整 9 页正文 + 2 页 reference + 附录
- [ ] 内部 review ≥ 2 轮（找 2 个非 stock-prediction 但懂 ML 的同行）
- [ ] LaTeX checklist：所有 figure / table 矢量、字号、配色一致
- [ ] Limitations 章节诚实交代失效情形

---

## 8. 风险管理与决策日志

### 8.1 主要风险

详见 `paper_plan.md` §11。本文档对应的执行风险：

| 风险 | 应对 |
|---|---|
| SOTA backbone 复现失败 | 改用 Qlib 集成版本作为 fallback |
| HSSM 训练不稳 | 三档方案（HSSM/SNLDS/经典 HSMM）准备好 |
| 数据 license 问题 | D2 公开版本必须最早准备好 |
| 跨市场失效 | D4 backup 启动 |
| 时间不够 | 优先保 T1/T6/T7，砍 T3/T5/T4 |

### 8.2 决策日志格式

`research/decision_log.md` 持续记录关键决策：

```
2026-MM-DD: <决策标题>
- Context: ...
- Options considered: ...
- Decision: ...
- Reason: ...
- Affected experiments: ...
- Revisit date: ...
```

---

## 9. 跟生产线 V12.31 的协同

### 9.1 物理隔离原则

| 维度 | 生产线 | 研究分支 |
|---|---|---|
| 仓库 | stockagent-analysis | stockagent-research（建议新建） |
| 数据 | Tushare 实时拉取，付费 | D2 公开版优先，D1 私有版次要 |
| 模型 | v3c LGBM | HSSM + 多 backbone |
| 节奏 | 每日推理 | weekly experiment runs |
| 决策权 | 实战驱动 | paper-quality 驱动 |

### 9.2 双向流动

| 方向 | 内容 |
|---|---|
| 生产 → 研究 | factor_lab、市场环境感知、概念热度作为 informative prior |
| 研究 → 生产 | Cycle 6 (在线学习) 落地后，HSSM filter 反哺 V12.32+ |

### 9.3 生产线不动期间

- 每日推理保持，daily_top20 脚本不变
- factor 更新 nightly 任务保持
- 仅添加：研究分支 weekly 报告链接到 memory，让两边都看得见

---

## 10. 附录：常用命令速查

```bash
# 实验启动
make exp_e1_1                       # 跑 E1.1
make gate_1                         # 跑 Gate 1 全部 cell

# 数据准备
make data_d1                         # 重建 D1
make data_d2                         # 重建 D2 (Qlib)
make data_d3                         # 重建 D3 (NASDAQ)

# 报告生成
make report                          # 汇总所有 run 生成全局报告
make table_t1                        # 汇总 T1 主表

# 测试
make test                            # 跑所有 pytest
make qa                              # 数据质量检查
```

---

## 11. 下一步立即行动（用户确认后）

按优先级 P0 → P1：

1. **本周内**：建 `stockagent-research` 仓库骨架（按 §6.2）
2. **下周内**：D1 整合 + L1/L6 移植 + LGBM baseline 跑通
3. **M2 第一周**：开始 LdP TB 实现（决定 Gate 1）
4. **M2 第二周**：Qlib + D2 + HIST 复现启动
5. **M2 末**：Gate 1 判定

需要我现在开始建仓库骨架，还是先消化这两份计划再决定？
