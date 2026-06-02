# Paper Plan — Movement Onset Classification under Short-Sale Constraints

**起始日期**：2026-05-28
**状态**：planning（未启动实现）
**生产线**：V12.31 不动，研究分支独立
**目标**：CCF A KDD ADS Track（首选）/ CIKM Long（兜底）

---

## 0. 文档导读（给消化用的目录）

| 节 | 内容 | 是否需要先掌握 |
|---|---|---|
| 1 | 现有 V12 工作的学术 reposition | ★ 必读 |
| 2 | **正负启动子的非平稳理论包装**（用户 2026-05-28 关键洞察） | ★★ 最核心 |
| 3 | 关键术语对照表（中英文 + 学术等价物） | 消化术语用 |
| 4 | 文献占用情况 + 必避雷区 | 用于命名和定位 |
| 5 | 必打 SOTA Baseline 矩阵 | 实验设计基础 |
| 6 | 7 张主表的精确设计 | 实验骨架 |
| 7 | 形式化（Notation + Definition + Proposition） | 写作时回查 |
| 8 | 已有实证资产 vs 待补 | 工作量评估 |
| 9 | 研究分支基础设施 | 工程结构 |
| 10 | Go/No-Go Gate 与节奏 | 进度管理 |
| 11 | 风险与对策 | 后续避坑 |

---

## 1. 学术 reposition：从"启动子"到 Movement Onset

### 1.1 中文术语的英文学术化

| 中文（生产线/对话用语） | 英文学术标准 | 备注 |
|---|---|---|
| 启动子 / pump | **Movement Onset** / **Onset Event** | pump 已被市场操纵研究占用，绝对避免 |
| 正启动子 | **Bullish Onset** / **Upward Onset** | |
| 负启动子 | **Bearish Onset** / **Downward Onset** | |
| 启动期 | **Onset Phase** (可变长) | 关键：非固定窗口（见 §2） |
| 启动强度 | **Onset Magnitude** | |
| 假启动 / 伪启动 | **Pseudo-Onset** / **Spurious Onset** | v3b 的污染样本就是它 |
| 启动持续时间 | **Onset Duration** | 隐含变量，非平稳的核心 |
| past_r5 ≤ 8% 约束 | **Backward-Context Filter** (first-order) / **Adaptive Onset-Bounded Filter** (推广版) | first-order = 你的 v3c |
| pump_ratio = P_up/P_down | **Mutual-Exclusion Ratio Score** | |

### 1.2 Paper 标题候选

**首选**：
> **"Non-Stationary Onset Modeling for Stock Movement Classification under Short-Sale Constraints"**

**备选**：
- "Adaptive Backward-Context Filtering for Stock Movement Onset Classification"
- "Latent-Duration Onset Detection in Multi-Class Stock Movement Prediction"

为什么用 "Non-Stationary"：见 §2，这是 paper 的真正理论创新所在，不能藏。

### 1.3 三大 Contribution（reposition 后版本）

| # | Contribution | 原始 evidence | 学术升级 |
|---|---|---|---|
| C1 | **Onset Non-Stationarity Hypothesis**：股票市场的"启动"是一个**变长隐含事件**，而非固定窗口里的回报阈值 | v3b → v3c 中间指标改善（误高% 砍半）但灾难月仍存在 | 形式化为 HSMM / change-point 模型 |
| C2 | **Adaptive Backward-Context Filter**：将 fixed-window past constraint 推广为 change-point-aware adaptive window | v3c 已是 fixed-window special case | 提出 learnable / adaptive 版本作为更优解 |
| C3 | **Mutual-Exclusion Confidence Score for Ranking**：softmax 互斥比值作为推理时的决断度信号 | v12.31 +0.16pp/月 实测 | 与 calibration 文献对话 |

---

## 2. 【核心】正负启动子的非平稳理论包装

> **用户 2026-05-28 关键洞察**：启动子不是固定长度窗口的捕获，而是非平稳的。

这一节是 paper 真正的理论创新焦点，必须深入。

### 2.1 现状的局限性（v3c 的软肋）

v3c 用 `past_r5 ≤ 8%` 过滤训练样本，潜台词是：

> 假设 1（不成立）：所有股票的"启动期"长度都是 5 天
> 假设 2（不成立）：所有股票启动期内的累积回报阈值都是 8%

但实际上：

- 小盘启动可能 2 天内完成 30% 涨幅
- 主升浪股可能慢爬 15 天累积 50%
- 横盘后的启动 vs 已有趋势中的二次加速，触发条件不同
- 不同板块（消费 vs 科技 vs 周期）启动节奏差异显著

**结论**：past_r5 是一个 first-order approximation，理论上不够。Paper 需要把"非平稳启动子"作为 framework 提出来，再展示 fixed-window 是它的退化特例。

### 2.2 五个可选的理论框架（按推荐度排序）

#### 框架 A：Hidden Semi-Markov Model (HSMM) ★★★ 强推

**核心建模**：
- 隐含状态 $z_{i,t} \in \{\text{rest}, \text{onset}, \text{trend}, \text{exhaustion}\}$
- HSMM 允许状态停留时间服从任意分布（HMM 限制为几何分布）
- 即"启动持续多久"是数据学出来的，不是预设的

**对应到你的问题**：
- v3c 的 "past_r5 ≤ 8%" → "假设 onset 状态停留时间 = 5 天的退化"
- HSMM 推广 → onset 停留时间 $T_{onset} \sim \text{Negative-Binomial}$ 或 learned distribution
- 过滤条件从 fixed-window 变为 "$P(z_{i,s} = \text{onset}, \forall s \in [t-T, t]) > 0.5$"

**优势**：
- 经典统计模型，有严格收敛性证明
- 在金融时间序列（regime switching）文献里已有大量应用
- 形式优雅，可写 propositions

**劣势**：
- 训练 HSMM 比 LGBM 慢得多（EM 算法）
- 与现代 deep learning baseline 对接需要桥接

**文献种子**：
- Hamilton 1989 (Markov-switching, 经典)
- Yu 2010 *Hidden Semi-Markov Models* (textbook)
- Bulla & Bulla 2006 (HSMM for financial returns)
- Nystrup et al. 2017 (regime-based portfolio under HSMM)

#### 框架 B：Bayesian Online Changepoint Detection (BOCPD) ★★

**核心建模**：
- 视"启动"为一个 change-point event
- 每个时刻维护"距离最近 change-point 的运行时长" $r_t$
- 后验 $P(r_t | x_{1:t})$ 在线更新
- 过滤条件：$r_t < \tau_{onset}$ 时排除

**对应到你的问题**：
- past_r5 → "最近一个 change-point 之后的窗口"
- 自适应：窗口长度由数据驱动
- 可与 Hazard function 结合，量化"启动概率"

**优势**：
- 算法清晰，online 可以跑实时
- 与 PWC filter 概念完美对接（PWC = 取 change-point 之后的样本）
- Adams & MacKay 2007 原 paper 引用过万，强权威

**劣势**：
- 对突发噪声敏感，需要选合适的 hazard prior
- 多维信号（价量同时变）的 BOCPD 较复杂

**文献种子**：
- Adams & MacKay 2007 (BOCPD 原论文)
- Saatçi et al. 2010 (GP-based extension)
- Knoblauch & Damoulas 2018 (multivariate BOCPD)

#### 框架 C：Self-Exciting Hawkes Process ★★

**核心建模**：
- "启动"是一个 spike event，spike 后短期内再 spike 概率提高（self-excitement）
- 强度函数 $\lambda(t) = \mu + \sum_{t_i < t} \alpha \cdot e^{-\beta(t - t_i)}$
- "启动子" = $\lambda(t)$ 局部峰值

**对应到你的问题**：
- 启动持续时间的"半衰期" $1/\beta$ 由数据学出来
- 同一只股不同时期的"启动半衰期"可以不同（time-varying β）
- 跨股票启动的相关性（板块联动）也能建模

**优势**：
- 金融高频数据里的经典工具，权威性强（Bowsher 2007, Bacry et al. 2015）
- 可以同时建模"启动密度"和"启动余波"
- 跨股票的传染效应 → 板块层面 onset

**劣势**：
- 主要用于事件流（订单到达），用于 daily bar 需要适配
- 与 PWC filter 的对接不如 BOCPD 直接

#### 框架 D：Survival Analysis / Time-to-Reversal ★

**核心建模**：
- 把启动期看作 survival time $T$
- 协变量为价量因子
- Cox 比例风险模型 / Neural Survival Network

**优势**：
- 与"启动期持续多久"问题直接对应
- 医学/工程里成熟方法

**劣势**：
- "事件"定义需要小心（什么算 reversal）
- 与 multiclass prediction 框架对接生硬

#### 框架 E：Shapelet / DTW (变长时间序列模式) ★

**核心建模**：
- 启动子 = 一个可变长的"特征性 shape"
- 用 DTW 做时间扭曲容忍
- Shapelet learning 自动发现关键 sub-sequence

**优势**：
- 直接面对"非平稳"的变长属性
- 在时间序列分类领域有 SOTA 工作（Ye & Keogh 2009, Grabocka et al. 2014）

**劣势**：
- 与传统 label-engineering 框架的接口不自然
- 多分类下扩展复杂

### 2.3 强推方案：HSMM + BOCPD 双层框架

让 HSMM 提供**生成模型** + BOCPD 提供**判别检测器**：

```
Level 1 (生成):  HSMM 假设 onset 持续时间服从 Negative-Binomial
                ↓
Level 2 (判别):  BOCPD 在线估计 P(当前在 onset 中 | 历史观测)
                ↓
Level 3 (过滤):  filter(i,t) = 1 if P(onset_running | x_{1:t}) < threshold
                ↓
Level 4 (训练):  在过滤后的 D_PWC 上训 multiclass classifier
                ↓
Level 5 (推理):  ratio score 排序
```

这样 v3c 的 `past_r5 ≤ 8%` 自然成为 **fixed-duration onset assumption** 下的特例。Paper 可以这样组织：

- Section 3.1：定义 onset event 与 onset duration（HSMM 框架）
- Section 3.2：fixed-window approximation (这就是 v3c，作为 Method baseline)
- Section 3.3：adaptive change-point filtering (BOCPD-based, 这是 paper 的方法升级)
- Section 4.X 实验：两个版本都跑，证明 adaptive 显著优于 fixed-window

### 2.4 paper 的真正卖点（reposition 后）

从"加个 past_r5 ≤ 8% 提了 0.4pp" → 升级为：

> **"我们提出 onset 是一个非平稳的隐含状态变量，其持续时间由市场状态共同决定。传统 fixed-window 假设（包括 LdP triple-barrier 的固定 horizon）是这一框架的退化特例。我们证明 HSMM-based adaptive filter 显著优于固定窗口，且这一改进在 4 个 SOTA backbone × 2 个市场上一致成立。"**

这种 framing 让 paper 同时和：
- LdP triple-barrier（fixed forward window）对话
- BOCPD 文献（变长窗口检测）对话
- HSMM 金融应用文献对话
- 卖空约束的实证经济学对话

故事厚度从"工程优化"升到"建模框架"。

### 2.5 实证锚点（你已有的数据如何支撑非平稳论点）

| 已有 evidence | 支撑"非平稳"的方式 |
|---|---|
| v3b → v3c 在 t+1/t+3/t+5/t+10 误高% 改善幅度不同（-12.6 / -27 / -28.6 / 较小） | 说明启动期长度本身不固定 |
| 灾难月（如 202602）即使 v3c 仍受影响 | 说明 fixed 8% 阈值在某些 regime 下失效 |
| pump_ratio 在灾难月 +2.30pp 翻盘 | 说明 inference 时的 confidence 能补救 label 的不完美 |
| 不同行业 cap 后的表现（你的双轨 v6 数据） | 不同板块启动节奏不同 → 行业特异性 onset duration |

### 2.6 形式化（写作时直接用）

**Definition (Onset Event)**:
设 $z_{i,t} \in \{0, 1, 2\}$ 为隐含状态（rest / bullish-onset / bearish-onset），其转移服从 semi-Markov：
$$
P(z_{i,t+1} = j, T_j \mid z_{i,t} = k) = P_{k \to j} \cdot f_j(T_j)
$$
其中 $T_j$ 是状态 $j$ 的停留时间，$f_j(\cdot)$ 是其分布（HMM 中 $f_j$ 必为几何，HSMM 放宽）。

**Definition (Adaptive Backward-Context Filter)**:
$$
\text{filter}_{\text{adapt}}(i, t) = \mathbb{1}\left[\sum_{s=\tau_{cp}(t)}^{t} r_{i,s} \leq \theta\right]
$$
其中 $\tau_{cp}(t)$ 是 BOCPD 估计的"最近 change-point 时刻"，$\theta$ 是累积阈值。

**Definition (Fixed-Window Special Case, v3c)**:
当 $\tau_{cp}(t) \equiv t - 5$（固定 5 天）且 $\theta = 0.08$ 时，adaptive filter 退化为 v3c 的 `past_r5 ≤ 8%`。

**Proposition (Onset Duration Heterogeneity)**:
在多市场/多 regime 数据上，stock-specific onset duration $\hat{T}^*_i$ 显著偏离常数（KS 检验 p<0.001）。
→ 这条命题用你已有数据就能验证，是 paper 的 first table 之一。

**Proposition (Adaptive Filter Dominance)**:
在 Onset Duration 非常数前提下，adaptive filter 的下游分类损失 strictly less than fixed-window filter 的损失。
→ 理论 + 实证 T1。

### 2.7 生成模型选型（2026-05-28 追加 / HSSM 主推）

> **本节回答**：HSMM 用什么现代深度模型实现？VAE / Flow / Diffusion / Mamba 哪个合适？
> **结论先放**：**Hierarchical Switching State-Space Model (HSSM)** 是主推，比单层 SNLDS 更适合本任务，因为层次化天然对接 [[project_market_context]] 已建的市场环境感知系统。

#### 2.7.1 三大候选生成框架对比

任务约束（决定模型选择的硬性条件）：
1. 离散隐状态（rest / onset / trend / exhaustion）必须可索引
2. 变长 duration（HSMM 核心，否则退化成 HMM）
3. 在线 filtering（每日推断 O(1)，不能是 O(diffusion steps)）
4. 下游接 PWC filter（必须能输出 $P(z_t = \text{onset} | x_{1:t})$）
5. 多变量观测（153 因子）
6. 可写 propositions（概率图清晰）

|  | VAE | Normalizing Flow | Diffusion |
|---|---|---|---|
| **离散隐变量支持** | ★★★ Gumbel-Softmax/Categorical 成熟 | ★ 需要 dequantization hack | ✗ 无显式 latent state |
| **变长 duration 建模** | ★★★ 显式 duration head | ★ 不自然 | ✗ |
| **在线推断速度** | ★★★ amortized O(1) | ★★ 每步 forward | ✗ 1000 步去噪 |
| **PWC filter 接口** | ★★★ 直接给后验 | ★★ 可逆 invert | ✗ 给不出 $P(z_t \| x_{1:t})$ |
| **与 HSMM 对应** | ★★★ SSM 的自然延伸 | ★ 生硬 | ✗ 完全不对应 |
| **金融领域 baseline** | ★★★ FactorVAE (NeurIPS'22) 等 | ★★ MAF/IAF for TS | ★ TimeGrad |
| **paper 可解释性** | ★★★ 离散状态保留语义 | ★ | ✗ |
| **训练稳定性** | ★★ ELBO 成熟 | ★★ 精确 likelihood | ★★★ 但慢 |

**结论**：VAE 主框架。Flow 作为 VAE 的 emission 分布参数化补充。Diffusion 仅做 supplementary（synthetic onset 数据增强）。

#### 2.7.2 VAE 内部进一步对比

| 模型 | 离散切换 | 层次结构 | 变长 duration | 与本任务匹配度 |
|---|---|---|---|---|
| **DKF** (Krishnan 2017) | ✗ 连续 | ✗ | ✗ | ★ |
| **VRNN** (Chung 2015) | ✗ | ✗ | ✗ | ★ |
| **RSSM / Dreamer** (Hafner 2019) | ✗ | ✗ | ✗ | ★ |
| **FactorVAE** (Duan 2022) | ✗ | ✗ | ✗ | ★★（已是 stock baseline，可作对照） |
| **SNLDS** (Dong 2020) | ✓ 单层 | ✗ | ✗（需扩展） | ★★ |
| **rSLDS** (Linderman 2017) | ✓ 单层 | ✗ | ✗ | ★★ |
| **HSSM** (Hierarchical Switching SSM, 2021-2024 多篇) | ✓ 多层 | ✓ | ✓（自然） | **★★★ 主推** |

#### 2.7.3 为什么 HSSM > SNLDS（修正之前的推荐）

之前主推 SNLDS 的理由是"有显式 switching 离散变量"。但深入思考后，**HSSM 是更优的选择**，理由：

1. **层次对应市场结构的层次性**：
   - 高层 $z^{macro}_t$：市场 regime（牛市/熊市/震荡） — 直接对接 [[project_market_context]] 已有的市场环境感知系统
   - 中层 $z^{stock}_{i,t}$：股票 onset 状态（rest/bullish_onset/bearish_onset/trend/exhaustion）
   - 低层 $h_{i,t}$：日内动力学连续隐变量
   - SNLDS 单层无法表达"市场 regime 影响 onset 概率"这一关键耦合

2. **解释你的"非平稳"洞察更彻底**：
   - 非平稳的**根源是 regime 变化**，不仅是 stock-level onset 的变长
   - 灾难月 202602 这类 tail risk 本质是 macro regime 切换 + onset 误判耦合
   - 单层 SNLDS 只能解释一层，HSSM 同时解释两层

3. **与生产线已有资产无缝衔接**：
   - 你已有"市场环境感知系统"（指数趋势分类 + 板块热度 + ETF + 视觉 LLM）
   - 这些恰好是 $z^{macro}_t$ 的天然 observation / prior
   - HSSM 框架下，市场环境感知系统的输出可作为 macro layer 的 informative prior

4. **Paper 故事更厚**：
   - "Joint regime-aware onset modeling" 比 "switching onset detection" 故事大
   - 与多市场（A 股 + NASDAQ）对比时，regime 层的跨市场差异本身就是一个 narrative

5. **理论 propositions 更丰富**：
   - 可以加 Proposition (Regime-Conditional Onset Asymmetry)：在熊市 regime 下，bearish onset 的 contamination ratio 显著高于牛市 — 这条命题用你的灾难月数据可证

#### 2.7.4 HSSM 概率图

```
高层 macro:    z^M_t ∈ {bull, bear, sideways}       (semi-Markov, slow)
                ↓ 影响
中层 onset:    z^O_{i,t} ∈ {rest, b_onset, s_onset, trend, exhaustion}   (semi-Markov, medium)
                ↓ 影响
低层 dynamics: h_{i,t} ∈ R^m                          (continuous, fast)
                ↓ 影响
观测:         x_{i,t} ∈ R^d                          (153 因子)
```

生成模型：
$$
p_\theta(x, z^M, z^O, h) = \prod_t \underbrace{p(z^M_t | z^M_{<t})}_{\text{macro semi-Markov}} \cdot \underbrace{p(z^O_{i,t} | z^O_{i,<t}, z^M_t)}_{\text{onset under macro}} \cdot \underbrace{p(h_{i,t} | h_{i,<t}, x_{i,<t})}_{\text{RNN dynamics}} \cdot \underbrace{p(x_{i,t} | z^O_{i,t}, h_{i,t})}_{\text{emission}}
$$

Duration heads：
$$
p(T^M | z^M) \sim \text{NegBinom}(\alpha_M, \beta_M), \quad p(T^O | z^O, z^M) \sim \text{NegBinom}(\alpha_O(z^M), \beta_O(z^M))
$$

注意 onset duration 是**条件在 macro regime 上**的 — 这正是"非平稳"的根源。

推断模型（amortized）：
$$
q_\phi(z^M_{1:T}, z^O_{1:T} | x_{1:T}) = \text{Bi-directional Transformer encoder}
$$

PWC adaptive filter（升级版）：
$$
\text{filter}(i, t) = \mathbb{1}\left[ P(z^O_{i,t} \neq \text{rest} \mid x_{i,1:t}, \text{market context}_{1:t}) < \theta \right]
$$

#### 2.7.5 关于 Mamba / S4 / S5 系列（不要混淆的另一种 "SSM"）

最近 deep learning 圈非常火的 Mamba (Gu & Dao 2023) / S4 / S5 也常被称为 "State Space Models"，需要特别澄清：

| 维度 | 我们要的 (HSSM) | Mamba/S4/S5 |
|---|---|---|
| 隐变量类型 | 离散 + 连续混合 | 纯连续 |
| 显式 latent state | ✓ z^M / z^O | ✗ 内部 hidden state 不可索引 |
| 适合 onset filter | ✓ | ✗ |
| 强项 | 概率推断 / 可解释 | 长序列效率 O(N log N) |
| 在本任务中的位置 | **主框架** | 可作 encoder 替代 Bi-LSTM/Transformer，但不是主框架 |

如果实验中 Bi-directional Transformer encoder 在长序列上瓶颈，可以**把 Mamba 当作 q_phi 的内部 encoder**，不影响主框架。但不能用 Mamba 替代 HSSM 的概率结构 — 那样会失去离散 onset state 的可索引性。

#### 2.7.6 关键参考文献

P0（HSSM 主框架直接参考）：
- **Linderman et al. 2017**, "Bayesian Learning and Inference in Recurrent Switching Linear Dynamical Systems" (rSLDS), AISTATS — 单层 switching SSM 的现代起点
- **Dong et al. 2020**, "Collapsed Amortized Variational Inference for Switching Nonlinear Dynamical Systems" (SNLDS), ICML — 提供 collapsed VI 训练技巧
- **Becker-Ehmck et al. 2019**, "Switching Linear Dynamics for Variational Bayes Filtering", ICML — 切换 SSM 的 VB filter
- **Karl et al. 2017**, "Deep Variational Bayes Filters", ICLR — DKF 的扩展

P1（层次化扩展）：
- **Saeedi et al. 2016**, "The Segmented iHMM: A Simple, Efficient Hierarchical Infinite HMM" — HSMM 的层次化变体
- **Johnson & Willsky 2013**, "Bayesian Nonparametric Hidden Semi-Markov Models", JMLR — HSMM 的贝叶斯非参数版
- 多篇 2022-2024 hierarchical state-space VAE 工作（写作时再 deep dive）

P2（对照 baseline）：
- **Duan et al. 2022**, "FactorVAE", NeurIPS — 金融 VAE baseline，对照用
- **Gu & Dao 2023**, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" — 仅在 encoder 替代时引用
- **Rasul et al. 2021**, "TimeGrad", NeurIPS — Diffusion 时间序列基线（supplementary）

#### 2.7.7 实施风险

| 风险 | 概率 | 对策 |
|---|---|---|
| HSSM 训练不稳定（两层离散切换 + 连续隐变量耦合） | 高 | 借鉴 SNLDS collapsed amortized VI；分阶段训练（先单层后层次） |
| 离散状态退化（崩到单一状态） | 中 | KL annealing + 状态熵正则 + 数据增强 |
| Bi-directional encoder 在 5000+ 股 × 1500 天太慢 | 中 | encoder 切到 Mamba/S5；或 batch by stock 训练 |
| 与 LGBM/MASTER baseline 共训复杂 | 中 | **两阶段**：先训 HSSM 出 filter，filter 静态后训 backbone（推荐） |
| HSSM 实现工作量高（>SNLDS） | 必然 | 时间预算：M1+M2 共 8 周给 HSSM；fallback 到 SNLDS 单层 |

#### 2.7.8 决策：HSSM vs SNLDS vs 经典 HSMM 三档方案

| 方案 | 工作量 | paper 上限 | 失败 fallback |
|---|---|---|---|
| **A. HSSM 双层（主推）** | 高（8 周） | KDD A / NeurIPS | 退到 B |
| **B. SNLDS 单层** | 中（4 周） | CCF B-A 边界 | 退到 C |
| **C. 经典 HSMM** | 低（2 周） | CIKM / 国内 | — |

**推荐路径**：M1 先做 C（建 baseline + 验证概率图正确），M2 升级到 B（加 deep encoder/decoder），M3 升级到 A（加 macro layer）。这样每个 milestone 都有可投稿的版本，**Gate 失败也有 paper 可写**。

#### 2.7.9 现有 V12 资产对接 HSSM 框架

| 已有 | 对接 HSSM 的位置 |
|---|---|
| 市场环境感知系统（指数 + 板块热度 + ETF + 视觉 LLM） | $z^M_t$ 的 informative prior / 监督信号 |
| 概念板块数据库（6.15M 热度因子） | 中层 $z^O$ 的 conditional context |
| 153 因子库 | 观测 $x_{i,t}$ |
| v3c past_r5 ≤ 8% | $z^O = \text{onset}$ 的弱监督 anchor |
| pump_ratio 推理信号 | 等价于推断后的 $P(z^O | x)$ 的简化形式（启发） |

这种对接说明 HSSM 不是"另起炉灶"，而是把你已有 V12 资产**纳入一个统一的概率框架**。这是 paper 写作的关键卖点。

---

## 3. 关键术语对照（消化检索用）

### 3.1 中→英 全对照

| 中文 | 英文 | 备注 |
|---|---|---|
| 启动子 | Movement Onset Event | 学术名 |
| 启动 (动作) | Onset | |
| 正/负启动子 | Bullish/Bearish Onset | |
| 启动期 | Onset Phase / Onset Duration | 变长 |
| 假启动 | Pseudo-Onset / Spurious Onset | |
| 池 | Candidate Pool / Filtered Universe | |
| 双轨架构 | Dual-Track Portfolio Construction | |
| 行业 cap | Industry Concentration Constraint | |
| 行业过滤 | Sector Filter / Industry Filter | |
| 灾难月 | Tail-Risk Month / Disaster Month | |
| 概念热度 | Concept Heat / Thematic Momentum | |
| 因子库 | Factor Zoo | "因子动物园"是行业惯用语 |
| 稀疏激活 | Sparse Activation (of factors) | |
| 走样验证 (walk-forward) | Walk-Forward Validation / Expanding-Window CV | |
| OOS 过拟合 | OOS Overfitting / Test-Set Hacking | |
| 龙虎榜 | Daily Block Trade Disclosure | A 股特有 |
| 振幅 | Daily Range | |
| 涨跌停 | Limit Up / Limit Down | A 股特色规则 |
| ST 股 | Special Treatment Stocks | A 股退市预警 |

### 3.2 学术专业术语 must-know

| 术语 | 含义 | 在 paper 中的角色 |
|---|---|---|
| **Triple-Barrier Method (TBM)** | López de Prado 提出的三屏障 label | 最强对照 baseline |
| **Fixed-Horizon Labeling (FH)** | 用 t+h 收益符号打 label | 最弱 baseline |
| **Change-Point Detection (CPD)** | 时间序列分割问题 | §2 核心工具 |
| **Hidden Markov Model (HMM)** | 隐马尔可夫，状态停留时间 = 几何分布 | §2 入门框架 |
| **Hidden Semi-Markov (HSMM)** | HMM 推广，状态停留任意分布 | §2 推荐主框架 |
| **Diebold-Mariano (DM) Test** | 比较两个预测的精度差异是否显著 | T5 统计检验 |
| **Benjamini-Hochberg (BH) FDR** | 多重比较的 FDR 控制 | T5 多次比较时用 |
| **Bootstrap CI** | 重采样估计置信区间 | T5 |
| **RankIC** | Rank Information Coefficient，秩相关 | 量化预测排序质量 |
| **Information Ratio (IR)** | α/σ(α) | 风险调整后的超额收益 |
| **Maximum Drawdown (MDD)** | 最大回撤 | 风险指标 |
| **Sharpe Ratio** | (E[R] - Rf)/σ(R) | 风险调整收益 |
| **Calmar Ratio** | 年化收益 / |MDD| | 抗回撤指标 |
| **Cross-Sectional IC** | 横截面 IC | 一个时刻所有股票预测 vs 实际 rank 相关 |
| **Calibration** | 概率校准（Platt / temperature scaling） | T4 对比 |
| **Self-Supervised Learning (SSL)** | 自监督 | Denoised Label baseline 用 |
| **Walk-Forward / Out-of-Sample (OOS)** | 时序严格分割 | Dataset 协议 |
| **Look-ahead Bias** | 未来函数泄漏 | 必须杜绝（V3 教训） |
| **Survivorship Bias** | 幸存者偏差 | dataset 必须包含退市股 |
| **Class Imbalance** | 类别不平衡 | onset 是稀有事件 |
| **Focal Loss** | 类别不平衡的损失函数 | 训练时可考虑 |
| **Label Smoothing** | 标签平滑正则化 | 训练时可考虑 |
| **Multi-Task Learning** | 多任务 | 同时预测涨跌 + 持续时间 |

---

## 4. 文献占用与避雷

### 4.1 命名雷区（不能用）

| 已被占用 | 占用方向 | 风险 |
|---|---|---|
| "pump" | crypto pump-and-dump 操纵检测 | 完全跑题 |
| "momentum ignition" | 高频操纵性激发 | 监管负面 |
| "spike" | 偶尔指操纵性短脉冲 | 歧义 |

### 4.2 必读 baseline（这周内开始读）

#### 优先级 P0（决定 paper 站位）
1. **López de Prado 2018**, *Advances in Financial Machine Learning*, Chapter 3 (Labeling)
   — Triple-Barrier 教科书
2. **Han et al. 2023**, "Optimal Trend Labeling in Financial Time Series", IEEE
   — 同领域 SOTA label engineering
3. **Adams & MacKay 2007**, "Bayesian Online Changepoint Detection", arXiv:0710.3742
   — §2 框架 B 的源头

#### 优先级 P1（baseline 实现）
4. **Xu et al. 2021**, "HIST: Hierarchical Information Aggregation Framework for Stock Prediction", KDD
5. **Li et al. 2024**, "MASTER: Market-Guided Stock Transformer", AAAI
6. **Fan et al. 2024**, "StockMixer", AAAI
7. **Duan et al. 2022**, "FactorVAE", NeurIPS
8. **Wang et al. 2021**, "Denoised Labels via SSL", arXiv:2112.10139

#### 优先级 P2（理论包装支撑）
9. **Yu 2010**, *Hidden Semi-Markov Models*, AIJ
10. **Bulla & Bulla 2006**, "Stylized facts of financial time series and HSMM", CSDA
11. **Nystrup et al. 2017**, "Regime-based portfolio under HSMM"
12. **Atilgan et al. 2022**, "Short-sale constraints and cross-predictability in China"
13. **Bacry et al. 2015**, "Hawkes processes in finance", Market Microstructure and Liquidity

### 4.3 必显式划清界限的领域

| 领域 | 划清方式 |
|---|---|
| Pump-and-dump 操纵检测 | Intro 明说："Our 'onset' refers to legitimate price movements, distinct from manipulative pump-and-dump schemes" |
| Sentiment-based prediction | Related Work 提一句不重叠 |
| Insider trading detection | 完全不提 |

---

## 5. 必打 SOTA Baseline 矩阵

### 5.1 矩阵设计（核心一击）

|  | FH | TB (LdP) | OTL | DL | **+ PWC** | **+ Adaptive (HSMM/BOCPD)** |
|---|---|---|---|---|---|---|
| LGBM | ✓ 已有 | 需跑 | 需跑 | 需跑 | ✓ 已有 (v3c) | 需做（**核心创新**） |
| HIST | 需跑 | 需跑 | — | — | 需跑 | 需做 |
| MASTER | 需跑 | 需跑 | — | — | 需跑 | 需做 |
| StockMixer | 需跑 | 需跑 | — | — | 需跑 | 需做 |
| FactorVAE | 需跑 | 需跑 | — | — | 需跑 | 需做 |

每个 cell 在 A 股 + NASDAQ 两市场跑，得到 5×6×2 = **60 个对比点**。审稿人压力测试时易于站立。

### 5.2 SOTA backbone 复现优先级

```
HIST (P0)        ── Qlib 官方集成，最易接入
MASTER (P0)      ── 作者放了完整 PyTorch 实现
StockMixer (P1)  ── 官方 GitHub，结构简单
FactorVAE (P2)   ── Qlib 集成，但训练慢
```

如果时间紧，砍 FactorVAE 不影响主结论。

---

## 6. 7 张主表的精确设计

### Table 1 — Main Result（核心）

子表 A：A股 / 子表 B：NASDAQ
每个子表：5 backbone × 6 label/filter 方案

| | FH | TB | OTL | DL | FH+PWC | FH+Adaptive |
|---|---|---|---|---|---|---|
| LGBM | 1.50 | ? | ? | ? | **1.91** | **?** |
| HIST | ? | ? | — | — | ? | ? |
| ... | | | | | | |

每个 cell：α (pp/月)，下方括号显示 Sharpe / RankIC。

### Table 2 — Ablation

```
(a) v3b baseline (FH, no filter)
(b) + past_r5 ≤ 8% (v3c, fixed-window)
(c) + adaptive window (BOCPD-based)
(d) + adaptive window + duration learning (HSMM)
(e) + bidirectional filter (also future_r5 constraint)
```

### Table 3 — τ Sensitivity

x 轴：τ ∈ {0.04, 0.06, 0.08, 0.10, 0.12, ∞}
y 轴：α, Sharpe, sample count after filtering
折线图主表示，附数字表。

### Table 4 — Ranking Signal

| Signal | RankIC | TopK Sharpe | α | Worst Month |
|---|---|---|---|---|
| P_up alone | base | base | 2.078 | -0.01 |
| max softmax | | | | |
| temperature-scaled | | | | |
| **ratio = P_up/(P_down+ε)** | | | **2.236** | **+0.97** |
| learned ranker on probs | | | | |

### Table 5 — Statistical Significance

| Comparison | DM Statistic | p-value | BH-adjusted | Bootstrap CI |
|---|---|---|---|---|
| LGBM-FH vs LGBM-FH+PWC | 2.34 | 0.019 | 0.025 | [0.31, 0.49] |
| ... | | | | |

### Table 6 — Backward Contamination Evidence

| Horizon | v3b error rate | v3c error rate | adaptive | Δ adaptive vs v3c |
|---|---|---|---|---|
| t+1 | 40.44% | 27.83% | ?  | ? |
| t+3 | 50.53% | 23.50% | ? | ? |
| t+5 | 56.69% | 28.04% | ? | ? |
| t+10 | ? | ? | ? | ? |

### Table 7 — Portfolio-Level Backtest (附录)

walk-forward 19 月：月度 α 序列、灾难月分析、MDD、行业暴露。

---

## 7. Formalization（写作时使用）

完整 notation：

- $N$：股票数，$T$：时间步数
- $P_{i,t}$：股票 $i$ 在时刻 $t$ 的价格
- $r^{(h)}_{i,t} = P_{i,t+h}/P_{i,t} - 1$：h-period 前向收益
- $\mathbf{x}_{i,t} \in \mathbb{R}^d$：特征向量（153 因子）
- $y_{i,t} \in \{\text{up, down, neutral}\}$：三分类 label
- $z_{i,t} \in \{0, 1, 2\}$：隐含 onset 状态
- $T^{onset}_{i,t}$：从 $t$ 开始（如果在 onset 中）的 onset 持续时间
- $\tau_{cp}(t)$：BOCPD 估计的最近 change-point 时刻

**Definition 1 (Movement Onset)**:
A *bullish onset* of stock $i$ at time $t$ is a maximal interval $[t, t+T]$ such that
1. $r^{(T)}_{i,t} \geq \Delta^+$ (cumulative return exceeds threshold)
2. $\max_{0 \leq h \leq T} \text{drawdown}(P_{i,t:t+h}) \leq \Delta^-$ (bounded drawdown)
3. $z_{i,s} = 1$ for $s \in [t, t+T]$ in the HSMM generative model.

Bearish onset 类比定义。

**Definition 2 (Backward-Context Filter, generalized)**:
$$
\text{filter}(i, t) = \mathbb{1}\left[\sum_{s=\tau_{cp}(t)}^{t-1} r_{i,s} \leq \theta\right]
$$

**Definition 3 (PWC v1, Fixed-Window — this is v3c)**:
$\tau_{cp}(t) \equiv t - W$, $W=5$, $\theta = 0.08$.

**Definition 4 (PWC v2, Adaptive — this paper's main proposal)**:
$\tau_{cp}(t)$ from online BOCPD with hazard rate $\lambda$ learned per stock.

**Proposition 1 (Onset Duration Non-Stationarity)**:
For stocks in $\mathcal{S}$ and time windows in $\mathcal{T}$, the empirical distribution of $T^{onset}_{i,t}$ has a coefficient of variation $> 0.5$ (i.e., far from constant). [可用数据验证]

**Proposition 2 (Filter Dominance)**:
Let $\mathcal{L}_{\text{fixed}}$ and $\mathcal{L}_{\text{adapt}}$ be the cross-entropy losses on test set after training on $D_{PWC, \text{fixed}}$ and $D_{PWC, \text{adapt}}$ respectively. Under Proposition 1, $\mathbb{E}[\mathcal{L}_{\text{adapt}}] < \mathbb{E}[\mathcal{L}_{\text{fixed}}]$.

**Proposition 3 (Asymmetry under Short-Sale)**:
In short-sale-constrained markets, the contamination ratio for *bearish* onset labels strictly exceeds that of bullish labels: $\eta_{\text{down}} > \eta_{\text{up}}$. [用你 v3b OOS 数据已可证实证]

---

## 8. 已有实证资产 vs 待补

### 已有
- ✓ v3b 三分类 baseline OOS 6 月
- ✓ v3c past_r5 ≤ 8% OOS 6 月
- ✓ OOS t+1/t+3/t+5/t+10 contamination 数据
- ✓ pump_ratio vs composite OOS 数据
- ✓ walk-forward 19 月（V11 阶段，需在 v3b/v3c 上重跑）
- ✓ A 股 5129 股 × 6 年因子库
- ✓ 双轨 portfolio 完整实现
- ✓ ST 排除 + zombie filter 数据

### 必补
- ✗ HSMM / BOCPD onset 检测器实现
- ✗ LdP Triple-Barrier label 实现 + 跑通
- ✗ Optimal Trend Labeling (IEEE 2023) 复现
- ✗ Denoised Label (arXiv 2021) 复现
- ✗ HIST / MASTER / StockMixer / FactorVAE 接入
- ✗ NASDAQ-100 数据管道
- ✗ Qlib Alpha158 数据兼容
- ✗ DM 检验 + bootstrap CI + BH FDR
- ✗ τ 完整网格 5-6 个点
- ✗ Adaptive PWC 实现 + 在所有 backbone 上跑

### 工作量估算（兼职）
| 阶段 | 工作 | 时长 |
|---|---|---|
| M1 | repo + LdP TB + HSMM 原型 + Gate 1 | 4-5 周 |
| M2 | HIST + Qlib + Gate 2 | 4 周 |
| M3 | MASTER + StockMixer + NASDAQ + Gate 3 | 4 周 |
| M4 | OTL/DL/FactorVAE + τ 网格 + 统计检验 + Gate 4 | 4 周 |
| M5 | Intro/Related/Method 写作 + Prop 形式化 | 4 周 |
| M6 | Experiments/Analysis 写作 + 内部 review | 4 周 |
| 总 | | **~6 个月兼职 / ~3 个月全职** |

---

## 9. 研究分支基础设施

### 9.1 仓库结构（建议新建 stockagent-research）

```
D:\aicoding\stockagent-research\
├─ src/
│   ├─ onset/                              ← §2 核心
│   │   ├─ hsmm.py                          HSMM 状态推断
│   │   ├─ bocpd.py                         Bayesian online change-point detection
│   │   ├─ duration_model.py                onset duration 分布学习
│   │   └─ filter.py                        fixed / adaptive backward filter
│   ├─ labels/
│   │   ├─ fixed_horizon.py
│   │   ├─ triple_barrier.py                LdP
│   │   ├─ optimal_trend.py                 IEEE 2023
│   │   ├─ continuous_trend.py              Entropy 2020
│   │   └─ denoised_label.py                arXiv 2021
│   ├─ models/
│   │   ├─ lgbm_baseline.py                 (v3c 移植版)
│   │   ├─ hist/
│   │   ├─ master/
│   │   ├─ stockmixer/
│   │   └─ factorvae/
│   ├─ ranking/
│   │   ├─ ratio_score.py                   ratio = P_up/(P_down+ε)
│   │   ├─ temperature_scale.py
│   │   └─ confidence_calibration.py
│   ├─ data/
│   │   ├─ ashares_tushare.py
│   │   ├─ ashares_qlib.py                  Alpha158 公开版
│   │   └─ nasdaq_qlib.py
│   ├─ evaluation/
│   │   ├─ metrics.py
│   │   ├─ dm_test.py
│   │   ├─ bootstrap_ci.py
│   │   └─ fdr_correction.py
│   └─ portfolio/
│       └─ dual_track.py                    生产线脱敏移植
├─ experiments/
│   ├─ t1_main/
│   ├─ t2_ablation/
│   ├─ t3_tau/
│   ├─ t4_ranking/
│   ├─ t5_significance/
│   ├─ t6_contamination/
│   └─ t7_portfolio/
├─ configs/
├─ checkpoints/
├─ results/
├─ paper/
│   ├─ main.tex
│   ├─ sections/
│   ├─ figures/
│   ├─ tables/
│   └─ refs.bib
└─ README.md
```

### 9.2 生产 → 研究代码移植清单

| 生产 | 研究 | 改动 |
|---|---|---|
| `src/stockagent_analysis/v12_scoring.py` | `src/portfolio/dual_track.py` + `src/onset/filter.py` | 拆分；去 Tushare 付费依赖 |
| `train_pump_classifier_3way_v3c.py` | `src/labels/fixed_horizon.py` + `src/onset/filter.py` | 抽出 PWC filter 函数 |
| `backtest_v12_dual_v6.py` | `src/portfolio/dual_track.py` + `experiments/t7_portfolio/` | 参数化；去公司私有逻辑 |
| `backtest_v12_ratio_sort.py` | `experiments/t4_ranking/` | 直接搬 |
| `update_factor_lab_from_tushare.py` | `src/data/ashares_tushare.py`（内部用） + `src/data/ashares_qlib.py`（公开版） | 替换底层数据源 |

---

## 10. Go/No-Go Gate 与节奏

### Gate 1 (M1 末) — TB + PWC 是否正交
**实验**：用 LdP Triple-Barrier label 训 LGBM，对比 +fixed PWC filter / 不加。
**Pass**：PWC 在 TB 之上仍带正向。
**Fail**：PWC 退化为 "FH label 的替代品"，paper 故事弱一档，考虑降级到 CIKM。

### Gate 2 (M2 末) — HIST + PWC 是否正交
**实验**：在 GNN 类 backbone 上看 fixed PWC 是否仍 model-agnostic 有效。
**Pass**：跨架构成立。
**Fail**：改写为 "Tree-based model-specific filter"，期刊路线。

### Gate 3 (M3 末) — Adaptive 是否显著优于 Fixed
**实验**：HSMM/BOCPD-based adaptive filter vs fixed-window PWC。
**Pass**：adaptive 在 A 股上 ≥80% backbone 上显著优于 fixed。
**Fail**：paper 主标题改为 "Fixed-Window Backward Filtering"，理论 framework 退到附录。**这个 gate 最关键**。

### Gate 4 (M4 末) — NASDAQ 是否泛化
**Pass**：跨市场，KDD ADS Track 合理目标。
**Fail**："Chinese A-share specific"，故事改写，考虑国内 CCF B。

### Gate 5 (M5 中) — 完整 T1 矩阵
**Pass**：≥75% cell 显著正向。
**Fail**：submission 推迟一轮，补实验。

---

## 11. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| HIST/MASTER 在 A 股复现性能差 | 中 | 先在 Alpha158 上复现作者数字 → 确认实现正确后再换数据 |
| Adaptive PWC vs Fixed PWC 提升不显著 | 中 | Fallback：把 fixed-window 论点强化为"理论上是 first-order Taylor approximation" |
| NASDAQ 上 PWC 失效（卖空相对自由） | 中-高 | 选 hard-to-borrow stocks 子集 / 改韩国 KOSPI / 印度 NIFTY |
| HSMM 训练慢 / 不收敛 | 中 | 用 Variational EM 或 Neural HSMM；BOCPD 是 fallback |
| 审稿人质疑"只对 down 单向约束" | 高 | T2 ablation row (e) 双向版本必做 |
| 数据可复现性 | 必然 | Alpha158 公开版 + NASDAQ 公开版必须有 |

---

## 12. 决策清单（用户消化后回答）

A. **理论框架主选**：HSMM / BOCPD / 双层组合 / 其他
B. **生产线节奏**：保持每日推理 / 暂停以全力做研究 / 部分暂停
C. **仓库**：新建 stockagent-research / 当前 repo 加 research/ 分支
D. **优先 Gate**：Gate 1 (TB+PWC) 先做 / Gate 3 (adaptive) 先做
E. **写作语言**：英文一稿 / 中文先写后翻译
F. **作者署名**：单一作者 / 邀请合作

---

## 附录 A：本次会话核心洞察归档

| 日期 | 洞察 | 影响 |
|---|---|---|
| 2026-05-28 | 启动子非平稳，非固定窗口 | 把 paper 从"工程优化"提升到"建模框架"，§2 是核心 |
| 2026-05-28 | pump 关键词被 crypto fraud 占用，必须改名 | 命名改为 Movement Onset |
| 2026-05-28 | LdP Triple-Barrier 是绕不开 baseline | T1 必跑 + 必显示正交性 |
| 2026-05-28 | v3c 的 past_r5 ≤ 8% 是 Adaptive 的退化特例 | 写作时定位为 first-order approximation |
| 2026-05-28 | ratio = P_up/(P_down+ε) 实战 +0.16pp，灾难月 +2.30pp | T4 已有完整数据 |
| 2026-05-28 | HSSM 双层比 SNLDS 单层更适合 | macro regime + stock onset 两层耦合; 直接对接生产线市场环境感知系统; §2.7 主推 |
| 2026-05-28 | Mamba/S4 类"现代 SSM"不能做主框架 | 无显式离散 latent；可作 encoder 替代 |

---

**TODO 优先级**（消化后再决定执行）：

1. [ ] 用户消化 §2 + §3 术语
2. [ ] 用户做 §12 决策
3. [ ] 决策后建 stockagent-research 仓库骨架
4. [ ] Gate 1 实验（TB + Fixed PWC）一周内启动
5. [ ] 同步开始读 P0 三篇论文
