# IRL/模仿学习：从历史完美买卖点学习交易经验 — 可行性分析与工程方案

> 版本: v2.0 | 日期: 2026-03-15 | 状态: 设计阶段（完整版）

---

## 目录

1. [项目概述与核心思路](#1-项目概述与核心思路)
2. [学术可行性评估](#2-学术可行性评估)
3. [技术可行性评估](#3-技术可行性评估)
4. [算法方案对比与推荐](#4-算法方案对比与推荐)
5. [推荐架构：三阶段Pipeline](#5-推荐架构三阶段pipeline)
6. [数据工程方案](#6-数据工程方案)
7. [模型训练方案](#7-模型训练方案)
8. [推理与实盘应用方案](#8-推理与实盘应用方案)
9. [风险与挑战](#9-风险与挑战)
10. [里程碑与路线图](#10-里程碑与路线图)
11. [附录：学术论文与开源项目](#11-附录学术论文与开源项目)
12. [结论与建议](#12-结论与建议)
13. [计算成本与硬件方案](#13-计算成本与硬件方案)
14. [交易截面特征空间全景](#14-交易截面特征空间全景通达信--tushare--akshare)
15. [Gemini Embedding 多模态方案](#15-gemini-embedding-多模态方案k线图--特征序列统一嵌入)
16. [项目工程化设计](#16-项目工程化设计)
17. [数据版本管理与实验追踪](#17-数据版本管理与实验追踪)
18. [完整里程碑与任务分解（修订版）](#18-完整里程碑与任务分解修订版)
19. [项目成功标准与退出条件](#19-项目成功标准与退出条件)

---

## 1. 项目概述与核心思路

### 1.1 目标

在100只A股上，通过算法标注历史"完美买卖点"，构造虚拟交易高手的操作记录（年化50%+），结合现有stockagent-analysis的多维度评分体系，用模仿学习方法从中提取交易经验，实现对这100只股票未来买卖点的预判。

### 1.2 核心流程

```
历史K线数据 ──→ 买卖点标注算法 ──→ 虚拟专家交易记录(Demonstrations)
                                          │
现有29 Agent评分 ─┐                       │
技术指标计算 ─────┼──→ 特征向量(State) ←──┘
K线图LLM评估 ────┘                       │
                                          ▼
                              IRL/模仿学习训练
                                          │
                                          ▼
                              学到的奖励函数/策略
                                          │
                                          ▼
                              实时预判买卖点信号
```

### 1.3 与现有项目的关系

| 维度 | 现有stockagent-analysis | 新项目 |
|------|------------------------|--------|
| 输入 | 单只股票实时数据 | 100只股票历史数据 |
| 核心 | 29个Agent独立评分 | 从评分+指标中学习买卖模式 |
| 输出 | 综合评分+决策建议 | 精确买卖点时机预判 |
| 定位 | 特征提取与打分引擎 | 决策优化层（消费评分结果） |

---

## 2. 学术可行性评估

### 2.1 IRL在金融交易中的学术现状

**结论：学术上可行，但处于早期探索阶段。** 直接将IRL用于股票交易的论文较少（约5-8篇），但趋势明确向好。

#### 核心支撑论文

| 论文 | 年份 | 方法 | 关键贡献 | 与本项目相关度 |
|------|------|------|---------|---------------|
| **TAIRL** (Sun et al., Applied Intelligence) | 2023 | MaxEnt IRL + transaction-aware reward | 解决交易中奖励稀疏+hold状态歧义 | ⭐⭐⭐⭐⭐ 最直接相关 |
| **iRDPG** (Liu et al., AAAI 2020) | 2020 | Imitative RDPG + LSTM | 用经典策略做专家示范，POMDP建模 | ⭐⭐⭐⭐ |
| **Pro Trader RL** (Gu et al., Expert Systems) | 2024 | 分解式RL(买入/卖出/止损独立Agent) | 模仿专业交易者的分阶段决策 | ⭐⭐⭐⭐⭐ 架构契合 |
| **Human-aligned Trading** (Ye & Schuller) | 2023 | Imitative Multi-Loss DQN | 将"人类对齐"引入交易RL | ⭐⭐⭐⭐ |
| **Reward Shaping via Expert Feedback** (KAIS) | 2025 | 专家反馈驱动的奖励塑造 | 技术指标信号作为专家反馈 | ⭐⭐⭐⭐ |
| **ElliottAgents** (Applied Sciences) | 2024 | LLM多Agent + Elliott Wave + DRL | 波浪理论+LLM+RL融合 | ⭐⭐⭐⭐ 波浪/缠论方向 |
| **CFA Institute IRL Guide** | 2025 | 综述: MaxEnt/Bayesian/AIRL/T-REX | 行业实践指南，确认IRL在投资中可用 | ⭐⭐⭐ |

#### 学术层面的关键洞察

1. **IRL的核心价值**：不需要手动设计奖励函数（传统RL最大难点），而是从专家行为中反推。这完美契合"从完美买卖点反推什么是好的交易决策"
2. **TAIRL的启发**：交易场景特有的三个问题（利润延迟/不同持仓周期/hold歧义）已有学术解决方案
3. **Pro Trader RL的架构启发**：将买入/卖出/止损分解为独立模块的思路，与本项目29个Agent的多维度评分天然契合
4. **缠论+ML缺乏学术标准化**：缠论的笔/线段/中枢算法没有统一标准，本项目的实现已算领先

### 2.2 方法论对比

| 方法 | 适用场景 | 数据需求 | 训练难度 | 本项目适配度 |
|------|---------|---------|---------|-------------|
| **MaxEnt IRL** | 从专家轨迹反推奖励函数 | 中（数百条轨迹） | 高 | ⭐⭐⭐⭐ 经典可靠 |
| **AIRL** | 对抗式奖励学习，可迁移 | 高（数千条轨迹） | 很高 | ⭐⭐⭐ 数据可能不足 |
| **GAIL** | 生成式对抗模仿，不需显式奖励 | 高 | 很高 | ⭐⭐⭐ 训练不稳定 |
| **Behavioral Cloning (BC)** | 直接监督学习模仿 | 低（数百条） | 低 | ⭐⭐⭐⭐⭐ 简单有效的baseline |
| **DAgger** | 交互式纠错模仿学习 | 中 | 中 | ⭐⭐ 需要在线专家 |
| **Decision Transformer** | 序列建模，离线RL | 中 | 中 | ⭐⭐⭐⭐⭐ 与LLM架构契合 |
| **CQL/IQL (离线RL)** | 从离线数据学习策略 | 中 | 中 | ⭐⭐⭐⭐ 不过拟合OOD动作 |

---

## 3. 技术可行性评估

### 3.1 开源生态成熟度

| 框架 | Stars | 维护状态 | 适用算法 | 推荐度 |
|------|-------|---------|---------|--------|
| **imitation** (HumanCompatibleAI) | ~1.6k | 活跃 | BC, DAgger, GAIL, AIRL | ⭐⭐⭐⭐⭐ 核心推荐 |
| **d3rlpy** | ~1.6k | 活跃 | CQL, IQL, Decision Transformer, 20+算法 | ⭐⭐⭐⭐⭐ 离线RL首选 |
| **FinRL** | ~6.6k | 活跃 | 交易环境 + RL pipeline | ⭐⭐⭐⭐ 环境定义参考 |
| **Stable-Baselines3** | ~10k | 活跃 | PPO, SAC, DQN等 | ⭐⭐⭐⭐ imitation的底层依赖 |
| **FinRL_Imitation_Learning** | — | 实验性 | BC初始化 + RL微调 | ⭐⭐⭐ pipeline思路参考 |

### 3.2 数据可行性分析

本项目数据来源充足：

| 数据类型 | 来源 | 可用性 | 说明 |
|----------|------|--------|------|
| 日/周/月K线 | 通达信本地/Tushare/AKShare | ✅ 完备 | 已有完整采集pipeline |
| 1h K线 | AKShare | ✅ 可用 | TDX无分钟线时自动降级 |
| 技术指标 | data_backend.py计算 | ✅ 完备 | MA/MACD/RSI/KDJ/布林等 |
| 缠论信号 | _detect_chanlun_signals() | ✅ 已实现 | 分型→笔→中枢→买卖点 |
| K线形态 | _detect_advanced_kline_patterns() | ✅ 已实现 | 15+种经典形态 |
| 背离/量价/支撑阻力 | 7个新技术Agent | ✅ 已实现 | 多周期信号 |
| 29 Agent评分 | stockagent-analysis | ✅ 核心资产 | 需批量回跑历史数据 |
| K线图LLM视觉评估 | Vision Agent | ⚠️ 成本高 | 可选，作为增强特征 |

### 3.3 计算资源评估

| 阶段 | 计算需求 | 本地可行性 |
|------|---------|-----------|
| 买卖点标注 | CPU，数分钟/只 | ✅ 完全可行 |
| 特征计算（技术指标） | CPU，秒级/只 | ✅ 完全可行 |
| Agent评分回跑 | LLM API调用，100只×N个历史点 | ⚠️ 成本和时间需评估 |
| BC/MaxEnt IRL训练 | CPU或单GPU，数小时 | ✅ 可行 |
| Decision Transformer训练 | 单GPU，数小时 | ⚠️ 需GPU |
| GAIL/AIRL训练 | GPU推荐，数小时-天 | ⚠️ 需GPU |

---

## 4. 算法方案对比与推荐

### 4.1 三条技术路线

#### 路线A：经典IRL（MaxEnt IRL → RL Policy）

```
专家轨迹 ──→ MaxEnt IRL ──→ 奖励函数R(s,a) ──→ RL训练(PPO) ──→ 交易策略π
```

- **优点**：学到的奖励函数可解释（"什么状态下买入是好的"），可迁移到新股票
- **缺点**：训练复杂，需要定义好状态空间和动作空间；两阶段训练可能误差累积
- **适用**：需要理解"为什么"买卖的场景

#### 路线B：直接模仿学习（BC + DAgger）

```
专家轨迹 ──→ BC(监督学习) ──→ 策略π(s) → action
                  │
                  └──→ DAgger(交互纠错) ──→ 改进策略π*
```

- **优点**：实现最简单，数据需求最低，快速迭代
- **缺点**：BC有复合误差(compounding error)问题；DAgger需要在线专家
- **适用**：快速验证可行性的baseline

#### 路线C：Decision Transformer（序列建模）

```
(s₁,a₁,R₁, s₂,a₂,R₂, ..., sT,aT,RT) ──→ Transformer ──→ 给定目标收益，生成动作序列
```

- **优点**：将RL问题转化为序列预测，天然契合LLM架构；不需要显式奖励函数设计；可条件生成（指定目标收益率）
- **缺点**：需要较多数据；Transformer训练需GPU
- **适用**：与现有LLM多Agent架构深度融合

### 4.2 推荐方案：渐进式三阶段

**Phase 1 → BC（Behavioral Cloning）作为baseline**
- 快速验证数据pipeline和特征工程是否有效
- 1-2周可出初步结果
- 无需GPU

**Phase 2 → MaxEnt IRL + Reward Shaping**
- 从专家买卖点反推奖励函数
- 结合现有Agent评分做Reward Shaping
- 学到可解释的奖励函数

**Phase 3 → Decision Transformer（目标方案）**
- 将全部历史轨迹（状态+动作+收益）作为序列输入
- 条件生成：指定目标收益率，输出买卖动作
- 与stockagent-analysis的LLM架构深度集成

---

## 5. 推荐架构：三阶段Pipeline

### 5.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 1: 数据准备与标注                        │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ K线数据   │→│ 买卖点标注算法 │→│ 专家交易记录(Demonstrations)│   │
│  │(100只×3年)│  │ (缠论/波浪/  │  │ {date, action, price,    │   │
│  │           │  │  多策略融合)  │  │  holding_period, pnl}    │   │
│  └──────────┘  └──────────────┘  └──────────────────────────┘   │
│        │                                      │                  │
│        ▼                                      ▼                  │
│  ┌──────────────────────┐  ┌────────────────────────────────┐   │
│  │ 特征提取(每个时间步)   │  │  轨迹构造                      │   │
│  │ • 29 Agent评分        │  │  τ = (s₁,a₁,r₁,...,sT,aT,rT) │   │
│  │ • 技术指标(MA/RSI/…) │  │  • 每只股票 ≈ 20-50条交易      │   │
│  │ • 缠论/形态/背离信号  │  │  • 共 2000-5000 条交易         │   │
│  │ • 市场环境特征        │  └────────────────────────────────┘   │
│  └──────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 2: 模型训练                              │
│  ┌─────────┐  ┌─────────────┐  ┌──────────────────────────┐    │
│  │ Phase 1 │  │   Phase 2   │  │       Phase 3            │    │
│  │   BC    │→│ MaxEnt IRL  │→│ Decision Transformer     │    │
│  │(baseline)│  │(奖励学习)   │  │(序列建模，目标方案)      │    │
│  └─────────┘  └─────────────┘  └──────────────────────────┘    │
│       │              │                    │                      │
│       └──────────────┴────────────────────┘                      │
│                      │ 模型评估：回测收益、夏普比率、最大回撤       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stage 3: 推理与应用                            │
│  ┌──────────────┐  ┌─────────────┐  ┌────────────────────┐     │
│  │ 实时特征计算   │→│  模型推理    │→│  买卖信号输出       │     │
│  │(复用现有Agent) │  │ (已训练模型) │  │  • 买入概率/卖出概率│     │
│  │              │  │             │  │  • 建议仓位         │     │
│  └──────────────┘  └─────────────┘  │  • 置信度           │     │
│                                      └────────────────────┘     │
│                                              │                   │
│                                              ▼                   │
│                                     融入现有报告PDF               │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 交易环境定义

```python
# 状态空间 State (约 80-120维)
state = {
    # === 第一类：现有Agent评分 (29维) ===
    "agent_scores": {
        "trend_score": float,         # TREND Agent评分 0-100
        "capital_flow_score": float,  # 资金流评分
        "chanlun_score": float,       # 缠论评分
        "divergence_score": float,    # 背离评分
        # ... 共29个Agent
    },

    # === 第二类：技术指标 (约30维) ===
    "technical": {
        "ma5_bias": float,            # MA5乖离率
        "ma20_bias": float,           # MA20乖离率
        "rsi_6": float,               # RSI(6)
        "rsi_12": float,              # RSI(12)
        "macd_hist": float,           # MACD柱
        "kdj_j": float,              # KDJ J值
        "boll_position": float,      # 布林带位置(0-1)
        "volume_ratio": float,        # 量比
        "turnover_rate": float,       # 换手率
        # ... 多周期
    },

    # === 第三类：结构化信号 (约20维) ===
    "signals": {
        "chanlun_buy_type": int,      # 缠论买点类型(0/1/2/3)
        "chanlun_sell_type": int,     # 缠论卖点类型
        "divergence_type": int,       # 背离类型(-2~2)
        "kline_pattern": int,         # K线形态编码
        "support_distance": float,    # 距支撑位距离%
        "resistance_distance": float, # 距阻力位距离%
        "timeframe_resonance": float, # 多周期共振分数
    },

    # === 第四类：持仓状态 (3维) ===
    "position": {
        "holding": bool,              # 是否持仓
        "holding_days": int,          # 持仓天数
        "unrealized_pnl": float,      # 未实现盈亏%
    },

    # === 第五类：市场环境 (约10维) ===
    "market": {
        "index_trend": float,         # 大盘趋势
        "sector_momentum": float,     # 板块动量
        "market_volatility": float,   # 市场波动率
        "northbound_flow": float,     # 北向资金
    },
}

# 动作空间 Action (离散)
action = {
    0: "hold/wait",    # 空仓等待 或 持仓持有
    1: "buy",          # 买入（全仓或半仓）
    2: "sell",         # 卖出（全部）
    # 扩展: 可细化为 buy_25%, buy_50%, sell_50%, sell_100%
}

# 奖励 Reward (由标注算法产生，IRL从中学习)
reward = realized_pnl_at_sell  # 卖出时的实际盈亏
```

---

## 6. 数据工程方案

### 6.1 买卖点标注算法（核心）

这是整个系统最关键的模块。标注质量直接决定学习效果。

#### 策略一：缠论买卖点标注

```
输入: 日线K线数据(3年)
步骤:
  1. 包含处理 → 分型识别 → 笔构造 → 线段 → 中枢
  2. 识别一类买点(中枢下方背离)、二类买点(中枢回拉不破)、三类买点(中枢突破回踩)
  3. 对应卖点同理
  4. 验证: 买点后是否上涨20%+，卖点后是否下跌10%+
  5. 剔除无效信号(假突破、快速反转)
输出: [(date, "buy", price, confidence), ...]
```

#### 策略二：波浪理论标注

```
输入: 日线K线数据
步骤:
  1. 识别推动浪(1-2-3-4-5)和调整浪(A-B-C)
  2. 第2浪末尾 → 买点(确认趋势开始)
  3. 第3浪中段 → 加仓点
  4. 第5浪末尾 → 卖点(趋势终结)
  5. 用Fibonacci回撤/扩展验证浪的比例关系
输出: [(date, "buy"/"sell", price, wave_context), ...]
```

#### 策略三：回溯最优标注（推荐作为主方案）

```
输入: 日线K线数据 + 约束条件
约束:
  - 最少持仓天数: 5天（避免日内交易）
  - 最多持仓天数: 60天（波段操作）
  - 最少间隔: 买卖之间至少隔3天
  - 目标: 最大化年化收益率，同时满足最大回撤<20%
算法:
  1. 动态规划 / 遗传算法 搜索最优买卖序列
  2. 约束条件确保是"合理的人类操作"而非"未来函数"
  3. 加入交易成本(印花税0.05% + 佣金0.025%)
  4. 验证收益率 ≥ 50%年化
输出: [(date, action, price, realized_pnl), ...]
```

#### 策略四：融合标注（最终方案）

```
综合三种策略，投票或加权:
  - 缠论买卖点 ∩ 回溯最优点 → 高置信度标注
  - 波浪理论确认 → 增强置信度
  - 单一策略标注 → 低置信度标注
  - 三种策略冲突 → 不标注（观望）
```

### 6.2 特征工程

#### 特征来源对照表

| 来源 | 特征数 | 获取方式 | 成本 |
|------|--------|---------|------|
| 现有29 Agent评分 | 29 | 需回跑历史数据（LLM API调用） | 高（每个时间步×29Agent×N Provider） |
| 技术指标 | ~30 | data_backend.py直接计算 | 低（纯计算） |
| 缠论/形态/背离信号 | ~20 | data_backend.py已有算法 | 低（纯计算） |
| 持仓状态 | 3 | 由标注算法产生 | 无 |
| 市场环境 | ~10 | 大盘指数+板块数据 | 低 |
| K线图LLM视觉评估 | ~5 | Vision Agent调用 | 很高 |

**成本优化策略**：

1. **Phase 1 不使用Agent评分和LLM视觉**：仅用技术指标+结构化信号（~60维），验证pipeline可行性
2. **Phase 2 引入Agent评分的简化版**：用`_simple_policy()`公式（已在agents.py实现）替代LLM调用，零成本获得29维评分近似值
3. **Phase 3 选择性回跑LLM评分**：仅对关键买卖点附近的时间窗口(±5天)调用LLM，减少90%调用量

### 6.3 数据规模估算

| 参数 | 估算值 |
|------|--------|
| 股票数量 | 100只 |
| 历史周期 | 3年（约750个交易日） |
| 每只股票买卖次数 | 20-50次（波段操作） |
| 总交易记录 | 2,000-5,000条 |
| 总时间步（日频） | 100 × 750 = 75,000 |
| 特征维度（Phase 1） | ~60维 |
| 特征维度（最终） | ~100维 |
| 状态-动作对总量 | 75,000条（含大量hold） |

---

## 7. 模型训练方案

### 7.1 Phase 1: Behavioral Cloning

```python
# 训练数据构造
# X: state vector (60维), Y: action label (0=hold, 1=buy, 2=sell)
# 注意: hold样本远多于buy/sell，需要处理类别不平衡

# 方案: 使用 imitation 库的 BC 实现
from imitation.algorithms import bc
from stable_baselines3 import PPO

# 或简化为标准分类器
from sklearn.ensemble import GradientBoostingClassifier
# SMOTE/下采样处理类别不平衡
# 交叉验证: 按时间前后分割（非随机分割，防止未来泄漏）
```

**评估指标**:
- 买卖点识别的Precision/Recall/F1
- 模拟交易的年化收益率、夏普比率、最大回撤
- 与"完美标注"的一致率

### 7.2 Phase 2: MaxEnt IRL

```python
# 使用 imitation 库的 AIRL 或自实现 MaxEnt IRL
from imitation.algorithms.adversarial import airl
from imitation.data import types

# 关键: 从专家轨迹反推奖励函数 R(s, a)
# 学到的R可以解释"在什么状态下买入/卖出是好的"
# 再用R训练PPO获得策略π

# 奖励函数可视化 → 可解释性分析
# 哪些特征维度对买入/卖出奖励贡献最大
```

### 7.3 Phase 3: Decision Transformer

```python
# 使用 d3rlpy 的 Decision Transformer
import d3rlpy

# 轨迹数据格式:
# (return-to-go, state, action) 序列
# return-to-go = 从当前步到轨迹结束的累积收益

# 推理时: 设定目标收益 → 模型生成相应动作序列
# 例如: return-to-go=0.5(50%收益) → 模型输出高收益策略的动作

# 与LLM的潜在融合:
# 将Decision Transformer的embedding与LLM的文本embedding对齐
# 实现"自然语言描述市场状态 → 交易动作"的端到端推理
```

### 7.4 训练策略

```
数据分割（严格按时间）:
├── 训练集: 前2年数据 (2023.01 - 2024.12)
├── 验证集: 第3年前半 (2025.01 - 2025.06)
└── 测试集: 第3年后半 (2025.07 - 2025.12)

绝对禁止: 随机分割（时间序列数据必须按时间顺序分割）

交叉验证: 滚动窗口(Walk-forward)
├── Fold 1: Train 2023.01-2024.06 → Test 2024.07-2024.12
├── Fold 2: Train 2023.01-2025.06 → Test 2025.01-2025.06
└── Fold 3: Train 2023.01-2025.06 → Test 2025.07-2025.12
```

---

## 8. 推理与实盘应用方案

### 8.1 信号生成流程

```
每日收盘后 (15:15)
    │
    ├─ 1. 采集100只股票最新K线数据
    ├─ 2. 计算技术指标 + 缠论/形态信号
    ├─ 3. (可选) 运行简化Agent评分
    ├─ 4. 构造state vector
    ├─ 5. 模型推理 → 输出 (buy_prob, sell_prob, hold_prob)
    ├─ 6. 过滤: buy_prob > 0.7 → 买入信号
    │              sell_prob > 0.7 → 卖出信号
    │              else → 观望
    └─ 7. 生成信号报告 (融入现有PDF或独立通知)
```

### 8.2 与现有系统集成

```
                  stockagent-analysis (现有)
                  ┌───────────────────────┐
                  │  29 Agent 评分引擎     │
                  │  → 综合评分 + 决策建议  │
                  └───────┬───────────────┘
                          │
                          │ 评分数据 + 决策
                          ▼
                  ┌───────────────────────┐
                  │  IRL策略模型 (新)       │
                  │  → 买卖时机精准预判     │
                  │  → 仓位建议            │
                  │  → 置信度              │
                  └───────┬───────────────┘
                          │
                          ▼
                  ┌───────────────────────┐
                  │  融合输出              │
                  │  • 现有评分 (基本面)    │
                  │  • IRL信号 (时机判断)   │
                  │  • 综合决策            │
                  └───────────────────────┘
```

### 8.3 信号质量监控

| 指标 | 预警阈值 | 动作 |
|------|---------|------|
| 信号胜率(rolling 30天) | < 50% | 暂停信号，重新训练 |
| 平均持仓收益 | < 0% | 检查特征漂移 |
| 信号频率 | > 5次/天(100只) | 检查是否过度交易 |
| 特征分布漂移 | KL散度 > 阈值 | 触发增量训练 |

---

## 9. 风险与挑战

### 9.1 核心风险

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| **过拟合历史数据** | 🔴 高 | 严格时间分割；Walk-forward验证；正则化 |
| **标注质量决定上限** | 🔴 高 | 多策略融合标注；人工抽检；约束条件防止不合理操作 |
| **市场制度变化(regime change)** | 🔴 高 | 滚动重训练；加入市场环境特征；监控特征漂移 |
| **Agent评分回跑成本高** | 🟡 中 | 用_simple_policy()公式近似；仅关键窗口调用LLM |
| **数据量可能不足(100只×3年)** | 🟡 中 | BC对数据量要求低；数据增强(噪声/时间偏移) |
| **类别不平衡(hold>>buy/sell)** | 🟡 中 | SMOTE/下采样；Focal Loss；分阶段建模(先识别"交易点"再判断方向) |
| **缠论标注标准化困难** | 🟡 中 | 以回溯最优标注为主，缠论为辅助验证 |

### 9.2 特有挑战

1. **"完美买卖点"的定义问题**：回溯标注本质上用了未来信息，模型需要从"可观测特征"中学习到接近未来信息的判断能力。这要求特征工程足够丰富，能捕捉到价格转折的先兆信号。

2. **交易成本与滑点**：标注算法中的完美买卖点假设以收盘价成交，实际滑点可能1-3%。需在标注和评估中加入合理的成本假设。

3. **A股特殊性**：T+1交易制度、10%涨跌停板、融资融券限制等，需要在环境和约束中体现。

4. **100只股票的同质性/异质性**：如果100只股票行业分布不均，模型可能学到行业特定而非通用的交易模式。建议覆盖至少10个行业。

---

## 10. 里程碑与路线图

### M0: 准备阶段（1-2周）

- [ ] 确定100只股票池（覆盖10+行业，流动性充足，3年数据完整）
- [ ] 确认数据源可用性（通达信/Tushare/AKShare覆盖率）
- [ ] 搭建新项目骨架（复用core/框架）
- [ ] 安装依赖：imitation, d3rlpy, stable-baselines3, gymnasium

### M1: 买卖点标注引擎（2-3周）

- [ ] 实现回溯最优标注算法（动态规划/遗传算法）
- [ ] 实现缠论买卖点标注（复用`_detect_chanlun_signals()`）
- [ ] 实现融合标注逻辑
- [ ] 标注质量验证：收益率、交易频率、回撤统计
- [ ] 人工抽检10只股票的标注结果

### M2: 特征工程与数据Pipeline（2周）

- [ ] 构建特征提取器（技术指标+结构化信号+持仓状态）
- [ ] 实现`_simple_policy()`批量评分（零LLM成本）
- [ ] 构建Gymnasium交易环境
- [ ] 数据格式转换（适配imitation/d3rlpy输入格式）
- [ ] 数据质量检查与EDA

### M3: Phase 1 — Behavioral Cloning（1-2周）

- [ ] BC baseline训练与评估
- [ ] 超参搜索（特征选择、采样策略、模型复杂度）
- [ ] Walk-forward回测验证
- [ ] 确认pipeline端到端跑通

### M4: Phase 2 — MaxEnt IRL（2-3周）

- [ ] IRL奖励函数学习
- [ ] 奖励函数可视化与分析
- [ ] PPO策略训练
- [ ] 对比BC baseline性能

### M5: Phase 3 — Decision Transformer（2-3周）

- [ ] 轨迹数据格式化(return-to-go序列)
- [ ] Decision Transformer训练
- [ ] 条件生成测试（不同目标收益率下的策略差异）
- [ ] 三种方法横向对比

### M6: 系统集成与验证（2周）

- [ ] 与stockagent-analysis集成
- [ ] 实时信号生成pipeline
- [ ] 信号质量监控仪表板
- [ ] 全量100只股票回测

### 总计预估：12-16周

---

## 11. 附录：学术论文与开源项目

### 11.1 核心参考论文

| # | 论文 | 年份 | 方法 | 发表venue |
|---|------|------|------|-----------|
| 1 | TAIRL: Transaction-aware IRL for Trading | 2023 | MaxEnt IRL | Applied Intelligence |
| 2 | iRDPG: Adaptive Quantitative Trading (AAAI) | 2020 | Imitative RDPG | AAAI 2020 |
| 3 | Pro Trader RL: Mimicking Professional Traders | 2024 | 分解式RL | Expert Systems with Applications |
| 4 | Human-aligned Trading via Imitative Multi-Loss DQN | 2023 | Imitative DQN | Expert Systems with Applications |
| 5 | PPGAIN: Privacy-Preserving GAIL for Trading | 2022 | GAIL | Neurocomputing |
| 6 | Reward Shaping via Expert Feedback for Stock Trading | 2025 | Reward Shaping | KAIS (Springer) |
| 7 | FlowHFT: Flow Matching for HFT | 2025 | Flow Matching IL | arXiv |
| 8 | ElliottAgents: LLM + Elliott Wave + DRL | 2024 | Multi-Agent LLM | Applied Sciences |
| 9 | LSIL: Latent Segmentation IL for Trading | 2020 | Segmented IL | JRFM (MDPI) |
| 10 | CFA Institute: RL/IRL Practitioner's Guide | 2025 | 综述 | CFA Research Foundation |

### 11.2 核心开源项目

| # | 项目 | Stars | 用途 |
|---|------|-------|------|
| 1 | **HumanCompatibleAI/imitation** | ~1.6k | BC/DAgger/GAIL/AIRL算法框架 |
| 2 | **takuseno/d3rlpy** | ~1.6k | 离线RL + Decision Transformer |
| 3 | **AI4Finance/FinRL** | ~6.6k | 金融交易环境 + RL pipeline |
| 4 | **AI4Finance/FinRL_Imitation_Learning** | — | BC→RL微调的金融应用示例 |
| 5 | **JurajZelman/airl-market-making** | — | AIRL做市策略(ICAIF 2024) |
| 6 | **chandc/Inverse_RL_for_Stocks** | — | MaxEnt IRL股票交易复现 |
| 7 | **Stable-Baselines3** | ~10k | RL算法底层框架 |

### 11.3 推荐技术栈

```
核心框架:
  ├── imitation (BC, AIRL)          → Phase 1-2
  ├── d3rlpy (Decision Transformer) → Phase 3
  ├── stable-baselines3 (PPO, SAC)  → RL策略训练
  └── gymnasium                     → 交易环境定义

数据与特征:
  ├── stockagent-analysis/core      → 复用LLM路由和并行框架
  ├── stockagent-analysis/data_backend → 复用数据采集和指标计算
  ├── pandas / numpy                → 数据处理
  └── scikit-learn                  → 特征工程、BC baseline

评估与可视化:
  ├── backtrader 或 自研回测引擎     → 策略回测
  ├── matplotlib / plotly           → 可视化
  └── mlflow / wandb                → 实验追踪
```

---

## 12. 结论与建议

### 可行性判定：**可行，但需渐进式实施**

| 维度 | 判定 | 说明 |
|------|------|------|
| 学术基础 | ✅ 有支撑 | 5+篇直接相关论文，IRL/IL在交易中已有验证 |
| 技术栈 | ✅ 成熟 | imitation/d3rlpy/SB3生态完善 |
| 数据可行性 | ✅ 充足 | 现有数据pipeline完全可复用 |
| 与现有项目集成 | ✅ 自然 | 29 Agent评分作为特征输入，架构契合 |
| 买卖点标注 | ⚠️ 是核心难点 | 建议以回溯最优标注为主，缠论/波浪为辅 |
| 数据量 | ⚠️ 需注意 | BC和MaxEnt IRL可应对，GAIL/AIRL可能不足 |
| 过拟合风险 | ⚠️ 需严格控制 | 强制时间分割+Walk-forward+正则化 |

### 首要行动项

1. **立即可做**：确定100只股票池 + 实现回溯最优标注算法
2. **1周内**：搭建特征工程pipeline（纯技术指标，零LLM成本）
3. **2周内**：BC baseline训练 + 回测验证
4. **后续根据BC结果决定**：是否推进IRL/Decision Transformer

> 核心原则：**先用最简单的方法验证数据和特征是否有效，再逐步引入复杂算法**。BC如果完全无效（F1<0.3），说明特征工程或标注质量有根本问题，此时引入IRL也无法解决。

---

## 13. 计算成本与硬件方案

> 基于实际硬件检测结果编写（2026-03-15）

### 13.1 当前硬件配置

| 组件 | 规格 | 对训练的影响 |
|------|------|------------|
| **CPU** | i5-13500H (12核/16线程) | 足够：数据处理、特征计算、BC/sklearn训练 |
| **内存** | 16GB DDR | 紧张：100只×750天×100维 ≈ 57MB（数据本身无压力），但训练框架+数据加载需 8-10GB |
| **GPU** | Intel Iris Xe（核显，无独显） | **瓶颈**：无法本地跑 PyTorch 神经网络训练 |
| **磁盘** | D盘 150GB 可用 | 充足：全量数据+模型+日志 < 5GB |

### 13.2 硬件限制对各Phase的影响

```
Phase 1 (BC/sklearn)    → ✅ 本地完全可行，CPU训练秒级完成
Phase 2 (MaxEnt IRL)    → ⚠️ 简单版本可本地跑(numpy)，完整AIRL需GPU
Phase 3 (Decision Trans)→ ❌ 必须使用云GPU
```

### 13.3 各阶段计算成本明细

#### 阶段零：数据采集与标注

| 任务 | 计算方式 | 耗时估算 | 成本 |
|------|---------|---------|------|
| K线数据下载(100只×3年) | 通达信本地读取/Tushare API | 10-30分钟 | **¥0**（本地/免费API） |
| 技术指标计算 | CPU纯计算(pandas) | 1-2分钟 | **¥0** |
| 缠论/形态/背离检测 | CPU纯计算(复用data_backend) | 3-5分钟 | **¥0** |
| 回溯最优标注(DP/GA) | CPU计算(100只×搜索) | 10-30分钟 | **¥0** |
| **小计** | | **< 1小时** | **¥0** |

#### 阶段一（关键成本项）：Agent评分的历史回跑

这是**成本最高的环节**，有三种方案：

##### 方案A：零成本 — 纯`_simple_policy()`公式评分

```
29个Agent × _simple_policy() 公式 → 直接从技术指标计算评分
无需LLM调用，纯CPU运算

数据量: 100只 × 750天 × 29 Agent = 2,175,000 次公式计算
耗时: < 5分钟 (CPU)
成本: ¥0
精度: 与LLM评分相关系数约 0.6-0.7（可能的估计值）
```

**推荐作为Phase 1首选。** 如果BC用公式评分就能达到可接受效果，则无需花费LLM成本。

##### 方案B：低成本 — 采样回跑LLM评分

```
策略: 仅对标注买卖点附近 ±5天 的窗口回跑LLM评分
买卖点数量: 100只 × 35次/只 = 3,500 个买卖点
窗口: 3,500 × 11天 = 38,500 个时间步
每个时间步: 29 Agent × 1次LLM调用(合并后) = 29 次调用
总LLM调用次数: 38,500 × 29 = 1,116,500 次

但每次只需1个Provider（非多Provider并行）：
选 deepseek（最便宜）: ¥1/M input + ¥2/M output
每次调用约 2K input + 0.5K output tokens
input成本: 1,116,500 × 2K × ¥1/M = ¥2.23
output成本: 1,116,500 × 0.5K × ¥2/M = ¥1.12
总计: ≈ ¥3.4 (deepseek极便宜)

如果用 grok-4-1-fast:
input: $3/M input, $15/M output (海外价格)
input成本: 1,116,500 × 2K × $3/M = $6.70
output成本: 1,116,500 × 0.5K × $15/M = $8.37
总计: ≈ $15 ≈ ¥110

耗时: 6路并发 × 120s超时 → 约 3-6小时
```

##### 方案C：全量回跑LLM评分（不推荐，仅供参考）

```
全部时间步: 100只 × 750天 = 75,000 个时间步
LLM调用: 75,000 × 29 = 2,175,000 次

deepseek: ≈ ¥6.5
grok: ≈ $290 ≈ ¥2,100
gemini: ≈ $130 ≈ ¥940

耗时: 按6路并发 → 约 50-100小时（2-4天不间断运行）
```

##### Agent评分回跑方案对比

| 方案 | LLM成本 | 耗时 | 数据覆盖 | 推荐度 |
|------|---------|------|---------|--------|
| A: 纯公式 | ¥0 | 5分钟 | 全量(近似) | ⭐⭐⭐⭐⭐ Phase 1必选 |
| B: 采样回跑(deepseek) | ¥3-5 | 3-6小时 | 关键窗口(精确) | ⭐⭐⭐⭐ Phase 2推荐 |
| B: 采样回跑(grok) | ¥110 | 3-6小时 | 关键窗口(精确) | ⭐⭐⭐ 可选 |
| C: 全量回跑 | ¥2,100+ | 2-4天 | 全量(精确) | ⭐ 不推荐 |

#### 阶段二：模型训练

| Phase | 算法 | 硬件需求 | 本地可行? | 云GPU方案 | 训练耗时 | 云成本 |
|-------|------|---------|----------|----------|---------|--------|
| **Phase 1** | BC (GBT/RF/MLP-sklearn) | CPU 16GB | ✅ 完全可行 | 不需要 | 1-10分钟 | ¥0 |
| **Phase 1+** | BC (小型MLP-PyTorch) | CPU可跑 | ✅ 慢但可行 | 可选 | 10-30分钟 | ¥0 |
| **Phase 2a** | MaxEnt IRL (numpy实现) | CPU 16GB | ✅ 可行 | 不需要 | 30-60分钟 | ¥0 |
| **Phase 2b** | AIRL (PyTorch+SB3) | GPU 4GB+ | ❌ 核显不够 | 需要 | 2-6小时 | ¥5-15 |
| **Phase 3** | Decision Transformer | GPU 8GB+ | ❌ 核显不够 | 需要 | 4-12小时 | ¥10-30 |

#### 阶段三：推理（日常运行）

```
每日收盘后推理100只股票:
  1. 数据采集 + 指标计算: CPU, 5-10分钟, ¥0
  2. 公式评分(可选): CPU, < 1分钟, ¥0
  3. 模型推理:
     - BC/sklearn: CPU, < 1秒, ¥0
     - Decision Transformer: CPU可推理(不需GPU), < 5秒, ¥0
  4. 信号过滤 + 报告生成: CPU, < 1分钟, ¥0

日常运行总成本: ¥0（纯本地CPU）
```

### 13.4 云GPU方案选型

由于本地无独显，Phase 2b/3 需要云GPU。以下为国内主流方案对比：

| 平台 | GPU | 显存 | 价格 | 适合阶段 | 备注 |
|------|-----|------|------|---------|------|
| **AutoDL** | RTX 4090 | 24GB | ¥2.58/时 | Phase 2b, 3 | 国内最便宜，预装PyTorch |
| **AutoDL** | RTX 3090 | 24GB | ¥1.68/时 | Phase 2b, 3 | 性价比最高 |
| **Google Colab Pro** | T4/V100 | 16GB | ¥70/月 | Phase 2b, 3 | Jupyter直接用，但网络受限 |
| **阿里云 PAI-DSW** | V100 | 16GB | ¥6.72/时 | Phase 3 | 大厂稳定 |
| **Kaggle** | P100/T4 | 16GB | 免费(30h/周) | Phase 1-2 | 免费额度足够实验 |

**推荐方案**：

```
实验阶段 → Kaggle免费T4（30小时/周，足够Phase 2b实验）
正式训练 → AutoDL RTX 3090（¥1.68/时）
  Phase 2b AIRL: 6小时 × ¥1.68 = ¥10
  Phase 3 DT:  12小时 × ¥1.68 = ¥20
  超参搜索:    24小时 × ¥1.68 = ¥40
  总计: ≈ ¥70
```

### 13.5 全项目成本汇总

#### 最低成本方案（仅Phase 1，验证可行性）

| 项目 | 成本 |
|------|------|
| 数据采集 | ¥0 |
| 特征计算（公式评分） | ¥0 |
| BC训练（本地CPU） | ¥0 |
| 回测验证 | ¥0 |
| **合计** | **¥0** |

#### 推荐成本方案（Phase 1-3完整流程）

| 项目 | 成本 |
|------|------|
| 数据采集与标注 | ¥0 |
| Phase 1 特征：公式评分 | ¥0 |
| Phase 2 特征：deepseek采样回跑 | ¥5 |
| Phase 1 训练：BC本地CPU | ¥0 |
| Phase 2 训练：AIRL云GPU(AutoDL) | ¥10 |
| Phase 3 训练：DT云GPU(AutoDL) | ¥20 |
| 超参搜索与实验 | ¥40 |
| 日常推理运行 | ¥0/天 |
| **合计** | **≈ ¥75** |

#### 上限成本方案（含grok多Provider回跑）

| 项目 | 成本 |
|------|------|
| 数据采集与标注 | ¥0 |
| LLM评分采样回跑(grok+deepseek) | ¥120 |
| 云GPU训练(含充分实验) | ¥150 |
| **合计** | **≈ ¥270** |

### 13.6 时间×硬件甘特图

```
周次   1    2    3    4    5    6    7    8    9   10   11   12
      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
M0    [准备]                                            硬件: 本地CPU
M1    ·····[标注引擎·····]                               硬件: 本地CPU
M2              ·····[特征工程···]                        硬件: 本地CPU + deepseek API
M3                        [BC训练·]                      硬件: 本地CPU ← 决策点①
M4                              ··[MaxEnt IRL]···        硬件: 本地CPU + AutoDL
M5                                      ····[Dec.Trans]  硬件: AutoDL RTX 3090
M6                                              ··[集成]  硬件: 本地CPU

决策点①: BC结果 F1>0.4 → 继续Phase 2-3
                F1<0.3 → 回到M1/M2改进标注和特征
```

### 13.7 内存使用估算

```
数据集内存占用:
  100只 × 750天 × 100维 × float32 = 28.6 MB（状态矩阵）
  标注数据 + 元信息 ≈ 5 MB
  小计: ≈ 35 MB ← 数据本身完全不是问题

训练时内存占用:
  sklearn (BC):     数据 35MB + 模型 < 50MB + 开销 ≈ 500MB    → OK
  imitation (AIRL): 数据 + SB3环境 + PyTorch ≈ 3-4GB           → OK（16GB够）
  d3rlpy (DT):      数据 + Transformer + Replay Buffer ≈ 6-8GB → 紧张但可行

建议: 训练时关闭浏览器/其他大程序，确保 10GB+ 可用内存
      如果d3rlpy内存不足 → 在云GPU实例上训练（AutoDL默认配32GB+内存）
```

### 13.8 数据存储估算

```
D盘空间: 150GB可用 → 完全充足

存储明细:
  K线原始数据: 100只 × 3年 × 4周期 ≈ 200MB
  技术指标: ≈ 100MB
  Agent评分缓存: ≈ 50MB
  标注数据: ≈ 10MB
  K线图(如需Vision): 100只 × 4周期 × N快照 → 可能 2-5GB
  模型checkpoint: ≈ 100-500MB
  实验日志(mlflow/wandb): ≈ 200MB
  总计: ≈ 3-6GB ← 远低于可用空间
```

### 13.9 建议的硬件升级路径（可选）

如果项目进展顺利、需要更频繁的实验迭代：

| 优先级 | 升级项 | 成本 | 收益 |
|--------|--------|------|------|
| 不需要 | 当前配置足够Phase 1 | ¥0 | BC/sklearn/numpy全部可跑 |
| 可选 | 内存 16→32GB | ¥200-300 | PyTorch本地训练更宽裕 |
| 可选 | 外接eGPU(RTX 4060) | ¥2,500+ | 本地也能跑Phase 2-3，省云费用 |
| 不推荐 | 换电脑 | ¥8,000+ | ROI太低，AutoDL更划算 |

> **结论**：当前硬件配置 + AutoDL云GPU **完全满足**全部三个Phase的需求。Phase 1零成本验证，Phase 2-3累计云GPU费用约¥70。整个项目从零到出结果的总成本控制在 **¥100以内** 完全可行。

---

## 14. 交易截面特征空间全景（通达信 / Tushare / AKShare）

> 系统梳理三大数据源在每个交易截面（买卖点时刻）可提取的全部特征维度。
> 分为"已实现"（现有 data_backend.py 已采集计算）和"待新增"两部分。

### 14.1 特征空间总览

```
特征总维度: ≈ 250-300 维（去重后）
├── 已实现（现有data_backend.py）: ≈ 150 维
└── 待新增（三大数据源可扩展）: ≈ 100-150 维

按来源拆分:
├── 通达信本地:   K线OHLCV（基础原料，非直接特征）
├── Tushare Pro:  行情 + 基本面 + 资金流 + 融资融券 + 股东 + 事件
├── AKShare:      实时盘口 + 板块 + 情绪 + 宏观 + 独有数据
└── 本地计算:     技术指标 + 缠论 + 形态 + 统计量（基于K线推导）
```

### 14.2 第一大类：价格与成交量（基础层）

#### 已实现 ✅

| # | 特征 | 维度 | 来源 | 周期 | data_backend位置 |
|---|------|------|------|------|-----------------|
| 1 | OHLCV (开高低收量额) | 6 | TDX→Tushare→AKShare | 日/周/月/1h | `_fetch_kline_*()` |
| 2 | 涨跌幅 pct_chg | 1 | 快照/K线 | 日 | `snapshot.pct_chg` |
| 3 | 量比 volume_ratio_5_20 | 1 | 本地计算 | 日 | `features.volume_ratio_5_20` |
| 4 | 20日动量 momentum_20 | 1 | 本地计算 | 日 | `features.momentum_20` |
| 5 | 20日波动率 volatility_20 | 1 | 本地计算 | 日 | `features.volatility_20` |
| 6 | 60日最大回撤 drawdown_60 | 1 | 本地计算 | 日 | `features.drawdown_60` |
| 7 | 趋势强度 trend_strength | 1 | 本地计算 | 日 | `features.trend_strength` |
| 8 | 10根动量 momentum_10 | 4 | 本地计算 | 日/周/月/1h | `kline_indicators.{tf}.momentum_10` |
| 9 | 20根振幅 amplitude_20 | 4 | 本地计算 | 日/周/月/1h | `kline_indicators.{tf}.amplitude_20` |
| 10 | 上/下影线占比 | 8 | 本地计算 | 日/周/月/1h | `kline_indicators.{tf}.upper/lower_shadow_ratio` |

**小计: ~28维**

#### 待新增 🆕

| # | 特征 | 维度 | 来源 | 说明 |
|---|------|------|------|------|
| 11 | 日内量比（盘口） | 1 | AKShare `stock_zh_a_spot_em` | `volume_ratio` 字段，比自算更精确 |
| 12 | 振幅（日内） | 1 | AKShare `stock_zh_a_spot_em` | 当日振幅 |
| 13 | 复权因子变动 | 1 | Tushare `adj_factor` | 除权除息事件标记 |
| 14 | N日涨跌幅序列 (5/10/20/60) | 4 | 本地计算 | 多窗口收益率 |
| 15 | 成交额相对排名(行业内) | 1 | AKShare 板块成分 | 行业内活跃度 |

---

### 14.3 第二大类：技术指标（已实现为主）

#### 已实现 ✅ — 多周期 (日/周/月/1h × 每个指标)

| # | 指标 | 维度 | 说明 |
|---|------|------|------|
| 1 | RSI(14) | 4 | `kline_indicators.{tf}.rsi` |
| 2 | MACD (DIF/DEA/HIST) | 12 | `kline_indicators.{tf}.macd_dif/dea/hist` |
| 3 | KDJ (K/D/J) | 12 | `kline_indicators.{tf}.kdj_k/d/j` |
| 4 | 布林带 (上/中/下) | 12 | `kline_indicators.{tf}.boll_upper/mid/lower` |
| 5 | Stoch RSI | 4 | `kline_indicators.{tf}.stoch_rsi` |
| 6 | 趋势斜率 | 4 | `kline_indicators.{tf}.trend_slope_pct` |

**小计: ~48维**

#### 待新增 🆕 — Tushare `stk_factor_pro` 直接提供

| # | 指标 | 维度 | 来源 | 说明 |
|---|------|------|------|------|
| 7 | RSI(6) / RSI(12) / RSI(24) | 3 | Tushare `stk_factor_pro` | 多周期RSI，比仅RSI(14)更丰富 |
| 8 | CCI 顺势指标 | 1 | Tushare `stk_factor_pro` | CCI>100超买, <-100超卖 |
| 9 | ATR 平均真实波幅 | 1 | 本地计算 | 波动率度量，可用于动态止损 |
| 10 | WILLR 威廉指标 | 1 | 本地计算 | 超买超卖补充 |
| 11 | OBV 能量潮 | 1 | 本地计算(已部分实现) | `volume_price.obv_trend` 已有趋势，可补数值 |
| 12 | VWAP 成交量加权均价 | 1 | 本地计算 | 日内交易基准线 |

---

### 14.4 第三大类：均线系统

#### 已实现 ✅

| # | 特征 | 维度 | 说明 |
|---|------|------|------|
| 1 | MA5/10/20/60/120/250 价格 | 24 | `kline_indicators.{tf}.ma_system.ma{N}.value` (4周期) |
| 2 | MA5/10/20/60/120/250 乖离率 | 24 | `kline_indicators.{tf}.ma_system.ma{N}.pct_above` (4周期) |

**小计: ~48维**

#### 待新增 🆕

| # | 特征 | 维度 | 说明 |
|---|------|------|------|
| 3 | 均线多头/空头排列编码 | 4 | MA5>MA10>MA20>MA60 → 1, 反之 → -1, 混合 → 0 |
| 4 | 金叉/死叉信号 (MA5×MA20, MA10×MA60) | 4 | 布尔值，事件标记 |
| 5 | 均线粘合度 (std of MAs / close) | 4 | 粘合→即将变盘信号 |

---

### 14.5 第四大类：形态与结构化信号

#### 已实现 ✅

| # | 特征类别 | 维度 | 说明 |
|---|---------|------|------|
| 1 | K线形态 (15+种) | 4×3 =12 | 每周期: 最强看涨形态confidence + 最强看跌形态confidence + 形态数量 |
| 2 | K线组合 kline_combo_5 | 4 | 编码: 三连阳=2, 早晨之星=3, 三连阴=-2, 无=0 |
| 3 | 连续性统计 | 4×7 =28 | consecutive_bull/bear, body_trend, higher_highs/lower_lows, gap_up/down_count |
| 4 | 相邻K线关系 | 4×4 =16 | 2对相邻K线 × (gap_pct + engulf_degree) |
| 5 | 缠论信号 chanlun_score | 4 | -50~+50, 买卖点类型编码 |
| 6 | 缠论买点/卖点类型 | 4×2 =8 | 0/1/2/3类买卖点 |
| 7 | 背离 divergence_score | 4 | -100~+100 |
| 8 | MACD/RSI 顶底背离标记 | 4×4 =16 | macd_top/bot, rsi_top/bot 各周期 |
| 9 | 量价信号 volume_price_score | 4 | -50~+50 |
| 10 | 量价事件标记 | 4×4 =16 | volume_breakout, shrink_pullback, volume_anomaly, climax_volume |
| 11 | 支撑阻力 sr_score | 1 | -30~+30 (仅日线) |
| 12 | 支撑/阻力距离 | 2 | nearest_support/resistance 距离% |
| 13 | 价格位置 price_position | 1 | near_support=1, middle=0, near_resistance=-1 |
| 14 | 图表形态 chart_pattern_score | 4 | -40~+40 |
| 15 | 趋势线突破 | 4×2 =8 | up/down trendline breakout 标记 |

**小计: ~132维**

#### 待新增 🆕

| # | 特征 | 维度 | 说明 |
|---|------|------|------|
| 16 | 缠论中枢位置 (距当前价%) | 4 | 当前价在中枢上方/内部/下方的相对位置 |
| 17 | 缠论笔的长度趋势 | 4 | 最近3笔长度变化(延伸/收缩) |
| 18 | 波浪计数状态 | 1 | 推动浪1-5 / 调整浪A-C 编码 |
| 19 | 跳空缺口数量与最近缺口距离 | 2 | 已有gaps列表，提取统计量 |

---

### 14.6 第五大类：Fibonacci与关键价位

#### 已实现 ✅

| # | 特征 | 维度 | 说明 |
|---|------|------|------|
| 1 | Fibonacci 回撤位 (23.6/38.2/50/61.8%) | 4 | `features.key_levels.retrace_*` |
| 2 | 当前价在Fib区间的位置 | 1 | (current - band_low) / (band_high - band_low) |

**小计: ~5维**

---

### 14.7 第六大类：基本面因子

#### 已实现 ✅

| # | 特征 | 维度 | 来源 | 说明 |
|---|------|------|------|------|
| 1 | PE_TTM | 1 | Tushare/AKShare | `features.pe_ttm` |
| 2 | PB | 1 | Tushare/AKShare | `features.pb` |
| 3 | 换手率 | 1 | Tushare/AKShare | `features.turnover_rate` |
| 4 | 总市值 | 1 | Tushare/AKShare | `features.total_mv` |
| 5 | ROE | 1 | Tushare | `features.roe` |
| 6 | 毛利率 | 1 | Tushare | `features.grossprofit_margin` |
| 7 | 净利率 | 1 | Tushare | `features.netprofit_margin` |
| 8 | 资产负债率 | 1 | Tushare | `features.debt_to_assets` |
| 9 | 营收同比 | 1 | Tushare | `features.revenue_yoy` |
| 10 | 净利同比 | 1 | Tushare | `features.netprofit_yoy` |
| 11 | EPS | 1 | Tushare | `features.eps` |
| 12 | 每股现金流 | 1 | Tushare | `features.cfps` |

**小计: 12维**

#### 待新增 🆕 — 估值与财务深度因子

| # | 特征 | 维度 | 来源 | 买卖点价值 |
|---|------|------|------|-----------|
| 13 | PS (市销率) | 1 | Tushare `daily_basic` | 亏损股估值替代指标 |
| 14 | PS_TTM | 1 | Tushare `daily_basic` | 同上(滚动) |
| 15 | 股息率 dv_ratio | 1 | Tushare `daily_basic` | 高股息=防御底仓 |
| 16 | 自由流通换手率 turnover_rate_f | 1 | Tushare `daily_basic` | 比总换手率更真实 |
| 17 | 流通市值 circ_mv | 1 | Tushare `daily_basic` | 小盘股弹性更大 |
| 18 | PE/PB历史百分位(3年) | 2 | 本地计算(需`daily_basic`历史数据) | **极重要**: 低百分位=低估买入，高百分位=高估卖出 |
| 19 | ROE变化趋势(连续4季) | 1 | Tushare `fina_indicator` | ROE连续提升=基本面改善 |
| 20 | 毛利率变化趋势(连续4季) | 1 | Tushare `fina_indicator` | 毛利率趋势反映竞争壁垒 |
| 21 | 营收加速度(本季同比-上季同比) | 1 | Tushare `fina_indicator` | 加速增长=戴维斯双击信号 |
| 22 | 经营现金流/净利润 | 1 | Tushare `cashflow`+`income` | >1=高质量盈利 |
| 23 | 流动比率/速动比率 | 2 | Tushare `fina_indicator` | 偿债风险预警 |
| 24 | 应收账款周转率变化 | 1 | Tushare `fina_indicator` | 下降=回款恶化 |
| 25 | 存货周转率变化 | 1 | Tushare `fina_indicator` | 下降=库存积压 |
| 26 | PEG (PE/利润增速) | 1 | 本地计算 | PEG<1=成长性低估 |

---

### 14.8 第七大类：资金流向

#### 已实现 ✅

| # | 特征 | 维度 | 来源 | 说明 |
|---|------|------|------|------|
| 1 | 融资余额 rzye | 1 | Tushare `margin_detail` | `features.margin_data.rzye` |
| 2 | 融资买入额 rzmre | 1 | Tushare | `features.margin_data.rzmre` |
| 3 | 融券余额 rqye | 1 | Tushare | `features.margin_data.rqye` |
| 4 | 融券卖出额 rqmcl | 1 | Tushare | `features.margin_data.rqmcl` |
| 5 | 5日融资余额变化% | 1 | Tushare(本地计算) | `features.margin_data.rzye_change_5d` |
| 6 | 10日融资余额变化% | 1 | Tushare(本地计算) | `features.margin_data.rzye_change_10d` |
| 7 | 北向持股数 | 1 | Tushare `hk_hold` | `features.hsgt_data.hk_hold_vol` |
| 8 | 北向持股比例 | 1 | Tushare | `features.hsgt_data.hk_hold_ratio` |
| 9 | 北向持股变化% | 1 | Tushare | `features.hsgt_data.hk_hold_change_pct` |

**小计: 9维**

#### 待新增 🆕 — 主力资金流 + 北向大盘

| # | 特征 | 维度 | 来源 | 买卖点价值 |
|---|------|------|------|-----------|
| 10 | 主力(超大+大单)净流入额 | 1 | Tushare `moneyflow` 或 AKShare `stock_individual_fund_flow` | **核心**: 主力持续净流入=吸筹 |
| 11 | 超大单净流入/总成交额 | 1 | 同上 | 标准化的主力资金强度 |
| 12 | 大单净流入/总成交额 | 1 | 同上 | 大户资金方向 |
| 13 | 小单净流入/总成交额 | 1 | 同上 | 散户方向(反向指标) |
| 14 | 5日主力净流入累计 | 1 | 本地计算 | 短期主力行为趋势 |
| 15 | 20日主力净流入累计 | 1 | 本地计算 | 中期主力行为趋势 |
| 16 | 融资/融券比 | 1 | 本地计算 | 多空力量比值 |
| 17 | 北向资金全市场净流入(日) | 1 | Tushare `moneyflow_hsgt` | 外资整体情绪 |
| 18 | 板块资金净流入排名 | 1 | AKShare `stock_sector_fund_flow` | 板块热度 |

---

### 14.9 第八大类：筹码与股东

#### 已实现 ✅

| # | 特征 | 维度 | 说明 |
|---|------|------|------|
| 1 | 获利盘比例 | 1 | `features.chip_distribution.profit_ratio` |
| 2 | 被套盘比例 | 1 | `features.chip_distribution.trapped_ratio` |
| 3 | 筹码浓度(TOP3) | 1 | `features.chip_distribution.concentration` |
| 4 | 平均成本 | 1 | `features.chip_distribution.avg_cost` |
| 5 | 筹码健康度 | 1 | `features.chip_distribution.health_score` |
| 6 | 当前价vs成本 | 1 | `features.chip_distribution.current_vs_cost` |

**小计: 6维**

#### 待新增 🆕

| # | 特征 | 维度 | 来源 | 买卖点价值 |
|---|------|------|------|-----------|
| 7 | 股东户数 | 1 | AKShare `stock_zh_a_gdhs` | **重要**: 户数减少=筹码集中=主力吸筹 |
| 8 | 股东户数变化% | 1 | AKShare | 连续减少=强信号 |
| 9 | 人均持股金额 | 1 | AKShare `stock_zh_a_gdhs_detail` | 户均市值上升=集中度提高 |
| 10 | 机构持仓比例 | 1 | AKShare `stock_institute_hold` / Tushare `top10_holders` | 机构占比上升=价值认可 |
| 11 | 机构持仓变化(季度环比) | 1 | 同上 | 机构增减持方向 |

---

### 14.10 第九大类：相对强弱

#### 已实现 ✅

| # | 特征 | 维度 | 基准 |
|---|------|------|------|
| 1-7 | RS当前值 + 5/10/20日动量 + 超额收益5/10/20日 | 7 | vs 沪深300 |
| 8-11 | RS趋势 + 新高/新低 + 看涨/看跌背离 | 4 | vs 沪深300 |
| 12-22 | 同上全部指标 | 11 | vs 行业板块 |
| 23-33 | 同上全部指标 | 11 | vs 板块龙头TOP3 |
| 34-44 | 同上全部指标 | 11 | vs 行业ETF |

**小计: ~44维**

---

### 14.11 第十大类：市场环境与宏观

#### 已实现 ✅

| # | 特征 | 维度 | 说明 |
|---|------|------|------|
| 1 | 市场状态 regime | 1 | bull/bear/range 编码 |
| 2 | 沪深300收盘价 | 1 | `features.market_regime.index_close` |
| 3 | 沪深300 MA20/MA60 | 2 | `features.market_regime.index_ma20/ma60` |
| 4 | 沪深300 20日收益率 | 1 | `features.market_regime.index_ret_20d` |
| 5 | 沪深300 20日波动率 | 1 | `features.market_regime.index_vol_20d` |

**小计: 6维**

#### 待新增 🆕 — 市场情绪 + 宏观因子

| # | 特征 | 维度 | 来源 | 买卖点价值 |
|---|------|------|------|-----------|
| 6 | 涨停家数 | 1 | AKShare `stock_zh_a_limit_up_em` | **核心情绪指标**: <20家=冰点, >100家=狂热 |
| 7 | 跌停家数 | 1 | AKShare `stock_zh_a_limit_down_em` | 跌停激增=恐慌底部信号 |
| 8 | 涨停/跌停比 | 1 | 本地计算 | 多空力量直观比值 |
| 9 | 连板最高板数 | 1 | AKShare `stock_zh_a_limit_up_detail_em` | 情绪高度指标 |
| 10 | 炸板率 | 1 | AKShare `stock_zh_a_brokeout_pool_em` | 高炸板率=承接弱 |
| 11 | 个股东财人气排名 | 1 | AKShare `stock_zh_a_hot_rank_em` | 散户关注度(可做逆向) |
| 12 | 行业板块涨跌幅排名 | 1 | AKShare `stock_board_industry_name_em` | 板块轮动位置 |
| 13 | 所属概念板块数量 | 1 | AKShare `stock_board_concept_name_em` | 多概念加持=题材股属性 |
| 14 | 最强所属概念板块涨幅 | 1 | AKShare `stock_board_concept_hist_em` | 概念热度 |
| 15 | PMI (制造业) | 1 | AKShare `macro_china_pmi_yearly` | 经济周期定位 |
| 16 | 社融增量(月) | 1 | AKShare `macro_china_shrzgm` | **重要**: 社融超预期=信用扩张→利好 |
| 17 | M2同比增速 | 1 | AKShare `macro_china_m2_yearly` | 流动性环境 |
| 18 | LPR(1Y/5Y) | 2 | AKShare `macro_china_lpr` | 利率环境→估值影响 |
| 19 | CPI同比 | 1 | AKShare `macro_china_cpi_monthly` | 通胀预期 |
| 20 | 上证指数RSI | 1 | 本地计算(需指数数据) | 大盘超买超卖 |

---

### 14.12 第十一大类：事件与催化剂

#### 已实现 ✅

| # | 特征 | 维度 | 说明 |
|---|------|------|------|
| 1 | 新闻情绪得分 | 1 | `features.news_sentiment` |
| 2 | 新闻数量 | 1 | `features.news_count` |

**小计: 2维**

#### 待新增 🆕

| # | 特征 | 维度 | 来源 | 买卖点价值 |
|---|------|------|------|-----------|
| 3 | 业绩预告方向 | 1 | Tushare `forecast` | 预增=1, 预减=-1, 扭亏=2, 首亏=-2 |
| 4 | 业绩预告利润变动幅度 | 1 | Tushare `forecast` | 变动幅度越大催化越强 |
| 5 | 龙虎榜净买入额 | 1 | Tushare `top_list` / AKShare `stock_lhb_em` | 机构净买入=短线强信号 |
| 6 | 龙虎榜机构席位数 | 1 | Tushare `top_inst` | 机构席位数量 |
| 7 | 大宗交易溢折价率 | 1 | Tushare `block_trade` / AKShare `stock_dzjy_mr` | 溢价=看好, 折价=出货 |
| 8 | 大宗交易金额/日均成交 | 1 | 本地计算 | 大宗占比过大=减持风险 |
| 9 | 限售解禁(未来30天) | 1 | Tushare `share_float` | 大比例解禁=抛压预警 |
| 10 | 限售解禁比例 | 1 | Tushare `share_float` | 解禁股/总股本 |
| 11 | 分红预案(每股分红) | 1 | Tushare `dividend` | 高分红=价值底 |
| 12 | 停复牌标记 | 1 | Tushare `suspend_d` | 复牌日特殊处理 |
| 13 | 一致预期EPS变化 | 1 | AKShare `stock_profit_forecast_em` | 预期上调=基本面改善 |
| 14 | 分析师评级变化 | 1 | AKShare | 首次覆盖/上调评级 |

---

### 14.13 第十二大类：持仓状态（由交易系统维护）

这类特征不来自外部数据源，而是由交易环境/标注系统动态维护。

| # | 特征 | 维度 | 说明 |
|---|------|------|------|
| 1 | 是否持仓 | 1 | 0=空仓, 1=持仓 |
| 2 | 持仓天数 | 1 | 空仓时=0 |
| 3 | 未实现盈亏% | 1 | 空仓时=0 |
| 4 | 距上次交易天数 | 1 | 避免过度交易 |
| 5 | 最近一次交易方向 | 1 | buy=1, sell=-1, none=0 |
| 6 | 累计已实现盈亏% | 1 | 本轮操作的累计收益 |

**小计: 6维**

---

### 14.14 第十三大类：现有29 Agent评分

每个Agent提供一个0-100的评分，在IRL模型中作为高级聚合特征。

| # | Agent | 维度 | 获取方式 |
|---|-------|------|---------|
| 1-29 | TREND, CAPITAL_FLOW, DERIV_MARGIN, CHANLUN, DIVERGENCE, VOLUME_PRICE, SUPPORT_RESISTANCE, CHART_PATTERN, TIMEFRAME_RESONANCE, TRENDLINE, KLINE_PATTERN, kline_1h, kline_day, kline_week, kline_month, 以及基本面/行业/政策等Agent | 29 | Phase 1: `_simple_policy()` 公式 (¥0) / Phase 2+: LLM调用 |

**小计: 29维**

---

### 14.15 特征空间汇总

| 大类 | 已实现 | 待新增 | 合计 | 数据来源 |
|------|--------|--------|------|---------|
| 价格与成交量 | 28 | 8 | 36 | TDX/Tushare/AKShare |
| 技术指标 | 48 | 6 | 54 | 本地计算 + Tushare |
| 均线系统 | 48 | 12 | 60 | 本地计算 |
| 形态与结构化信号 | 132 | 11 | 143 | 本地计算 |
| Fibonacci关键位 | 5 | 0 | 5 | 本地计算 |
| 基本面因子 | 12 | 14 | 26 | Tushare + 本地计算 |
| 资金流向 | 9 | 9 | 18 | Tushare + AKShare |
| 筹码与股东 | 6 | 5 | 11 | 本地计算 + AKShare |
| 相对强弱 | 44 | 0 | 44 | 本地计算 + AKShare |
| 市场环境与宏观 | 6 | 20 | 26 | AKShare |
| 事件与催化剂 | 2 | 14 | 16 | Tushare + AKShare |
| 持仓状态 | 0 | 6 | 6 | 交易系统维护 |
| Agent评分 | 29 | 0 | 29 | 现有系统 |
| **合计** | **~369** | **~105** | **~474** | |

> **注意**：形态与结构化信号的132维中有大量布尔型和多周期重复，实际有效信息量远低于维数。经PCA/特征选择后，预计有效特征维度在 **80-150维** 之间。

---

### 14.16 特征优先级分层（IRL训练推荐）

按对买卖点判断的信息量和可获取性排序：

```
P0 — 必须（Phase 1 即用，零成本）
├── 技术指标 (RSI/MACD/KDJ/BOLL) — 48维
├── 均线乖离率 — 24维
├── 缠论评分 + 背离评分 + 量价评分 — 12维
├── 形态信号 (最强看涨/看跌置信度) — 8维
├── 支撑/阻力距离 + 价格位置 — 4维
├── 持仓状态 — 6维
├── 市场状态 regime — 6维
└── 小计: ≈ 108维

P1 — 重要（Phase 1 可选，低成本扩展）
├── 29 Agent公式评分 — 29维 (¥0, _simple_policy)
├── PE/PB + 历史百分位 — 4维 (需拉daily_basic历史)
├── 主力资金净流入 — 6维 (Tushare/AKShare)
├── 筹码健康度 — 6维 (已实现)
├── 相对强弱 vs 大盘/行业 — 14维 (已实现)
├── 涨停家数/跌停家数/炸板率 — 5维 (AKShare, 免费)
└── 小计: ≈ 64维

P2 — 辅助（Phase 2+ 引入）
├── 股东户数变化 — 2维 (AKShare)
├── 业绩预告方向/幅度 — 2维 (Tushare)
├── 龙虎榜净买入 — 2维 (Tushare/AKShare)
├── 融资/融券详细指标 — 6维 (已部分实现)
├── 北向资金全市场 — 1维 (Tushare)
├── 宏观因子 (PMI/社融/M2/LPR) — 5维 (AKShare)
├── 分析师一致预期 — 2维 (AKShare)
├── 大宗交易信号 — 2维 (Tushare/AKShare)
├── 限售解禁预警 — 2维 (Tushare)
└── 小计: ≈ 24维

P3 — 锦上添花（特定场景才有价值）
├── 波浪计数状态 — 1维
├── 概念板块热度 — 2维
├── 人气排名(散户情绪) — 1维
├── 财务深度因子(周转率/现金流质量) — 6维
└── 小计: ≈ 10维
```

---

### 14.17 三大数据源差异化价值

| 数据源 | 独有优势 | 劣势 | IRL项目中的角色 |
|--------|---------|------|----------------|
| **通达信本地** | 零延迟、零成本、无限频；日线数据最完整 | 无基本面/资金流；1h分钟线可能缺失 | K线原始数据主力源 |
| **Tushare Pro** | 基本面因子最全(160+财务指标)；资金流/融资融券个股级；龙虎榜；技术因子`stk_factor_pro` | 需积分(2000-5000)；有频率限制 | 基本面 + 资金流主力源 |
| **AKShare** | 完全免费；情绪数据独有(涨跌停/炸板/人气)；概念板块独有；宏观数据丰富；股东户数免费 | 部分接口不稳定；数据更新偶有延迟 | 情绪 + 宏观 + 独有数据补充源 |

**最优组合策略**：
```
K线数据:    TDX本地 → Tushare → AKShare（与现有data_backend一致）
基本面:     Tushare daily_basic + fina_indicator（已实现大部分）
资金流:     Tushare moneyflow（个股级） + AKShare（板块级）
情绪:       AKShare 涨跌停/炸板/人气（Tushare无此数据）
宏观:       AKShare 社融/PMI/M2/LPR（免费且全）
股东筹码:   AKShare 股东户数（免费） + Tushare 十大股东（已有）
事件:       Tushare 业绩预告/解禁 + AKShare 龙虎榜/大宗
```

---

### 14.18 特征获取成本估算（100只×3年历史回填）

| 数据类别 | 接口 | 调用次数 | 成本 | 耗时 |
|----------|------|---------|------|------|
| K线 (日/周/月) | TDX本地 | 100×3 | ¥0 | 1分钟 |
| 技术指标 | 本地计算 | — | ¥0 | 2分钟 |
| daily_basic (3年历史) | Tushare | 100×1 | ¥0(2000积分) | 5分钟(限频) |
| fina_indicator (12季) | Tushare | 100×1 | ¥0(2000积分) | 3分钟 |
| moneyflow (750天) | Tushare | 100×8 | ¥0(2000积分) | 30分钟(限频) |
| margin_detail (750天) | Tushare | 100×4 | ¥0(2000积分) | 15分钟 |
| 股东户数 | AKShare | 100×1 | ¥0 | 5分钟 |
| 涨跌停统计 (750天) | AKShare | 750×1 | ¥0 | 15分钟 |
| 宏观数据 (PMI/社融/M2) | AKShare | 5×1 | ¥0 | 1分钟 |
| 业绩预告 | Tushare | 100×1 | ¥0 | 3分钟 |
| **合计** | | | **¥0** | **≈ 1-1.5小时** |

> **结论**：特征回填的全部数据获取成本为 **¥0**，仅需 Tushare 2000积分（现有项目已具备）。主要时间消耗在 Tushare 限频等待上，总耗时约1-1.5小时。

---

## 15. Gemini Embedding 多模态方案：K线图 + 特征序列统一嵌入

> 评估使用 `gemini-embedding-2-preview` 将每个买卖截面的多模态数据（K线图 + 技术指标图 + 高维特征）映射为统一向量，作为IRL/ML的输入特征。

### 15.1 gemini-embedding-2-preview 模型能力

| 规格 | 值 |
|------|-----|
| **多模态支持** | 文本 + 图片(≤6张) + 视频(≤128s) + 音频(≤80s) + PDF(≤6页) |
| **统一向量空间** | 所有模态映射到同一embedding空间 |
| **输出维度** | 默认3072维；支持Matryoshka降维至128/256/512/768/1536 |
| **最大输入** | 8192 tokens（文本+图片token合计） |
| **图片输入** | 每次请求最多6张，PNG/JPEG，通过base64或URI传入 |
| **task_type** | 支持8种：CLASSIFICATION, CLUSTERING, RETRIEVAL_*, SEMANTIC_SIMILARITY等 |
| **价格** | $0.20/百万tokens；批量API $0.10/百万tokens |
| **Benchmark** | Elo 1605，综合排名#1，80%场景击败前代 |

**关键能力**：可以在单次请求中同时传入 K线图(图片) + 特征描述(文本)，生成一个统一的3072维向量。这个向量同时编码了视觉形态信息和数值特征语义。

### 15.2 可行性评估

#### 可行 ✅ 的方面

| 维度 | 评估 | 理由 |
|------|------|------|
| **技术可行性** | ✅ 高 | 原生支持图片+文本混合输入，API调用简单 |
| **学术支撑** | ✅ 有 | ViT图像因子在股票预测中已验证优于CNN和规则方法(SSRN 2025) |
| **多模态融合** | ✅ 验证 | 图像+数值融合比单模态提升3-5%(PLOS ONE, PeerJ CS 2025) |
| **相似度检索** | ✅ 天然适配 | "找到历史上与当前走势最相似的时刻"是embedding的核心能力 |
| **与现有系统集成** | ✅ 自然 | chart_generator.py已生成K线图PNG，直接复用 |
| **成本可控** | ✅ 低 | 批量API $0.10/M tokens，全量回填约$5-15 |

#### 需要注意 ⚠️ 的方面

| 维度 | 评估 | 理由 |
|------|------|------|
| **embedding质量** | ⚠️ 未验证 | Gemini Embedding在金融K线图上的效果**无先例**，需要实验验证 |
| **图片理解深度** | ⚠️ 不确定 | 通用多模态embedding vs 专用ViT在K线细节捕捉上的差距未知 |
| **信息密度** | ⚠️ 可能不足 | 3072维向量能否充分编码一张复杂K线图(含MA/MACD/RSI/KDJ)的全部信息？ |
| **可解释性** | ⚠️ 黑盒 | embedding向量不可解释，无法知道模型"看到了"什么 |

#### 不建议 ❌ 的做法

| 做法 | 理由 |
|------|------|
| **仅用embedding替代全部数值特征** | embedding是压缩表示，必然丢失精度；数值特征(RSI=23.5)比embedding更精确 |
| **把474维数值特征转成文本再embedding** | 浪费——数值直接用比通过语言模型绕路更高效 |
| **完全依赖Gemini理解K线图中的具体数值** | 模型可能无法精确读取图中RSI=72.3这样的数值 |

### 15.3 推荐算法架构：双塔融合 + 相似度增强

**核心思想**：embedding不替代数值特征，而是作为**补充的视觉语义特征**，与数值特征融合后输入下游ML模型。

```
                     ┌─────────────────────────────────────────┐
                     │        每个交易截面的输入数据              │
                     │                                         │
                     │  ┌───────────┐  ┌──────────────────┐   │
                     │  │ 4张K线图   │  │ 结构化文本描述     │   │
                     │  │ (日/周/月  │  │ "RSI超卖28,MACD  │   │
                     │  │  /1h)     │  │  金叉,缠论一买,   │   │
                     │  │ + MACD图  │  │  量能放大1.8倍"   │   │
                     │  │ + RSI图   │  │                  │   │
                     │  └─────┬─────┘  └────────┬─────────┘   │
                     │        │                  │             │
                     └────────┼──────────────────┼─────────────┘
                              │                  │
                              ▼                  ▼
                     ┌────────────────────────────────────┐
                     │   Gemini Embedding 2 API           │
                     │   (图片×N + 文本 → 统一embedding)   │
                     │   task_type = CLASSIFICATION       │
                     └──────────────┬─────────────────────┘
                                    │
                                    ▼
                          ┌──────────────────┐
                          │ 视觉语义向量      │
                          │ (768维, MRL降维)  │  ← 塔A: 视觉+语义
                          └────────┬─────────┘
                                   │
                                   │ concat
                                   │
                          ┌────────┴─────────┐
                          │ 数值特征向量      │
                          │ (108维, P0层)     │  ← 塔B: 精确数值
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │ 融合向量          │
                          │ (876维)           │
                          └────────┬─────────┘
                                   │
                          ┌────────┴──────────────────┐
                          │                           │
                          ▼                           ▼
                  ┌──────────────┐          ┌─────────────────┐
                  │ 下游任务A     │          │ 下游任务B        │
                  │ BC/IRL/DT    │          │ 相似度检索       │
                  │ (买卖点分类)  │          │ (历史形态匹配)   │
                  └──────────────┘          └─────────────────┘
```

### 15.4 每个截面的Embedding输入设计

#### 输入组合（单次API调用）

```python
# 每个交易截面发送1次 Gemini Embedding API 请求
request = {
    "model": "models/gemini-embedding-2-preview",
    "content": {
        "parts": [
            # Part 1: 日线K线图（含MA/布林带/成交量）
            {"inline_data": {"mime_type": "image/png", "data": kline_day_base64}},
            # Part 2: 周线K线图
            {"inline_data": {"mime_type": "image/png", "data": kline_week_base64}},
            # Part 3: 月线K线图
            {"inline_data": {"mime_type": "image/png", "data": kline_month_base64}},
            # Part 4: MACD+RSI+KDJ综合指标图（单独一张）
            {"inline_data": {"mime_type": "image/png", "data": indicator_chart_base64}},
            # Part 5: 结构化文本摘要（将关键数值特征编码为自然语言）
            {"text": structured_summary}
        ]
    },
    "task_type": "CLASSIFICATION",
    "output_dimensionality": 768  # MRL降维，平衡质量和效率
}
```

#### 结构化文本摘要模板

```python
def build_snapshot_text(features: dict, kline_ind: dict) -> str:
    """将关键数值特征转为自然语言描述，供Gemini Embedding编码"""
    parts = []

    # 趋势与动量
    regime = features.get("market_regime", {}).get("regime", "unknown")
    slope = kline_ind.get("day", {}).get("trend_slope_pct", 0)
    parts.append(f"市场状态{regime}，日线趋势斜率{slope:.2f}%/日")

    # 技术指标状态（文字化而非数值）
    rsi = kline_ind.get("day", {}).get("rsi", 50)
    if rsi < 30: parts.append("RSI超卖区域")
    elif rsi > 70: parts.append("RSI超买区域")
    else: parts.append(f"RSI中性{rsi:.0f}")

    macd_hist = kline_ind.get("day", {}).get("macd_hist", 0)
    parts.append(f"MACD柱{'翻红放大' if macd_hist > 0 else '翻绿缩小'}")

    # 缠论信号
    chanlun = kline_ind.get("day", {}).get("chanlun", {})
    buy_sigs = chanlun.get("buy_signals", [])
    sell_sigs = chanlun.get("sell_signals", [])
    if buy_sigs: parts.append(f"缠论{buy_sigs[0]['type']}类买点")
    if sell_sigs: parts.append(f"缠论{sell_sigs[0]['type']}类卖点")

    # 量价关系
    vp = kline_ind.get("day", {}).get("volume_price", {})
    if vp.get("volume_breakout"): parts.append("放量突破")
    if vp.get("shrink_pullback"): parts.append("缩量回踩")
    if vp.get("climax_volume"): parts.append("天量滞涨警告")

    # 均线位置
    ma_sys = kline_ind.get("day", {}).get("ma_system", {})
    ma20_bias = ma_sys.get("ma20", {}).get("pct_above", 0)
    parts.append(f"MA20乖离{ma20_bias:+.1f}%")

    # 背离
    div = kline_ind.get("day", {}).get("divergence", {})
    if div.get("macd_bot_div"): parts.append("MACD底背离")
    if div.get("macd_top_div"): parts.append("MACD顶背离")

    # 筹码与资金
    chip = features.get("chip_distribution", {})
    if chip.get("profit_ratio", 50) > 80: parts.append("获利盘超80%")
    if chip.get("profit_ratio", 50) < 20: parts.append("套牢盘密集")

    # 基本面概要
    pe = features.get("pe_ttm")
    if pe and pe < 15: parts.append(f"低估值PE{pe:.1f}")
    elif pe and pe > 60: parts.append(f"高估值PE{pe:.1f}")

    return "。".join(parts)
```

### 15.5 三种应用模式

#### 模式A：Embedding作为IRL/BC的增强特征

```
原始数值特征 (108维) + Gemini Embedding (768维) → concat (876维) → BC / IRL / DT

优点: 最简单的融合方式，embedding提供"视觉直觉"补充
缺点: 768维中大部分可能是噪声，需要降维或特征选择
建议: 先用PCA将768维降至32-64维，再与数值特征concat
```

#### 模式B：相似形态检索 + K近邻投票（最具创新性）

```
核心思想: 不直接预测买卖，而是"找到历史上最相似的N个时刻，看它们后来涨了还是跌了"

步骤:
1. 对100只股票×750天的每个交易日，生成 Gemini Embedding (768维)
2. 建立向量数据库（如FAISS/ChromaDB），索引全部 75,000 个embedding
3. 对标注的"买点"和"卖点"打标签
4. 实时推理时:
   a. 生成当前截面的embedding
   b. 在向量库中搜索最相似的K个历史时刻 (K=20)
   c. 统计这K个时刻中:
      - 有多少是"买点后N天上涨"的
      - 有多少是"卖点后N天下跌"的
      - 后续5/10/20日平均收益率
   d. 加权投票生成买卖信号

             当前截面                    向量数据库 (75,000条)
          ┌───────────┐               ┌──────────────────────┐
          │ embedding │──cosine sim──→│ 最相似的20个历史截面   │
          │ (768维)   │               │ ├─ #1: 买点, +12.3%  │
          └───────────┘               │ ├─ #2: 买点, +8.7%   │
                                      │ ├─ #3: 卖点, -5.2%   │
                                      │ ├─ ...               │
                                      │ └─ #20: 买点, +15.1% │
                                      └──────────┬───────────┘
                                                  │
                                                  ▼
                                      买点相似度: 14/20 = 70%
                                      预期收益: +9.8%
                                      → 发出买入信号 ✅

优点:
  - 直观可解释: "当前走势与2024年3月15日的XXX股最相似，当时随后涨了12%"
  - 不需要训练ML模型，纯检索
  - 跨股票迁移: 同一形态在不同股票上的表现可以互相参考
  - 自然支持增量更新: 新数据直接入库

缺点:
  - 依赖embedding质量（Gemini对K线图的理解深度）
  - K值和阈值需要调参
  - 向量库维护成本
```

#### 模式C：对比学习微调（最有学术前景）

```
步骤:
1. 构造正负样本对:
   - 正样本对: 同为"买点"的两个截面embedding
   - 负样本对: 一个"买点" + 一个"卖点"的embedding
   - 参考 ICAIF 2024 论文的统计采样策略
2. 用对比损失(InfoNCE)微调一个投影头(Projection Head):
   Gemini Embedding (768维) → MLP → 微调后向量 (128维)
3. 微调后的向量空间中, 买点聚类在一起, 卖点聚类在一起
4. 新截面的向量位置直接决定买/卖/观望

             Gemini Embedding (768维, 冻结)
                       │
                       ▼
              ┌─────────────────┐
              │ Projection Head │ ← 可训练 (2层MLP)
              │ 768 → 256 → 128 │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ 微调后向量空间    │
              │                 │
              │   ● ● ●        │  ● = 买点 (聚类)
              │     ●  ●       │
              │                 │
              │        ○ ○     │  ○ = 卖点 (聚类)
              │      ○   ○    │
              │         ○      │
              │                 │
              │   △ △ △ △      │  △ = 观望 (散布)
              └─────────────────┘

优点:
  - Gemini大模型冻结，只训练轻量投影头 → 训练快、数据需求低
  - 对比学习适合小样本(2000-5000个买卖点足够)
  - 微调后的向量空间语义清晰: 买卖点自然分离
  - 可与模式B(检索)结合: 在微调后的空间做KNN

缺点:
  - 需要GPU训练投影头(但极轻量, CPU也可跑)
  - 正负样本构造策略需要实验
```

### 15.6 推荐实施方案：渐进式三步

```
Step 1 (1周) → 模式B: KNN相似度检索 (零训练, 快速验证embedding质量)
  ├─ 对10只股票×1年数据, 生成Gemini Embedding
  ├─ 建FAISS索引, 查询最相似历史截面
  ├─ 评估: 相似截面的后续走势是否真的相似?
  ├─ 成本: ~$2 API费用
  └─ 决策点: embedding相似度与走势相关性 > 0.3 → 继续

Step 2 (1-2周) → 模式A: Embedding + 数值特征融合
  ├─ 将768维embedding PCA降至32维
  ├─ concat到P0层108维数值特征 → 140维
  ├─ 用BC(sklearn)训练买卖点分类器
  ├─ 对比: 有/无embedding的F1差异
  └─ 决策点: F1提升 > 3% → embedding有价值

Step 3 (2周) → 模式C: 对比学习微调
  ├─ 构造正负样本对(买点vs卖点)
  ├─ 训练投影头(2层MLP, CPU可跑)
  ├─ 在微调后空间做KNN + BC
  └─ 与Step 1/2对比
```

### 15.7 成本估算

#### Embedding生成成本

```
每个截面1次API调用:
  4张K线图(日/周/月/1h): 每张约 258 tokens → 1032 tokens
  1张指标图: ~258 tokens
  文本描述: ~200 tokens
  合计: ~1490 tokens/截面

全量回填 (100只 × 750天):
  75,000 截面 × 1490 tokens = 111,750,000 tokens ≈ 112M tokens
  批量API价格: $0.10/M tokens
  总成本: 112 × $0.10 = $11.2 ≈ ¥80

仅标注点附近 (±5天窗口):
  38,500 截面 × 1490 tokens ≈ 57M tokens
  总成本: 57 × $0.10 = $5.7 ≈ ¥41

Step 1 实验 (10只 × 250天):
  2,500 截面 × 1490 tokens ≈ 3.7M tokens
  总成本: $0.37 ≈ ¥3
```

#### 存储成本

```
每个embedding: 768维 × float32 = 3072 bytes ≈ 3KB
全量 75,000 个: 75,000 × 3KB = 225MB
FAISS索引额外开销: ~50MB
总计: < 300MB → 完全可本地存储
```

#### 时间成本

```
Gemini Embedding API 速率:
  约 300-500 requests/min (取决于配额)
  75,000 截面 → 150-250 分钟 ≈ 3-4小时

可优化:
  batchEmbedContents 批量接口 → 减少网络开销
  异步批量 batches.create_embeddings → 后台处理，无需等待
```

### 15.8 与IRL Pipeline的集成点

```
原方案 (第5章):
  State = [数值特征 108维]
              │
              └──→ BC / MaxEnt IRL / Decision Transformer

增强方案:
  State = [数值特征 108维] + [Gemini Embedding PCA 32维] = 140维
              │                        │
              │                        └── 编码了K线视觉形态+指标图形态+文本语义
              │
              └──→ BC / MaxEnt IRL / Decision Transformer

额外增强 (模式B叠加):
  State 140维 + [KNN历史相似度特征 5维] = 145维
                        │
                        ├── top_k_avg_return_5d: 最相似20个截面的5日平均收益
                        ├── top_k_avg_return_10d: 10日平均收益
                        ├── top_k_buy_ratio: 相似截面中买点占比
                        ├── top_k_avg_similarity: 平均余弦相似度
                        └── top_k_same_stock_ratio: 同一只股票的相似截面占比
```

### 15.9 与现有Vision Agent的关系

```
现有方案 (已实现):
  K线图 ──→ Gemini/Grok Vision ──→ 文本分析报告 ──→ LLM评分
  优点: 可解释(文本输出)
  缺点: 每次调用需10-30秒, 输出不稳定, 不可直接用于ML

Embedding方案 (新增):
  K线图 ──→ Gemini Embedding 2 ──→ 768维向量 ──→ ML模型输入
  优点: 快速(<1秒), 稳定(确定性输出), 可直接用于ML/IRL
  缺点: 不可解释(向量无语义)

两者关系: 互补而非替代
  ├── Vision Agent: 用于生成人类可读的分析报告(现有功能不变)
  └── Embedding: 用于IRL/ML的特征工程(新增能力)
```

### 15.10 风险与注意事项

| 风险 | 严重度 | 缓解措施 |
|------|--------|---------|
| **Gemini对K线图理解深度不足** | 🔴 核心风险 | Step 1快速验证；对比Random Embedding baseline排除偶然 |
| **embedding维度冗余** | 🟡 中 | PCA/UMAP降维；L1正则化特征选择 |
| **API依赖(网络/限额/价格变动)** | 🟡 中 | 预计算全量embedding缓存本地；批量API降低成本 |
| **跨模态对齐质量** | 🟡 中 | 文本描述需精心设计，避免与图片信息冲突 |
| **图片质量影响embedding** | 🟢 低 | 现有chart_generator已生成标准化PNG |

### 15.11 关键验证实验（Step 1必须回答的问题）

在投入大规模embedding之前，用10只股票×1年数据（成本¥3）回答：

| # | 实验 | 预期 | 通过标准 |
|---|------|------|---------|
| 1 | **形态相似性**: 同一K线形态(如"早晨之星")在不同股票上的embedding距离 | 应该较近 | cosine_sim > 0.7 |
| 2 | **趋势区分度**: 上涨趋势 vs 下跌趋势的embedding | 应该分离 | t-SNE可视化可见聚类 |
| 3 | **时间稳定性**: 同一只股票相邻交易日的embedding | 应该平滑变化 | 相邻日cosine_sim > 0.8 |
| 4 | **买卖点可分性**: 标注买点 vs 卖点的embedding | 应该有区分 | KNN分类准确率 > 60% |
| 5 | **Random Baseline**: 随机向量替代embedding | F1应显著低于真实embedding | p-value < 0.05 |

> **结论**：Gemini Embedding 多模态方案**技术上完全可行**，且是当前唯一能将K线图视觉形态与文字语义统一编码的商用API。推荐以**模式B(KNN相似检索)**作为快速验证起点（成本¥3，1周出结果），确认embedding质量后再融入IRL主Pipeline。最大风险是Gemini对金融K线图的理解深度——需通过5个验证实验排除。全量回填成本约¥80，与LLM评分回跑(¥0-¥120)属于同一量级。

### 15.12 学术参考

| 论文 | 年份 | 核心发现 |
|------|------|---------|
| From Vision to Value: Stock Chart Image-Driven Factors (SSRN) | 2025 | **ViT图像因子产生正且经济显著的收益差**，优于CNN |
| Learning Stock Price Signals via ViT (SSRN 5224805) | 2025 | ViT self-attention有效捕捉价量交互和均线偏离 |
| Enhancing Trend Prediction with CNN on Candlestick (PeerJ CS) | 2025 | CNN+技术指标测试准确率92.83%，F1>92% |
| Feature Fusion LSTM-CNN for Stock Prices (PLOS ONE) | — | 融合模型优于单模态 |
| Contrastive Learning of Asset Embeddings (ICAIF 2024) | 2024 | 基于假设检验的对比学习，行业分类和组合优化显著优于基线 |
| STONK: Cross-Modal Attention Fusion (arXiv) | 2025 | 数值指标+新闻情感跨模态注意力融合 |
| Encoding Candlesticks as Images for CNN (Financial Innovation) | 2020 | 将K线编码为图像用于CNN预测的开创性工作 |

---

## 16. 项目工程化设计

### 16.1 新项目目录结构

```
stockagent-irl/                         # 新项目根目录
├── configs/
│   ├── project.json                    # 主配置（股票池、训练参数、阈值）
│   ├── stocks.json                     # 100只股票池定义（代码/名称/行业/权重）
│   └── feature_schema.json             # 特征维度定义（名称/来源/归一化方式）
├── src/
│   ├── stockagent_irl/
│   │   ├── __init__.py
│   │   ├── labeler/                    # 买卖点标注引擎
│   │   │   ├── __init__.py
│   │   │   ├── backtrack_optimal.py    # 回溯最优标注（DP/GA）
│   │   │   ├── chanlun_labeler.py      # 缠论买卖点标注
│   │   │   ├── wave_labeler.py         # 波浪理论标注
│   │   │   ├── fusion_labeler.py       # 多策略融合标注
│   │   │   └── validator.py            # 标注质量验证（收益率/回撤/频率）
│   │   ├── features/                   # 特征工程
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py            # 统一特征提取入口
│   │   │   ├── technical.py            # 技术指标特征（复用data_backend）
│   │   │   ├── agent_scores.py         # Agent公式评分（复用_simple_policy）
│   │   │   ├── market_env.py           # 市场环境特征
│   │   │   ├── embedding.py            # Gemini Embedding特征
│   │   │   └── normalizer.py           # 特征归一化/标准化
│   │   ├── env/                        # Gymnasium交易环境
│   │   │   ├── __init__.py
│   │   │   ├── trading_env.py          # 核心环境定义
│   │   │   └── reward.py               # 奖励函数（标注驱动 / IRL学习）
│   │   ├── models/                     # 模型定义与训练
│   │   │   ├── __init__.py
│   │   │   ├── bc_trainer.py           # Phase 1: Behavioral Cloning
│   │   │   ├── irl_trainer.py          # Phase 2: MaxEnt IRL / AIRL
│   │   │   ├── dt_trainer.py           # Phase 3: Decision Transformer
│   │   │   └── evaluator.py            # 统一评估框架
│   │   ├── backtest/                   # 回测引擎
│   │   │   ├── __init__.py
│   │   │   ├── engine.py               # 核心回测逻辑
│   │   │   ├── metrics.py              # 评估指标（夏普/回撤/胜率等）
│   │   │   └── report.py               # 回测报告生成
│   │   ├── inference/                  # 推理与信号生成
│   │   │   ├── __init__.py
│   │   │   ├── predictor.py            # 模型推理入口
│   │   │   └── signal_filter.py        # 信号过滤与置信度
│   │   └── data/                       # 数据管理
│   │       ├── __init__.py
│   │       ├── fetcher.py              # 数据采集（复用stockagent-analysis的data_backend）
│   │       ├── cache.py                # 本地数据缓存（Parquet）
│   │       └── dataset.py              # 训练数据集构造（PyTorch Dataset / numpy）
│   └── core/                           # → 软链接或pip依赖 stockagent-analysis/src/core
├── data/                               # 数据目录（git忽略）
│   ├── raw/                            # 原始K线/基本面数据
│   │   └── {symbol}/
│   │       ├── kline_day.parquet
│   │       ├── kline_week.parquet
│   │       ├── fundamentals.json
│   │       └── charts/                 # K线图PNG
│   ├── labels/                         # 标注结果
│   │   └── {symbol}_labels.parquet     # date, action, price, confidence, strategy
│   ├── features/                       # 计算好的特征矩阵
│   │   └── {symbol}_features.parquet   # date × 150维
│   ├── embeddings/                     # Gemini Embedding缓存
│   │   └── {symbol}_emb.npy           # (N, 768) numpy数组
│   ├── trajectories/                   # 训练用轨迹数据
│   │   └── all_trajectories.pkl        # imitation/d3rlpy格式
│   └── models/                         # 训练好的模型
│       ├── bc_v1/
│       ├── irl_v1/
│       └── dt_v1/
├── notebooks/                          # 实验Notebook
│   ├── 01_eda.ipynb                    # 数据探索
│   ├── 02_label_quality.ipynb          # 标注质量检查
│   ├── 03_feature_analysis.ipynb       # 特征分析
│   ├── 04_embedding_validation.ipynb   # Embedding验证实验
│   └── 05_backtest_comparison.ipynb    # 模型对比回测
├── scripts/
│   ├── fetch_all_data.py               # 一键采集100只股票数据
│   ├── label_all.py                    # 一键标注全部股票
│   ├── extract_features.py             # 一键提取全部特征
│   ├── train.py                        # 训练入口（参数化选择算法）
│   ├── backtest.py                     # 回测入口
│   └── daily_inference.py              # 每日推理信号生成
├── tests/
│   ├── test_labeler.py
│   ├── test_features.py
│   ├── test_env.py
│   └── test_backtest.py
├── output/                             # 输出目录（git忽略）
│   ├── reports/
│   ├── signals/
│   └── experiments/
├── .env                                # API密钥（Tushare/Gemini等）
├── .gitignore
├── pyproject.toml
├── CLAUDE.md                           # Claude Code项目指引
└── README.md
```

### 16.2 核心配置文件设计

#### `configs/project.json`

```json
{
  "project_name": "stockagent-irl",
  "version": "0.1.0",

  "stock_pool": {
    "config_file": "stocks.json",
    "min_stocks": 100,
    "min_industries": 10,
    "history_years": 3,
    "min_daily_volume_1m": 5000,
    "exclude_st": true,
    "exclude_new_stocks_days": 365
  },

  "labeling": {
    "primary_strategy": "backtrack_optimal",
    "secondary_strategies": ["chanlun", "wave"],
    "fusion_mode": "intersection",
    "constraints": {
      "min_hold_days": 5,
      "max_hold_days": 60,
      "min_interval_days": 3,
      "target_annual_return": 0.50,
      "max_drawdown": 0.20,
      "transaction_cost_pct": 0.075,
      "slippage_pct": 0.10
    }
  },

  "features": {
    "schema_file": "feature_schema.json",
    "tiers": {
      "P0": true,
      "P1": true,
      "P2": false,
      "P3": false
    },
    "normalization": "robust_scaler",
    "embedding": {
      "enabled": false,
      "model": "gemini-embedding-2-preview",
      "dimensions": 768,
      "pca_target_dims": 32
    }
  },

  "training": {
    "data_split": {
      "train_end": "2024-12-31",
      "val_end": "2025-06-30",
      "test_end": "2025-12-31"
    },
    "walk_forward": {
      "enabled": true,
      "train_window_months": 18,
      "test_window_months": 6,
      "step_months": 6
    },
    "class_balance": {
      "method": "smote",
      "buy_sell_ratio": 1.0,
      "hold_downsample_ratio": 0.1
    },
    "bc": {
      "model": "gradient_boosting",
      "n_estimators": 500,
      "max_depth": 6,
      "learning_rate": 0.05
    },
    "irl": {
      "algorithm": "maxent",
      "n_iterations": 200,
      "learning_rate": 0.01
    },
    "dt": {
      "context_length": 20,
      "n_heads": 4,
      "n_layers": 3,
      "embed_dim": 128,
      "epochs": 100,
      "batch_size": 64
    }
  },

  "inference": {
    "buy_threshold": 0.70,
    "sell_threshold": 0.70,
    "min_confidence": 0.60,
    "cooldown_days": 3,
    "max_signals_per_day": 5
  },

  "backtest": {
    "initial_capital": 1000000,
    "commission_rate": 0.00025,
    "stamp_tax_rate": 0.0005,
    "slippage_pct": 0.001,
    "position_size": "equal_weight",
    "max_positions": 5
  }
}
```

### 16.3 股票池选择标准

#### 筛选条件

| 条件 | 阈值 | 理由 |
|------|------|------|
| 上市时间 | ≥ 3年（2023-01前上市） | 确保历史数据充足 |
| 日均成交额 | ≥ 5000万元 | 流动性保障，避免价格操纵 |
| 非ST/*ST | 排除 | 退市风险股行为异常 |
| 非北交所 | 排除 | 交易规则差异大（30%涨跌停） |
| 停牌天数 | < 总交易日的5% | 数据连续性 |
| 行业覆盖 | ≥ 10个申万一级行业 | 避免行业偏倚 |
| 市值分布 | 大/中/小盘各占约1/3 | 覆盖不同市值风格 |

#### 推荐行业分布（100只）

| 行业 | 数量 | 代表票（示例） |
|------|------|--------------|
| 电子/半导体 | 12 | 立讯精密、韦尔股份、北方华创 |
| 医药生物 | 10 | 恒瑞医药、药明康德、迈瑞医疗 |
| 电力设备/新能源 | 10 | 宁德时代、阳光电源、汇川技术 |
| 食品饮料 | 8 | 贵州茅台、五粮液、伊利股份 |
| 计算机/软件 | 8 | 海康威视、金山办公、中科创达 |
| 机械设备 | 8 | 三一重工、先导智能、杰瑞股份 |
| 银行/非银金融 | 8 | 招商银行、宁波银行、东方财富 |
| 汽车 | 8 | 比亚迪、长城汽车、拓普集团 |
| 化工/基础材料 | 8 | 万华化学、恩捷股份、华鲁恒升 |
| 有色/钢铁/采矿 | 6 | 紫金矿业、北方稀土、中国神华 |
| 地产/建筑 | 6 | 保利发展、中国建筑、海螺水泥 |
| 传媒/通信 | 5 | 分众传媒、中际旭创、中兴通讯 |
| 国防军工 | 3 | 中航沈飞、航发动力、中航光电 |

#### 股票池的3种子集

```
full_100:    全部100只 — 训练主集
top_30:      流动性最好的30只 — 快速实验集
sector_10:   每行业1只代表 — 行业覆盖验证集
```

### 16.4 Gymnasium 交易环境详细设计

```python
class StockTradingEnv(gymnasium.Env):
    """
    A股波段交易环境，适配IRL/BC/DT训练。

    核心设计决策:
    - 离散动作空间: hold(0), buy(1), sell(2)
    - 日频决策: 每个交易日收盘后做出决策，次日开盘执行
    - T+1约束: 买入当天不可卖出
    - 单股单仓: 每只股票独立环境，不涉及组合
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, symbol, features_df, labels_df, config):
        super().__init__()

        self.symbol = symbol
        self.features = features_df      # (T, feature_dim) DataFrame
        self.labels = labels_df          # (T,) 标注序列 {0:hold, 1:buy, 2:sell}
        self.config = config

        self.feature_dim = features_df.shape[1]
        self.position_dim = 6            # 持仓状态维度

        # === 空间定义 ===
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.feature_dim + self.position_dim,),
            dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(3)  # hold, buy, sell

        # === 交易约束 ===
        self.commission = config["backtest"]["commission_rate"]
        self.stamp_tax = config["backtest"]["stamp_tax_rate"]
        self.slippage = config["backtest"]["slippage_pct"]
        self.min_hold_days = config["labeling"]["constraints"]["min_hold_days"]
        self.cooldown_days = config["inference"]["cooldown_days"]

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.holding = False
        self.buy_price = 0.0
        self.buy_step = 0
        self.holding_days = 0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.last_trade_step = -self.cooldown_days  # 允许首次交易
        return self._get_obs(), {}

    def _get_obs(self):
        """构造观测向量: [市场特征 | 持仓状态]"""
        market_features = self.features.iloc[self.current_step].values.astype(np.float32)
        position_features = np.array([
            float(self.holding),
            float(self.holding_days),
            self.unrealized_pnl,
            float(self.current_step - self.last_trade_step),  # 距上次交易天数
            float(self.total_trades),
            self.realized_pnl
        ], dtype=np.float32)
        return np.concatenate([market_features, position_features])

    def step(self, action):
        current_price = self.features.iloc[self.current_step]["close"]
        reward = 0.0
        info = {"action_taken": "hold", "valid": True}

        # === 动作合法性检查 ===
        if action == 1 and self.holding:
            action = 0  # 已持仓不可重复买入
            info["valid"] = False
        if action == 2 and not self.holding:
            action = 0  # 未持仓不可卖出
            info["valid"] = False
        if action == 1 and (self.current_step - self.last_trade_step) < self.cooldown_days:
            action = 0  # 冷却期内不可买入
            info["valid"] = False
        if action == 2 and self.holding_days < self.min_hold_days:
            action = 0  # T+1 + 最低持仓天数
            info["valid"] = False

        # === 执行动作 ===
        if action == 1:  # BUY
            self.buy_price = current_price * (1 + self.slippage)
            self.buy_step = self.current_step
            self.holding = True
            self.holding_days = 0
            self.last_trade_step = self.current_step
            self.total_trades += 1
            info["action_taken"] = "buy"

        elif action == 2:  # SELL
            sell_price = current_price * (1 - self.slippage)
            cost = self.buy_price * self.commission + sell_price * (self.commission + self.stamp_tax)
            pnl = (sell_price - self.buy_price) / self.buy_price - cost / self.buy_price
            self.realized_pnl += pnl
            reward = pnl  # 卖出时获得实际盈亏作为奖励
            self.holding = False
            self.holding_days = 0
            self.last_trade_step = self.current_step
            self.total_trades += 1
            info["action_taken"] = "sell"
            info["pnl"] = pnl

        # === 更新持仓状态 ===
        if self.holding:
            self.holding_days += 1
            self.unrealized_pnl = (current_price - self.buy_price) / self.buy_price
        else:
            self.unrealized_pnl = 0.0

        # === 步进 ===
        self.current_step += 1
        terminated = self.current_step >= len(self.features) - 1
        truncated = False

        # 强制平仓：到期未卖出
        if terminated and self.holding:
            sell_price = self.features.iloc[self.current_step]["close"] * (1 - self.slippage)
            pnl = (sell_price - self.buy_price) / self.buy_price
            self.realized_pnl += pnl
            reward = pnl
            self.holding = False

        return self._get_obs(), reward, terminated, truncated, info

    def get_expert_action(self):
        """返回标注的专家动作（用于BC/IRL）"""
        return int(self.labels.iloc[self.current_step])
```

### 16.5 评估指标体系

#### 分类指标（买卖点识别质量）

| 指标 | 公式 | 目标 | 说明 |
|------|------|------|------|
| Precision (买点) | TP_buy / (TP_buy + FP_buy) | > 0.50 | 预测买点中真正好买点的比例 |
| Recall (买点) | TP_buy / (TP_buy + FN_buy) | > 0.40 | 真正好买点中被识别出的比例 |
| F1 (买点) | 2×P×R/(P+R) | > 0.40 | 买点综合指标 |
| F1 (卖点) | 同上 | > 0.40 | 卖点综合指标 |
| Accuracy (整体) | correct / total | 参考 | 因类别不平衡，仅作参考 |

#### 交易指标（回测表现）

| 指标 | 公式 | 目标 | 说明 |
|------|------|------|------|
| 年化收益率 | (1+total_return)^(252/days) - 1 | > 30% | 低于标注的50%是预期内的 |
| 夏普比率 | (Rp - Rf) / σp | > 1.5 | 风险调整后收益 |
| 最大回撤 | max(peak - trough) / peak | < 20% | 最大亏损幅度 |
| 胜率 | 盈利交易数 / 总交易数 | > 55% | 每笔交易的胜率 |
| 盈亏比 | 平均盈利 / 平均亏损 | > 1.5 | 赚的比亏的多 |
| 交易频率 | 年交易次数 | 15-40次 | 波段操作合理范围 |
| Calmar比率 | 年化收益 / 最大回撤 | > 2.0 | 收益/风险效率 |
| 超额收益 | 策略收益 - 沪深300收益 | > 15% | 是否跑赢大盘 |

#### 模型对比矩阵

```
                 BC      MaxEnt IRL    Decision Trans.   Buy&Hold
年化收益率        ?%         ?%             ?%             ?%
夏普比率          ?          ?              ?              ?
最大回撤          ?%         ?%             ?%             ?%
胜率              ?%         ?%             ?%             -
F1(买点)          ?          ?              ?              -
训练时间          ?min       ?h             ?h             -
推理时间/只       ?ms        ?ms            ?ms            -
```

### 16.6 回测框架设计

```python
class BacktestEngine:
    """
    回测引擎设计要点:
    - 严格避免未来信息泄漏
    - 模拟真实A股交易规则
    - 支持多种仓位管理策略
    """

    # === A股交易规则 ===
    RULES = {
        "t_plus_1": True,                # T+1: 买入次日才能卖出
        "price_limit_pct": 10.0,         # 涨跌停板10% (创业板20%, 科创板20%)
        "min_lot": 100,                  # 最小交易单位100股
        "trading_hours": "09:30-15:00",  # 交易时间
        "settlement": "T+1",            # 资金交收T+1
        "no_short": True,               # 不允许融券做空（简化）
    }

    # === 回测模式 ===
    MODES = {
        "single_stock": "单股策略: 100%仓位单只股票轮动",
        "equal_weight": "等权策略: 同时持有≤5只, 等权分配",
        "signal_rank": "信号排序: 按置信度排序, 优先买入top信号",
    }

    # === 滑点模型 ===
    SLIPPAGE_MODELS = {
        "fixed": "固定百分比滑点 (默认0.1%)",
        "volume_impact": "成交量冲击: slippage = k * sqrt(order_size / daily_volume)",
        "spread": "买卖价差: 按盘口5档估算",
    }
```

### 16.7 完整依赖清单

```toml
# pyproject.toml

[project]
name = "stockagent-irl"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    # === 数据采集（复用现有项目） ===
    "mootdx>=0.11.7",           # 通达信本地数据
    "tushare>=1.4.0",           # Tushare Pro
    "akshare>=1.14.0",          # AKShare

    # === 数据处理 ===
    "pandas>=2.0",
    "numpy>=1.24",
    "pyarrow>=14.0",            # Parquet读写
    "scipy>=1.11",              # 统计计算

    # === 机器学习 ===
    "scikit-learn>=1.3",        # BC baseline + 特征工程
    "imbalanced-learn>=0.11",   # SMOTE类别平衡

    # === 强化学习 / 模仿学习 ===
    "gymnasium>=0.29",          # 交易环境
    "stable-baselines3>=2.2",   # RL算法 (PPO/SAC)
    "imitation>=1.0",           # BC / DAgger / GAIL / AIRL
    "d3rlpy>=2.2",              # 离线RL + Decision Transformer

    # === 可视化 ===
    "matplotlib>=3.8",
    "plotly>=5.18",

    # === 工具 ===
    "python-dotenv>=1.0",
    "loguru>=0.7",              # 日志
    "tqdm>=4.66",               # 进度条
    "joblib>=1.3",              # 并行计算
]

[project.optional-dependencies]
embedding = [
    "google-genai>=1.0",        # Gemini Embedding API
    "faiss-cpu>=1.7",           # 向量检索
]
gpu = [
    "torch>=2.1",               # Phase 2-3 GPU训练
]
notebook = [
    "jupyter>=1.0",
    "seaborn>=0.13",
]
```

---

## 17. 数据版本管理与实验追踪

### 17.1 数据版本管理

```
采用轻量方案: 文件名含版本号 + metadata JSON

data/labels/
├── 002571_labels_v1_backtrack.parquet        # 回溯最优标注
├── 002571_labels_v2_fusion.parquet           # 融合标注
├── labels_manifest.json                      # 版本清单
│   {
│     "v1_backtrack": {
│       "created": "2026-04-01",
│       "strategy": "backtrack_optimal",
│       "constraints": {"min_hold": 5, "max_hold": 60},
│       "stats": {"avg_return": 0.52, "avg_trades": 28}
│     },
│     "v2_fusion": {
│       "created": "2026-04-15",
│       "strategy": "fusion(backtrack+chanlun)",
│       "stats": {"avg_return": 0.48, "avg_trades": 24}
│     }
│   }
└── ...
```

### 17.2 实验追踪（轻量方案）

```
不引入MLflow/WandB, 用JSON文件追踪:

output/experiments/
├── exp_001_bc_p0_features.json
│   {
│     "id": "exp_001",
│     "algorithm": "bc_gradient_boosting",
│     "features": "P0_108dim",
│     "label_version": "v1_backtrack",
│     "train_period": "2023-01 ~ 2024-12",
│     "test_period": "2025-07 ~ 2025-12",
│     "hyperparams": {"n_estimators": 500, "max_depth": 6},
│     "results": {
│       "f1_buy": 0.42, "f1_sell": 0.38,
│       "annual_return": 0.31, "sharpe": 1.8,
│       "max_drawdown": 0.15
│     },
│     "model_path": "data/models/bc_v1/",
│     "notes": "baseline, P0特征足够区分买卖点"
│   }
├── exp_002_bc_p0p1_features.json
└── ...
```

---

## 18. 完整里程碑与任务分解（修订版）

> 融合第13章硬件方案、第14章特征空间、第15章Embedding方案后的修订路线图。

### M0: 项目初始化（第1-2周）

**目标**: 项目骨架就绪，股票池确定，数据可用性验证通过。

| # | 任务 | 产出 | 硬件 | 依赖 |
|---|------|------|------|------|
| 0.1 | 创建项目目录结构（16.1节） | stockagent-irl/ 骨架 | 本地 | 无 |
| 0.2 | 编写 pyproject.toml + 安装依赖 | 虚拟环境就绪 | 本地 | 无 |
| 0.3 | 编写 configs/project.json | 全局配置 | 本地 | 无 |
| 0.4 | 确定100只股票池（16.3节标准） | configs/stocks.json | 本地 | 无 |
| 0.5 | 验证数据源覆盖率（TDX/Tushare/AKShare） | 数据可用性报告 | 本地 | 0.4 |
| 0.6 | 实现 data/fetcher.py（复用 data_backend） | 100只×3年数据入库 | 本地 | 0.4, 0.5 |
| 0.7 | 编写 CLAUDE.md 项目指引 | 开发规范文档 | 本地 | 无 |

**M0交付标准**: `python scripts/fetch_all_data.py` 可成功获取100只股票的日/周/月K线 + 基本面数据。

---

### M1: 买卖点标注引擎（第3-5周）

**目标**: 对100只股票生成高质量的"完美买卖点"标注。

| # | 任务 | 产出 | 硬件 | 依赖 |
|---|------|------|------|------|
| 1.1 | 实现回溯最优标注算法(DP) | labeler/backtrack_optimal.py | 本地CPU | M0 |
| 1.2 | 实现缠论买卖点标注（复用_detect_chanlun_signals） | labeler/chanlun_labeler.py | 本地CPU | M0 |
| 1.3 | 实现波浪理论标注（可选） | labeler/wave_labeler.py | 本地CPU | M0 |
| 1.4 | 实现融合标注逻辑 | labeler/fusion_labeler.py | 本地CPU | 1.1, 1.2 |
| 1.5 | 实现标注质量验证器 | labeler/validator.py | 本地CPU | 1.4 |
| 1.6 | 运行全量标注 + 验证 | data/labels/*.parquet | 本地CPU | 1.4, 1.5 |
| 1.7 | Notebook: 人工抽检10只股票标注结果 | notebooks/02_label_quality.ipynb | 本地 | 1.6 |

**M1交付标准**:
- 100只股票全部完成标注
- 平均年化收益率 ≥ 50%
- 平均每只股票每年交易 10-30次
- 最大回撤 ≤ 20%
- 人工抽检10只无明显不合理标注

**M1决策点**: 如果回溯最优标注无法在约束条件下达到50%年化 → 放宽约束（如max_hold_days从60→90），或降低目标至30%。

---

### M2: 特征工程与数据Pipeline（第5-7周）

**目标**: 构建完整的特征提取器，生成训练数据集。

| # | 任务 | 产出 | 硬件 | 依赖 |
|---|------|------|------|------|
| 2.1 | 实现P0层技术指标特征提取 | features/technical.py | 本地CPU | M0 |
| 2.2 | 实现Agent公式评分批量计算 | features/agent_scores.py | 本地CPU | M0 |
| 2.3 | 实现市场环境特征 | features/market_env.py | 本地CPU | M0 |
| 2.4 | 实现特征归一化器 | features/normalizer.py | 本地CPU | 2.1-2.3 |
| 2.5 | 实现统一特征提取入口 | features/extractor.py | 本地CPU | 2.1-2.4 |
| 2.6 | 运行全量特征提取 | data/features/*.parquet | 本地CPU(5min) | 2.5 |
| 2.7 | Notebook: 特征EDA + 相关性分析 | notebooks/03_feature_analysis.ipynb | 本地 | 2.6 |
| 2.8 | 实现Gymnasium交易环境（16.4节） | env/trading_env.py | 本地CPU | 2.6, M1 |
| 2.9 | 实现数据集构造器（轨迹格式） | data/dataset.py | 本地CPU | 2.6, 2.8 |
| 2.10 | 拉取P1层新增特征（moneyflow/涨跌停） | Tushare+AKShare补充数据 | 本地(1h) | 2.5 |

**M2交付标准**:
- `python scripts/extract_features.py` 输出100只×750天×108维（P0）特征矩阵
- Gymnasium环境可正常 `reset() / step()` 循环
- 特征EDA无异常（无全零列、无极端偏态、缺失率<5%）

---

### M3: Phase 1 — Behavioral Cloning（第7-9周）

**目标**: BC baseline，验证数据和特征的有效性。

| # | 任务 | 产出 | 硬件 | 依赖 |
|---|------|------|------|------|
| 3.1 | 实现BC训练器（sklearn + imitation库） | models/bc_trainer.py | 本地CPU | M2 |
| 3.2 | 实现统一评估器 | models/evaluator.py | 本地CPU | M2 |
| 3.3 | 实现回测引擎 | backtest/engine.py + metrics.py | 本地CPU | M2 |
| 3.4 | BC训练: GradientBoosting, P0特征 | exp_001 | 本地CPU(1min) | 3.1-3.3 |
| 3.5 | BC训练: RandomForest, P0特征 | exp_002 | 本地CPU(1min) | 3.1-3.3 |
| 3.6 | BC训练: MLP(sklearn), P0特征 | exp_003 | 本地CPU(5min) | 3.1-3.3 |
| 3.7 | BC训练: P0+P1特征（加入资金流/情绪） | exp_004-006 | 本地CPU | 3.4-3.6, 2.10 |
| 3.8 | Walk-forward交叉验证 | 3折结果 | 本地CPU | 3.4-3.7 |
| 3.9 | Notebook: 模型对比 + 特征重要性 | notebooks/05_backtest.ipynb | 本地 | 3.8 |

**M3交付标准**:
- 买点F1 > 0.35（底线），目标 > 0.45
- 回测年化收益 > 15%（底线），目标 > 25%
- Walk-forward 3折结果稳定（标准差 < 10%）

**M3关键决策点 ⚠️**:

```
F1(买点) ≥ 0.45  →  Phase 2-3 大概率有意义，继续
0.35 ≤ F1 < 0.45 →  尚可，先尝试Phase 2看IRL能否提升
F1 < 0.35         →  特征或标注有根本问题，回到M1/M2迭代
F1 < 0.25         →  方案可能不可行，需重新评估
```

---

### M3.5: Gemini Embedding 验证实验（第8-9周，与M3并行）

**目标**: 验证Gemini Embedding对K线图的理解质量。

| # | 任务 | 产出 | 硬件 | 成本 |
|---|------|------|------|------|
| 3.5.1 | 实现Embedding提取器 | features/embedding.py | 本地+API | — |
| 3.5.2 | 10只股票×250天 Embedding生成 | 2500个768维向量 | Gemini API | ¥3 |
| 3.5.3 | 验证实验1: 形态相似性 | cosine_sim矩阵 | 本地CPU | ¥0 |
| 3.5.4 | 验证实验2: t-SNE可视化 | 趋势聚类图 | 本地CPU | ¥0 |
| 3.5.5 | 验证实验3: KNN买卖点分类 | 准确率 | 本地CPU | ¥0 |
| 3.5.6 | 对比: 有/无Embedding的BC F1差异 | 增量提升% | 本地CPU | ¥0 |

**M3.5决策点**:

```
KNN准确率 > 65% 且 BC F1提升 > 3%  →  全量回填100只Embedding(¥80)
KNN准确率 55-65%                    →  保留方案，Phase 2再评估
KNN准确率 < 55%                     →  放弃Embedding方案
```

---

### M4: Phase 2 — MaxEnt IRL（第9-12周）

**目标**: 从专家轨迹反推奖励函数，学到可解释的"什么时候买卖是好的"。

| # | 任务 | 产出 | 硬件 | 依赖 |
|---|------|------|------|------|
| 4.1 | 实现MaxEnt IRL训练器(numpy版) | models/irl_trainer.py | 本地CPU | M3 |
| 4.2 | MaxEnt IRL训练 + 奖励函数学习 | learned_reward.pkl | 本地CPU(30min) | 4.1 |
| 4.3 | 奖励函数可视化（特征维度贡献） | 可解释性分析报告 | 本地CPU | 4.2 |
| 4.4 | 用学到的R训练PPO策略 | ppo_policy.pkl | 本地CPU | 4.2 |
| 4.5 | 回测 + 与BC baseline对比 | exp_010-012 | 本地CPU | 4.4, 3.3 |
| 4.6 | (可选)AIRL对抗训练 | airl_model/ | AutoDL GPU(6h,¥10) | 4.1 |

**M4交付标准**:
- 学到的奖励函数可解释（能说清哪些特征维度驱动买/卖奖励）
- PPO策略回测年化 ≥ BC baseline × 1.1（至少提升10%）
- MaxEnt IRL在本地CPU上可在1小时内完成训练

---

### M5: Phase 3 — Decision Transformer（第11-14周）

**目标**: 序列建模方式学习交易决策，支持条件生成。

| # | 任务 | 产出 | 硬件 | 依赖 |
|---|------|------|------|------|
| 5.1 | 实现轨迹数据格式化(return-to-go序列) | trajectories.pkl | 本地CPU | M2 |
| 5.2 | 实现DT训练器(d3rlpy) | models/dt_trainer.py | AutoDL | 5.1 |
| 5.3 | DT训练（P0特征） | dt_model/ | AutoDL GPU(12h,¥20) | 5.2 |
| 5.4 | 条件生成测试(目标收益率30%/50%/80%) | 不同目标下的策略差异 | 本地CPU推理 | 5.3 |
| 5.5 | 回测 + 三方法横向对比 | exp_020-025 | 本地CPU | 5.3, M3, M4 |
| 5.6 | (如M3.5通过) DT + Embedding特征 | dt_emb_model/ | AutoDL GPU | 5.3, M3.5 |

**M5交付标准**:
- DT训练收敛（loss稳定下降）
- 条件生成有效（高目标收益 → 更激进策略）
- 三方法对比表填完（16.5节矩阵）

---

### M6: 系统集成与全量验证（第13-16周）

**目标**: 打通从数据到信号的完整pipeline，100只股票全量回测。

| # | 任务 | 产出 | 硬件 | 依赖 |
|---|------|------|------|------|
| 6.1 | 实现推理入口 | inference/predictor.py | 本地CPU | M3-5最优模型 |
| 6.2 | 实现信号过滤（置信度/冷却期） | inference/signal_filter.py | 本地CPU | 6.1 |
| 6.3 | 实现每日推理脚本 | scripts/daily_inference.py | 本地CPU | 6.1, 6.2 |
| 6.4 | 100只股票全量Walk-forward回测 | 全量回测报告 | 本地CPU | 6.1 |
| 6.5 | 回测报告生成（PDF/HTML） | backtest/report.py | 本地CPU | 6.4 |
| 6.6 | 与stockagent-analysis集成接口 | 信号输出 → 现有报告PDF | 本地CPU | 6.3 |
| 6.7 | 信号质量监控（rolling指标计算） | 监控脚本/仪表板 | 本地CPU | 6.3 |
| 6.8 | 文档收尾: README + 使用说明 | README.md | — | 全部 |

**M6交付标准**:
- `python scripts/daily_inference.py` 可在15分钟内完成100只股票的信号生成
- 全量回测年化收益 > 20%，夏普 > 1.5
- 最优模型确定，与stockagent-analysis可联动

---

### 修订后的甘特图

```
周次   1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
M0    [初始化···]                                                    CPU ¥0
M1         ····[标注引擎········]                                     CPU ¥0
M2                   ····[特征工程····]                                CPU ¥0
M3                              [BC训练···]                           CPU ¥0
M3.5                              ··[Emb验证]·                       API ¥3
                                          ↑ 决策点①
M4                                   ····[MaxEnt IRL····]             CPU+GPU ¥10
M5                                             ····[Dec.Trans···]     GPU ¥20
                                                          ↑ 决策点②
M6                                                    ····[集成验证····] CPU ¥0

决策点①(第9周): BC F1 ≥ 0.35 → 继续; < 0.35 → 回到M1/M2
决策点②(第13周): 选定最优模型(BC/IRL/DT) → 全量回测
                  所有模型均不如Buy&Hold → 重新评估方案

总预算: ¥33-113 (取决于Embedding和GPU用量)
```

---

## 19. 项目成功标准与退出条件

### 19.1 成功标准（分级）

| 等级 | 条件 | 含义 |
|------|------|------|
| **A级（完全成功）** | 年化>30%, 夏普>2.0, 回撤<15%, F1(买)>0.50 | 可实盘使用 |
| **B级（基本成功）** | 年化>20%, 夏普>1.5, 回撤<20%, F1(买)>0.40 | 可作为辅助信号 |
| **C级（有价值）** | 年化>10%, 夏普>1.0, F1(买)>0.35 | 证明方向可行，需继续优化 |
| **D级（不可行）** | 年化<10% 或 夏普<1.0 或 F1(买)<0.30 | 方案需根本调整 |

### 19.2 退出/转向条件

| 检查点 | 条件 | 动作 |
|--------|------|------|
| M1完成后 | 标注收益率<30%（约束下无法构造好专家） | 放宽约束 或 改用趋势跟踪策略替代"完美标注" |
| M3完成后 | BC F1<0.25 | 停止IRL路线，转为纯监督学习(LSTM/Transformer直接预测涨跌) |
| M3完成后 | BC F1在0.25-0.35之间 | 改进特征工程(加入P2/P3特征)，而非推进Phase 2 |
| M5完成后 | 三种方法均不如Buy&Hold | 方案整体不可行，总结经验，考虑替代方案 |
| 任何时刻 | 累计花费超过¥500 | 暂停评估ROI |

### 19.3 备选方案（如果IRL路线不可行）

| 替代方案 | 复杂度 | 复用程度 | 说明 |
|----------|--------|---------|------|
| **纯监督学习（LSTM/TCN预测N日涨跌）** | 低 | 特征工程100%复用 | 不需要标注买卖点，直接预测涨跌幅 |
| **因子选股（多因子排序打分）** | 低 | 特征工程100%复用 | 传统量化方法，不需要RL/IL |
| **Embedding相似度纯检索** | 低 | Embedding方案100%复用 | 第15.5节模式B独立运行 |
| **现有29 Agent评分优化（Contextual Bandit）** | 中 | 100%在现有项目内 | docs/contextual-bandit-optimization.md 已有方案 |
