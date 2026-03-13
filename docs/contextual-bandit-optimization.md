# 方案B：Contextual Bandit 在线参数优化

> 目标：用 Contextual Bandit 算法实现每日在线学习，根据市场状态自动选择最优参数组合
> 与方案A的核心区别：**方案A是批量回测优化（离线），方案B是每日增量在线学习（在线）**
> 两者可组合：方案A做冷启动，方案B做持续适应

---

## 1. 为什么需要方案B

### 1.1 方案A的局限

方案A（Optuna批量优化）在以下场景表现不佳：

| 场景 | 方案A的问题 |
|------|------------|
| 市场风格快速切换（如突然转熊） | 需等60天数据积累才能重新优化 |
| 新参数组合未被探索 | Optuna搜索完就固定，不再探索 |
| 参数对不同市场状态应有不同最优值 | 全局最优 ≠ 当前市场最优 |
| 收盘后需快速给出次日参数 | Optuna需重跑200 trials |

### 1.2 Contextual Bandit 的优势

```
              方案A (Optuna)              方案B (Bandit)
学习模式:     离线批量回测               在线逐日学习
适应速度:     60天窗口                   每天更新，3-5天响应
探索机制:     无（搜索完就固定）          持续探索（ε-greedy / UCB）
上下文感知:   无（全局最优）              有（根据市场状态选参数）
计算量:       200 trials × 全量回放       1次前向推理
```

---

## 2. 核心概念映射

### 2.1 Contextual Bandit 框架

```
经典定义：
  每轮 t=1,2,...:
    1. 观察上下文 context_t     （市场状态）
    2. 选择动作 action_t        （参数组合）
    3. 获得奖励 reward_t        （评分准确度）
    4. 更新策略                 （调整参数偏好）
```

### 2.2 映射到股票评分系统

| Bandit 概念 | 映射 | 具体内容 |
|-------------|------|----------|
| **Context** | 市场状态特征向量 | market_regime(bull/bear/range) + 指数20日收益 + 指数波动率 + 北向资金流向 + 行业轮动状态 |
| **Action** | 参数组合 | 从K个预定义参数集中选一个（如：进攻型/平衡型/防守型/趋势型/震荡型） |
| **Reward** | 评分预测质量 | 当日评分与T+N日实际涨跌的IC + 方向准确率 |
| **Policy** | LinUCB / Thompson Sampling | 学习"什么市场状态下用什么参数组合最好" |

### 2.3 关键洞察：离散动作空间

不直接优化77个连续参数（那是方案A的事），而是：

```
Step 1 (方案A/离线): 用Optuna在不同市场条件下分别优化，得到K=5组参数集
Step 2 (方案B/在线): 每天根据市场状态，用Bandit选择最优的那组参数
```

这样把连续优化问题转化为 **K-armed contextual bandit** 问题，大幅降低复杂度。

---

## 3. 参数集设计（Arms）

### 3.1 五组预定义参数集

每组参数对应一种市场策略假设：

```python
PARAM_PRESETS = {
    "aggressive": {
        # 进攻型：重趋势、重动量、轻防御
        "trend_mom_coef": 1.5,    # 高动量权重
        "trend_trend_coef": 1.0,
        "tech_vr_coef": 15.0,     # 放量敏感
        "bias_penalty_5": -8,     # 轻乖离惩罚（允许追涨）
        "w_TREND": 0.25,
        "w_TECH": 0.15,
        "w_FUNDAMENTAL": 0.03,
        # ...
    },
    "balanced": {
        # 平衡型：当前默认参数
        "trend_mom_coef": 1.0,
        "trend_trend_coef": 0.8,
        "tech_vr_coef": 12.0,
        "bias_penalty_5": -15,
        "w_TREND": 0.18,
        "w_TECH": 0.12,
        "w_FUNDAMENTAL": 0.05,
        # ...
    },
    "defensive": {
        # 防守型：重基本面、重风控、轻动量
        "trend_mom_coef": 0.5,
        "trend_trend_coef": 0.4,
        "tech_vr_coef": 8.0,
        "bias_penalty_5": -25,    # 重乖离惩罚
        "w_TREND": 0.10,
        "w_TECH": 0.08,
        "w_FUNDAMENTAL": 0.15,
        # ...
    },
    "momentum": {
        # 趋势追踪型：极重动量和突破信号
        "trend_mom_coef": 2.0,
        "trend_trend_coef": 1.2,
        "tech_vr_coef": 18.0,
        "bias_penalty_5": -5,     # 几乎不惩罚追高
        "w_TREND": 0.28,
        "w_TRENDLINE": 0.10,
        "w_TIMEFRAME_RESONANCE": 0.12,
        # ...
    },
    "mean_revert": {
        # 均值回归型：重超卖/超买信号、重支撑阻力
        "trend_mom_coef": 0.3,    # 弱化动量（反向思维）
        "tech_vr_coef": 6.0,
        "bias_penalty_5": -30,    # 强烈惩罚追高
        "w_SUPPORT_RESISTANCE": 0.15,
        "w_DIVERGENCE": 0.15,
        "w_BOTTOM_STRUCTURE": 0.10,
        # ...
    },
}
```

### 3.2 参数集来源

- **初始值**：人工设计（基于交易经验）
- **方案A优化**：在不同市场状态的历史子集上分别跑Optuna，自动生成
- **持续迭代**：Bandit学习到某组效果差时，可用方案A重新优化替换

---

## 4. 上下文特征（Context）

### 4.1 市场状态特征向量

每日收盘后提取，作为Bandit的输入上下文：

```python
def extract_market_context() -> dict:
    """提取当日市场状态特征向量。"""
    return {
        # 大盘趋势
        "regime": "bull/bear/range",       # 来自 _detect_market_regime()
        "index_ret_5d": float,             # 沪深300 5日收益
        "index_ret_20d": float,            # 沪深300 20日收益
        "index_vol_20d": float,            # 沪深300 20日波动率

        # 市场情绪
        "advance_decline_ratio": float,    # 涨跌比（涨家数/跌家数）
        "limit_up_count": int,             # 涨停数
        "limit_down_count": int,           # 跌停数

        # 资金面
        "northbound_net_5d": float,        # 北向资金5日净流入(亿)

        # 波动特征
        "vix_proxy": float,               # 波动率指数代理(可用期权隐含波动率)
        "turnover_rate_market": float,     # 全市场平均换手率

        # 行业轮动
        "sector_dispersion": float,        # 行业收益离散度（高=轮动快）
    }
```

### 4.2 特征向量化（供LinUCB使用）

```python
def context_to_vector(ctx: dict) -> np.ndarray:
    """将市场状态转换为数值特征向量。"""
    regime_onehot = {
        "bull": [1, 0, 0],
        "bear": [0, 1, 0],
        "range": [0, 0, 1],
    }.get(ctx["regime"], [0, 0, 1])

    return np.array([
        *regime_onehot,                           # 3维
        ctx["index_ret_5d"] / 10,                 # 归一化
        ctx["index_ret_20d"] / 20,
        ctx["index_vol_20d"] / 5,
        ctx.get("advance_decline_ratio", 1.0),
        ctx.get("northbound_net_5d", 0) / 100,
        ctx.get("sector_dispersion", 0) / 5,
        1.0,                                      # bias项
    ])  # 共10维
```

---

## 5. Bandit 算法选型

### 5.1 三种候选算法对比

| 算法 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| **ε-Greedy** | 以ε概率随机探索，1-ε概率选当前最优 | 最简单，易实现 | 不利用上下文 |
| **LinUCB** | 线性回归+置信上界，利用上下文选arm | 上下文感知，理论保证 | 假设线性关系 |
| **Thompson Sampling** | 贝叶斯后验采样 | 自适应探索，效果最优 | 稍复杂 |

**推荐：LinUCB**，理由：
- 天然利用市场状态上下文（"牛市选进攻型、熊市选防守型"是线性可分的）
- 5个arms + 10维context，规模很小，LinUCB足够
- 有成熟实现（Vowpal Wabbit 或自写100行）
- 自带探索机制（UCB置信上界），无需手调ε

### 5.2 LinUCB 核心逻辑

```python
class LinUCBBandit:
    """
    LinUCB Contextual Bandit for parameter set selection.

    每个arm（参数集）维护一个线性模型：
      predicted_reward = theta_a^T @ context
      UCB = predicted_reward + alpha * confidence_bound

    选择 UCB 最高的 arm。
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        # 每个arm维护 A (d×d矩阵) 和 b (d×1向量)
        self.A = [np.eye(self.d) for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]

    def select_arm(self, context: np.ndarray) -> int:
        """选择UCB最高的arm。"""
        ucb_values = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            pred = theta @ context
            confidence = self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_values.append(pred + confidence)
        return int(np.argmax(ucb_values))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """观察到reward后更新模型。"""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

    def save(self, path: str) -> None:
        """持久化模型状态。"""
        ...

    def load(self, path: str) -> None:
        """加载模型状态。"""
        ...
```

### 5.3 Adaptive Discounted Thompson Sampling (ADTS)

进阶选项：当市场发生结构性变化时，LinUCB可能反应慢（因为历史数据权重均等）。
ADTS通过几何衰减旧数据来加速适应：

```python
class ADTSBandit:
    """
    Adaptive Discounted Thompson Sampling.
    用衰减因子降低旧观测权重，更快适应市场变化。
    """

    def __init__(self, n_arms: int, gamma: float = 0.95):
        self.n_arms = n_arms
        self.gamma = gamma  # 衰减因子：0.95 = 20天半衰期
        # Beta分布参数 (alpha_a, beta_a) for each arm
        self.alpha = [1.0] * n_arms
        self.beta = [1.0] * n_arms

    def select_arm(self) -> int:
        """从后验分布采样选arm。"""
        samples = [np.random.beta(self.alpha[a], self.beta[a])
                   for a in range(self.n_arms)]
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """更新后验，旧数据衰减。"""
        for a in range(self.n_arms):
            self.alpha[a] = self.gamma * self.alpha[a] + (1 if a == arm and reward > 0.5 else 0)
            self.beta[a] = self.gamma * self.beta[a] + (1 if a == arm and reward <= 0.5 else 0)
```

---

## 6. 奖励信号（Reward）设计

### 6.1 延迟奖励问题

Bandit的挑战：选了参数组合后，要等T+5或T+10天才知道评分准不准。

```
Day 1: 选参数集"aggressive" → 给100只股票评分
Day 6: 回填5日涨跌 → 计算IC → 得到Day 1的reward
Day 1的reward要到Day 6才能用于更新模型
```

### 6.2 多时间尺度奖励

```python
def compute_bandit_reward(
    scores: list[float],       # 当日评分
    ret_1d: list[float],       # T+1日涨跌
    ret_5d: list[float],       # T+5日涨跌
    ret_10d: list[float],      # T+10日涨跌
) -> float:
    """
    综合多时间尺度的奖励信号。
    短期奖励（T+1）反馈快但噪声大，长期奖励（T+10）更可靠但延迟高。
    """
    ic_1d = np.corrcoef(scores, ret_1d)[0, 1] if len(scores) > 5 else 0
    ic_5d = np.corrcoef(scores, ret_5d)[0, 1] if len(scores) > 5 else 0
    ic_10d = np.corrcoef(scores, ret_10d)[0, 1] if len(scores) > 5 else 0

    # 加权：短期快速反馈 + 长期稳定信号
    reward = 0.2 * ic_1d + 0.3 * ic_5d + 0.5 * ic_10d

    # 方向准确率加分
    dir_acc = sum(
        (s > 60 and r > 0) or (s < 40 and r < 0)
        for s, r in zip(scores, ret_10d)
    ) / max(len(scores), 1)
    reward += 0.3 * dir_acc

    return max(0.0, min(1.0, reward))  # 归一化到[0,1]
```

### 6.3 快速反馈：盘中奖励代理

不等T+10天，用盘中数据构造"即时奖励代理"：

```python
def compute_intraday_proxy_reward(
    scores: list[float],
    intraday_pcts: list[float],  # 当天盘中涨跌幅
) -> float:
    """
    盘中代理奖励：不用等10天。
    虽然噪声大，但提供快速反馈。
    """
    # 评分高的涨了=好，评分低的跌了=好
    agreement = sum(
        (s > 60 and p > 0) or (s < 40 and p < 0) or (40 <= s <= 60 and abs(p) < 2)
        for s, p in zip(scores, intraday_pcts)
    ) / max(len(scores), 1)
    return agreement
```

### 6.4 奖励时间线

```
Day T:   选arm → 评分100只 → [等待]
Day T+1: intraday_check → proxy_reward (权重0.1, 噪声大但快)
Day T+5: backfill_5d → ic_5d reward  (权重0.3)
Day T+10: backfill_10d → ic_10d reward (权重0.5, 最可靠)
         ↓
     更新Bandit模型（加权组合三个奖励信号）
```

---

## 7. 每日工作流

```
16:00 收盘后：
  ┌─────────────────────────────────────────────────────┐
  │ 1. extract_market_context()                         │
  │    提取今日市场状态 → context向量                    │  ← 秒级
  │                                                     │
  │ 2. bandit.select_arm(context)                       │
  │    LinUCB选择最优参数集                             │  ← 毫秒
  │    输出: "今日推荐参数集=aggressive (UCB=0.73)"      │
  │                                                     │
  │ 3. batch_score(pool, params=selected_preset)        │
  │    用选中参数集对100只股票评分                       │  ← 30秒
  │                                                     │
  │ 4. backfill_returns() + compute_delayed_rewards()   │
  │    回填历史涨跌，计算历史arm的延迟奖励              │  ← 1分钟
  │                                                     │
  │ 5. bandit.update(past_arm, past_context, reward)    │
  │    用延迟奖励更新Bandit模型                         │  ← 毫秒
  │                                                     │
  │ 6. select_top(n=5) → full_analysis (可选)           │
  │    Top候选跑LLM全量分析                             │  ← 可选
  └─────────────────────────────────────────────────────┘

次日 9:30-15:00：
  ┌─────────────────────────────────────────────────────┐
  │ 7. intraday_check()                                 │
  │    盘中预警 + 计算proxy_reward                      │
  │    bandit.update(yesterday_arm, context, proxy)      │
  └─────────────────────────────────────────────────────┘
```

---

## 8. 模块设计

### 8.1 新增文件

```
src/stockagent_analysis/
  ├── bandit.py              # LinUCB / ADTS 算法实现
  ├── market_context.py      # 市场状态特征提取
  ├── param_presets.py       # 5组参数预设定义
  └── bandit_pipeline.py     # Bandit日终流水线

configs/
  ├── param_presets.json     # 5组参数集（可被方案A优化更新）
  └── bandit_state.json      # Bandit模型持久化状态（A矩阵/b向量）

output/
  └── bandit_history.csv     # Bandit决策历史（日期/context/arm/reward）
```

### 8.2 bandit.py — 算法核心

```python
class LinUCBBandit:
    def __init__(self, n_arms: int, context_dim: int, alpha: float = 1.0): ...
    def select_arm(self, context: np.ndarray) -> int: ...
    def update(self, arm: int, context: np.ndarray, reward: float) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
    def arm_stats(self) -> dict: ...         # 各arm累计选择次数/平均reward
    def exploration_rate(self) -> float: ... # 当前探索率估计
```

### 8.3 market_context.py — 市场特征提取

```python
def extract_market_context() -> dict:
    """从akshare提取当日市场状态特征。"""
    # 沪深300指数数据
    # 涨跌家数统计
    # 北向资金
    # 行业离散度
    ...

def context_to_vector(ctx: dict) -> np.ndarray:
    """转换为数值向量供LinUCB使用。"""
    ...
```

### 8.4 param_presets.py — 参数预设管理

```python
PRESET_NAMES = ["aggressive", "balanced", "defensive", "momentum", "mean_revert"]

def load_presets(path="configs/param_presets.json") -> dict[str, dict]: ...
def get_preset(name: str) -> dict: ...
def update_preset(name: str, params: dict) -> None: ...  # 方案A优化后更新
```

### 8.5 bandit_pipeline.py — 日终流水线

```python
def run_bandit_pipeline(
    top_n: int = 5,
    providers: list[str] | None = None,
    skip_full_analysis: bool = False,
) -> dict:
    """
    Bandit日终流水线：
    1. 提取市场上下文
    2. LinUCB选arm
    3. 用选中参数集批量评分
    4. 回填+计算延迟奖励+更新模型
    5. 选Top N候选
    6. (可选) 对Top N跑LLM
    """
```

---

## 9. CLI 命令设计

```bash
# Bandit日终流水线（推荐）
python run.py bandit-run
python run.py bandit-run --top 10 --providers grok,gemini

# 查看Bandit状态
python run.py bandit-status
# 输出:
#   当前推荐参数集: aggressive (UCB=0.73)
#   各arm统计: aggressive(选32次,avg_reward=0.68) balanced(选45次,avg_reward=0.61) ...
#   探索率: 12%
#   累计训练天数: 47

# 手动指定参数集（跳过Bandit选择）
python run.py bandit-run --force-arm defensive

# 重置Bandit模型（市场结构性变化时）
python run.py bandit-reset

# 查看决策历史
python run.py bandit-history --last 20
# 输出:
#   日期        市场状态   选择     奖励    IC
#   2026-03-13  range     balanced  0.65   0.18
#   2026-03-12  range     balanced  0.58   0.12
#   2026-03-11  bull      aggressive 0.72  0.25
```

---

## 10. bandit_state.json 格式

```json
{
  "algorithm": "LinUCB",
  "alpha": 1.0,
  "context_dim": 10,
  "n_arms": 5,
  "arm_names": ["aggressive", "balanced", "defensive", "momentum", "mean_revert"],
  "total_rounds": 47,
  "last_updated": "2026-03-13T16:05:00",
  "arm_stats": {
    "aggressive": {"count": 12, "avg_reward": 0.68, "last_used": "2026-03-11"},
    "balanced": {"count": 20, "avg_reward": 0.61, "last_used": "2026-03-13"},
    "defensive": {"count": 8, "avg_reward": 0.55, "last_used": "2026-03-08"},
    "momentum": {"count": 5, "avg_reward": 0.72, "last_used": "2026-03-10"},
    "mean_revert": {"count": 2, "avg_reward": 0.45, "last_used": "2026-03-05"}
  },
  "model_state": {
    "A_matrices": "base64_encoded...",
    "b_vectors": "base64_encoded..."
  }
}
```

---

## 11. 方案A + 方案B 组合使用

```
┌─────────── 方案A (离线优化) ──────────────┐
│                                            │
│  每周末/月末运行:                           │
│  Optuna优化 → 更新5组参数预设              │
│  configs/param_presets.json                │
│                                            │
└──────────────────┬─────────────────────────┘
                   │ 参数预设（5组arms）
                   ▼
┌─────────── 方案B (在线选择) ──────────────┐
│                                            │
│  每日收盘后运行:                            │
│  市场状态 → LinUCB选最优参数集 → 评分      │
│  reward反馈 → 更新Bandit模型               │
│                                            │
└────────────────────────────────────────────┘

分工:
  方案A: 回答"每组参数应该是什么值"（参数内容优化）
  方案B: 回答"今天该用哪组参数"（参数选择优化）
```

---

## 12. 预期效果

### 12.1 学习曲线

| 阶段 | 天数 | Bandit行为 | 预期效果 |
|------|------|-----------|----------|
| 冷启动 | 1-10天 | 均匀探索5个arm | IC ≈ 方案A水平 |
| 快速学习 | 10-30天 | 开始偏好高reward arm | IC提升10-20% |
| 稳态 | 30天+ | 80%exploitation + 20%exploration | 持续适应市场变化 |

### 12.2 对比基线

| 指标 | 固定参数(现状) | 方案A(Optuna) | 方案A+B(组合) |
|------|---------------|---------------|---------------|
| IC(10日) | ~0.03 | ~0.08 | ~0.12 |
| 方向准确率 | ~52% | ~58% | ~62% |
| 市场切换适应 | 无 | 60天滞后 | 3-5天适应 |
| 每日计算量 | 0 | 5秒(可选) | 30秒+毫秒 |

---

## 13. 依赖与成本

### 新增依赖

```
numpy          # 已有
# 无额外依赖 — LinUCB自写100行，不需要Vowpal Wabbit
# 如需VW: vowpalwabbit>=9.0（可选，更高级功能）
```

### 计算成本

| 操作 | 耗时 | API费用 |
|------|------|---------|
| 市场状态提取 | ~5秒 | 0 |
| LinUCB select_arm | <1毫秒 | 0 |
| 批量评分(100只) | ~30秒 | 0 |
| Bandit更新 | <1毫秒 | 0 |
| Top5 LLM分析(可选) | ~25分钟 | ~$0.5 |

---

## 14. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 5个arm太少，覆盖不够 | 方案A定期优化可扩展到8-10个arm |
| 延迟奖励导致学习慢 | 用盘中proxy_reward加速（权重0.1） |
| 极端市场下所有arm都差 | 设置reward下限，触发时回退到默认balanced |
| 上下文特征获取失败 | 降级为non-contextual（普通ε-Greedy） |
| LinUCB线性假设不够 | 可升级为Kernel UCB或Neural Bandit（Phase 4） |

---

## 15. 实施步骤

### Phase 1: 方案A先行（前提）
- [ ] 完成方案A的 stock_pool + batch_scorer + rl_optimizer
- [ ] 用Optuna在不同市场状态下优化出5组参数集
- [ ] 存入 configs/param_presets.json

### Phase 2: Bandit核心（1次commit）
- [ ] 新建 `bandit.py`（LinUCB实现，~150行）
- [ ] 新建 `market_context.py`（市场特征提取）
- [ ] 新建 `param_presets.py`（参数集管理）
- [ ] 验证：单元测试LinUCB select/update

### Phase 3: 流水线集成（1次commit）
- [ ] 新建 `bandit_pipeline.py`
- [ ] CLI: `bandit-run`, `bandit-status`, `bandit-history`
- [ ] 集成延迟奖励计算和盘中proxy奖励
- [ ] 验证：`python run.py bandit-run --skip-llm`

### Phase 4: 进阶（后续迭代）
- [ ] ADTS算法替代LinUCB（更快适应市场变化）
- [ ] 动态arm数量（自动添加/淘汰参数集）
- [ ] 行业级Bandit（不同行业用不同Bandit实例）
- [ ] 可视化：arm选择频率随时间变化图

---

## 16. 不做的事

1. **不做全市场扫描** — 仅100只池内股票
2. **不直接优化连续参数** — 那是方案A的事，B只做离散选择
3. **不用Vowpal Wabbit** — 5 arms + 10维context，自写LinUCB更轻量
4. **不做高频（日内）切换** — 每日选一次参数集，盘中不切换
5. **不替代方案A** — A优化参数内容，B选择参数组合，两者互补
