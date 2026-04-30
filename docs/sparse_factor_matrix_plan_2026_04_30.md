# 动态稀疏因子矩阵设计方案

> 日期: 2026-04-30
> 背景: 基于 2026-04-29 完成的 153 因子 × 102 万样本回测, 提炼"分层有效"特性, 落地为运行时稀疏打分系统
> 状态: 设计文档, 待实施

---

## 一、需求

### 1.1 用户原话
> 我关注胜率高于 60% 的因子表现条件, 所以评分矩阵, 在每一个时刻 (某日收盘后), 要根据各种组合调节, 只能选择最有效的因子进行评分 (加分或者扣分), 而无效因子不参与评分。

### 1.2 关键约束
1. **稀疏**: 大部分因子在某个具体股票的具体上下文下应该静默, 不参与打分
2. **动态**: 因子集随股票上下文变化 (大盘股 vs 小盘股, 题材股 vs 价值股)
3. **高胜率门槛**: 仅胜率≥60% 的"强信号"段激活
4. **多变量上下文**: PE / 市值 / 板块 / ETF 持仓 / 短期资金流 / 大盘环境
5. **大盘环境**: 当前是 2025-2026 年的"慢牛 + 板块分化"

---

## 二、核心 Insight

传统打分:
```
score = Σ(weight_i × factor_value_i for all i)
```

新设计:
```
score = Σ(weight_i × factor_value_i for i in active_factors(stock, context))
```

激活函数:
```
active(factor, stock) = effective_win_rate(factor, stock_context) >= 60%
```

`stock_context` = 这只股票当下所属的多个分层段 (市值段 / PE 段 / 行业 / ETF 持有 / 资金流状态)

**60% 阈值物理含义**:
- 基准 D+20 胜率 = 55.0% (从 102 万样本算出)
- 60% 胜率 = +5pp = 比抛硬币好得多
- 65% = +10pp = 强信号
- 70%+ = +15pp = 极强信号 (在小盘 Q1 段才出现)

---

## 三、为什么不能直接叉乘所有维度

5 市值 × 6 PE × 110 行业 × 2 ETF × 4 资金流状态 = **26,400** 个 context 桶 × 153 因子 = **400 万格子**

我们只有 102 万样本, 平均**每格 0.25 个样本** → 严重过拟合, 统计无意义。

**正确做法: 各维度独立计算, 运行时合并**

| 矩阵 | 维度 | 格子数 | 平均样本/格 |
|---|---|---|---|
| W_mv  | factor × mv_seg(5) × q(5) | 153×5×5 = 3,825 | ~270 |
| W_pe  | factor × pe_seg(6) × q(5) | 153×6×5 = 4,590 | ~220 |
| W_ind | factor × top30 行业 × q(5) | 153×30×5 = 22,950 | ~30 (够用) |

每格 30+ 样本, 统计意义足够。

---

## 四、运行时合并策略

### 4.1 几何平均 + 阈值

对一只股票, 因子 F 的有效胜率:

```python
W_eff = (W_mv[F][q_bucket] × W_pe[F][q_bucket] × W_ind[F][q_bucket]) ** (1/3)

if W_eff >= 0.60:
    factor 激活
    sign  = +1 if (该股在因子的"高胜率桶") else -1
    delta = (W_eff - 0.55) × scale_factor   # 例如 scale=20: 65% → +2 分
    score += sign × delta
else:
    factor 静默, 不参与
```

### 4.2 为什么用几何平均, 而不是 max / avg

| 合并方法 | 效果 | 评价 |
|---|---|---|
| AND (要求全部维度 >=60%) | 太严, 激活率极低 | ❌ |
| OR (任一维度 >=60%) | 太松, 边缘信号都激活 | ❌ |
| 算术平均 | 强信号被弱信号稀释 | 平庸 |
| **几何平均** | 任一维度低则整体低 (惩罚冲突) | ✅ 推荐 |
| Max | 偏好极致, 忽略冲突 | 过激进 |

### 4.3 因子方向 (sign)

每个因子有"最佳分位桶" (max win 的 Q 几):
- ma_ratio_60 在小盘段最佳桶是 Q1, 大盘段最佳桶是 Q5 → 这个**方向也要存在 validity_matrix 里**
- 运行时: 如果当前股票的 q_bucket == best_q_bucket → sign = +1 (买点); 否则 sign = -1 (卖点) 或 sign = 0 (中间桶不激活)

---

## 五、ETF 持仓 / 资金流的特殊地位

它们**不当作分层维度** (数据稀疏 + 没有相应回测样本), 而是作为**调节器**。

### 5.1 ETF 持仓 → 因子权重调节器

| 情境 | 调节 |
|---|---|
| 被主流宽基 ETF 持有 + 大盘股 | 动量因子 (mfi/rsi24/macd_hist) 权重 ×1.3 |
| 被主流宽基 ETF 持有 + 小盘股 | 反转因子 (channel_pos_60 等) 权重 ×0.7 |
| 不被持有 | 用回测原始权重 (基线) |
| 北向重仓 +5pp 且增持 | 动量因子额外 +10% |

数据源: 复用 `fetch_etf_holdings.py` 已有的 ETF 成分数据。

### 5.2 短期资金流 → 门控信号

资金流不当主因子, 当**信号过滤器** (gating signal):

```python
# 因子: ma_ratio_60 (小盘 Q1)
# 基础: W_eff = 66.1%, 激活 +6 分

if 主力净流入 MA3 > 0 AND 主力连续净流入 ≥ 3 日:
    delta_final = +6 分   # 信号确认
elif 主力净流出 MA3 < 0:
    delta_final = +0 分   # 假信号, 不激活
else:
    delta_final = +3 分   # 未确认, 减半
```

**回测过的资金流分量价值** = 单独 IC 弱, 但作为门控能过滤"假反转"。

### 5.3 大盘环境 → 全局乘数

不分上下文, 对所有因子作用:

```python
regime = {
    "trend": "slow_bull",          # 慢牛
    "dispersion": "high_industry", # 板块分化
}

# 慢牛: 动量类全局 ×1.1, 反转类 ×0.9
# 分化: 行业相对强弱因子 ×1.5
```

**慢牛逻辑**: 长期向上漂移让"涨多了"不一定立刻跌, 反转效应**减弱**, 动量效应**增强**。

**板块分化逻辑**: 行业内强者更强, 弱者更弱。**行业内排名 (rs_industry) 因子权重大幅提升**。

---

## 六、完整数据流

```
┌─────────────────────────────────────────────┐
│  离线 (每月/季度跑一次)                     │
│  102 万 parquet → validity_matrix.json      │
│    ├─ W_mv [factor × mv_seg × q_bucket]     │
│    ├─ W_pe [factor × pe_seg × q_bucket]     │
│    └─ W_ind[factor × industry × q_bucket]   │
│  + best_q_bucket / direction 表             │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  在线 (每只股票每天一次)                    │
│                                             │
│  1. 算 stock_features (153 因子值)          │
│  2. 确定 context: (mv, pe, industry, etf)   │
│  3. 对每个因子 F:                           │
│     a. 找它在这只股票上属于 Q几             │
│     b. 查 W_mv/W_pe/W_ind 三维胜率          │
│     c. 几何平均 W_eff                       │
│     d. W_eff >= 60% → 激活, 加 delta        │
│  4. 资金流门控过滤假信号                    │
│  5. ETF / regime 全局乘数                   │
│  6. 输出: layered_score + active_factors    │
└─────────────────────────────────────────────┘
```

---

## 七、可解释输出 (每只股票)

```json
{
  "stock": "300750.SZ",
  "context": {
    "mv_seg": "1000亿+", "pe_seg": "30-50",
    "industry": "电气设备", "etf_held": true,
    "mf_state": "main_inflow_3d"
  },
  "layered_score": 67.5,
  "regime": "slow_bull + high_dispersion",
  "active_factors": [
    {"name": "mfi_14", "q_bucket": "Q5", "w_eff": 0.628,
     "delta": +5.2, "reason": "大盘+电气设备 MFI 高位 (慢牛动量)"},
    {"name": "macd_hist", "q_bucket": "Q5", "w_eff": 0.612,
     "delta": +4.8, "reason": "大盘 MACD 转强"},
    {"name": "ma_ratio_60", "q_bucket": "Q5", "w_eff": 0.601,
     "delta": +4.5, "reason": "大盘 60日均线偏离 (机构惯性)"}
  ],
  "silent_factors": [
    {"name": "channel_pos_60", "w_eff": 0.512,
     "reason": "大盘段反转无效 (W_eff<60%)"},
    {"name": "trix", "w_eff": 0.488,
     "reason": "大盘段 TRIX 信号弱"}
  ],
  "gates_applied": ["main_inflow_3d_consensus"],
  "regime_multipliers": {"slow_bull": 1.1}
}
```

**特性**:
- **稀疏**: 153 因子里通常只有 5-15 个激活
- **可解释**: 每个 delta 都说得清原因
- **自适应**: 大盘股自动用动量集, 小盘股自动用反转集

---

## 八、与现有系统的关系

不要替换现有 `compute_quant_score`, **新增** `compute_sparse_layered_score`:

```
final = 0.50 × agent_avg            (LLM 4 专家)
      + 0.20 × sparse_layered       ⬅️ 新加, 替代部分 quant 权重
      + 0.10 × quant_score          (旧 4 维)
      + 0.10 × resonance / dissent
      + 0.10 × bonus
```

新分量的核心价值是给 LLM 一个**量化锚** —— LLM 看到"这股票被激活了 8 个胜率 65%+ 的因子"会比看一个孤立 quant_score 数字更有信心。

---

## 九、落地路径 (3 步)

| 步骤 | 工作 | 时间 |
|---|---|---|
| 1. 离线: 从 parquet 算 validity_matrix.json | `factor_lab.py --phase 3` 加一个 phase | ~20 分钟 |
| 2. 在线: 写 `compute_sparse_layered_score` | 查表 + 几何平均 + 门控 + 全局调节 | ~半天编码 |
| 3. 集成: orchestrator 接入 | 在 final score 加新分量 | ~1 小时 |

---

## 十、待决策问题

| 问题 | 倾向 |
|---|---|
| W_ind 用 Top 30 还是全部 110 行业 | **Top 30** (>10000 样本), 其余股票降级到只用 mv+pe |
| 几何平均 vs 加权平均 | **几何** (惩罚冲突, 符合"稀疏"哲学) |
| 阈值 60% 是固定还是动态 | 第一版**固定 60%**, 第二版可按"激活率目标 8-15%"调阈值 |
| ETF 持仓数据从哪取 | 复用 `fetch_etf_holdings.py` 数据 |
| 资金流门控放在因子内还是顶层 | **因子内** (更精细, 不同因子用不同门控规则) |
| validity_matrix 多久更新 | **每月** (季度有点慢, 月度抓住市场风格切换) |
| **是否引入 DS 证据理论** | **见下节专题讨论, 倾向于"借用冲突度 K"轻量集成** |

---

## 十一、DS 证据理论是否引入 (2026-04-30 讨论)

### 11.1 概念契合度

我们的 sparse_layered 设计与 DS (Dempster-Shafer) 证据理论高度契合:

| sparse_layered | DS 证据理论 |
|---|---|
| 多个因子是多个独立证据源 | 多 source evidence |
| 因子激活 / 静默 | mass on hypothesis vs mass on Θ (未知) |
| 几何平均胜率 | 简化版 mass 融合 |
| 60% 阈值 | Belief 阈值 |
| 因子冲突未处理 | DS 的 K 冲突系数 |

### 11.2 DS 能给 sparse_layered 的额外价值

1. **冲突度量 K** ⭐ 最有价值
   - 当多个因子给出冲突信号 (一个说买, 一个说卖), K 越大说明证据矛盾越大
   - 简单加权: "+5 -3 = +2" 看起来有信号
   - DS 下 K 高 → 提示"信号不可靠, 别信"
   - 这是**简单加权做不到的可解释性**

2. **"未知"质量分配**
   - 贝叶斯必须把概率分到具体类别
   - DS 可以为"市场异常情境"分配 mass to {未知}
   - 适合"今天某只股票特征罕见, 没有历史可参考"的场景

3. **三态决策**
   - 我们的输出不必是 [-100, +100] 一维分数
   - 可以是 (Bel(做多), Bel(做空), Bel(观望)) 三元组
   - 决策规则: `if Bel(做多) > 0.4 AND K < 0.3 → 强买; if K > 0.5 → 观望`

### 11.3 DS 的代价

1. **mass 函数定义难** — 怎么把"胜率 66%"映射成 mass on {做多} vs mass on {未知}? 需要工程主观决定
2. **计算复杂** — N 个证据源的 mass 函数定义在 2^|焦元| 上, 全 DS 是指数级
3. **冲突悖论** — 经典 Zadeh 反例 (两个独立诊断都很自信但完全冲突, 经典 DS 给出荒谬结果)
4. **Overengineer 风险** — 我们已经做了 Q 桶分箱 + 上下文分层, 多重处理后再叠加 DS 可能没必要

### 11.4 推荐方案: **轻量集成 — 借用 K, 不上完整 DS**

不全盘 DS, 而是**借用其冲突度 K 概念**作为新分量:

```python
# 把每个激活因子的 (sign, delta) 换算成 mass:
# mass(做多) = max(0, sign × delta) / max_score
# mass(做空) = max(0, -sign × delta) / max_score
# mass(未知) = 1 - mass(做多) - mass(做空)

# 多因子 mass 用 DS 组合规则简化版融合
# 主要计算 K (冲突度):
K = sum(m1(A) × m2(B) for A, B 互斥)

if K > 0.5:
    confidence = "low"   # 因子冲突大
elif K < 0.2:
    confidence = "high"  # 因子高度一致
else:
    confidence = "med"
```

### 11.5 给输出加一个 confidence 字段

```json
{
  "stock": "300750.SZ",
  "layered_score": 67.5,
  "confidence": "high",      ⬅️ 新增
  "conflict_K": 0.18,        ⬅️ 新增
  "active_factors": [...],
  ...
}
```

**用法**:
- LLM 看到 `confidence=low` + `K=0.6` → 知道是矛盾信号, 在分析时会保守
- 排序选股时, 高分但低置信度的股票降权或筛掉
- 给用户的报告里展示"信号一致性"

### 11.6 不引入完整 DS 的理由

我们的场景里:
1. 因子已经经过分层胜率过滤 (大部分冲突在分层后已经消解, 比如小盘和大盘的因子方向冲突, 不会同时出现在一只股票上)
2. mass 函数定义没有理论依据 (拍脑袋设计)
3. 全 DS 计算和调试成本高, 不一定换来 3pp+ 的预测提升
4. **轻量 K 已经能拿到 80% 的可解释性收益**

### 11.7 决策

| 项 | 决策 |
|---|---|
| 完整 DS 替代几何平均 | ❌ 否, overengineer |
| 借用 K 冲突度作为辅助分量 | ✅ 是 |
| 三态决策 (做多/做空/观望) | ⚠️ 第二版考虑 (先用一维 score) |
| mass 函数手工定义 | ✅ 用上述简化版 |

### 11.8 延伸思考 — DS 在 LLM 专家融合层更合适

DS 证据理论的更大价值, 其实在**多 LLM 专家融合**那一层, 而不是因子层:

| 比较维度 | 因子层 (153 因子) | LLM 专家层 (4 专家) |
|---|---|---|
| 证据源数 | 153 (太多, DS 计算成本高) | 4 (TA / Fundamental / Sentiment / Risk) |
| 是否经过预处理 | ✅ Q 桶 + 上下文分层 (冲突大部分消解) | ❌ 各专家独立给信号, 冲突未处理 |
| mass 函数定义 | ❌ 主观 | ✅ LLM 已自带 confidence 输出 |
| Zadeh 悖论风险 | 高 (因子可能极端冲突) | 低 (专家间分歧分布更平滑) |
| 可解释收益 | 中 (K 已能拿大部分价值) | 高 (4 专家分歧是用户最想看的) |

**未来如果在哪里上完整 DS, 应该在 LLM 专家层做**:
- 4 专家 = 4 证据源 (规模合适)
- 每个专家的 confidence 可直接当 mass on {正向}, 1-confidence 可分给 {未知}
- K 冲突度直接量化"专家分歧" — 比现有 dissent_score 更严谨
- 输出 (Bel(买), Bel(卖), Bel(观望)) 给 orchestrator 决策

**推荐路线**:
- v1 (现在): 因子层用几何平均 + 借 K, 专家层维持现有 dissent_score
- v2 (将来): 专家层升级为完整 DS, 因子层维持轻量 K
- 不要在因子层上完整 DS, 这是 ROI 最低的选择

---

## 十二、第一步具体动作

实施步骤:

1. 给 `factor_lab.py` 加 `--phase 3` 子命令, 计算 `validity_matrix.json`:
   - 对每个 factor × (mv_seg / pe_seg / industry) × q_bucket, 算 D+20 胜率
   - 标注 best_q_bucket (该上下文下最佳桶) 和 sign

2. 写 `src/stockagent_analysis/sparse_layered_score.py`:
   - `compute_sparse_layered_score(stock_features, context, validity_matrix)` 返回 layered_score + active_factors + confidence + K

3. 在 `compute_quant_score` 之后, 加一行调用并把 layered_score 合并到 enrich 字段。

4. orchestrator_v3 调整 final 公式:
   - `0.50 × agent_avg + 0.20 × sparse_layered + 0.10 × quant_score + ...`

5. 在结果页 / PDF 里展示 active_factors 和 confidence (LLM context 也用)。

---

*文档版本: 2026-04-30 v1*
*基于回测: 2026-04-29 (102 万样本 / 153 因子)*
*先决条件: factor_lab.py 已建成, 数据可复用*
