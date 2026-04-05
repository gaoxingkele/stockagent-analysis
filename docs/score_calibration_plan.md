# 评分系统校准方案

> 基于 2026-04-04 校准报告，223样本，196条有效回填

## 问题诊断

**现象**：评分标准差仅9.3，74%样本挤在45-65观望区，无法产生有效买卖信号。

**根因**：四层平均压缩方差（中心极限定理）
```
单智能体 σ≈30 → 12个加权平均 σ≈12 → 5模型均值 σ≈10 → 辩论混合 σ≈9.3
```

**关键校准发现**：
- 整体IC(5日)=+0.051，IC(10日)=+0.083
- trend_momentum IC=-0.253（反向指标！趋势末端而非起点）
- chanlun IC=+0.159（最有效维度）
- 卖出信号准确率远高于买入信号

---

## 实施计划：方案三 + 方案二（主线）

### Phase 1: 方案三 — 权重校准（基于IC数据驱动）

**原理**：IC高的维度加权，IC负的维度降权或反转。改动最小、确定性最高。

**具体调整**：

| 智能体 | IC(5d) | IC(10d) | 当前权重 | 调整后 | 操作 |
|--------|--------|---------|----------|--------|------|
| trend_momentum | -0.253 | -0.073 | 0.15 | 0.05 (反转) | 降权+反转：`100-score` |
| chanlun | +0.159 | +0.082 | ~0.04 | 0.10 | 加权 |
| divergence | +0.095 | +0.094 | ~0.04 | 0.08 | 加权 |
| fundamental | +0.080 | +0.024 | 0.02 | 0.05 | 加权 |
| capital_liquidity | +0.074 | +0.076 | 0.12 | 0.12 | 保持 |
| kline_vision | +0.048 | +0.111 | ~0.06 | 0.08 | 加权（中期IC好）|
| deriv_margin | +0.002 | +0.034 | ~0.04 | 0.04 | 保持 |
| sentiment_flow | -0.085 | +0.117 | 0.08 | 0.06 | 略降（短期负，中期正）|
| resonance | -0.060 | +0.006 | ~0.04 | 0.03 | 略降 |
| tech_quant | -0.130 | -0.124 | 0.10 | 0.04 | 降权 |
| volume_structure | -0.094 | -0.173 | ~0.04 | 0.02 | 降权 |
| pattern | -0.088 | -0.084 | ~0.04 | 0.02 | 降权 |

**实施方式**：
1. 修改 `configs/agents/{agent_id}.json` 的 `weight` 字段
2. 在 `orchestrator.py` 中对 trend_momentum 启用反转（已有 `_INVERT_DIMS` 机制）
3. 后续每周/每月跑 `backfill_tracking.py --all` 重新校准，持续迭代权重

**trend_momentum 反转的依据**：
- 80-100分段：13只，未来5日全部下跌，均跌8.4%
- 0-20分段：3只，未来5日全部上涨，均涨4.5%
- 本质是"趋势末端检测器"——高分=已涨很多→即将回调，低分=已跌很多→即将反弹
- A股5日维度均值回归效应强于趋势延续
- 注意：20日维度可能转正（趋势延续），待数据积累后分周期验证

---

### Phase 2: 方案二 — 融合层改造

目标：让强信号突破"平均的平均"的束缚。

#### 2a. 模型间聚合：均值 → 加权中位数 + 离群加成

```python
# 现状（orchestrator.py L727）
final_score = sum(model_totals.values()) / len(model_totals)

# 改为
import statistics
scores = list(model_totals.values())
median = statistics.median(scores)

# 离群信号加成：有模型给出极端分时，向极端方向拉
outlier_bonus = 0
for s in scores:
    diff = s - median
    if abs(diff) > 12:  # 离群阈值
        outlier_bonus += diff * 0.25  # 25%的拉力
outlier_bonus /= len(scores)

final_score = median + outlier_bonus
```

**效果**：5个模型共识温和→温和；3个温和+2个极端→向极端方向偏移。保留共识信号。

#### 2b. 关键维度主导机制

当高IC维度给出极端评分时，允许额外拉力突破加权束缚：

```python
# 在计算 total (L724) 之后
KEY_DIMS = {
    "chanlun_agent": 0.15,        # IC最高，拉力最大
    "divergence_agent": 0.12,
    "fundamental_agent": 0.10,
}

bonus = 0
for agent_id, pull_strength in KEY_DIMS.items():
    dim_score = model_scores.get(provider, {}).get(agent_id, 50)
    if dim_score > 75:
        bonus += (dim_score - 75) * pull_strength
    elif dim_score < 25:
        bonus -= (25 - dim_score) * pull_strength

total += bonus
```

**效果**：缠论打90分时，额外加 (90-75)×0.15 = +2.25分。12个维度平均后这2.25分的增量足以把最终分从60推到62-63，进入买入区。

#### 2c. 辩论权重动态化

```python
# 现状：固定 _debate_w = 0.40
# 改为：根据辩论置信度动态调整
_ds = float(_debate_score_raw)
if _debate_decision in ("buy", "sell") and _ds > 75:
    _debate_w = 0.50  # 强信号时辩论权重提升
elif _debate_decision == "hold":
    _debate_w = 0.25  # 犹豫时降低辩论干扰
else:
    _debate_w = 0.40  # 默认
```

**效果**：辩论给出明确观点时放大其声音，犹豫不决时让数据说话。

---

## 备选对照：方案一 — 后置展宽

独立于主线方案，可同时部署用于A/B对比。

```python
# 在 final_score 确定后、输出前
# 用滑动窗口的 running_mean/running_std 自适应展宽
HISTORY_FILE = Path("output/score_stretch_history.json")

def stretch_score(raw_score: float) -> float:
    history = _load_history()  # 最近100次的 final_score
    if len(history) < 30:
        return raw_score  # 冷启动不展宽
    
    mu = statistics.mean(history)
    sigma = statistics.stdev(history)
    
    if sigma < 12:  # 只在分布过窄时展宽
        target_sigma = 18
        z = (raw_score - mu) / sigma
        stretched = 55 + z * target_sigma  # 目标均值55
        return max(5, min(95, stretched))
    
    return raw_score  # sigma已经够大，不处理
```

**关键**：`sigma < 12` 条件使其自适应。如果方案二+三已经把σ提到15+，这段代码自动不生效。

---

## 验证计划

1. **Phase 1 上线后**：跑50只新股票，对比校准前后的 IC 和分数分布
2. **Phase 2 上线后**：同样50只，三组对比（旧/Phase1/Phase1+2）
3. **方案一对照**：单独开一列 `stretched_score`，不影响主评分，纯对比
4. **持续校准**：每周 `python backfill_tracking.py --all`，观察IC趋势

## 注意事项

- 样本集中在2026年3月（单边下跌行情），IC可能有环境偏差
- trend_momentum 反转在牛市中可能失效（趋势延续 > 均值回归），需持续监控
- 权重调整应渐进，每次变动不超过±50%，避免过拟合单一时期数据
