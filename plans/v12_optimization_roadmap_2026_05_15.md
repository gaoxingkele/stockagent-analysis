# V12 优化路线图

**版本**: V12.7 (2026-05-15) → V13
**当前状态**: r10_v16 IC=0.23 / r20_v16 IC=0.32 / V7c 6 铁律 (sell 屏蔽) / LLM 政策面已集成
**已知短板**: 灾难月 202508 -4.95pp 失效; 短期 4 日 alpha 仍为负 -3.20pp; 错过爆发股 (普冉+33%, 澜起+26%)

---

## 股票池子分层方案 (7 池: 1 反思 + 6 实战, 2026-05-15 新增, v3 增池 0)

> 核心原则: **每池对应一种明确市场模式 + 不同持有期 + 不同仓位**
> 每股只能进一个实战池 (优先级 池2 > 池1 > 池4/5/6 > 池3), 仓位由所在池决定
> **池 0 是反思学习池**, 与实战池正交 (同一股可能既在 池 0 也在实战池)

### 池子全景

| 池 | 类型 | 对应市场模式 | 判据 | 持有期 | 单股仓位 | 池占比 |
|---|---|---|---|---|---|---|
| **池0 看走眼 (反思)** ⭐ | 学习 | 历史 V12 判错的股 | 见下 §池0 详细 | — (复盘) | **0% (不入仓)** | — |
| **池1 V7c 主推 (核心)** | 实战 | 整理后突破 (旗形/三角形) | V7c 6 铁律全过 | 20 日 | ≤ 2.5% | 50% |
| **池2 三引擎共识 (重点)** | 实战 | 政策驱动 ∩ 技术形态 | V7c 主推 ∩ policy_heat ≥ 70 | 30-60 日 | ≤ 5% | 15% |
| **池3 超跌反弹 (短线)** | 实战 | 短期反弹 (5-10日) | 距 MA60 < -15% + RSI6<30 + 5日量比>1.5 | 5-10 日 | ≤ 1.5% | 5% |
| **池4 底部突破 (左侧)** | 实战 | W底/U底 突破 | 历史 zombie>60% + MA60↑ + 突破 60 日新高 | 20-40 日 | ≤ 2.5% | 10% |
| **池5 政策风口 (主题)** | 实战 | 板块爆发 | policy_heat≥70 + 板块5日涨幅>5% + buy>60 | 10-20 日 | ≤ 1.5% | 10% |
| **池6 强势股回调 (接力)** | 实战 | 龙头休整 | 60日涨幅>30% + 5日跌5~15% + close>MA20 + 缩量 | 10-15 日 | ≤ 2% | 10% |

### 🪞 池 0 看走眼 (反思学习池) — 详细设计

**本质**: 与实战池**正交**, 不参与建仓, 用于**模型迭代 + 风险预警 + 业务透明**。

#### 三类典型"看走眼"

| 类型 | 定义 | 例子 (0508→0514) |
|---|---|---|
| **A. False Positive (推荐却跌)** | V7c 主推 (T 日), 实际 T+20 日 r20 < -3% | 0508 V7c 主推 54 股里 8 只跑输 hs300 -5%+ |
| **B. Missed Rocket (漏判的爆发股)** | T 日不在 V7c 主推 (buy<70 或 sell>30 或矛盾段), 实际 T+20 日涨幅 > +20% | 普冉股份 +32.84% / 澜起 +26.07% / 长川科技 +14.76% |
| **C. LLM Reverse (LLM 视觉判反)** | T 日 V11 视觉 bull < 0.30, 实际 T+20 日涨幅 > +10% | V11 给长川 bear=-12%, 实际涨停 +20% |

#### 量化判据

```python
def find_review_pool(date_range, lookback_days=20):
    """找过去 N 日 V12 看走眼的股."""
    pool = []
    for snap_date in date_range:
        v12_csv = load_v12_inference(snap_date)
        actual_r20 = load_actual_returns(snap_date, snap_date + lookback_days)

        for row in v12_csv:
            real_r20 = actual_r20[row['ts_code']]

            # Type A: 推荐却跌
            if row['v7c_recommend'] and real_r20 < -3.0:
                pool.append({**row, 'wrong_type': 'false_positive',
                             'real_r20': real_r20,
                             'gap': real_r20 - row['r20_pred']})

            # Type B: 漏判爆发
            elif (not row['v7c_recommend']) and real_r20 > 20.0:
                pool.append({**row, 'wrong_type': 'missed_rocket',
                             'real_r20': real_r20,
                             'gap': real_r20 - row['r20_pred']})

            # Type C: LLM 看反 (需 V11 结果)
            v11 = load_v11_result(snap_date, row['ts_code'])
            if v11 and v11['bull_prob'] < 0.30 and real_r20 > 10.0:
                pool.append({**row, 'wrong_type': 'llm_reverse',
                             'real_r20': real_r20,
                             'gap': real_r20 - v11.get('bull_target', 0)})
    return pool
```

#### 池 0 的四大用法

| # | 用法 | 实施 |
|---|---|---|
| 1 | **模型迭代 (hard negative mining)** | 重训时 sample_weight × 2 for 池 0 股, 让模型重点学习这些样本 |
| 2 | **风险预警 (相似度匹配)** | 当前评分高的股, 如果与池 0 历史"看走眼"股 cosine similarity > 0.8 → 减仓 50% |
| 3 | **业务透明度** | WEB 端每周展示 "V12 上周看走眼清单", 用户能看到系统局限性 |
| 4 | **LLM 反思** | 喂池 0 数据给 LLM, 让 LLM 总结"看走眼的共同模式" (如 "化工股集中误判" / "矛盾段+主力流出大涨") |

#### 池 0 输出格式

```json
{
  "snapshot_date": "20260508",
  "ts_code": "300604.SZ",
  "name": "长川科技",
  "v12_score": {
    "buy_score": 81.6, "sell_score": 61.1, "r20_pred": 2.27,
    "v7c_recommend": false, "quadrant": "中性区"
  },
  "actual_outcome": {
    "real_r20": 14.76,  // 实际 20 日涨幅
    "max_gain_20": 22.5,
    "max_dd_20": -2.5
  },
  "wrong_type": "missed_rocket",
  "gap": 12.49,  // 实际 - 预测
  "post_mortem": {
    "key_factors": ["sell_score 偏高", "pyr_velocity 不吸筹"],
    "industry": "半导体",
    "regime": "bull_fast"
  }
}
```

#### 池 0 实施任务

| 任务 | 工时 | 归属 |
|---|---|---|
| `review_pool_builder.py` 模块 | 1 天 | Sprint 1.6 |
| 跑历史 0508 / 0511 / 0512 / 0513 / 0514 找出"看走眼"清单 | 0.5 天 | Sprint 1.6 |
| WEB 端 v12.html 加"看走眼" tab | 0.5 天 | Sprint 1.6 |
| Hard negative mining 接入 V17 重训 | 1 天 | Sprint 4 |
| LLM 反思模块 (post_mortem LLM 总结) | 1 天 | Sprint 3 (顺带做) |

### 池子详细判据 (实现伪代码)

```python
def assign_pool(row):
    """每股按优先级分配到唯一池子."""
    # 池 2 优先 (三引擎共识)
    if row['v7c_recommend'] and row.get('policy_heat_score', 0) >= 70:
        return 'pool2_triple_consensus'
    # 池 1 (V7c 主推)
    if row['v7c_recommend']:
        return 'pool1_v7c_main'
    # 池 4 (底部突破, 左侧)
    if (row.get('zombie_days_pct_60d', 0) > 0.6
        and row.get('ma60_slope_short', 0) > 0.005
        and row['close'] > row['high_60d']):
        return 'pool4_breakout'
    # 池 5 (政策风口)
    if (row.get('policy_heat_score', 0) >= 70
        and row.get('industry_ret_5d', 0) > 0.05
        and row['buy_score'] > 60):
        return 'pool5_policy_wave'
    # 池 6 (强势股回调)
    if (row.get('ret_60d', 0) > 0.30
        and -0.15 < row.get('ret_5d', 0) < -0.05
        and row['close'] > row['ma20']
        and row.get('vol_ratio_5', 1) < 0.8):
        return 'pool6_pullback'
    # 池 3 (超跌反弹)
    if (row.get('close_to_ma60', 0) < -0.15
        and row.get('rsi_6', 50) < 30
        and row.get('vol_ratio_5', 1) > 1.5
        and row['close_change'] > 0
        and not row['is_zombie']):
        return 'pool3_oversold'
    return None  # 不入任何池
```

### 池子与 V12 现有能力对照

| 池 | V12 现状 | 缺什么 | 实现 Sprint |
|---|---|---|---|
| 池 1 | ✅ 已实现 | 无 | 已上线 |
| 池 2 | ✅ 已实现 | 无 | 已上线 |
| 池 3 | ⚠️ 缺 r5 短期模型置信度 | r5_v17_all 模型 | Sprint 4 |
| 池 4 | ⚠️ 缺 "突破 60 日新高" 判据 | 加 break_high_60 / zombie_days_60d 历史 | Sprint 1 加 1 天 |
| 池 5 | ⚠️ 缺板块涨幅统计 | industry_ret_5d 因子 | Sprint 1 加 1 天 |
| 池 6 | ⚠️ 缺历史涨幅 + 缩量缩量判据 | ret_60d / vol_ratio_5 (factor_lab 已有) | Sprint 1 加 1 天 |

### 5 种"继续上涨"模式 -> 池子映射

| 用户模式 | 映射池子 | 量化判据要点 |
|---|---|---|
| 1. 超跌反弹 | 池 3 | RSI6<30 + MA60 偏离 + 5日量比 |
| 2. 筑底突破 (W/U) | 池 4 | 历史 zombie + MA60 转上 + 突破新高 |
| 3. 宏观概念整体涨 | 池 5 (+ 池 2 增强) | LLM 政策面 + 板块涨幅 |
| 4. 整理后突破 (旗形/三角形) | 池 1 (+ 池 2 增强) | V7c 6 铁律 (pyr吸筹+f1/f2静默+非僵尸) |
| 5. 其他 (建议补 模式 6+7) | 池 6 / 待开发 | 龙头回调 / 北向资金 |
| 反思 (历史误判) | 池 0 | 见 §池 0 详细 (false_positive / missed_rocket / llm_reverse) |

### 仓位分配模板 (100% 总仓位)

```
总仓位 100%
├─ 池 1 V7c 主推 (50%) ─── 20 只 × 2.5% [核心]
├─ 池 2 三引擎共识 (15%) ─── 3-5 只 × 3-5% [重点]
├─ 池 4 底部突破 (10%) ─── 4 只 × 2.5% [左侧布局]
├─ 池 5 政策风口 (10%) ─── 7 只 × 1.5% [跟风]
├─ 池 6 强势股回调 (10%) ─── 5 只 × 2% [接力]
└─ 池 3 超跌反弹 (5%) ─── 3 只 × 1.5% [高风险]
```

### 与用户已有 4 池方案对比

| 用户原池 | 数量 | 对应本方案池 | 评价 |
|---|---|---|---|
| 池1 量价异动 (34) | 缓存 | 池 6 强势股回调 (部分) | 范围太窄, 缺技术形态约束 |
| 池2 逆市抗跌 (211) | 缓存 | 池 1 V7c 主推 (太宽) | 211 只过多, 应严格 V7c 6 铁律 |
| 池3 超跌反弹 (1) | 缓存 | 池 3 超跌反弹 | 一致, 但判据可能过严 |
| 池4 强势股回调 (2) | 缓存+5API | 池 6 强势股回调 | 一致 |

**用户 4 池方案缺失**:
- 政策面池 (LLM 数据未利用)
- 底部突破池 (W底/U底)
- 三引擎共识池 (核心仓位)

### 实施清单 (落地到 Sprint)

| 任务 | 工时 | 归属 |
|---|---|---|
| 加 `assign_pool()` 函数到 V12Scorer | 0.5 天 | Sprint 1 |
| 加 industry_ret_5d 因子 (池 5 用) | 0.5 天 | Sprint 1 |
| 加 zombie_days_60d / break_high_60 (池 4 用) | 1 天 | Sprint 1 |
| 加 ret_60d / vol_ratio_5 提取 (池 6 用) | 0.5 天 | Sprint 1 |
| WEB v12.html 加池子筛选 tab (含池0) | 1 天 | Sprint 1 |
| **池 0 看走眼 review_pool_builder** ⭐ | **1 天** | **Sprint 1** |
| **池 0 历史扫描 + WEB 展示** ⭐ | **0.5 天** | **Sprint 1** |
| 池 3 接 r5_v17_all (训完后) | 0.5 天 | Sprint 4 后置 |
| 池 0 Hard negative 接入 V17 重训 | 1 天 | Sprint 4 |
| 池 0 LLM 反思模块 | 1 天 | Sprint 3 顺带 |
| **总计 (Sprint 1)** | **5.5 天** | **Sprint 1 内完成 6 实战池 + 池0** |

---

## 排序原则

按 **预期 alpha 改善 / 工作量** 排序。优先做"低投入高回报"任务。

| 优先级 | 类型 | 预期 alpha 改善 |
|---|---|---|
| 🔴 高 | 策略层 + 验证层 + 个股新闻 LLM | +2-4pp/月 |
| 🟡 中 | 模型扩展 + 基本面因子 | +0.5-2pp/月 |
| 🟢 低 | 跨市场 + WEB 交互 | 0-0.5pp/月 |

---

## Sprint 1: 仓位 + 风控 + 池子分层 (1 周, 9.5 工作日)

> 当前最大问题是"V7c 给出选股清单后无仓位/止损规则", 灾难月直接 -5pp。
> 同时引入 6 实战池 + 池 0 反思池 (共 7 池) 分层架构, 让推荐结构化 + 可复盘。

### 1.1 行业分散硬约束 (半天)

**问题**: V16 0514 Top 5 全部化工原料, 单一板块过度集中风险大。

**方案**: V12Scorer 输出推荐池时, 加 `industry_cap` 参数 (默认 0.30):
- 同一申万二级行业的股票占比 ≤ 30%
- 按 r20_pred 排序后, 超出占比的股票自动剔除

**文件**:
- `src/stockagent_analysis/v12_scoring.py` 加 `apply_industry_diversification(df, cap=0.30)` 方法
- WEB 端 `/api/v12/recommend` 加 `industry_cap` query param

**预期收益**: 回撤减少 5-8%

---

### 1.2 止盈止损规则 (1 天)

**当前**: V7c 推荐后无止盈止损规则, 持有期不明。

**方案 (基于历史 OOS 数据)**:
- **持有期上限**: 20 个交易日 (r20 模型周期)
- **止盈触发**: 实际涨幅 ≥ r20_pred × 80% 即减半仓
- **止损触发**: 实际跌幅 ≥ -8% 即清仓 (历史 OOS dd<-8% 段命中率约 12%)
- **特殊**: zombie 触发 (持有中股票进入横盘僵尸区) 减半

**文件**:
- 新增 `src/stockagent_analysis/position_manager.py`
- 类 `V12PositionManager`: 跟踪持仓 + 自动出场信号

**预期收益**: +0.5-1pp/月 (主要救回 dd<-8% 的损失)

---

### 1.3 Kelly Criterion 动态仓位 (2 天)

**当前**: 单股 ≤5% 固定上限, 无置信度区分。

**方案**:
```python
# 单股仓位 = base_size × confidence_multiplier
# confidence = (r20_pred / r20_p95) × (1 - sell_v6_prob)
# 截断到 [0.5%, 5%]
```
- 三引擎共识股 (V7c+政策) 仓位 × 1.5
- 仅 V7c 主推但无政策面 × 1.0
- 仅 V15 看好但未通过 6 铁律 × 0.5

**文件**:
- `position_manager.py` 加 `calc_kelly_size(score, total_portfolio)`

**预期收益**: +1-2pp/月 (high-conviction 股仓位放大)

---

### 1.4 池子分层架构 (4 天) ⭐⭐ 用户 0515 提出, 必做

**目标**: V12Scorer 输出含 `pool` 字段, 每股归属唯一池子 (优先级 池2>池1>池4/5/6>池3)。

**子任务**:
- (1) 加 `assign_pool()` 函数到 V12Scorer (0.5 天)
- (2) 加 industry_ret_5d 因子 (0.5 天, 池 5)
- (3) 加 zombie_days_60d / break_high_60 (1 天, 池 4)
- (4) 加 ret_60d / vol_ratio_5 因子完善 (0.5 天, 池 6)
- (5) WEB v12.html 加池子筛选 tab + 推荐分池展示 (1 天)
- (6) 加池子-仓位映射规则到 position_manager.py (0.5 天)

**预期收益**: 推荐清晰分层, 避免混仓; 灾难月可按池子选择性减仓 (例如保留池 2 政策共识, 减仓池 3 超跌反弹)。

---

### 1.6 池 0 看走眼池 (反思学习池) (1.5 天) ⭐⭐ 用户 0515 提出, 必做

**目标**: 每日扫描历史 V12 评分 vs 实际表现, 找出"看走眼"的股, 用于复盘+模型迭代。

**子任务**:
- (1) 新建 `src/stockagent_analysis/review_pool_builder.py` (1 天)
  - `find_v12_wrongs(snap_date, lookback=20)` 找三类误判
  - 输出 `output/v12_review/wrongs_{date}.parquet`
- (2) 跑历史 0508/0511/0512/0513/0514 五日清单 (0.5 天)
  - 输出累积 review_pool.csv

**Sprint 3/4 后置**:
- LLM 反思模块 (1 天, Sprint 3)
- Hard negative mining 接入 V17 重训 (1 天, Sprint 4)

**预期收益**:
- 业务透明度 (用户看到模型局限)
- V17 重训时 hard negative 加权可能 +0.5-1pp/月
- 风险预警: 当前推荐与历史误判相似度匹配减仓

---

### 1.5 Regime 触发减仓 (灾难月救命) (2 天) ⭐ 最重要

**问题**: 202508 月度 -4.95pp 系统性失效, V7c 无法识别。

**方案**:
- 监控 `regime_id` 切换 (5 个状态: bull_policy / bull_fast / bull_slow_diverge / bear / sideways / mixed)
- 触发条件: 连续 2 日 regime_id ∈ {bear, sideways} AND hs300 当日跌幅 > -2%
- 行动: 减仓至 30% (而非 100%)
- 退出: regime_id 回到 bull_* 状态恢复满仓

**文件**:
- `src/stockagent_analysis/regime_monitor.py`
- 集成到 V12Scorer.score_market() 输出 `recommended_position_ratio` 字段

**预期收益**: 灾难月 +3pp (从 -4.95 救回到 -2pp 左右)

---

## Sprint 2: 真实 Backtest 引擎 (1 周, 5 工作日)

> V12 当前回测是"模型 IC OOS",  不是"模拟交易"。无法验证策略层改进 (止盈止损/仓位/regime) 的实战效果。

### 2.1 backtest_v12.py 模拟交易引擎 (3 天)

**功能**:
- 输入: 起始日期 / 结束日期 / 初始资金 / 策略参数
- 流程:
  ```
  for date in trading_days:
    1. 跑 V12Scorer.score_market(date)
    2. 应用行业分散 + Kelly Criterion
    3. 检查持仓股的止盈止损
    4. 计算当日 PnL + 净值
    5. regime 监控
  返回: 净值曲线 + 回撤曲线 + 交易日志
  ```

**约束模拟**:
- T+1 (买入次日才能卖)
- 滑点 0.1% (开盘买 + 收盘卖)
- 手续费 0.05% 双向
- 涨跌停限制 (不能买/卖涨跌停板)

**文件**:
- `src/stockagent_analysis/backtest_engine.py`
- `scripts/run_backtest.py` 入口

---

### 2.2 多策略对比 (1 天)

跑 5 组对比:
- V4 baseline (旧模型)
- V15 (新模型, 无策略)
- V16 (最新模型, 无策略)
- V16 + Sprint 1 全套 (行业分散 + 止盈止损 + Kelly + regime)
- V16 + Sprint 1 + 个股新闻 LLM (Sprint 3 后再补)

输出对比表 + 净值曲线图。

---

### 2.3 净值曲线可视化 (1 天)

WEB 端 `/v12/backtest` 页面:
- 上传策略参数 (或选预设)
- 显示净值曲线 / 回撤曲线 / 月度收益柱状图
- 与 hs300 / cyb / zz500 对比线

---

## Sprint 3: 个股新闻 LLM (1-2 周, 5 工作日)

> 当前 LLM 只看 cctv 政策面 (板块级), 个股级催化未利用。

### 3.1 数据源接入 (2 天)

**akshare 个股新闻**:
```python
import akshare as ak
df = ak.stock_news_em(symbol="300604")  # 长川科技
# 返回: 标题, 时间, 内容, 来源
```

**覆盖度** (按当日 V12 主推 + 矛盾段, 约 200-300 股):
- 主推股 (~84): 必跑
- 矛盾段 (~200): 可选, V11 视觉曾在矛盾段有效

**成本估算**: 200 股 × $0.005 = $1/日

---

### 3.2 LLM 情感+催化分析 (2 天)

`src/stockagent_analysis/stock_news_llm.py`:

输入: 单股最近 7 日新闻
LLM Prompt 输出:
```json
{
  "sentiment_score": -1.0 ~ +1.0,
  "catalyst_type": "业绩预增" | "股权激励" | "回购" | "中标订单" | "诉讼" | "无",
  "key_events": ["事件1", "事件2"],
  "hotness": 0-100  // 新闻数量+情感强度
}
```

集成到 V12Scorer:
- 加 3 字段: `news_sentiment` / `catalyst_type` / `news_hotness`
- 触发 `catalyst_type ∈ {业绩预增, 中标订单}` 时, buy_score 加 5 分

---

### 3.3 爆发股二分类器 (1 天)

**问题**: V7c r20 连续模型对极端涨幅 (+20%) 识别不足。

**方案**: 训练独立二分类器
- label: `max_gain_20 >= 20.0`
- 训练: 20230101-20260413 (与 V16 同窗口)
- 输出: P(超大涨)

加进 V12 综合分:
- 当 `boom_proba > 0.8` 且 V7c 6 铁律全过 → 标 ⭐⭐⭐ 重点推荐

---

## Sprint 4: 多尺度共识 + 1H 向下钻取 (重组 2026-05-17)

> **核心创新** (用户 0517 提出):
> - 嵌套预测窗口: r5/r10/r20 三层独立, 而非当前 r10+r20 强制等权
> - 1H 跨尺度: **不全市场跑 1H 模型** (算力爆炸), 而是对**日线已推荐的 80 只**做 1H 二次验证 (向下钻取)
> - 实现"宽度选股 (日线) + 深度选时 (1H)" 的工程最优

### 4.1 训练 r5_v17 短期模型 (2-3 小时) ⭐⭐⭐

**目标**: 训练 5 个交易日 forward 预测 (r5)

**实施**:
- forward label: r5 = close[t+5] / open[t+1] - 1 (从 daily cache 现算)
- 232 因子 + 时序 split (后 10% val)
- 模型 r5_v17_all 输出 IC + 锚点

**预期**:
- r5 IC > 0.20 (比 r20 IC 略低, 因为短期噪声大)
- 锚点 ~[-2%, 0.5%, 4%] (r5 幅度比 r20 小 3-4 倍)

---

### 4.2 V12Scorer 拆分三层 buy/sell (1 小时)

**当前**: `buy_score = 0.5 × score(r10) + 0.5 × score(r20)`
**改造**:
```
buy_r5_score   = anchor_map(r5_pred, R5_ANCHOR)
buy_r10_score  = anchor_map(r10_pred, R10_ANCHOR)
buy_r20_score  = anchor_map(r20_pred, R20_ANCHOR)
buy_score (向后兼容) = 0.5 × buy_r10 + 0.5 × buy_r20   # 老逻辑保留
```

---

### 4.3 多窗口池分级 (2 小时)

**信号金字塔**:
| r5 | r10 | r20 | 新池子 | 含义 |
|---|---|---|---|---|
| 高 | 高 | 高 | `pool7_pyramid_strong` | 全周期共识 → 高仓位 ★★★ |
| 低 | 中 | 高 | `pool8_pullback_long`  | 先跌后涨 → 等回调 ★★ |
| 高 | 中 | 低 | `pool9_pulse_short`    | 短线脉冲 → 5 日快进快出 ★ |
| 高 | 低 | 低 | (已 V7c 排除) | 上涨过头 |

集成到 pool_classifier.py, 新加 3 池.

---

### 4.4 backtest 验证多窗口收益 (1 天)

跑 60 日 OOS, 对比:
- V12.12 (老 buy_score 等权混合)
- V12.16 (多窗口共识)

预期 alpha +1pp/月 (从 4 池升级到 9 池, 信号区分度提升)

---

### 4.5 1H 向下钻取 (3-4 天) — 用户 0517 创新 ⭐⭐⭐

**核心架构**:
```
Layer 1 (日线宽度): 全市场 5000+ 股 → V7c 主推 ~80
Layer 2 (1H 深度): 仅对这 80 只跑 1H 模型 → 二次验证
```

**1H 模型范围**:
- 训练数据: 仅 V7c 主推过的历史股 (~500-1000 只独立股)
- 1H 因子: 232 因子里**仅价格/量能类** (约 80 个, 不需要全部 232)
- 1H labels: r20_1h = 20 个 1H bar ≈ 5 日 forward return
- 与日线 r5 交叉验证: 同 5 日窗口, 两模型预测应一致

**实施**:
- (1) Tushare `pro_bar` 拉 1H 数据 (V7c 候选池历史 + 当前) - 1 天
- (2) 重新计算 1H 价格/量能因子 (MA/RSI/MACD/BOLL 等 80 个) - 1 天
- (3) 训练 r20_1h 模型 (1H × 20 bar = 5 日, 与日线 r5 同窗口) - 0.5 天
- (4) 二次验证流水线: 日线推荐 + 1H 钻取双重共识 - 0.5 天
- (5) backtest 集成 - 1 天

**输出**: V12 推荐池每股加 `r20_1h_pred` / `dual_consensus` (日线 r5 vs 1H r20 是否同向)

---

### 4.6 (旧) 基本面季报因子 (2 天)

接入 Tushare `pro.fina_indicator()` ROE/ROA/营收增速.
作为 232 → 240+ 因子池, V18 重训.

---

### 4.7 (旧) Regime-Conditional 模型 (2 天)

5 个 regime 训练 5 套子模型, 推理时按当日 regime 切换.

---

## 旧 Sprint 4 内容 (回收, 不删除)

### 4.1 多周期模型集成 (3 天)

- r5_v17_all (5 日短期波段, 抓爆发首日)
- r60_v17_all (60 日中期趋势, 抓持续主升)

集成方法:
```python
v12_total_score = 0.2 * buy_r5 + 0.5 * buy_r10 + 0.3 * buy_r60
```

---

### 4.2 基本面季报因子 (2 天)

接入 Tushare:
- `pro.fina_indicator()` ROE/ROA/营收增速/利润增速
- `pro.income()` 利润表

加进 232 → 240+ 因子池, V17 重训。

---

### 4.3 Regime-Conditional 模型 (2 天)

针对 5 个 regime 状态训练 5 套子模型:
- bull_policy → r10_v17_bp / r20_v17_bp
- bull_fast → r10_v17_bf / r20_v17_bf
- ...

V12Scorer 推理时按当日 regime 自动切换。

预期: regime_0 (灾难月) +2pp; bull_fast +1pp。

---

## Sprint 5: 长期探索 (Q3+)

### 5.1 跨市场映射
- AH 溢价: 同公司 A股/港股价差
- 中概股映射: 百度→搜索/AI/广告板块

### 5.2 研报情感 LLM
卖方研报评级变化 (从"持有"升"买入" → 短期催化)

### 5.3 期权情绪
50ETF / 300ETF 期权 IV / Skew → 隐含恐惧/贪婪指数

### 5.4 WEB 高级交互
- 实时推送 (WebSocket)
- 单股画像页面 (K线 + V12 评分 + 政策 + 新闻一体化)
- 用户自定义回测页面

---

## 工时与收益总览

| Sprint | 工时 | 预期 alpha 改善 | ROI |
|---|---|---|---|
| **Sprint 1** 仓位+风控+7池子 (含池0) | **9.5 天** | +2-3pp/月 + 结构化推荐 + 模型迭代基础 | ⭐⭐⭐⭐⭐ |
| **Sprint 2** Backtest 引擎 | 5 天 | 0 (基础设施) | ⭐⭐⭐⭐ |
| **Sprint 3** 个股新闻 LLM | 5 天 | +1-2pp/月 | ⭐⭐⭐⭐ |
| **Sprint 4** 模型扩展 (r5 + 基本面 + Regime条件模型) | 7 天 | +0.5-1pp/月 | ⭐⭐⭐ |
| **Sprint 5** 长期探索 | 30+ 天 | +0-2pp/月 | ⭐⭐ |

**总工时**: Sprint 1+2+3 ≈ 18 天 (~3.5 周), 预期实战 alpha 从 +4.83pp/月 提升到 **+7-9pp/月**, 灾难月不再 -5pp。

---

## 推荐启动: Sprint 1.4 (Regime 触发减仓)

最高 ROI 单点: 仅 2 天工作量, 解决 V7c 唯一的硬伤 (灾难月 -4.95pp)。

**第一步代码**:
```python
# src/stockagent_analysis/regime_monitor.py
class RegimeMonitor:
    def get_recommended_position(self, date: str) -> float:
        """返回当日建议持仓比例 (0.0 ~ 1.0)."""
        rg = pd.read_parquet("output/regimes/daily_regime.parquet")
        recent = rg[rg["trade_date"] <= date].tail(5)
        bear_days = (recent["regime"].isin(["bear", "sideways"])).sum()
        if bear_days >= 3:
            return 0.30  # 减仓到 30%
        if bear_days >= 2:
            return 0.60  # 减仓到 60%
        return 1.0  # 满仓
```

---

## 决策清单

[ ] 立即启动 Sprint 1 (仓位+风控+6池子)?
[ ] 跳过 Sprint 2, 直接做 Sprint 3 (个股新闻)?
[ ] 或并行 Sprint 1 + Sprint 3?
[ ] 等 0605 看 V16 真实 alpha 后再决定?

---

## 修订历史

| 日期 | 版本 | 改动 |
|---|---|---|
| 2026-05-15 | v1 | 初版 (5 Sprint, 总览) |
| 2026-05-15 | v2 | 加 "股票池子分层方案" (6 池) + Sprint 1 加 1.4 池子分层任务 |
| 2026-05-15 | v3 | 增加池 0 看走眼 (反思学习池) + Sprint 1 加 1.6 任务 (1.5 天) |
| 2026-05-17 | v4 | Sprint 1 + 2 + 3 全部完成. 重组 Sprint 4 为 "多尺度共识 + 1H 向下钻取" |

**关联 commit**: a74015c (V16 + V12.7), e8c80e4 (路线图初版)
