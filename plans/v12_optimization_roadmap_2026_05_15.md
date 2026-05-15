# V12 优化路线图

**版本**: V12.7 (2026-05-15) → V13
**当前状态**: r10_v16 IC=0.23 / r20_v16 IC=0.32 / V7c 6 铁律 (sell 屏蔽) / LLM 政策面已集成
**已知短板**: 灾难月 202508 -4.95pp 失效; 短期 4 日 alpha 仍为负 -3.20pp; 错过爆发股 (普冉+33%, 澜起+26%)

---

## 排序原则

按 **预期 alpha 改善 / 工作量** 排序。优先做"低投入高回报"任务。

| 优先级 | 类型 | 预期 alpha 改善 |
|---|---|---|
| 🔴 高 | 策略层 + 验证层 + 个股新闻 LLM | +2-4pp/月 |
| 🟡 中 | 模型扩展 + 基本面因子 | +0.5-2pp/月 |
| 🟢 低 | 跨市场 + WEB 交互 | 0-0.5pp/月 |

---

## Sprint 1: 仓位 + 风控 (1 周, 4 工作日)

> 当前最大问题是"V7c 给出选股清单后无仓位/止损规则", 灾难月直接 -5pp。先把基础风控做齐。

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

### 1.4 Regime 触发减仓 (灾难月救命) (2 天) ⭐ 最重要

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

## Sprint 4: 模型扩展 (1-2 周, 7 工作日)

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
| **Sprint 1** 仓位+风控 | 4 天 | +2-3pp/月 (灾难月救命) | ⭐⭐⭐⭐⭐ |
| **Sprint 2** Backtest 引擎 | 5 天 | 0 (基础设施) | ⭐⭐⭐⭐ |
| **Sprint 3** 个股新闻 LLM | 5 天 | +1-2pp/月 | ⭐⭐⭐⭐ |
| **Sprint 4** 模型扩展 | 7 天 | +0.5-1pp/月 | ⭐⭐⭐ |
| **Sprint 5** 长期探索 | 30+ 天 | +0-2pp/月 | ⭐⭐ |

**总工时**: Sprint 1+2+3 = 2 周, 预期实战 alpha 从 +4.83pp/月 提升到 **+7-9pp/月**, 灾难月不再 -5pp。

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

[ ] 立即启动 Sprint 1 (仓位+风控)?
[ ] 跳过 Sprint 2, 直接做 Sprint 3 (个股新闻)?
[ ] 或并行 Sprint 1 + Sprint 3?
[ ] 等 0605 看 V16 真实 alpha 后再决定?

---

**最后更新**: 2026-05-15
**作者**: V12 系统
**关联 commit**: a74015c (V16 切换 + V12.7 上线)
