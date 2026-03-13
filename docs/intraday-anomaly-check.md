# 方案C：盘中异常预警系统

> 目标：在交易时段对比前日评分与实时走势，自动检测"评分-走势"严重背离，及时预警
> 原则：零LLM调用、零API费用、纯规则引擎、秒级响应
> 状态：**已实现** — `intraday_check.py` + CLI子命令 `intraday-check`

---

## 1. 核心思路

### 1.1 解决的问题

多智能体评分系统在T日收盘后生成评分和建议，但T+1日的实际走势可能与评分严重背离：

| 场景 | 风险 |
|------|------|
| 评分≥70（买入）但T+1跌停 | 追高被套 |
| 评分<40（卖出）但T+1涨停 | 踏空行情 |
| 评分60（观望）但突发利空暴跌 | 未及时止损 |

### 1.2 方案定位

```
T日 16:00    多智能体分析 → 评分/建议写入 signal_history.csv
T+1日 盘中    方案C → 读评分 + 读实时行情 → 规则比对 → 异常预警
                        ↑           ↑              ↑
                    本地CSV     AKShare免费       纯Python
                    (已有)     (1次API全市场)    (0 LLM调用)
```

### 1.3 与方案A/B的关系

| | 方案A (Optuna) | 方案B (Bandit) | 方案C (盘中预警) |
|---|---|---|---|
| 核心功能 | 离线优化评分参数 | 在线选择参数组 | 实时预警评分背离 |
| 执行时机 | 每日收盘后 | 每日收盘后 | 盘中任意时刻 |
| 输入 | 历史评分+收益 | 市场上下文+奖励 | 昨日评分+实时价格 |
| 输出 | 优化后的系数 | 今日最优参数组 | 异常等级+预警消息 |
| LLM费用 | 0 | 0 | 0 |
| 实现状态 | 设计文档 | 设计文档 | **已实现** |

三者互补：A/B改善未来评分，C监控当前评分的实时有效性。

---

## 2. 系统架构

### 2.1 数据流

```
signal_history.csv ──→ load_recent_signals()
                           │
                           │  (symbol, score, decision, close_price)
                           ▼
                      classify_anomaly() ◄── fetch_realtime_snapshot()
                           │                        │
                           │                    AKShare全市场
                           │                   stock_zh_a_spot_em()
                           ▼                   (带缓存，只调1次)
                      AnomalyResult
                           │
                           ▼
                    print_anomaly_table()
```

### 2.2 模块结构

```
src/stockagent_analysis/
├── intraday_check.py          # 核心模块（~250行）
│   ├── AnomalyResult          # 结果数据类
│   ├── load_recent_signals()  # 读CSV、去重、取均值
│   ├── fetch_realtime_snapshot()  # AKShare实时行情（全局缓存）
│   ├── classify_anomaly()     # 规则分类引擎
│   ├── run_intraday_check()   # 主流程
│   └── print_anomaly_table()  # 格式化输出
├── main.py                    # CLI: intraday-check 子命令
└── backtest.py                # signal_history.csv 路径定义
```

---

## 3. 异常分类规则

### 3.1 规则矩阵

| 前日评分 | 前日建议 | 盘中走势 | 异常等级 | 含义 |
|---------|---------|---------|---------|------|
| ≥70 | 买入 | 跌≥5% | 🔴 severe | 评分看多但暴跌，严重背离 |
| ≥70 | 买入 | 跌≥3% | ⚠️ warning | 评分看多但下跌，需关注 |
| <40 | 卖出 | 涨≥8% | 🔴 severe | 评分看空但暴涨，严重背离 |
| <40 | 卖出 | 涨≥5% | ⚠️ warning | 评分看空但上涨，需关注 |
| 其他 | — | — | ✅ normal | 走势符合预期 |

### 3.2 阈值配置

`configs/project.json` → `intraday_check.thresholds`:

```json
{
  "intraday_check": {
    "thresholds": {
      "buy_warn_drop_pct": 3.0,
      "buy_severe_drop_pct": 5.0,
      "sell_warn_rise_pct": 5.0,
      "sell_severe_rise_pct": 8.0,
      "buy_score_min": 70,
      "sell_score_max": 40
    }
  }
}
```

所有阈值均可通过配置文件调整，无需修改代码。

### 3.3 分类逻辑伪代码

```python
if score >= buy_score_min:          # 买入信号
    if pct <= -severe_drop:   → severe  "评分{score}(买入)但盘中跌{pct}%，严重背离"
    elif pct <= -warn_drop:   → warning "评分{score}(买入)但盘中跌{pct}%，需关注"
elif score < sell_score_max:         # 卖出信号
    if pct >= severe_rise:    → severe  "评分{score}(卖出)但盘中涨{pct}%，严重背离"
    elif pct >= warn_rise:    → warning "评分{score}(卖出)但盘中涨{pct}%，需关注"
else:                                # 观望
    → normal "走势符合预期"
```

---

## 4. 数据源

### 4.1 历史评分（输入1）

来源：`output/signal_history.csv`

每次 `run_analysis()` 完成后，`backtest.record_signal()` 自动追加一行：

```csv
date,symbol,name,final_score,final_decision,close_price,provider,...
2026-03-12,300827,上能电气,62.3,hold,18.50,grok,...
2026-03-12,300827,上能电气,58.1,hold,18.50,gemini,...
```

同一只股票同一天可能有多个Provider的评分记录。`load_recent_signals()` 按 `(date, symbol)` 分组取均值。

### 4.2 实时行情（输入2）

来源：AKShare `stock_zh_a_spot_em()` — 东方财富全市场实时快照

```python
# 一次调用获取全A股实时数据（~5000行），全局缓存不重复调用
df = ak.stock_zh_a_spot_em()
# 字段：代码、名称、最新价、涨跌幅、昨收、量比...
```

特点：
- **免费**：东方财富公开接口
- **高效**：1次API调用覆盖所有股票，通过全局 `_spot_cache` 避免重复请求
- **实时**：交易时段每3秒更新（AKShare缓存约30秒延迟）

### 4.3 费用分析

| 组件 | 调用方式 | 次数/天 | 费用 |
|------|---------|--------|------|
| signal_history.csv | 本地文件读取 | 不限 | 0 |
| AKShare全市场快照 | HTTP → 东方财富 | 1次（缓存） | 0 |
| 异常分类 | 本地Python计算 | 不限 | 0 |
| 终端输出 | print() | 不限 | 0 |
| **日均总费用** | | | **¥0** |

---

## 5. CLI使用

### 5.1 命令格式

```bash
# 检查所有最近评分的股票
python run.py intraday-check

# 指定股票代码
python run.py intraday-check --symbols 300827,300274,002571

# 指定信号日期（默认取最新日期）
python run.py intraday-check --date 2026-03-12
```

### 5.2 输出示例

```
==========================================================================================
  盘中异常预警  |  信号日期: 2026-03-12  |  检查时间: 2026-03-13 10:35
==========================================================================================
 #  代码     名称     评分  建议   信号价     现价    涨跌幅  量比  状态     说明
------------------------------------------------------------------------------------------
 1  300274   阳光电源  72.5  买入    48.20    45.60   -5.4%   1.8  🔴严重   评分73(买入)但盘中跌-5.4%，严重背离
 2  300827   上能电气  60.2  观望    18.50    17.85   -3.5%   0.9  ✅正常   走势符合预期
 3  002571   索菲亚    45.8  观望    12.30    12.65   +2.8%   1.2  ✅正常   走势符合预期
------------------------------------------------------------------------------------------
  合计: 3只 | 🔴严重:1 | ✅正常:2
```

### 5.3 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--symbols` | str | 全部 | 逗号分隔的股票代码，不传则检查所有有信号记录的股票 |
| `--date` | str | 最新日期 | 信号日期，格式 YYYY-MM-DD |

---

## 6. 数据结构

### 6.1 AnomalyResult

```python
@dataclass
class AnomalyResult:
    symbol: str              # 股票代码
    name: str                # 股票名称
    signal_date: str         # 信号日期
    signal_score: float      # 前日评分（多Provider均值）
    signal_decision: str     # 前日建议（buy/hold/sell）
    close_at_signal: float   # 信号日收盘价
    current_price: float     # 当前实时价格
    intraday_pct: float      # 盘中涨跌幅(%)
    volume_ratio: float|None # 量比
    level: str               # "severe" / "warning" / "normal" / "unknown"
    message: str             # 预警描述
```

### 6.2 排序规则

输出表按 `(level优先级, 评分降序)` 排列：
```
severe (0) > warning (1) > unknown (2) > normal (3)
```

同等级内，评分高的排前面（高评分股票的背离更值得关注）。

---

## 7. 边界处理

### 7.1 非交易时段

```python
if hour < 9 or hour >= 16:
    print("当前时间不在交易时段(9:30-15:00)，数据可能为昨日收盘价")
```

非交易时段仍可运行，但返回的"实时价格"实际为昨收，涨跌幅≈0%，不会触发异常。

### 7.2 无历史信号

未找到 `signal_history.csv` 或目标日期无记录时：
```
[盘中预警] 未找到匹配的历史信号记录
```

### 7.3 实时数据获取失败

AKShare接口不可用或股票代码未匹配时：
- `level = "unknown"`
- `message = "无法获取实时数据"`

### 7.4 多Provider评分去重

同一股票同一天可能被多个Provider评分，处理逻辑：
1. 按 `(date, symbol)` 分组
2. 取所有Provider的 `final_score` 算术平均
3. 根据均值重新推断建议：≥65 买入 / <40 卖出 / 其他观望

---

## 8. 未来扩展方向

### 8.1 量比异常检测（P1）

当前已获取量比数据但未用于分类，可扩展：

```python
# 量比>3 + 评分买入 + 放量下跌 → 主力出货嫌疑
if volume_ratio > 3 and score >= 70 and pct < 0:
    level = "severe"
    message += "，量比异常(主力出货?)"
```

### 8.2 分时走势监控（P2）

当前只对比单一时刻的涨跌幅，可扩展为分时序列监控：

```python
# 每30分钟检查一次，检测V型反转、高开低走等日内形态
schedule:
  09:45 — 开盘30分钟快照
  10:30 — 上午第一波检查
  13:30 — 午后趋势确认
  14:30 — 尾盘预警
```

### 8.3 与方案A/B联动（P3）

异常预警结果可作为方案A/B的快速反馈信号：

```
方案C检测到背离 → 当日评分标记为"异常"
                → 方案A：该样本在Optuna优化时获得更高损失权重
                → 方案B：Bandit当日reward直接用盘中proxy reward(-1)
```

这比等T+5/T+10收益反馈快得多，可加速参数修正。

### 8.4 推送通知（P4）

严重异常时自动推送：

```
severe → 企业微信/钉钉机器人
warning → 本地弹窗/系统通知
```

### 8.5 自适应阈值（P5）

基于历史波动率动态调整阈值：

```python
# 高波动股票（如日均波幅>5%）放宽阈值
# 低波动股票（如日均波幅<1.5%）收紧阈值
stock_volatility = hist_df["close"].pct_change().std() * 100
if stock_volatility > 5:
    thresholds["buy_severe_drop_pct"] *= 1.5
elif stock_volatility < 1.5:
    thresholds["buy_severe_drop_pct"] *= 0.7
```

---

## 9. 实现清单

### 已完成 ✅

| 模块 | 文件 | 内容 |
|------|------|------|
| 核心引擎 | `intraday_check.py` | AnomalyResult + 5个函数 |
| CLI入口 | `main.py` | `intraday-check` 子命令 |
| 配置项 | `project.json` | `intraday_check.thresholds` |
| 信号记录 | `backtest.py` | `record_signal()` 写CSV |

### 待实现（扩展）

| 优先级 | 功能 | 预估工作量 |
|--------|------|-----------|
| P1 | 量比异常检测 | ~20行 |
| P2 | 分时多轮监控 | ~80行 + cron |
| P3 | 与方案A/B联动 | 依赖A/B实现 |
| P4 | 推送通知（企微/钉钉） | ~60行 |
| P5 | 自适应阈值 | ~40行 |
