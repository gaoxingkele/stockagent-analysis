# 方案A：本地评分参数RL优化

> 目标：通过历史数据回放，自动优化 `_simple_policy()` 中50+个评分系数，提升评分与实际涨跌的相关性(IC)
> 原则：零LLM调用、零API费用、全本地计算、每日自动迭代
> 范围：100只以内的股票跟踪池

---

## 1. 核心思路

### 1.1 当前评分架构

```
最终评分 = 本地评分(_simple_policy) × 60% + LLM评分(enrich_and_score) × 40%
                    ↑
              RL只优化这部分（纯公式，毫秒级）
```

### 1.2 优化对象

`_simple_policy()` 中所有可调系数，约50+个参数：

```python
# 示例：TREND维度当前固定系数
"TREND": 50 + 1.0*mom + 0.8*trend + 0.3*pct + 0.12*dd + bias_penalty
#             ^^^       ^^^        ^^^        ^^^^
#           全部可优化

# 优化后可能变为：
"TREND": 50 + 1.3*mom + 0.5*trend + 0.4*pct + 0.08*dd + bias_penalty
```

完整参数清单（按维度分组）：

| 维度 | 可调参数 | 数量 |
|------|----------|------|
| TREND | mom系数, trend系数, pct系数, dd系数, bias惩罚阈值(3/5/8), bias惩罚值(-5/-15/-25), 硬封顶阈值 | ~10 |
| TECH | mom系数, trend系数, vr系数, vol系数 | 4 |
| LIQ | vr系数, pct系数, vol系数, turn奖惩 | 4 |
| CAPITAL_FLOW | pct系数, vr系数, news系数, mom系数 | 4 |
| FUNDAMENTAL | pe阈值(6档), pb阈值(5档), ROE/成长性/负债率阈值, chip_bias系数 | ~15 |
| RELATIVE_STRENGTH | 超额收益阈值(6档), RS趋势奖惩, 多层修正系数 | ~10 |
| Agent权重 | 30个Agent的weight | 30 |
| **合计** | | **~77** |

### 1.3 奖励函数(Reward)

```python
def reward(params, history_records):
    """
    params: 候选参数向量
    history_records: [(features, actual_ret_10d), ...] 历史数据
    """
    scores = [simple_policy(features, params) for features, _ in history_records]
    returns = [ret for _, ret in history_records]

    ic = correlation(scores, returns)           # 评分与涨跌相关性
    direction_acc = mean(                        # 方向准确率
        (score > 60 and ret > 0) or
        (score < 40 and ret < 0)
        for score, ret in zip(scores, returns)
    )
    # 综合reward：IC为主，方向准确率为辅
    return 0.6 * ic + 0.4 * direction_acc
```

---

## 2. 股票池设计

### 2.1 池规模与选择

维护一个 **100只以内** 的跟踪池，存储在 `configs/stock_pool.json`：

```json
{
  "stocks": [
    {"symbol": "300274", "name": "阳光电源", "sector": "新能源"},
    {"symbol": "300827", "name": "上能电气", "sector": "新能源"},
    {"symbol": "600519", "name": "贵州茅台", "sector": "白酒"}
  ],
  "updated": "2026-03-13"
}
```

选股建议：
- 覆盖5-8个主要行业（新能源、半导体、白酒、医药、银行、地产等）
- 每行业10-15只，大中小盘均衡
- 包含近期关注的自选股
- 避免ST/退市风险股

### 2.2 池管理CLI

```bash
python run.py pool --add 300274:阳光电源:新能源
python run.py pool --remove 300274
python run.py pool --import watchlist.txt    # 每行: 代码 名称 [行业]
python run.py pool --list
```

---

## 3. 历史回放评分（无LLM）

```
对每只股票的历史每一天：
  1. 从本地K线数据(TDX/AKShare)读取那天的features
  2. 用 _simple_policy(features, params) 算score        ← 纯公式，毫秒
  3. 查表获取实际N日涨跌                                ← 已知结果
  4. 得到 (predicted_score, actual_return) 配对

100只股票 × 60个交易日 = 6000对数据
全部跑完 < 1分钟，零API费用
```

### 数据积累预期

| 时间节点 | 信号量 | 可做什么 |
|----------|--------|----------|
| 第1天 | 100条(无回填) | 只记录 |
| 第5天 | 500条(回填100条) | 初步IC统计 |
| 第10天 | 1000条(回填500条) | **开始Optuna优化** |
| 第30天 | ~2200条(回填1500条) | 参数基本收敛 |
| 第60天 | ~4400条 | 分行业/分市场状态优化 |

---

## 4. 每日工作流

```
收盘后自动执行（约16:00）：

  Step 1: batch_score（30秒）
  ┌────────────────────────────────────────┐
  │  拉取100只股票当日K线+快照             │
  │  _simple_policy() 本地评分             │
  │  记录到 output/pool_signals.csv        │
  └────────────────────────────────────────┘
         ↓
  Step 2: backfill（1分钟）
  ┌────────────────────────────────────────┐
  │  回填历史信号的实际5/10/20日涨跌       │
  └────────────────────────────────────────┘
         ↓
  Step 3: optimize（几秒）
  ┌────────────────────────────────────────┐
  │  读取历史信号+实际涨跌                 │
  │  Optuna优化50+个系数                   │
  │  计算IC/胜率/方向准确率                │
  │  写回 configs/rl_params.json           │
  └────────────────────────────────────────┘
         ↓
  Step 4: select_top（秒级）
  ┌────────────────────────────────────────┐
  │  用优化后参数重新评分100只             │
  │  筛选Top 5-10只候选                    │
  │  输出排名表                            │
  └────────────────────────────────────────┘
         ↓
  Step 5: full_analysis（可选，25-50分钟）
  ┌────────────────────────────────────────┐
  │  对Top候选跑完整LLM分析+PDF           │
  │  生成投资者报告                        │
  └────────────────────────────────────────┘

  次日盘中：
  ┌────────────────────────────────────────┐
  │  intraday_check 盘中预警               │
  │  验证昨日评分与实际走势                │
  └────────────────────────────────────────┘
```

---

## 5. 模块设计

### 5.1 新增文件

```
src/stockagent_analysis/
  ├── stock_pool.py        # 股票池管理（增删/分组/导入导出）
  ├── batch_scorer.py      # 批量本地评分（无LLM，纯_simple_policy）
  ├── rl_optimizer.py      # 参数优化引擎（Optuna）
  └── daily_pipeline.py    # 每日自动化流水线编排

configs/
  ├── stock_pool.json      # 股票池定义（≤100只）
  └── rl_params.json       # RL优化后的参数快照（每日更新，含版本号）

output/
  └── pool_signals.csv     # 池内评分历史（每日100条追加）
```

### 5.2 stock_pool.py — 股票池管理

```python
class StockPool:
    """股票池管理，支持增删改查和分组。"""

    def __init__(self, config_path="configs/stock_pool.json"):
        ...

    def add(self, symbol: str, name: str, sector: str = "") -> None: ...
    def remove(self, symbol: str) -> None: ...
    def list_all(self) -> list[dict]: ...
    def import_from_file(self, path: str) -> int: ...   # 从txt/csv导入
    def export_to_file(self, path: str) -> None: ...
    def get_symbols(self) -> list[str]: ...              # 纯代码列表
    def by_sector(self, sector: str) -> list[dict]: ...  # 按行业筛选
```

### 5.3 batch_scorer.py — 批量本地评分

```python
def batch_score(
    pool: list[dict],           # [{symbol, name, sector}]
    params: dict | None = None, # RL优化参数，None=用默认
) -> pd.DataFrame:
    """
    对池内所有股票执行本地评分（无LLM调用）。

    流程：
    1. 批量拉取当日K线+快照（akshare批量接口）
    2. compute_features() 计算因子
    3. _simple_policy(features, params) 评分

    返回 DataFrame:
      symbol | name | score | decision | TREND | TECH | ... | bias_ma5
    """

def batch_score_historical(
    pool: list[dict],
    start_date: str,
    end_date: str,
    params: dict | None = None,
) -> pd.DataFrame:
    """
    历史回放评分：对每只股票在历史每个交易日重算评分。
    用于RL优化的训练数据生成。

    返回 DataFrame:
      date | symbol | score | TREND | ... | actual_ret_5d | actual_ret_10d
    """
```

### 5.4 rl_optimizer.py — 参数优化引擎

```python
# 依赖: optuna (pip install optuna)

# 参数空间定义
PARAM_SPACE = {
    # TREND维度
    "trend_mom_coef":     (0.3, 2.0),    # 当前1.0
    "trend_trend_coef":   (0.2, 1.5),    # 当前0.8
    "trend_pct_coef":     (0.1, 0.8),    # 当前0.3
    "trend_dd_coef":      (0.05, 0.3),   # 当前0.12
    # TECH维度
    "tech_mom_coef":      (0.2, 1.5),    # 当前0.7
    "tech_trend_coef":    (0.2, 1.2),    # 当前0.6
    "tech_vr_coef":       (5.0, 20.0),   # 当前12
    # ... 其他维度类似
    # bias阈值
    "bias_penalty_3":     (-10, 0),       # 当前-5
    "bias_penalty_5":     (-25, -5),      # 当前-15
    "bias_penalty_8":     (-40, -15),     # 当前-25
    # Agent权重（归一化）
    "w_TREND":            (0.05, 0.30),
    "w_TECH":             (0.03, 0.20),
    # ...
}

class RLOptimizer:
    """基于Optuna的评分参数优化器。"""

    def __init__(self, history_df: pd.DataFrame):
        """history_df: 含 score + actual_ret_10d 的历史数据。"""
        self.history = history_df
        self.study = None

    def objective(self, trial) -> float:
        """Optuna目标函数：用候选参数重算历史评分，与实际涨跌对比。"""
        params = {}
        for name, (lo, hi) in PARAM_SPACE.items():
            params[name] = trial.suggest_float(name, lo, hi)

        # 用候选参数重算所有历史评分
        new_scores = recalc_scores(self.history, params)
        actual_rets = self.history["actual_ret_10d"].values

        # 计算reward
        ic = np.corrcoef(new_scores, actual_rets)[0, 1]
        direction_acc = calc_direction_accuracy(new_scores, actual_rets)
        return 0.6 * ic + 0.4 * direction_acc

    def optimize(self, n_trials: int = 200) -> dict:
        """运行优化，返回最优参数。"""
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=n_trials)
        return self.study.best_params

    def report(self) -> dict:
        """输出优化报告：最优IC、参数变化、提升幅度。"""
        ...

    def save_params(self, path="configs/rl_params.json") -> None:
        """保存最优参数到配置文件。"""
        ...

    def compare(self) -> dict:
        """对比优化前后：IC变化、胜率变化、各维度贡献。"""
        ...
```

### 5.5 daily_pipeline.py — 日终流水线

```python
def run_daily_pipeline(
    top_n: int = 5,
    providers: list[str] | None = None,
    skip_optimize: bool = False,
    skip_full_analysis: bool = False,
) -> dict:
    """
    一键日终流水线：
    1. batch_score: 本地评分100只 (30s)
    2. backfill: 回填历史涨跌 (1min)
    3. optimize: Optuna优化参数 (几秒) [可跳过]
    4. select_top: 筛选Top N (秒级)
    5. full_analysis: 对Top N跑LLM (可选)

    返回: {rankings, optimized_params, top_candidates, analysis_results}
    """
```

---

## 6. CLI 命令设计

```bash
# 股票池管理
python run.py pool --add 300274:阳光电源:新能源
python run.py pool --remove 300274
python run.py pool --import watchlist.txt
python run.py pool --list

# 批量本地评分（无LLM，30秒）
python run.py batch-score
python run.py batch-score --date 2026-03-13

# 参数优化
python run.py optimize --backfill             # 先回填再优化
python run.py optimize --trials 200           # Optuna试验次数
python run.py optimize --report               # 查看优化报告

# 一键日终流水线
python run.py daily-run                       # 默认：评分+回填+优化+Top5
python run.py daily-run --top 10 --providers grok,gemini
python run.py daily-run --skip-llm            # 只做本地，不跑LLM

# 盘中预警（次日）
python run.py intraday-check
```

---

## 7. rl_params.json 格式

```json
{
  "version": 12,
  "optimized_at": "2026-03-13T16:05:00",
  "training_samples": 1500,
  "metrics": {
    "ic_before": 0.12,
    "ic_after": 0.28,
    "direction_accuracy_before": 0.54,
    "direction_accuracy_after": 0.63,
    "reward_improvement": "+38%"
  },
  "params": {
    "trend_mom_coef": 1.32,
    "trend_trend_coef": 0.55,
    "trend_pct_coef": 0.41,
    "tech_mom_coef": 0.85,
    "bias_penalty_3": -7,
    "bias_penalty_5": -18,
    "w_TREND": 0.22,
    "w_TECH": 0.10
  },
  "history": [
    {"version": 11, "date": "2026-03-12", "ic": 0.25, "accuracy": 0.61},
    {"version": 10, "date": "2026-03-11", "ic": 0.22, "accuracy": 0.59}
  ]
}
```

---

## 8. _simple_policy 参数化改造

当前 `_simple_policy()` 中系数是硬编码的，需改造为可接受外部参数覆盖：

```python
def _simple_policy(self, snap, analysis_context, params_override=None):
    """
    params_override: RL优化后的参数dict，覆盖默认系数。
    None时使用原始硬编码值（向后兼容）。
    """
    p = params_override or {}

    # 原来：50 + 1.0*mom + 0.8*trend + 0.3*pct + 0.12*dd
    # 改为：
    base_dim["TREND"] = (50
        + p.get("trend_mom_coef", 1.0) * mom
        + p.get("trend_trend_coef", 0.8) * trend
        + p.get("trend_pct_coef", 0.3) * pct
        + p.get("trend_dd_coef", 0.12) * dd
        + bias_penalty)
```

改造要点：
- 所有硬编码系数替换为 `p.get("key", 默认值)`
- `params_override=None` 时行为完全不变（向后兼容）
- 不改变函数签名对外接口，仅内部参数化

---

## 9. 依赖与成本

### 新增依赖

```
optuna>=3.0       # 参数优化框架（MIT协议）
```

### 计算成本

| 操作 | 耗时 | API费用 |
|------|------|---------|
| 每日批量评分(100只) | ~30秒 | 0 |
| 历史回填 | ~1分钟 | 0 |
| Optuna优化(200trials) | ~5秒 | 0 |
| Top5完整LLM分析(可选) | ~25分钟 | ~$0.5 |

### 数据存储

- `pool_signals.csv`: ~100条/天 × 365天 ≈ 3.6万条 ≈ 5MB/年
- `rl_params.json`: ~2KB/版本，含历史追溯

---

## 10. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 过拟合历史数据 | 分训练集/验证集(80/20)；Optuna early stopping |
| 参数漂移（市场风格切换） | 滚动窗口优化（仅用近60天数据）；牛/熊/震荡分别优化 |
| 优化后反而变差 | 保留原始默认参数作为baseline；每次对比前后IC，变差则回滚 |
| 数据不足时过早优化 | 设门槛：已回填信号 ≥ 500条才启动优化 |
| 前视偏差 | 严格按日期截断K线数据，基本面用已公告财报 |
| 视觉/舆情Agent无法回测 | 对应Agent在回测中权重设为0，仅用于实时分析 |

---

## 11. 实施步骤

### Phase 1: 基础设施（1次commit）
- [ ] 新建 `stock_pool.py` + `configs/stock_pool.json`
- [ ] 新建 `batch_scorer.py`
- [ ] CLI: `pool`, `batch-score` 子命令
- [ ] 验证：`python run.py pool --list` + `python run.py batch-score`

### Phase 2: 优化引擎（1次commit）
- [ ] 新建 `rl_optimizer.py`
- [ ] 改造 `_simple_policy()` 支持 `params_override`
- [ ] 新建 `configs/rl_params.json`
- [ ] CLI: `optimize` 子命令
- [ ] 验证：`python run.py optimize --report`

### Phase 3: 日终流水线（1次commit）
- [ ] 新建 `daily_pipeline.py`
- [ ] CLI: `daily-run` 子命令
- [ ] 集成 batch_score → backfill → optimize → select_top → full_analysis
- [ ] 验证：`python run.py daily-run --top 5 --skip-llm`

### Phase 4: 持续迭代
- [ ] 分市场状态(牛/熊/震荡)分别维护参数集
- [ ] 分行业优化（不同行业不同系数）
- [ ] 优化效果可视化（IC趋势图、参数演化图）

---

## 12. 与现有系统的关系

```
                    ┌──────────────────────────┐
                    │   configs/rl_params.json  │ ← RL每日更新
                    └────────────┬─────────────┘
                                 │ 读取优化参数
                                 ▼
  数据采集 → _simple_policy(params) → 本地评分 → LLM增强 → 最终评分
              ↑                        │
              │                        ▼
          参数化改造              output/pool_signals.csv
          (向后兼容)                    │
                                       ▼
                               rl_optimizer.py
                               (Optuna优化)
                                       │
                                       ▼
                             configs/rl_params.json ← 写入新参数
```

核心特点：
- **无侵入性**：`params_override=None` 时完全兼容现有行为
- **渐进式**：积累数据 → 开始优化 → 验证提升 → 扩大应用
- **零LLM依赖**：优化过程纯本地计算
- **可回滚**：每版参数有版本号，变差即回退

---

## 13. 不做的事

1. **不做深度RL/神经网络** — 参数空间~77维，Optuna完全够用，不需要GPU
2. **不做实时交易** — 系统定位是分析研判，不执行交易
3. **不优化LLM prompt** — prompt优化需要LLM调用，成本高且难以回测
4. **不做个股级别参数** — 统一参数集，避免过拟合到特定股票
5. **不做全市场扫描** — 100只池内跟踪，计算量和信号质量可控
6. **不用OpenClaw-RL** — 当前用付费API而非自托管模型，无法RL微调LLM本身
