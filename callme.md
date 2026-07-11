# 调用 stock_benchmark 生成每日股票推荐清单

本项目需要从 `D:\aicoding\stock_benchmark` 获取每日最新股票推荐清单。调用流程是：先更新 A 股日线数据到最新可用交易日，再基于最新数据生成统一推荐 CSV。

## 目标

每天在最后一个交易日收盘数据拉取完成后，生成最新股票排序清单。本项目只需要读取生成后的 CSV 文件，并按 `merged_rank` 升序使用。

## 来源项目

```text
D:\aicoding\stock_benchmark
```

## 每日调用流程

### 1. 进入来源项目目录

```powershell
cd /d D:\aicoding\stock_benchmark
```

### 2. 更新 A 股日线数据到最新

```powershell
python scripts\update_lingxi_v2_cn_daily_latest.py --sleep 0.05
```

说明：

- 该步骤会调用 Tushare / 本地数据更新流程。
- 目标是把研究股票池的日线数据更新到当前可获得的最新交易日。
- 建议在 A 股收盘且数据源完成更新后执行。
- 如果数据源当天尚未发布完整数据，后续推荐会继续使用本地最新可用交易日。

### 3. 生成当前全量最优组合统一股票清单

```powershell
python scripts\generate_final_best_combo_stock_list.py
```

说明：

- 该脚本会自动使用本地数据中的最新可用交易日作为 `signal_date`。
- 例如本地最新数据是 `2026-07-07`，则输出文件名中会包含 `2026-07-07`。
- 生成的是收盘后信号清单，通常用于下一交易日开盘前后的外部评分、过滤或执行流程。

## 一条命令完成每日更新和生成

可由本项目或调度器直接执行：

```powershell
powershell -ExecutionPolicy Bypass -Command "cd /d D:\aicoding\stock_benchmark; python scripts\update_lingxi_v2_cn_daily_latest.py --sleep 0.05; python scripts\generate_final_best_combo_stock_list.py"
```

## 主要输出文件

统一推荐清单：

```text
D:\aicoding\stock_benchmark\experiments\final_best_combo_stock_list\final_best_combo_unified_stock_list_YYYY-MM-DD.csv
```

分周期原始清单：

```text
D:\aicoding\stock_benchmark\experiments\final_best_combo_stock_list\final_best_combo_per_horizon_stock_list_YYYY-MM-DD.csv
```

元信息文件：

```text
D:\aicoding\stock_benchmark\experiments\final_best_combo_stock_list\final_best_combo_unified_stock_list_YYYY-MM-DD_meta.json
```

其中 `YYYY-MM-DD` 是信号日期，也就是本地数据中的最新可用交易日。

## 本项目读取方式

1. 进入输出目录：

```text
D:\aicoding\stock_benchmark\experiments\final_best_combo_stock_list
```

2. 查找最新日期的文件：

```text
final_best_combo_unified_stock_list_*.csv
```

3. 选择日期最新的 CSV。

4. 按 `merged_rank` 升序读取。

5. 使用 `ts_code` 作为股票唯一代码。

6. 如只需要前 N 只股票，直接取：

```text
merged_rank <= N
```

## 关键字段

| 字段 | 含义 |
|---|---|
| `merged_rank` | 最终合并排序，`1` 表示最高优先级 |
| `ts_code` | Tushare 股票代码，例如 `688121.SH` |
| `symbol` | 来源项目内部 symbol |
| `stock_name` | 股票名称 |
| `signal_date` | 推荐信号日期，通常应等于最新可用交易日 |
| `recommended_horizons` | 该股票由哪些周期推荐 |
| `strategy_horizons` | 该股票落入哪些周期的回测 TopK；空值表示只是 Top30 候选补位 |
| `hit_count` | 命中的周期数量，越高表示多周期共同推荐 |
| `strategy_hit_count` | 命中回测 TopK 的周期数量，排序时优先级最高 |
| `rank_h5` | 该股票在 H5 推荐列表中的排名，空值表示 H5 未推荐 |
| `rank_h10` | 该股票在 H10 推荐列表中的排名，空值表示 H10 未推荐 |
| `rank_h20` | 该股票在 H20 推荐列表中的排名，空值表示 H20 未推荐 |
| `score_z_h5` | H5 标准化得分 |
| `score_z_h10` | H10 标准化得分 |
| `score_z_h20` | H20 标准化得分 |
| `avg_rank` | 命中周期内的平均排名，越小越好 |
| `rank_score` | 基于排名的综合得分 |
| `score_z_sum` | 多个周期 `score_z` 的合计值 |
| `in_strategy_topk_h5` | 是否进入 H5 回测 TopK |
| `in_strategy_topk_h10` | 是否进入 H10 回测 TopK |
| `in_strategy_topk_h20` | 是否进入 H20 回测 TopK |

## recommended_horizons 可能值

```text
H5
H10
H20
H5;H10
H10;H20
H5;H10;H20
```

## 当前推荐组合

| Horizon | 方法 | TopK |
|---|---|---:|
| H5 | DDG-DA + AlphaAgent imported factors + FAMA seed formulas | 10 |
| H10 | DoubleAdapt + AlphaAgent imported factors + FAMA seed formulas | 20 |
| H20 | AdaRNN + AlphaAgent imported factors + FAMA seed formulas | 10 |

说明：

- 回测策略 TopK 分别是 10 / 20 / 10。
- 为了给外部软件稳定提供 TOP30 评分清单，每个周期会取候选 Top30 合并。
- `strategy_horizons` 和 `in_strategy_topk_h5/h10/h20` 用于识别是否属于回测 TopK 内的强推荐。

## 最终排序规则

统一清单按以下规则排序：

1. `strategy_hit_count` 降序
2. `hit_count` 降序
3. `avg_rank` 升序
4. `score_z_sum` 降序

## 数据新鲜度校验

生成完成后，本项目应读取最新的 `*_meta.json`，检查：

```text
signal_date
data_max_date
```

正常情况下：

```text
signal_date == data_max_date
```

并且该日期应等于最近一个已完成数据更新的交易日。

如果 `signal_date` / `data_max_date` 不是预期的最近交易日，说明数据源尚未更新或本地更新失败，此时不应把该清单当作最新推荐。

## Python 读取示例

```python
from pathlib import Path
import json
import pandas as pd

out_dir = Path(r"D:\aicoding\stock_benchmark\experiments\final_best_combo_stock_list")

csv_files = sorted(out_dir.glob("final_best_combo_unified_stock_list_*.csv"))
csv_files = [p for p in csv_files if not p.name.endswith("_meta.csv")]
if not csv_files:
    raise FileNotFoundError("No stock recommendation CSV found")

latest_csv = csv_files[-1]
date_part = latest_csv.stem.replace("final_best_combo_unified_stock_list_", "")
meta_path = out_dir / f"final_best_combo_unified_stock_list_{date_part}_meta.json"

df = pd.read_csv(latest_csv)
df = df.sort_values("merged_rank")

if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("signal_date") != meta.get("data_max_date"):
        raise RuntimeError(f"stale or inconsistent data: {meta}")

top20 = df[df["merged_rank"] <= 20]
stocks = top20[["merged_rank", "ts_code", "stock_name", "recommended_horizons", "strategy_horizons", "hit_count", "strategy_hit_count"]]
print(stocks)
```
