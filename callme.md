# 调用 stock_benchmark 生成每日股票推荐清单（池E）

本项目池E需要从 `D:\aicoding\stock_benchmark` 获取每日 Top100 多策略推荐清单。清单由来源项目的独立收盘后任务生产；本项目通过稳定导出 CLI 读取权威发布指针，不扫描目录猜测 latest。

## 来源项目

```text
D:\aicoding\stock_benchmark
```

## 调用入口（由 stock_benchmark 侧调度执行）

触发完整流水线（幂等，当天已生成则跳过）：

```powershell
D:\Python314\python.exe D:\aicoding\stock_benchmark\scripts\run_daily_top100_pipeline.py
```

固定补算某个交易日：

```powershell
D:\Python314\python.exe D:\aicoding\stock_benchmark\scripts\run_daily_top100_pipeline.py --end 20260720
```

同一天强制重新计算：

```powershell
D:\Python314\python.exe D:\aicoding\stock_benchmark\scripts\run_daily_top100_pipeline.py --end 20260720 --force
```

首次生成可能耗时约 30 分钟，外部调用超时建议设置为至少 7200 秒。

Python 外部调用示例：

```python
import subprocess

result = subprocess.run(
    [
        r"D:\Python314\python.exe",
        r"D:\aicoding\stock_benchmark\scripts\run_daily_top100_pipeline.py",
    ],
    cwd=r"D:\aicoding\stock_benchmark",
    timeout=7200,
    check=True,
)
```

## 发布状态检查

触发后先检查两个状态文件：

```text
D:\aicoding\stock_benchmark\experiments\daily_top100_multi_strategy\latest_manifest.json
D:\aicoding\stock_benchmark\experiments\data_update\latest_daily_top100_pipeline.json
```

调用方应要求：

- 进程退出码为 0。
- 状态 JSON 中 `status` 为 `ok`。
- `unique_recommendations` 为 100。
- `signal_date` 是预期交易日。

## 本项目读取方式

稳定读取入口：

```powershell
D:\Python314\python.exe D:\aicoding\stock_benchmark\scripts\export_daily_top100_list.py --group strategy --format json
```

读取规则（`daily_dashboard.py::load_pool_e` 已实现）：

1. 导出器先读取权威 `latest_manifest.json`，再返回其指向的完整 JSON 快照。
2. 本项目复验：100只唯一股票、H5/H10/H20配额30/35/35、15个策略及15份权重、有效 `signal_date`。
3. 按 `overall_rank` 升序排序，使用 `ts_code` 作为唯一代码，取前 `POOL_E_TOPN` 只（默认30）。
4. 成功后同时覆写 `config/pool_e.json` 和 `config/pool_e_meta.json`；任何校验失败都回退最近一份已知良好缓存。

## 关键字段

| 字段 | 含义 |
|---|---|
| `overall_rank` | 去重合并后的总排名，`1` 表示最高优先级 |
| `ts_code` | Tushare 股票代码，例如 `688059.SH` |
| `stock_name` | 股票名称 |
| `signal_date` | 推荐信号日期，应等于最新已完成数据更新的交易日 |
| `assigned_horizon` | 该股票分配到的周期（H5/H10/H20，配额 30/35/35） |
| `top100_strategy_votes` | 命中 Top100 的策略数量（共15个策略） |
| `assignment_score` | 周期分配得分 |
| `overall_consensus` | 多策略一致性得分 |
| `composite_h5/h10/h20` | 各周期综合得分 |

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `POOL_E_EXPORT_SCRIPT` | `D:\aicoding\stock_benchmark\scripts\export_daily_top100_list.py` | 权威稳定导出入口 |
| `POOL_E_PIPELINE_SCRIPT` | `D:\aicoding\stock_benchmark\scripts\run_daily_top100_pipeline.py` | 触发入口 |
| `POOL_E_STATUS_FILE` | `D:\aicoding\stock_benchmark\experiments\data_update\latest_daily_top100_pipeline.json` | 状态 JSON |
| `POOL_E_TOPN` | `30` | 池E取前 N 只（源文件共 100 只） |
