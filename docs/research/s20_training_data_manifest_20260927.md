# S20 训练数据清单与可得性（2026-09-27）

性质：数据资产盘点，不是数据准入结论。H01 的历史可得性证据缺口依然存在（见 `config/s20_v4_data_scope_decision_20260914.json`），本页只回答"训练需要哪些数据、现在在哪、GitHub 上有没有"。

## 结论

GitHub 仓库**不含市场数据**。`output/` 在 `.gitignore` 中被整体排除，仓库只携带代码、配置、文档和少量小型结果文件。训练所需的原始行情与派生产物都在本地 `output/` 下，其中本地 `output/` 总大小约 **10 GB**，仓库 `.git` 对象库已约 **1.9 GB**。

## 训练需要的数据

| 数据 | 本地路径 | 规模 | 是否在 GitHub |
|---|---|---:|---|
| 日线行情（原始） | `output/tushare_cache/daily/*.parquet` | 663 个交易日（2024-01-02～2026-09-24），约 178 MB | 否 |
| S20-20R 确认因子 | `output/experiments/s20_20r_confirmation/factor_groups/` | 15 组 parquet，与 unS20 训练产物合计约 460 MB | 否 |
| unS20 标签与模型 | `output/experiments/s20_uns20_20260927/` | 同上 | 否 |
| 生产模型文件 | `output/production/**` | 部分已跟踪（44 个小型文件） | 是（旧版本） |

unS20 训练入口：`scripts/build_uns20_labels.py`（读日线缓存建 `down_first20` 标签）→ `scripts/train_uns20.py`（三种子 LightGBM + Platt）。

## 时点可得性

按交易所日历（SSE），2026-09-24 是最近的交易日；09-25 为中秋/国庆假期、09-26/27 为周末，故**本地日线已是最新，当前无缺失交易日**。缓存文件本身没有"当日获取"回执，不能据此声称历史可得性已证明。

## 补数与重建

缺日补数（需要 `.env` 中的 `TUSHARE_TOKEN`，该文件不入库）：

```powershell
python scripts/fetch_daily_range.py --start 20260924 --end 20260930
```

脚本只拉取本地缺失的交易日并跳过已存在文件。拉全后按上面两个脚本重建标签和模型。

## 入库现状与限制

1. 原始日线（178 MB）与派生产物（约 460 MB 起）都未入库。
2. 仓库 `.git` 已约 1.9 GB；再纳入原始数据会显著超过 GitHub 的舒适区间（推荐 <1 GB），并进一步拖慢 clone 与 push。
3. Tushare 数据再分发受其服务条款约束；在公开仓库中上传原始行情前需确认许可。

结论：GitHub 目前**不能**直接满足训练；正确路径是"清单 + 补数脚本 + 本地重建"，或把数据包放到 GitHub Release / LFS / 私有存储，而不是塞进 git 历史。
