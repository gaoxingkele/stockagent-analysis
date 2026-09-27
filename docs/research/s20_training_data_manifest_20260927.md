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

## 入库现状（2026-09-27 起）

1. **原始日线已纳入 Git LFS**：`output/tushare_cache/daily/*.parquet` 由 `.gitattributes` 以 LFS 跟踪。
2. **四个派生产物目录也已纳入 LFS**：`output/experiments/s20_uns20_20260927`、`output/experiments/s20_20r_confirmation`、`output/experiments/s20_safe_v4/sources/v4-paired-campaign`、`output/experiments/s20_safe_v4/sources/v4-rsi-campaign`。
3. LFS 总占用约 **1,012 MB**，已贴近免费额度上限（1 GB 存储）；后续大文件需先扩容 LFS 额度。
4. 仍未入库、可由脚本重建或非当前工作：`moneyflow`（约 393 MB）、`s20_v3`（约 1 GB）、`s20_20r_calibration*`（约 500 MB）、`identity-panel`（约 261 MB）、其余 `s20_safe_v4` 运行产物。
5. 仓库 `.git` 已约 1.9 GB，LFS 指针不会继续膨胀 git 历史。
6. Tushare 数据再分发受其服务条款约束；公开仓库上传原始行情前需确认许可。

另一台电脑拉取：

```powershell
git lfs install
git clone https://github.com/gaoxingkele/stockagent-analysis.git
git lfs pull   # 已在 clone 时自动拉取，除非 --skip-smudge
```

注意：拉取全部 LFS 数据约消耗 1 GB 下载流量，与免费额度（每月 1 GB）相当。

结论：GitHub 现在**可以直接满足当前全部训练与复算**（原始日线 + 当前派生产物 + 重建脚本）；数据本体走 LFS，不走 git 历史。
