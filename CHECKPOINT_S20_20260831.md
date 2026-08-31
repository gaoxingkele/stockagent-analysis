# S20 checkpoint — 2026-08-31

## 当前目标

S20 是 R20 之外的独立研究指数。首要业务目标是每天从全市场找到少量“未来 20 个交易日
上涨达标置信度最高，同时先触发下跌风险概率较低”的股票，不替代、不修改 R20。

目标曲线：

```text
S20-15 = P(20日内先触达 +15%，而非先触达 -15%)
S20-20 = P(20日内先触达 +20%，而非先触达 -15%)
S20-25 = P(20日内先触达 +25%，而非先触达 -15%)
Risk20 = P(20日内先触达 -15%)
```

主评估改为每日 Precision@10/20/50、lift、最差折/市场阶段、正命中交易日比例；
Brier/ECE 是把分数解释为“置信度”的必要约束。没有行业数量上限，也不按趋势、反弹或
形态做硬筛选。

## Git 状态

- 工作仓库：`D:/aicoding/stockagent-analysis`
- 当前分支：`experiment/s20-first-passage-v1`
- 已推送最新提交：`aea6c2e research: align S20 with high-confidence stock selection`
- 上一核心提交：`afb88ee research: validate S20 competing-risk baseline`
- 生产基线仍为 `publish/web-fourpool` 的 `0cff868`；S20 没有生产影响。
- 工作区仅有 `config/pool_e_meta.json` 的运行时间戳改动，属于既有日更产物，未提交、勿覆盖。

## mylib 状态

- 路径：`D:/aicoding/mylib`
- `main` 已从 GitHub 快进到 `e3b5eaa`，与 `origin/main` 同步且干净。
- 已安装 `uv/uvx 0.12.7`。
- `paper_search` 的 31 项自测通过；OpenAlex、arXiv、Crossref 可用。
- Semantic Scholar 因当前进程无 `S2_API_KEY` 会 429。
- OpenReview 因 Python 3.14 下 `editdistance` 依赖无法构建，当前不可用。
- Windows 运行检索前设置 `$env:PYTHONUTF8='1'`，否则作者 Unicode 姓名可能触发 GBK 错误。
- 完整 40 篇检索记录：`allinone.md`；文献路线：
  `docs/research/s20_literature_and_algorithm_roadmap_20260831.md`。

## 已完成：标签

实现：

- `src/stockagent_analysis/s20.py`
- `scripts/build_s20_first_passage_labels.py`
- `config/s20_first_passage_label_audit.json`

全量结果：

- 3,200,156 条路径，5,299 只沪深证券，零重复。
- 信号日 2024-01-02 至 2026-07-31，路径数据到 2026-08-28。
- +25% 先触达 14.396%，-15% 先触达 19.911%，未触达 65.692%。
- +25% 同日双触达歧义 0.000125%。
- 先达 +25% 后又破 -15% 占 0.288%。
- 与旧标签重叠 2,441,863 行：entry_open 完全一致；max gain/dd/r20 close 最大差异
  0.00005，等于旧标签四位小数舍入上界。

本地产物（gitignored）：

```text
output/experiments/s20_first_passage_v1/first_passage_labels.parquet
output/experiments/s20_first_passage_v1/label_audit.json
```

## 已完成：B1 竞争风险模型

实现：`scripts/explore_s20_b1_competing_risk.py`。

口径：50% 固定哈希样本，与旧 B0 的 350,608 条测试预测逐行一致；将 20 日拆为
`1 / 2-3 / 4-5 / 6-10 / 11-15 / 16-20` 六个离散区间，每段预测无事件、上涨
先触达、下跌先触达三个 conditional hazard，再累积为 CIF。

汇总结果：

| 候选 | Brier Skill | ECE | ROC-AUC | PR-AUC | 顶十分位命中率 |
|---|---:|---:|---:|---:|---:|
| B1 raw CIF | +1.62% | 2.68% | 0.6295 | 0.1956 | 24.07% |
| B1 Platt | +1.11% | 1.76% | 0.6568 | 0.2026 | 23.98% |
| B0 按 S20 标签重评 | +2.16% | 1.42% | 0.6390 | 0.1926 | 22.74% |

B1 排序信号成立，但静态 Platt 在 wf1 的 Brier Skill 为 -5.20%，概率校准失败。
calibration → test 基准率分别反转为 `13.88%→10.64% / 11.49%→14.81% /
10.93%→13.70%`。状态固定为 `offline_completed_not_shadow_eligible`。

测试后发现的固定 50/50 `B1 Platt + B0` 汇总 Brier Skill +2.80%、ECE 1.25%、
ROC-AUC 0.6542，且三折 Brier Skill 为正；但这是 post-hoc 发现，只能冻结为 C1，在新时期
一次性确认，不能用当前测试晋级。

结果与报告：

- `config/s20_b1_competing_risk_results.json`
- `docs/research/s20_b1_competing_risk_report_20260831.md`
- 本地产物：`output/experiments/s20_b1_competing_risk_50pct/`

## 已完成：每日高置信选股重评

实现：`scripts/evaluate_s20_high_confidence.py`；机器结果：
`config/s20_high_confidence_evaluation.json`。

50% 实验样本、145 个测试交易日：

| 候选 | 每日 K | 命中率 | lift | 至少一只命中的交易日比例 |
|---|---:|---:|---:|---:|
| B1 raw | 10 | 26.14% | 1.99x | 88.28% |
| B1 raw | 20 | 26.41% | 2.01x | 97.24% |
| B1 raw | 50 | 26.73% | 2.04x | 99.31% |
| B0 | 20 | 23.41% | 1.79x | 91.72% |

B1 Top20 分折命中率 `20.24% / 29.89% / 28.22%`；B0 为
`24.17% / 23.86% / 22.54%`。B1 总体更强，但 wf1 退化约 3.9 个百分点，因此还不稳定。
这些 Top-K 来自 50% 样本，正式结论必须用全市场推理重算。

## 测试状态

最近一次相关测试：`13 passed`。

```powershell
python -m pytest tests/test_s20.py tests/test_r20_target_prob.py -q
```

## 下一步（按顺序）

1. 实现带 20 个交易日标签成熟延迟的滚动校准；严禁使用尚未成熟的测试标签。
2. 同时建模 S20-15/20/25 和 Risk20，而不是只训练 +25%。
3. 在训练/调参窗口学习市场状态权重，解决 B1 在 wf1 退化的问题；禁止按测试折手工切换。
4. 冻结 C1 的 50/50 权重，只在新的未触碰时期一次性确认。
5. 做全市场每日 Top10/20/50 推理和评估；50% 样本结果不能用于上线。
6. 评估 B2 DLinear；若不能改善每日 precision/lift、最差折和校准中的至少两项，停止增加
   时序网络复杂度。
7. 只有逐折 Brier Skill 为正、每日 Top20 lift 每折 >1.5、相对 B0 无超过 2pp 的单折
   precision 退化，才进入至少 60 个交易日影子运行。

## 明确禁止误读

- 当前没有生产 S20 清单，也没有网页 S20 展示。
- 26.41% 是 +25% first 的历史样本 Top20 命中率，不是 60%/80% 的绝对保证。
- R20、池 A、池 G、每日 21:00 更新任务均未被 S20 修改。
- B1+B0 的好结果是后验线索，不是已通过验证的正式模型。
