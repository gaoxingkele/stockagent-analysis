# S20 Harness 端到端执行改造（2026-09-24）

状态：工程改造已完成并验证。**未重训冻结排序、未授权正式训练、未改标签、未接生产。** H01 仍是 `FAILED_VALIDITY`；本次改造让链条能够**在诚实记录阻断的前提下走到 H10**，而不是把 H01 改成通过。

## 1. 为什么原来看不到头

原 harness 是一台**证据闸门机**，不是流水线。四条结构性阻断：

| 阻断 | 代码位置 | 后果 |
|---|---|---|
| 阶段白名单 | `runtime.run` 只允许 `H00/H01/H02/H03/H05`，其余 `NotImplementedError` | H04、H06–H10 永远无法执行 |
| 依赖必须是 COMPLETED | `run()` 要求所有 `depends_on` 为 `COMPLETED`；`verify()` 同样要求上游 COMPLETED | H01 失败后 H02 起全部不可达，与数据无关 |
| 没有流水线驱动 | 只有 `run-stage` 单阶段入口，`resume` 只做崩溃对账 | 每一步都要人手工衔接 |
| intake 永不通过 | `h02/h03/h05_intake` 硬编码 `formal_gate_passed=False` | 即使能跑也必然终止于 FAILED_VALIDITY |

结论：**它按设计走不到底。** 需要的是轨道语义，不是新的 executor。

## 2. 设计

### 2.1 两条显式轨道

| 轨道 | 语义 | 能否记录正式通过 |
|---|---|---|
| `formal` | 冻结语义。上游必须 `COMPLETED`；闸门失败即停 | 可以 |
| `diagnostic` | 登记式研究轨道。可跨过**已定格**的阻断继续，并继承阻断 | **不可以** |

`formal` 轨道行为完全未变：已有测试 `test_run_stage_h03_is_refused_until_foundation_passes`、`test_runtime` 全部原样通过。

### 2.2 阶段终止状态

保留原有 `COMPLETED / FAILED_VALIDITY / FAILED_INFRA`，新增：

- `WAITING_IMPLEMENTATION`：该阶段尚无执行路线，登记缺口而不是崩溃
- `WAITING_MATURITY`：需要真正未来的数据（H09），无法由存量历史算出

`WAITING_*` 的 manifest 不声明产物，且**永远不等于完成**，不会满足任何正式闸门。

### 2.3 关键规则

1. `formal_accepted` 只有在 `formal` 轨道且 `COMPLETED` 时才可能为真；diagnostic 轨道恒为 `false`。
2. 下游 manifest 必须写入 `inherited_blockers`，列出每一个非 `COMPLETED` 的上游及其状态。
3. diagnostic 轨道下，执行不了的阶段登记为缺口并**继续**，避免整条分支停在 `PLANNED` 而看起来"还没轮到"。
4. 已记录 `FAILED_VALIDITY` 的阶段不会被重复重跑；重跑需要新证据并用 `run-stage` 显式发起。

## 3. 执行路线表

| 阶段 | 路线 | 模块 | 现状 |
|---|---|---|---|
| H00 | builtin | — | 快照 + 资源探针，`COMPLETED` |
| H01 | builtin | `full_data_audit` | `FAILED_VALIDITY`（需源证据，非代码问题） |
| H02 | **compute（新）** | `abcds_labels` | 已实现并实跑 |
| H03 | **compute（新）** | `h03_baseline` | 已实现并实跑 |
| H04 | **compute（新）** | `h04_competition` | 已实现并实跑 |
| H05 | **compute（新）** | `h05_abcds` | 已实现并实跑 |
| H06 | unimplemented | `feature_group_ablation` | 需激活决策合同 |
| H07 | unimplemented | `portfolio_replay` | 需锁定评估装配 |
| H08 | unimplemented | `search_decision` | 需冠军冻结与 holdout 封存 |
| H09 | prospective | — | `WAITING_MATURITY`，需前瞻采集 |
| H10 | unimplemented | — | 一次性裁决；diagnostic 轨道只能产出不可晋级结论 |

H04/H05 的 `default_mode` 仍为 `intake`，所以 `run-stage` 的原有行为不变；只有流水线显式请求 `compute` 路线。

## 4. 本次实跑

```powershell
python -m research.s20_harness.cli run-pipeline --track diagnostic --plan config/s20_v4_harness_plan.json
python -m research.s20_harness.cli pipeline-status --plan config/s20_v4_harness_plan.json
```

结果：11 个阶段全部到达终止状态（`COMPLETED` 1 / `FAILED_VALIDITY` 10 / `WAITING_*` 6 中的其余），`promotion_eligible=false`，`formal_accepted_stages=[]`。

### 4.1 H02 五类标签（真实产物）

`S20.hit15_risk10_silence8.v1`，来源为冻结诊断样本的 `hit15 / path_risk / max_gain`：

| 类别 | 计数 |
|---|---:|
| A 安全达标 | 15,098 |
| B 带风险达标 | 1,075 |
| C 安全未达标 | 9,981 |
| D 有风险未达标 | 13,690 |
| S 沉默 | 15,685 |
| 未知（同日双触） | 1 |

与 `s20_abcds_restart_plan_20260921.md` 表格逐项一致。`label_contracts.json` 同时钉住旧定义对照：legacy a-only 的 S = **5,996**，新契约 S = **15,685**，两者必须不同，作为 golden check 常驻。

黄金检查 8 项全过，含：8% 用严格小于（恰好 8% 归 C）、+15% 触达永不归沉默、未解析行保留不插补、已解析行 A+B≡H 与 B+D≡R（唯一同日双触行保留局部掩码且不分类）。

### 4.2 H03 时间切分与参考拟合（真实产物）

纯日历分位切分，不掺杂结果信息：

| 段 | 行 | 已成熟可监督 | A 实际率 | 参考概率均值 |
|---|---:|---:|---:|---:|
| fit | 27,289 | 25,130 | 29.4% | — |
| tune | 7,151 | 5,034 | 20.9% | 30.0% |
| calibration | 6,509 | 4,334 | 28.8% | 26.7% |
| selection-policy | 7,279 | 5,095 | 23.9% | 28.3% |
| outer-test | 7,301 | 0（封存） | 27.1% | 28.1% |

只拟合 `fit` 段；fit 段预测被**扣留而非冒充 OOF**；outer 标签保持封存。

## 5. 仍然缺什么（诚实清单）

代码解决不了的部分：

- **H01 正式准入**：历史可得性、复权/公司行动、停牌、历史 ST、可执行性仍需源证据。
- **H09 前瞻**：必须真实等待未来信号日与 20 日成熟期。

代码可继续补的部分（按依赖顺序）：

1. H04：把 `registered_search_run` + `development_frontier` 接到 H03 的 `split_manifest.json`/`baseline_oof.parquet`。
2. H05：给 `joint_calibration` + `recommendation_policy` 写 compute 路线，替代只有 intake 的现状。
3. H06/H07/H08/H10：分别在 `feature_group_ablation`、`portfolio_replay`、`search_decision` 之上补装配层。

## 6. 改动清单

| 文件 | 变更 |
|---|---|
| `research/s20_harness/pipeline.py` | 新增：轨道、阶段顺序、执行路线注册表、`run_pipeline` |
| `research/s20_harness/abcds_labels.py` | 新增：H02 五类标签构建器 + 契约 + 黄金检查 |
| `research/s20_harness/h03_baseline.py` | 新增：H03 切分 + 参考拟合 |
| `research/s20_harness/runtime.py` | 轨道与续跑策略、`record_gap`、`_run_compute`、兼容两种注册表形状的写入 |
| `research/s20_harness/cli.py` | `run-pipeline`、`pipeline-status`、`run-stage --track/--mode` |
| `tests/s20_harness/test_pipeline.py` | 新增 8 项测试 |

冻结产物路径：`output/experiments/s20_safe_v4/<protocol_hash>/`，本次写入 `track=diagnostic` 的 H02/H03 运行与 H04–H10 缺口登记。生产池、网页、21:00 任务、R20/S20 冻结排序均未改动。

## 7. 复现

```powershell
python -m pytest tests/s20_harness/test_pipeline.py tests/s20_harness/test_runtime.py tests/s20_harness/test_cli.py -q
```

2026-09-24 结果：**28 passed**（含既有 runtime/CLI 冻结语义测试）。

全量回归：

```powershell
python -m pytest tests/s20_harness -q
```

2026-09-24 结果：**1128 passed**（26 分 17 秒），无回归。

## 8. 追加（2026-09-25）：H04 / H05 计算路线

### 8.1 新增模块

| 文件 | 作用 |
|---|---|
| `research/s20_harness/abcds_model.py` | 版本化五类多项模型与零拟合频率控制。**不修改冻结的 `joint_model.CLASSES`**，类别顺序必须精确匹配 `[A,B,C,D,S]`，否则拒绝。 |
| `research/s20_harness/abcds_calibration.py` | 任意 K 类 bounded scalar temperature，显式接收类别顺序，只在成熟校准段拟合。 |
| `research/s20_harness/policy_support.py` | 包装冻结 `recommendation_policy` 门；候选按 `sample_id` 对齐，绝不用行位置。 |
| `research/s20_harness/h04_competition.py` | H04 有界竞赛：登记配置、计费拟合、独立风险头、开发前沿。 |
| `research/s20_harness/h05_abcds.py` | H05 延迟校准 + 登记式政策网格 + 覆盖/风险 + 入选可靠性。 |

依赖解析：H04 的冻结依赖只声明 H03，但执行器还需要 H02 标签。`Runtime._lineage` 通过**回溯已登记 manifest 的血缘**取得祖先目录，而不是放宽冻结依赖或复制数据。

### 8.2 H04 真实结果（`H04-1ebca55257ad4207b71418c62bd881aa`）

5 个登记配置，计费拟合 **3 / 上限 18**（频率控制 0 次、二元参考头复用 H03 计 0 次、独立风险头 1 次、两个五类正则档各 1 次）。风险门阈值取校准段风险分 **q50 = 0.2481**，评估段 20 个信号日，Top20 = 400 个名额。

| 配置 | 门 | 实际 A | 实际 +15% | 实际 B+D | 均值分 |
|---|---|---:|---:|---:|---:|
| 频率控制 | 无 | 23.0% | 23.5% | 17.8% | 0.329 |
| 频率控制 | q50 | 19.5% | 20.0% | 7.3% | 0.329 |
| 二元 P(A) 参考 | 无 | 40.8% | 49.3% | 51.0% | 0.476 |
| 二元 P(A) 参考 | q50 | 27.5% | 28.3% | 21.0% | 0.309 |
| 五类 C=1.0 | 无 | 40.0% | 49.3% | **53.8%** | 0.519 |
| 五类 C=1.0 | q50 | 28.0% | 29.0% | 21.8% | 0.328 |
| 五类 C=0.1 | q50 | 27.8% | 28.8% | 21.8% | 0.329 |

读法：按安全达标概率排序能把 A 率从 23% 提到 40%、+15% 触达从 23.5% 提到 49.3%，但真实 B+D 同步升到 51–54%。风险门把 B+D 压回 21–22%，代价是 A 率回落到 28%。这与 09-20 诊断的"高上涨分必须被风险门抵消"是同一结构，现在由流水线自己产出。

### 8.3 H05 真实结果（`H05-b3020dcf918b4487bb7dbdf8f2f2b29f`）

2 个五类配置各做一次温度校准（**2 / 上限 6**），登记 6 个政策。

| 政策 | 入选 | 校准预测 +15% | 实际 +15% | 偏离 |
|---|---:|---:|---:|---:|
| C=0.1 无门 Top20 | 400 | 0.5067 | 0.5025 | **+0.4pp** |
| C=0.1 q50 Top20 | 400 | 0.3237 | 0.2925 | **+3.1pp** |
| C=0.1 q50 Top10 | 200 | 0.3334 | 0.3000 | +3.3pp |
| C=1.0 无门 Top20 | 400 | 0.5076 | 0.4975 | +1.0pp |
| C=1.0 q50 Top20 | 400 | 0.3231 | 0.2850 | **+3.8pp** |
| C=1.0 q50 Top10 | 200 | 0.3327 | 0.3000 | +3.3pp |

读法：全名单上温度校准后偏离只有 +0.4～1.0pp，但**风险门筛出的子集上校准偏乐观 +3.1～3.8pp**。这正是计划 §7 要求单独报告"入选子集可靠性"的原因——全样本校准好不代表入选子集可信。

### 8.4 仍然缺什么

代码可继续补：H06 特征组消融、H07 执行/组合压力与配对区间、H08 冠军冻结、H10 一次性裁决。

代码解决不了：H01 正式准入（源证据）与 H09 前瞻（真实未来数据 + 20 日成熟期）。

### 8.5 本次追加的验证

```powershell
python -m pytest tests/s20_harness/test_pipeline.py tests/s20_harness/test_abcds_executors.py tests/s20_harness/test_runtime.py tests/s20_harness/test_cli.py -q
python -m pytest tests/s20_harness/test_baseline_model.py tests/s20_harness/test_feature_pipeline.py tests/s20_harness/test_joint_model.py tests/s20_harness/test_splits.py tests/s20_harness/test_joint_calibration.py tests/s20_harness/test_policy_replay.py tests/s20_harness/test_recommendation_policy.py -q
```

2026-09-25 结果：**33 passed**、**53 passed**。其中 `test_candidate_alignment_never_uses_row_position` 是针对本次修掉的一个真实缺陷（政策层曾拿到位置索引而非 `sample_id`，导致全空选）加的回归测试。

## 9. 追加（2026-09-26）：H06 / H07 / H08 / H10 计算路线

### 9.1 路线表最终状态

| 阶段 | 路线 | 结果 |
|---|---|---|
| H00 | builtin | `COMPLETED` |
| H01 | builtin | `FAILED_VALIDITY`（需源证据） |
| H02–H08、H10 | compute | 全部执行并如实报告继承阻断 |
| H09 | prospective | `WAITING_MATURITY`（需真实未来数据） |

新增模块：`h06_ablation`（特征组消融 + 高级路线激活决策）、`h07_execution`（并发上限账本 + 费用/容量压力 + 配对块 bootstrap）、`h08_freeze`（G2 闸门 + 封存 + 功效规划）、`h10_review`（一次性裁决）。

### 9.2 H06 消融（`H06-f3ec01c0adc24878b7b0291bb9d4d42e`，6 次拟合）

| 配置 | 特征数 | 校准 log loss | 相对基线 | 实际 A | 实际 B+D |
|---|---:|---:|---:|---:|---:|
| base_five | 5 | 1.3933 | 0 | 28.0% | 21.8% |
| drop_trend | 3 | 1.3983 | +0.0050 | 28.3% | 18.0% |
| drop_volatility | 3 | **1.5060** | **+0.1127** | 25.0% | 17.0% |
| drop_liquidity | 4 | 1.3870 | −0.0063 | 30.3% | 21.3% |
| add_rsi_wilder | 9 | 1.3873 | −0.0060 | 29.0% | 18.5% |
| add_shuffled_control | 9 | 1.3919 | −0.0014 | 27.8% | 22.3% |

读法：只有**波动率组**的移除明确变差（+0.113）。移除量比反而略微改善损失，说明 `volume_ratio20` 信息增量很小。加真 Wilder RSI（−0.0060）比加打乱对照（−0.0014）只好 0.0046——与仓库既有"Wilder RSI 打乱对照不差"的否决结论一致，因此**没有任何高级家族被激活**。

### 9.3 H07 执行与容量（`H07-2bebbe6d601a4b3bac467afd80f23096`）

候选 = 五类 C=1.0 + q50 风险门 Top20；控制 = 零拟合频率参考，同门同名额。

| 角色 | 容量 | 费用倍数 | 实际建仓 | 因容量拒绝 | 期末净值 | 均值净收益 |
|---|---:|---:|---:|---:|---:|---:|
| 候选 | 20 | ×1 | 100 | **300** | 1.0559 | 0.0112 |
| 候选 | 20 | ×3 | 100 | 300 | 1.0309 | 0.0062 |
| 控制 | 20 | ×1 | 100 | 300 | **1.1490** | 0.0298 |
| 控制 | 10 | ×1 | 50 | 350 | 1.2046 | 0.0409 |
| 控制 | 5 | ×1 | 25 | 375 | 1.2564 | 0.0513 |
| 候选 | 5 | ×1 | 25 | 375 | **0.9394** | −0.0121 |

两个硬结论：

1. **容量是硬约束。** 20 日持有 + 20 个并发名额，每日 Top20 的 400 次选择里只有 100 次能真正建仓，300 次被容量拒绝（约 75%）。"每天出 20 只"在当前持有期下不可执行。
2. **候选在净值上输给零拟合控制。** 每个容量档控制都更高，而且容量收紧时控制变好（1.1490→1.2564）、候选变差（1.0559→0.9394，容量 5 时亏钱）。配对差异：安全达标率 +8.5pp、配对风险率 **+14.5pp**、已实现净收益 −0.15pp——即用更多风险换来更多达标，净值没有改善。

### 9.4 H08 与 H10

- H08：G2 要求跨折跨种子方向一致，本链只有 1 个折、1 个种子 → **不冻结冠军**。功效规划给出：在观测到的每日配对标准差 0.176 下，检测 +5pp 需要约 **48 个配对信号日**（低于重训计划里 120 日的初始猜测）。
- H08 同时如实记录：外层段已被诊断链消费，**不再是干净留出**，唯一干净路径是前瞻采集。
- H10：`promotion_decision.json` 终态为 **`PROMOTION_BLOCKED_BY_VALIDITY`**，`promotable=false`；阻断原因四条（H01 未准入、G2 未达、无干净留出、无前瞻证据），并与发现并列记录。

### 9.5 工程注记

`Runtime.base` 在 Windows 上带 `\\?\` 前缀。用普通相对路径去读深层的 run 目录会因 `MAX_PATH` 失败（`is_dir()` 返回假、`scandir` 抛 WinError 3），核对产物请从 `Runtime(...).base` 出发。

`Runtime._lineage` 现在按注册表顺序取每个阶段的**最新已定格运行**。早期版本只沿直接父级的绑定回溯，会解析到执行器存在之前登记的缺口运行，导致下游读到不存在的产物。

### 9.6 本次追加的验证

```powershell
python -m pytest tests/s20_harness/test_abcds_executors.py tests/s20_harness/test_pipeline.py tests/s20_harness/test_runtime.py tests/s20_harness/test_cli.py -q
```

2026-09-26 结果：**37 passed**。集成测试现在覆盖 H02→H10 全链，并断言：消融不得激活高级家族、容量必须拒绝而不是超限、无冠军可冻结、裁决态不得是可晋级态。
