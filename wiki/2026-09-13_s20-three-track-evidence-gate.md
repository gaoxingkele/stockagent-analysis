# 三轨评价 + 选择层安全盈利，复杂度由证据放行

- **日期**: 2026-09-13
- **provenance**: ai-executed（用户提交已冻结的 S20-v4 长计划，要求再跑 SmartGoal；规划讨论与执行切片讨论同日完成）
- **裁决**: 🧩框架（协议已冻结，runner 未实现，无新训练证据）
- **关联代码/实验**: `docs/research/s20_v4_harness_plan_20260913.md`；`config/s20_v4_harness_plan.json`；负结果 `config/s20_v3_results.json`、`docs/research/s20_b1_competing_risk_report_20260831.md`

## 一句话
在披露覆盖率的前提下让 TopN 尽量安全盈利：旧“曾经涨过”、新“持有到期扣费后赚且少跌”、以及真实成交账户，必须分成三条轨道；模型由简单到复杂，只有残差假说成立才消耗预算。

## 出处 (必填, 无出处不写)
- 选择性分类: Geifman & El-Yaniv 路线的后续工作，*"Towards Better Selective Classification"*，arXiv:2206.09034，https://arxiv.org/abs/2206.09034 — 拒绝推荐必须画风险–覆盖，缩小覆盖不等于更赚钱。
- 共形风险控制: Angelopoulos, Bates, Fisch, Lei, Schuster, *"Conformal Risk Control"*，ICLR 2024，https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html — 有假设的风险界，不是“选中的股票不会跌”。
- 在线共形: Angelopoulos, Candes, Tibshirani, *"Conformal Inference for Online Prediction with Arbitrary Distribution Shifts"*，JMLR 25 (2024)，https://www.jmlr.org/beta/papers/v25/22-1218.html — 漂移下可维持覆盖，仍不等于条件安全。
- 表格/时序简单基线: Zeng, Li, Yan, Lin, *"Are Transformers Effective for Time Series Forecasting?"* (DLinear)，AAAI 2023，https://ojs.aaai.org/index.php/AAAI/article/view/26317 — 复杂序列模型必须先输给简单基线。
- 股票漂移适配: Zhao et al., *"DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting"*，KDD 2023，https://github.com/SJTU-DMTai/DoubleAdapt — 仅在跨期失效被定位后才启动。
- 市场引导关联: Li et al., *"MASTER: Market-Guided Stock Transformer for Stock Price Forecasting"*，AAAI 2024，https://arxiv.org/abs/2312.15235 — 图/关联必须对照随机边。
- 小样本表格: Hollmann et al., *"Accurate predictions on small data with a tabular foundation model"*，Nature 2025，https://doi.org/10.1038/s41586-024-08328-6 — 不在 325 万行上盲目全量。
- 竞争风险/首次触达: Nagpal, Li, Dubrawski, *"Deep Survival Machines"*，2021，https://publications.ri.cmu.edu/deep-survival-machines-fully-parametric-survival-regression-and-representation-learning-for-censored-data-with-competing-risks-2 — 本仓 B1 已用离散时间 hazard，证明路径顺序有排序信息但校准失败。
- 反过拟合脚手架: 见 [[2026-06-24_anti-overfitting-scaffold]]（ICBINB / Bailey PBO / López de Prado AFML / ARA）。

## 为什么引入 (第一性原理)
用户要的是“推荐的股票大多能赚钱，涨跌双高要扣分”，不是“窗口内曾经涨过”。v3 把机会触达当训练目标后，诊断窗 Top20 仍约 35% 即时有效、56% 先行下跌，且概率严重向外乐观。若继续堆网络，只会在错误标签时钟上过拟合。真正卡住的是：数据时点、联合风险、选择后校准，而不是模型族清单。

## 核心思想 (讲直觉, 不堆公式)
1. **三条轨道回答三个问题。** O 轨：有没有明显上涨机会（保留旧信息）。P 轨：固定 20 个市场交易日扣费后是否盈利、期间是否跌破 -5%。T 轨：考虑成交、退出、资金占用后账户是否赚钱。用 O 轨赢面证明不了 T 轨赚钱。
2. **四类不是四个独立故事。** 安全盈利 / 危险盈利 / 安全不盈利 / 危险不盈利是 U 与 B5 的联合。排名用 `pA - λ pB - μ pD - ν pC`，且 μ 最重；同时设风险硬门槛，防止极高上涨分抵消明显风险。N 是上限，允许空缺。
3. **选择会改变分布。** 全市场校准好，不等于 TopN 校准好。拒绝推荐必须报告覆盖代价；空选的盈利率是 null，不是 0 风险 100% 成功。
4. **复杂度是被放行的，不是被清单驱动的。** 先证明复杂方法胜过 ATR/频率/浅树和旧锚；序列对照时间打乱，图对照随机边，新风险特征对照 ATR。负结果和 `INSUFFICIENT_EVIDENCE` 都是完成态。

## 我们怎么吸纳 / 改造
- 选择性分类：拿风险–覆盖曲线和可拒绝层，不拿“高置信=高盈利”的口号。
- 共形：只作 E08 候选，并写明交换性/单调损失在 TopN 时序上不自动成立。
- DLinear 精神：强制 E01 简单基线，包括 ATR+流动性；v3 已显示 ATR 是最稳双向信号。
- DoubleAdapt / MASTER / TabPFN：保留在目录，启动条件是已定位残差 + 预算，而不是论文年份。
- B1 竞争风险：不把首次触达模型静默换成 P 轨四分类，也不把 E07 强行提前；H02 先把到达时间存下来。
- 本仓改造：市场日历而非股票会话计 20 日；单位名义研究标签与真实组合账本分开；已知窗 2026-01-27～08-05 永久曝光。

## 结果与裁决
- **规划完成，执行未开始。** 无 v4 训练数字。v3 未改善、不可当绝对概率；B1 未达影子运行门槛。这些负结果仍约束 v4。
- 本轮 SmartGoal 追加执行切片：薄 H00 → H01 数据 → H02 标签/单位名义执行器。不实现完整训练栈，不改生产。
- 80%/5% 与 90%/1% 是挑战档，不是能力承诺；主确认是相近覆盖下相对基线的 G3。

## 思想谱系 (演化)
- 取代了: [[2026-08-31_s20-research-index]] 把首次触达概率当作唯一主任务、以及 v3 用触达+风险权重重训却未改善的路线。
- 被取代 / 下一步: 薄 H00 落地；数据审计后才能谈模型。诊断选择层见 [[2026-09-20_s20-causal-screen-specified-rise]]（指定 +15% / 因果沉默，发运仍 π0）。若日历修正后旧优势消失，回到数据，不上网络。
- 同源 / 对照: [[2026-09-12_s20-v3-opportunity-risk]] 负结果；[[2026-06-24_anti-overfitting-scaffold]] 的 gate/证伪优先；规划讨论 `wiki/smartgoal/2026-09-13_s20_safe_profit_harness.md`；执行切片 `wiki/smartgoal/2026-09-13_s20_v4_execution_kickoff.md`。

## 移植提示 (必填)
换市场或项目时只搬这四条：
1. 把“曾经发生过的事件”“持有到期的经济结果”“可成交账户结果”分成独立标签，禁止用前一个给后一个刷分。
2. 联合事件不要用边际概率相乘；选择层和评分层分开，空选不得记成功。
3. 先修时钟与可成交性，再训模型；简单波动/流动性基线是复杂模型的淘汰器。
4. 程序跑完 ≠ 科学成功；把 `NO_GAIN` / `TRADEOFF` / `INSUFFICIENT_EVIDENCE` 写成一等产出。
**本项目特有不可移植：** A 股 T+1、涨跌停、ST/退市时点、20 个市场交易日窗口、以及本仓已看窗 2026-01-27～08-05。
