# S20：近五年文献审查与独立算法路线（2026-08-31）

## 决策

S20 是独立研究指数，不替代、不修改、也不向当前有效 R20 注入信号。旧的
`R20-P v2` 实验保留原文件和名称，作为 S20 的 B0 基线证据；它证明“达标概率”有信号，
但它只知道 20 日窗口的最高涨幅和最大回撤，不知道两者发生的先后次序。

S20 的主问题改为竞争风险（competing risks）/首次触达（first passage）：

```text
S20_25 = 100 * P(20 个交易日内先触达 +25%，而非先触达 -15%)
S20_20 = 100 * P(20 个交易日内先触达 +20%，而非先触达 -15%)
S20_15 = 100 * P(20 个交易日内先触达 +15%，而非先触达 -15%)
```

模型不按趋势、反弹、形态或行业做硬筛选。这些变量可以提供信息，但最终输出必须是
经过时间外校准的绝对概率。日线同一天同时触达上下边界时无法判断盘中先后，标签记为
`ambiguous`，不进入二元损失；后续可用分钟数据消除歧义。

## 工具与检索边界

本环境没有暴露 `mylib` 命令、MCP 资源或连接器，仓库和环境变量中也没有对应入口，
因此没有冒充使用 mylib。本次改用 2021-2026 年会议/期刊官网、论文页和作者官方仓库；
以后若接入 mylib，应按本文检索式补做引用追踪和复现代码核验。

检索主题包括：probabilistic time-series forecasting、competing risks、first-passage、
concept drift、stock return/drawdown、adaptive conformal prediction、market-guided stock
transformer。优先保留能改变标签、验证或漂移处理的论文，而不是单纯堆叠网络结构。

## 核心文献与对 S20 的作用

| 方法 | 年份 | 可用于 S20 的部分 | 决策 |
|---|---:|---|---|
| [Deep Survival Machines](https://publications.ri.cmu.edu/deep-survival-machines-fully-parametric-survival-regression-and-representation-learning-for-censored-data-with-competing-risks-2) | 2021 | 时间到事件、删失、竞争风险 | 用来重构标签和损失，不直接照搬医疗分布假设 |
| [Temporal Routing Adaptor (TRA)](https://www.microsoft.com/en-us/research/publication/learning-multiple-stock-trading-patterns-with-temporal-routing-adaptor-and-optimal-transport/) | 2021 | 多种股票行为模式、路由专家、最优传输分配 | 作为“趋势/反弹不硬分类”的软路由候选 |
| [TimeGrad](https://proceedings.mlr.press/v139/rasul21a.html) | 2021 | 多变量联合概率路径 | 仅作为后段高成本挑战者 |
| [Adaptive Conformal Inference](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) | 2021 | 分布漂移下在线覆盖率控制 | 用于滚动不确定性校准 |
| [TACTiS](https://proceedings.mlr.press/v162/drouin22a.html) | 2022 | Transformer + attentional copula 的联合预测分布 | 若独立 hazard 明显低估路径依赖再试 |
| [Temporal Quantile Adjustments](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c8d2860e1b51a1ffadc7ed0a06f8d8f5-Abstract-Conference.html) | 2022 | 相关时间序列的 conformal 区间 | 校准候选 |
| [DLinear](https://ojs.aaai.org/index.php/AAAI/article/view/26317) | 2023 | 简单线性时序基线可胜复杂 Transformer | 强制进入基准，防止“新架构即提升”的错觉 |
| [DoubleAdapt](https://doi.org/10.1145/3580305.3599315) / [官方代码](https://github.com/SJTU-DMTai/DoubleAdapt) | 2023 | 股票概念漂移、数据与模型双适配、增量学习 | 漂移阶段首选候选 |
| [Conformal PID](https://proceedings.neurips.cc/paper_files/paper/2023/hash/47f2fad8c1111d07f83c91be7870f8db-Abstract.html) | 2023 | 趋势、季节性、系统误差和分布漂移下的在线 conformal | 优先于固定月份一次性校准 |
| [MASTER](https://arxiv.org/abs/2312.15235) | 2023 | 市场状态、股票间相关性、动态特征有效性 | 作为横截面上下文候选，不先定为主模型 |
| [TFT](https://doi.org/10.1016/j.ijforecast.2021.03.012) | 2021 | 多步分位数预测、变量选择和可解释门控 | 用于联合预测 1-20 日风险路径 |
| [iTransformer](https://proceedings.iclr.cc/paper_files/paper/2024/file/2ea18fdc667e0ef2ad82b2b4d65147ad-Paper-Conference.pdf) | 2024 | 变量维度注意力 | 表征挑战者，不预设胜出 |
| [MOMENT](https://proceedings.mlr.press/v235/goswami24a.html) | 2024 | 通用时间序列预训练表征 | 只做冻结表征/微调对照，防止高成本无增益 |

金融任务的直接证据也支持同时预测收益与风险，而不是只预测收益：2022 年
[横截面收益与最大回撤研究](https://doi.org/10.1016/j.jfds.2022.11.002)报告最大回撤的可预测性明显高于收益。
但 S20 不把“预测最大回撤”当最终答案，而是进一步预测上下边界的先后顺序。

## 实验阶梯

1. **B0：旧窗口极值概率基线。** 保留已有 350,608 条严格时间外预测；P25 顶十分位
   命中率 22.52%、基准率 12.94%、lift 1.74，作为最低比较线。
2. **B1：离散时间竞争风险 LightGBM。** 每个样本展开为 1-20 日 hazard，分别预测
   `up_first`、`down_first`、未触达；这是首个必须完成的新基线。
3. **B2：DLinear。** 用同一输入窗和标签验证复杂时序模型是否真的必要。
4. **B3：TFT 多期限模型。** 同时预测 1-20 日累计发生概率和收益分位数。
5. **B4：TRA 软路由。** 让模型自行路由趋势、反弹、震荡等潜在模式，不人为筛形态。
6. **B5：DoubleAdapt。** 只在固定模型出现跨阶段退化后加入滚动增量适配。
7. **B6：MASTER 上下文。** 加入指数、行业和横截面关联，仍不设行业数量上限。
8. **B7：TACTiS/扩散路径。** 仅当联合路径分布相对 B1-B6 有明确增益且成本可接受时进入。

所有候选使用相同的 purged walk-forward 划分、特征可用时点和样本池。模型选择先看每折
Brier Skill 是否为正，再看 ECE、top-decile lift 和不同市场阶段的校准，不以单一 AUC
或回测收益决定。概率校准使用滚动窗口，并比较 Adaptive Conformal 与 Conformal PID。

## 交付顺序与闸门

- S0：生成 `up15/up20/up25/down15` 首次触达日，审计同日双触达率和数据泄漏。
- S1：完成 B0 与 B1 的同样本公平复验；B1 若不能逐折胜过常数概率，不继续加深网络。
- S2：完成 B2-B4；只有相对 B1 的 Brier、ECE、lift 至少两项改善才保留。
- S3：在漂移证据成立后测试 B5-B6；B7 是可选项。
- S4：至少 60 个交易日影子运行；R20 与 S20 并排展示但互不影响。
- S5：任何生产使用必须另行审批；本研究分支不得自动修改池 A 或每日 21:00 任务。

机器可读契约位于 `config/s20_experiment_v1.json`，首次触达标签实现位于
`src/stockagent_analysis/s20.py`。

## mylib 实跑复核（2026-08-31）

`D:/aicoding/mylib` 更新至 `e3b5eaa` 后，使用其 `paper_search` 统一检索器，对
2021-2026 年的首次触达、最大回撤和股票概念漂移做了三组联合检索。稳定来源
OpenAlex、arXiv、Crossref、DBLP 共返回 40 篇去重记录；完整表和错误披露见仓库根目录
`allinone.md`。

检索重新找回了 DoubleAdapt，并补充了
[金融专用概念漂移检测](https://doi.org/10.48550/arxiv.2103.14079)、
[金融概念漂移的超平面角度量](https://doi.org/10.1007/s10489-025-06292-w)、
[离散时间最大回撤约束](https://doi.org/10.1007/978-3-030-98519-6_5)以及
[FinTSBridge](http://arxiv.org/abs/2503.06928v2)。这些结果加强了 B1 离散时间基线、
B5 漂移适配和分阶段校准的必要性，但没有推翻当前实验阶梯。

本次还确认了一个检索风险：`first passage` 会命中大量物理学、结构安全和渗流论文，
不能因数学术语相同就直接迁移到股票任务。S20 后续只接受经过金融数据、严格时间外验证
或可复现实验支持的方法。Semantic Scholar 无密钥请求触发 429；OpenReview 的 Python
3.14 依赖未安装成功，因此本轮完整结果明确限定为上述四个来源。
