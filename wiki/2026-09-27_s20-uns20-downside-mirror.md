# 上涨榜和下跌榜必须分开训，对称方法不是自动的 SOTA

- **日期**: 2026-09-27
- **provenance**: user-revised（用户要求把这几天的 S20/unS20 思考写入 wiki，并问学术 SOTA 能否支撑、stage1 能否借 SOTA 改算法）
- **裁决**: 🧩框架（标签拆分和分层使用有文献支撑；「stage1 最好所以镜像 unS20 就是最佳下跌榜」以及「换成 2025 深度排序器」都不成立）
- **关联代码/实验**: `scripts/build_uns20_labels.py`、`scripts/train_uns20.py`、`src/stockagent_analysis/s20.py`；产物 `output/experiments/s20_uns20_20260927/`；记录 `docs/research/s20_uns20_down_first_20260927.md`

## 一句话
S20 stage1 是上涨事件的最佳单信号，并不自动给出最佳下跌榜；下跌要按「20 日内先触 −10%」重打标签单独训练，而且它只能当名次门，不能和 stage1 平均，也不能换成更大的排序网络来绕开振幅问题。

## 出处 (必填, 无出处不写)
- Zhang, Perote, Vidal-García, Vidal, "First passage times in portfolio optimization: A novel nonparametric approach", *European Journal of Operational Research* 312(3), 2024. DOI: [10.1016/j.ejor.2023.07.044](https://doi.org/10.1016/j.ejor.2023.07.044) — 持有期内首次触线（intra-horizon）和期末收益不是同一个目标。
- Gao et al., "Progressive Dependency Representation Learning for Stock Ranking in Uncertain Risk Contrasting", KDD 2025. [10.1145/3690624.3709189](https://dl.acm.org/doi/10.1145/3690624.3709189) — 排序和不确定风险要分开学，不是把风险平均进收益名次。
- "When Alpha Breaks: Two-Level Uncertainty for Safe Deployment of Cross-Sectional Stock Rankers", arXiv:2603.13252, 2026. <https://arxiv.org/abs/2603.13252> — 强排序器会在行情切换时失效；部署应分成「今天做不做」和「这一只信不信」两层。
- Wang et al., "Risk-aware Stock Recommendation via Split Variational Adversarial Training", *ACM TOIS* 42(4), 2024. DOI: [10.1145/3643131](https://doi.org/10.1145/3643131) — 只优化收益排序不够控风险；他们的做法是在**同一个**排序器里压波动，不是再训一个对称下跌榜然后平均。
- "On Evaluating Loss Functions for Stock Ranking", arXiv:2510.14156, 2025. <https://arxiv.org/abs/2510.14156> — listwise/pairwise 损失可以改变风险，但实验是小股票池上的 Transformer，不能外推成「stage1 该换成深度模型」。
- 本仓已证伪的融合：[[2026-09-26_s20-amplitude-not-direction]]、`wiki/smartgoal/2026-09-26_s20_failure_prune_and_pump.md`。

## 为什么引入 (第一性原理)
stage1 在确认窗是最强的单信号（Top20 命中 26.9%）。把 R20、pump、stage1 平均后命中降到 22.8%。旧 unS20 学的是「入选后没有涨成」，AUC 约 0.66，裁剪只比删掉最低分多 1.4 个百分点。用户因此提出对称想法：上涨榜最好，下跌榜就该是它的镜像。这个想法对在**训练手续**，错在**结论自动成立**。价量特征更相关于振幅（分数与破位相关 0.33，与安全上涨相关 0.19）。镜像模型很容易把同一批大波动票排到两边的前面。

## 核心思想 (讲直觉, 不堆公式)
文献支持的是三件已有做法，不是一张新网络：

1. **触线，不是期末涨跌。** EJOR 2024 把「多久第一次摸到目标」和「持有期内会不会先摸到亏损线」分成两个路径量。对应本仓：stage1 的 `positive20` 是安全摸到上涨目标；unS20 的 `down_first20` 是 −10% 先于 +20%。闷住没破线、达标后再跌回入场价，都不是这个下跌事件。
2. **风险不要平均进上涨名次。** KDD 2025 的 PDU 和 TOIS 2024 的 SVAT 都把风险从纯收益排序里拆出来。本仓更硬的证据是平均会稀释强信号，所以拆开之后用漏斗：硬门、一条主排序、确认、软降权，不融合训练。
3. **排序器会在行情切换时失效。** arXiv:2603.13252 记录一个 20 日 LightGBM 排序器全样本很强、换一段主题行情就坏。这和 unS20 在 2025-03～04 下跌基数 44% 时 lift 0.95 是同一类失败。SOTA 启发是少做或不做，不是把 stage1 换成 Transformer。

不支持的命题：「stage1 最好 ⇒ 同构 unS20 就是 SOTA 下跌榜」。没有任何一篇上面的论文证明镜像自动最优。本仓 wf1 已经给出反例。

## 我们怎么吸纳 / 改造
- 照搬触线标签，不照搬他们的组合优化器或深度骨干。下跌标签在 `build_uns20_path_labels`：T+1 开盘，20 个交易日，−10% 先于 +20%；同一天双触记 −1 丢弃。
- 从头训练的是二分类 LightGBM 集成（三种子 + Platt），特征沿用 stage1 的可移植因子，不用 R20 锚，也不把 `positive20==0` 当标签。
- stage1 的算法改良**不**采纳 KDD 2025 / arXiv:2510.14156 的深度或 listwise 替换。本仓 stage2 已经是按日 LambdaRank，确认窗里仍是 stage1 单信号最强。下一刀如果做，只许在新窗口上让 listwise 或两层弃权去打 stage1，不许在已消费窗上换骨干。
- Jev 仍不进入。它不是牛熊分类器，历史 AUC 在随机附近。

## 结果与裁决
全市场 2,459,120 行，已解析样本先跌破率 31.6%。三段测试的下跌 Top20：

| 折 | 基数 | Top20 | lift | AUC |
|---|---:|---:|---:|---:|
| wf1 2025-03～04 | 44.4% | 42.1% | 0.95 | 0.46 |
| wf2 2025-07～08 | 13.6% | 52.7% | 3.89 | 0.59 |
| wf3 2025-11～2026-01 | 20.5% | 61.7% | 3.01 | 0.76 |

后两段名次有用，概率三段都没校准好（wf3 预测均值 0.44，实际 0.20）。确认窗 2026-01-27～08-05 和 2026-08-06 之后没碰。`promotion_eligible=false`。未与 `pump_down` 对照，因此还不能叫最佳下跌榜，更不能踢 S20。

裁决：做法里「分开的触线标签 + 分层使用」有 SOTA 文献支撑；「镜像即 SOTA」和「用 2025 排序网络改良 stage1」没有。stage1 保持点式概率主排序，直到新窗口上有对照赢过它。

## 思想谱系 (演化)
- 取代了: 「未涨成 = 下跌标签」以及「多评分器平均成一个精选类」。
- 被取代 / 下一步: 在 2026-08-06 之后的成熟窗，把 unS20 顶部分位当硬门去打 stage1，并与 `pump_down` 对照。赢不了就留 stage1。
- 同源 / 对照: [[2026-09-26_s20-amplitude-not-direction]]、[[2026-09-20_s20-causal-screen-specified-rise]]。

## 移植提示 (必填)
换市场时先写清上涨触线和下跌触线是两个事件，同一天双触必须丢弃。下跌模型用和时间切分的走样训练，报告名次 lift 和概率校准，不要只报 AUC。行情段基数突变而 lift 掉到 1 以下，就不要部署。不要因为另一篇深度排序论文在美股 S&P 上更好，就替换已经赢过平均融合的点式主排序。

不可移植：A 股 T+1、+20%/−10% 这一对阈值、50% 哈希抽样、以及 stage1 在本仓确认窗的 26.9% 命中。
