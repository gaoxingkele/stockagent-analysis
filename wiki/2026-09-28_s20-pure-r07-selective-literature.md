# R07 "推荐的都要涨"在统计上叫什么：选择性预测与 conformal selection（控制名单里的错误率）

- **日期**: 2026-09-28
- **provenance**: ai-executed（10 轮思考链第 7 轮：把用户目标翻译成统计学里已有的问题，找 CCF-A/顶刊的现成工具）
- **裁决**: 🧩框架
- **关联代码/实验**: 本轮只是文献，下一轮 [[2026-09-28_s20-pure-r08-conformal-selection]] 实测
- **链路**: [[2026-09-28_s20-pure-r06-day-gate]] → 本轮 → [[2026-09-28_s20-pure-r08-conformal-selection]]

## 一句话
用户要的"Top20/Top50 大概率都涨"，统计上的准确说法是：**选出一个长度可变的名单，使名单里失败的比例（错误发现率 FDR）不超过 q**。这正是 conformal selection（Jin & Candès, JMLR 2023）解决的问题；它的保证依赖"可交换性"，而 R06 已经说明 A 股日级冲击会打破这个前提。分布漂移下只能退而求"长期平均"的保证（Gibbs & Candès, NeurIPS 2021）。

## 出处 (必填, 无出处不写)
- Y. Jin, E. J. Candès, "Selection by Prediction with Conformal p-values", *JMLR* 24(244), 2023. arXiv:2210.01408 <https://arxiv.org/abs/2210.01408>；代码 <https://github.com/ying531/conformal-selection>——包在任何预测模型外，按"结果超过阈值 c"做 conformal p 值，再用 Benjamini–Hochberg 选名单，控制名单内的错选比例。
- Y. Jin, Z. Ren, "Confidence on the Focal: Conformal Prediction with Selection-Conditional Coverage", arXiv:2403.03868, 2024 <https://arxiv.org/abs/2403.03868>——对 Top-K 这类"先选再看"的规则给出选择条件下的覆盖保证。
- I. Gibbs, E. J. Candès, "Adaptive Conformal Inference Under Distribution Shift", *NeurIPS* 2021（CCF-A）<https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html>——分布随时间漂移时，用一个在线更新的参数维持长期平均覆盖率。
- A. N. Angelopoulos et al., "Conformal Risk Control", *ICLR* 2024, arXiv:2208.02814 <https://arxiv.org/pdf/2208.02814>——把"覆盖"推广成任意单调损失的期望控制。
- Y. Geifman, R. El-Yaniv, "Selective Classification for Deep Neural Networks", *NeurIPS* 2017（CCF-A）——以覆盖率换风险：只在高把握时作答，风险—覆盖曲线是标准评估。
- M. Kato, "Conformal Predictive Portfolio Selection", arXiv:2410.16333, 2024 <https://arxiv.org/pdf/2410.16333>——conformal 用在组合选择上的一个金融例子。
- 本仓前序：[[2026-09-13_s20-three-track-evidence-gate]]、[[2026-09-20_s20-causal-screen-specified-rise]] 已经提过 Selective classification / Conformal Risk Control，但没有落到"名单 FDR"这个具体形式。

## 为什么引入 (第一性原理)
前 6 轮把问题逼到一个角落：个股方向信息弱（R01、R04 待验），日级风险看不清（R06），胜率被交易规则钉在 50% 附近（R03）。于是需要换个问法：**不是"怎么让模型更准"，而是"在模型这么准的前提下，名单该多长，才能保证名单里的失败比例不超过 q"**。这句话把用户的目标变成了一个可验证的保证，而不是一个愿望。

## 核心思想 (讲直觉, 不堆公式)
1. **Conformal selection**：拿历史上已成熟的样本做"校准集"。对今天的每个候选问：历史上分数比它还高、结果却失败的有多少？这个比例就是它的 conformal p 值。p 值越小，"它失败"的证据越少。然后用 BH 程序决定今天收多少只，使收进来的失败比例期望 ≤ q。名单长度由数据决定，可以是 0。
2. **可交换性**：保证成立的前提是"校准集和今天是同一个分布"。股市里今天和 20 天前不是同一个分布，特别是 R06 看到的日级冲击。所以保证会被打破，这正是要实测的东西。
3. **Adaptive conformal**：把"阈值"当成在线学习的参数，错多了就收紧、错少了就放宽，只保证长期平均。它对应"最近 N 天名单失败率偏高就缩短名单"。
4. **选择性分类**：用风险—覆盖曲线（名单越短，失败率应越低）评估，这是比 AUC 更贴近用户目标的指标。

## 我们怎么吸纳 / 改造
- 用户目标的正式形式：**给定失败定义 F（纯跌 / 止损出场 / 非纯涨）和容忍 q，每天输出失败比例期望 ≤ q 的最长名单。**Top20/Top50 不再是固定长度，而是"在 q 下能给出多少只就给多少只"。
- 校准集只用"在今天之前已经走完 20 个交易日"的样本（间隔 21 个交易日），与本仓 purged walk-forward 纪律一致。
- 先用 stage1 分数实测保证是否成立（R08）。如果不成立，要看失败出在哪：是分数在池内没区分力，还是分布漂移。

## 结果与裁决
🧩框架。给出了用户目标的**正式定义和评估曲线**（风险—覆盖），以及一个可以直接实现的算法。能否兑现，看下一轮。

## 思想谱系 (演化)
- 取代了: "固定 Top20 + 看命中率"的评估方式。
- 被取代 / 下一步: [[2026-09-28_s20-pure-r08-conformal-selection]]
- 同源 / 对照: [[2026-09-13_s20-three-track-evidence-gate]]（TopN 可空缺的早期版本）

## 移植提示 (必填)
任何"推荐的都要对"的需求，先改写成"名单 FDR ≤ q"，再选 conformal selection 当外壳。关键前提是校准集和今天可交换；金融时间序列上必须实测实际 FDR，而不是相信名义 q。不可移植：无。这一条是通用的统计框架。
