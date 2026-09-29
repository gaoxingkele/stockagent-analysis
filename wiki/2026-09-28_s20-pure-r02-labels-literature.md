# R02 标签文献碰撞：三重障碍、中性类弃权、元标签——"去掉震荡"在训练端和推荐端是两件事

- **日期**: 2026-09-28
- **provenance**: ai-executed（10 轮思考链第 2 轮：文献检索 + 对 R01 数字的反思）
- **裁决**: 🧩框架
- **关联代码/实验**: 复用 R01 产物 `output/experiments/s20_pure_20260928/r01_states.txt`
- **链路**: [[2026-09-28_s20-pure-r01-three-states]] → 本轮 → [[2026-09-28_s20-pure-r03-asymmetry]]

## 一句话
文献给了三件现成工具：可调三重障碍、中性类（弃权）、元标签（二级模型只在主模型选中的票里学"这次对不对"）。它们合起来说明：**"去掉震荡"应该在训练标签里做，而不能指望在推荐时靠删震荡票得到高胜率**；反向减值的正确形态是元标签，不是对称的全市场下跌榜。

## 出处 (必填, 无出处不写)
- M. López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, Ch.3：三重障碍标签 + meta-labeling（主模型决定方向，二级模型决定做不做/做多大）。
- J. F. Joubert, "Meta-Labeling: Theory and Framework", *Journal of Financial Data Science* 4(3), 2022. <https://jfds.pm-research.com/content/early/2022/06/23/jfds.2022.1.098>；代码 <https://github.com/hudson-and-thames/meta-labeling>（同系列 "Meta-Labeling Architecture" JFDS 4(4) 2022、"Ensemble Meta-Labeling" JFDS 2022）。
- S. Kang, "Stock Price Prediction Using Triple Barrier Labeling and Raw OHLCV Data: Evidence from Korean Markets", arXiv:2504.02249, 2025. <https://arxiv.org/abs/2504.02249> —— 障碍宽度和窗口要按"标签均衡"来调（韩股给出 29 日 / 9%），说明障碍是数据相关的参数，不是常数。
- "From Patterns to Predictions: A Shapelet-Based Framework for Directional Forecasting in Noisy Financial Markets", arXiv:2509.15040, 2025. <https://arxiv.org/html/2509.15040v1> —— 三分类（涨/中性/跌）中，中性类的作用是让系统在证据弱时弃权。
- Gao et al., "Progressive Dependency Representation Learning for Stock Ranking in Uncertain Risk Contrasting", KDD 2025, DOI 10.1145/3690624.3709189（CCF-A）—— 排序和风险分开学。
- Zhang et al., "DoubleEnsemble: A New Ensemble Method Based on Sample Reweighting and Feature Selection for Financial Data Analysis", ICDM 2020, arXiv:2010.01265 —— 训练时按"难学/噪声"给样本降权，是"训练时剔除震荡"的温和版本。

## 为什么引入 (第一性原理)
R01 的数字给了一个硬约束：就算 stage1 已经把震荡挤到 16.5%，Top20 里纯涨也只有 29%，纯跌 43%。用户的推论是"去掉震荡 → 剩下纯涨"。要检验这句话，得先问清楚：**"去掉"发生在哪里？**
- 在**训练标签**里去掉：模型只从"明确的涨"和"明确的跌"学方向，不被一大堆 ±5% 的噪声样本稀释。
- 在**推荐结果**里去掉：我们根本不知道哪只票未来会震荡，只能预测它；预测错的那部分照样会进清单。

## 核心思想 (讲直觉, 不堆公式)
1. **三重障碍（可调）**：上障碍、下障碍、时间障碍谁先到就是标签；撞到时间障碍的就是"震荡"。Kang 2025 的经验是按数据把障碍调到各类比较均衡，印证了用户"参数不写死"。
2. **中性类 = 弃权**：三分类里中性类的意义不是"又一个要预测的类"，而是"证据不够时不下注"。在选股里，它对应**"这只不进名单"**，而不是"它会横盘"。
3. **元标签**：主模型先给候选（保召回），二级模型**只在主模型选中的样本上**训练，标签是"这次主模型对不对"。它提高的是精度，代价是召回。这正是用户说的"反向评分把里面可能下跌的减值掉"，而且它规定了**反向模型该在哪个分布上训练**：在主模型的候选池里，而不是全市场。

对照 R01：全市场训练的下跌模型会给大波动票打高分，而 stage1 头部本来就是大波动票，于是会误杀头部。元标签在候选池内训练，天然扣掉了"振幅"这个共同因子，剩下的才是方向。

## 我们怎么吸纳 / 改造
- **训练端**：方向模型只用"先涨 +U（且回撤没过 −M）"对"先跌 −D"这两类样本训练，震荡样本**不进方向模型**（这是用户的"去掉震荡"在正确位置的落点）。震荡交给振幅轴（stage1 已经做好）。
- **推荐端**：不承诺"删掉震荡"。震荡票进了名单，经济后果是小亏小赚，要在 R05 用收益量出来，而不是当成失败。
- **反向减值 = 元标签**：二级模型在 stage1 每日 Top-K（比如 100）里训练，标签是"纯跌 vs 非纯跌"或"纯涨 vs 其他"，用 purged walk-forward。本仓 09-26 的失败分类器（AUC 0.66，只比朴素裁剪多 1.4pp）其实已是元标签的雏形，但它的标签是 `positive20==0`，把震荡、脏涨、纯跌混成了一类，这正是它弱的一个可疑原因，R05 要单独拆开验。
- **不照搬**：元标签原文用在单一资产的时序信号上，一个样本就是一次交易；我们是截面选股，同一天 100 个候选共享一个大盘，R01 已证明纯跌 29% 的方差在"哪一天"。所以二级模型必须带日级特征，或者日级另设一层（R06/R07）。

## 结果与裁决
🧩框架。本轮没有新数字，推出的是**两个待验假设**：
- H-a：方向模型剔除震荡样本训练后，在候选池里对"纯涨 vs 纯跌"的区分力高于把震荡混进去训练的模型。（R04 验）
- H-b：候选池内训练的元标签，比全市场训练的 unS20 更适合做减值。（R05 验）

## 思想谱系 (演化)
- 取代了: "全市场对称训练一个下跌榜去扣分"的直觉版本（[[2026-09-27_s20-uns20-downside-mirror]] 的实验就是这种形态）。
- 被取代 / 下一步: [[2026-09-28_s20-pure-r03-asymmetry]]
- 同源 / 对照: [[2026-09-26_s20-amplitude-not-direction]]、`wiki/smartgoal/2026-09-26_s20_failure_prune_and_pump.md`

## 移植提示 (必填)
先分清"标签里的中性"和"推荐里的中性"：前者可以删，后者只能预测。反向模型一律按元标签的方式在主模型候选池里训练，并在报告里同时给"扣掉振幅后"的区分度。不可移植：A 股截面同日强相关，元标签原文的单资产独立样本假设在这里不成立。
