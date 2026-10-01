# 理论体检：我们的选股为什么有效、哪里没有理论撑腰、近两年研究给的方向

- **日期**: 2026-09-30
- **provenance**: user（"选股方式是否符合市场、有无理论支撑；理论上的改进空间；近一两年研究的启发与计划"）/ ai-executed
- **裁决**: 🧩框架（理论对照 + 改进路线，尚未执行）
- **关联代码/实验**: 本条不新增实验，引用 [[2026-09-28_s20-pure-chain]]、[[2026-09-29_s20-pure-r13-vol-band]]、[[2026-09-29_market-valve-monitor]]、[[2026-09-30_r20-feature-store-rebuild]] 的结果

## 一句话
我们的三件主要武器——**截掉极端波动、低波动稳健版、普跌预警阀门**——都能在经典资产定价文献里找到支撑（特质波动率之谜 / 彩票股效应、低风险异象、波动聚集与波动管理组合）；**真正没有理论撑腰的是"方向"**：价量信息主要刻画振幅，这与中国市场机器学习研究的结论一致。最大的理论缺口是**风格暴露未审计**（规模、价值、换手），以及阀门用"开关"而不是文献支持的"连续调仓"。

## 出处 (必填, 无出处不写)
- 机器学习选股：S. Gu, B. Kelly, D. Xiu, "Empirical Asset Pricing via Machine Learning", *RFS* 33(5), 2020；M. Leippold, Q. Wang, W. Zhou, "Machine learning in the Chinese stock market", *JFE* 145(2), 2022, DOI 10.1016/j.jfineco.2021.08.017。
- 中国定价因子：J. Liu, R. F. Stambaugh, Y. Yuan, "Size and value in China", *JFE* 134(1), 2019（CH-3：规模、EP 价值；剔除最小 30% 壳价值污染）。
- 波动与彩票：A. Ang, R. Hodrick, Y. Xing, X. Zhang, *JF* 61(1), 2006；T. Bali, N. Cakici, R. Whitelaw, "Maxing out", *JFE* 99(2), 2011；中国 MAX 效应综述性证据（散户主导、套利受限股票更强）：<https://www.tandfonline.com/doi/full/10.1080/23322039.2023.2175471>、<https://onlinelibrary.wiley.com/doi/10.1111/acfi.13354>。
- 低风险异象：A. Frazzini, L. H. Pedersen, "Betting Against Beta", *JFE* 111(1), 2014。
- 波动聚集与波动管理：R. F. Engle, *Econometrica* 1982；A. Moreira, T. Muir, "Volatility-Managed Portfolios", *JF* 72(4), 2017；中国证据 "Volatility-managed portfolios in the Chinese equity market", *Pacific-Basin Finance Journal* 88, 2024 <https://www.sciencedirect.com/science/article/abs/pii/S0927538X24003263>（考虑涨跌停后效果更强）。
- 恐慌后反弹：K. Daniel, T. Moskowitz, "Momentum Crashes", *JFE* 122(2), 2016；W. De Bondt, R. Thaler, *JF* 1985。
- 近两年：B. Kelly, B. Kuznetsov, S. Malamud, T. A. Xu, "Artificial Intelligence Asset Pricing Models", NBER w33351, 2025 <https://www.nber.org/papers/w33351>；Y. Shi et al., "Kronos: A Foundation Model for the Language of Financial Markets", arXiv:2508.02739（AAAI 2026）<https://arxiv.org/abs/2508.02739>；Z. Tang et al., "AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay", KDD 2025, arXiv:2502.16789 <https://arxiv.org/abs/2502.16789>；Y. Li et al., "R&D-Agent-Quant", NeurIPS 2025, arXiv:2505.15155 <https://arxiv.org/abs/2505.15155>；LLM 读中文新闻预测收益（ABFER 讲座 "Large Language Models and Return Prediction in China" <https://www.abfer.org/component/edocman/main-webinar-series/large-language-models-and-return-prediction-in-china>；"News Sentiment and Overnight Return Prediction" 2025 <https://www.researchgate.net/publication/395405756>）。

## 为什么引入 (第一性原理)
十几轮实验告诉了我们"什么有效"，但没回答"为什么有效、会不会失效"。只有能落到已知经济机制上的效果，才有理由相信它在新数据上继续存在；落不到机制上的，要么是新发现，要么是过拟合。

## 核心思想 (讲直觉, 不堆公式)
**一、逐项体检**

| 我们的做法 | 对应理论 | 支撑强度 | 我们的证据是否一致 |
|---|---|---|---|
| stage1：166 个价量因子的机器学习排序 | Gu–Kelly–Xiu 2020；Leippold 等 2022（中国市场 ML 可预测性强，但主要来自流动性/小盘/散户相关特征，且随时间衰减） | 中 | 一致：它主要识别"会动"，方向弱（[[2026-09-27_s20-two-axis-decomposition]]） |
| 截掉池内最高波动（振幅上限） | 特质波动率之谜（Ang 2006）+ 彩票股/MAX 效应（Bali 2011，中国散户市场更强） | **强** | 一致：一行规则追平所有学习式反向模型（R09），新窗口兑现 |
| 稳健版：只在低波动 40% 里选 | 低风险异象（Frazzini–Pedersen 2014） | 强，但**依赖行情** | 一致：开发期全面占优；2026 年高波动主导的行情里落后——低风险异象在投机性上涨中本就会阶段性失效 |
| 普跌阀门：近 5 日跌停合计 | 波动聚集（Engle 1982）+ 波动管理组合（Moreira–Muir 2017；中国 2024 证据） | 强（对"风险可预测"） | 一致：短期普跌可预测（AUC 0.59–0.64） |
| 阀门的"最高警报常是恐慌尾声" | 恐慌状态后的反转（Daniel–Moskowitz 2016；De Bondt–Thaler 1985） | 强 | 一致：开发期红色日之后反弹，这不是噪声，是有名字的现象 |
| 首达标签、区间出场 | 三重障碍（López de Prado 2018） | 实务 | — |

**二、理论指出的缺口**
1. **风格暴露没审计。**中国最稳的定价因子是规模、价值（EP）和换手（Liu–Stambaugh–Yuan 2019）。我们的可移植特征**排除了市值和估值**，名单的超额收益有多少只是"押小盘/押低估值/押高换手"，目前不知道。小盘溢价在中国还带着"壳价值"污染（CH-3 剔除最小 30%）。这也决定容量：如果收益集中在微盘股，资金一大就做不出来。
2. **阀门用的是"开关"，文献支持的是"连续调仓"。**Moreira–Muir 的做法是仓位与近期波动成反比，平滑地降仓，而不是红色日整体切换名单；中国 2024 的证据还说考虑涨跌停后效果更强。
3. **阀门分不清"恐慌开始"和"恐慌尽头"。**Daniel–Moskowitz 的 panic state 是"高波动 + 已大跌"，之后反弹最强。我们的信号只看跌停数量的水平，没看它在**上升还是回落**。这是一个有理论指向、可以事先登记的改进。
4. **方向信息依然空缺。**价量做不出方向；文献里能做出方向的是**带时点的公司/市场新闻**（多篇 2024–2025 研究显示 LLM 读中文新闻的语气能预测收益）。我们用新闻联播失败，是文本源错了，不是方法错了。

**三、近两年研究的启发**

| 研究 | 新意 | 对我们的意义 | 风险 |
|---|---|---|---|
| Kronos（AAAI 2026）K 线基础模型 | 120 亿条 K 线预训练，零样本预测 | 最可能有用的地方是**波动/振幅预测**，正好服务振幅上限和阀门；不该指望它给方向 | 同信息（价量）的天花板；要和 natr 这类朴素量对比 |
| AIPM（Kelly 等 2025）把 Transformer 放进定价核 | 跨资产共享信息，非线性 | 支持"同一天的股票不是独立的"——行业共振、日级风险都是跨资产信息 | 工程重，样本要求高 |
| AlphaAgent（KDD 2025）、R&D-Agent-Quant（NeurIPS 2025） | LLM 自动挖因子，用原创性约束、假设一致性、复杂度控制对抗 alpha 衰减 | 借它们的**约束**（新因子必须与已有因子不相似、必须先写出经济假设）来管住我们自己的研究循环 | 本仓 20+ 次否定说明"多挖"不是瓶颈，"信息源"才是 |
| LLM 读中文新闻 / 隔夜新闻 | 新闻语气对 A 股收益和 CSI300 隔夜有预测力 | 方向轴唯一有文献支撑的入口 | 历史回测有记忆泄漏；需要带时间戳的文本，只能前瞻积累 |

## 我们怎么吸纳 / 改造 —— 计划（按"理论把握 × 成本"排序）
1. **风格暴露体检（低成本，现在就能做）**：用 Tushare 补齐日度市值、EP、换手，算出进攻版/稳健版/R20 名单的规模、价值、换手暴露；构建 CH-3 因子，回归名单日收益，看剩下的 alpha；加一档"剔除最小 30% 市值"的稳健性检验。决定我们是不是在押小盘，以及容量上限。
2. **阀门 v2（有理论指向，需事先登记）**：(a) 连续调仓：仓位 = 目标波动 / 近期市场波动，上限 100%；(b) 区分恐慌阶段：跌停数量处于高位且**仍在上升**才降仓，高位但**已回落**视为恐慌尾声不降仓。现在有 Tushare token，可以把市场数据拉回 2015 年，独立窗口从十几个增加到一百多个，再在新数据上前瞻验证。
3. **MAX 显式化（低成本）**：把"过去 20 日最大单日涨幅"作为振幅上限的第二个维度，与 natr 对比，按朴素消融纪律决定取舍。
4. **方向轴的新信息源（前瞻）**：确认 Tushare 新闻/公告接口权限，从今天起带时间戳采集，用 LLM 打语气分；只在新数据上评估，验收指标用池内逐日方向 AUC。
5. **Kronos 做波动预测（中成本）**：零样本预测未来 5/20 日波动，和 natr、近 5 日跌停数比较，用于振幅上限和阀门；不用它预测方向。
6. **研究流程借鉴 AlphaAgent 的约束**：任何新特征上线前，先写经济假设、再查与已有特征的相似度、最后过朴素对照。

## 结果与裁决
🧩 框架。结论：现有的有效部分都站在成熟的异象之上（彩票/特质波动、低风险、波动聚集），这既是信心来源，也意味着它们是"已知、可能拥挤、在投机行情中会阶段失效"的收益源。下一步优先补上风格暴露审计和阀门 v2。

## 思想谱系 (演化)
- 取代了: 只用实证结果判断方案好坏的做法。
- 被取代 / 下一步: 风格暴露体检、阀门 v2 预登记。
- 同源 / 对照: [[2026-09-26_s20-amplitude-not-direction]]、[[2026-09-29_jev-market-selloff-probe]]

## 移植提示 (必填)
换市场时，先把每个有效组件对应到一个已知机制；对应不上的，默认按过拟合处理，直到在新数据上证明。在散户主导、有涨跌停的市场里，彩票效应、低风险异象和波动聚集通常更强；在机构主导的市场里要重新验证。风格暴露审计应该是任何选股方案的第一道检查。
