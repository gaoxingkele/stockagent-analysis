# 思维演化 Wiki — StockAgent

> Karpathy 风格的"思想吸纳层"。这里记录的不是"做了什么"（那在 git log / memory / README），
> 而是**为什么这样想、从哪借来的思想、怎么改造、它取代/被谁取代、怎么移植复现**。
> 全局自动行为，定义见全局 `CLAUDE.md` → "思维演化 Wiki" 一节。

## 这是什么 / 为什么存在
本项目两年挖了 20+ 个假设，绝大多数 REJECT。真正值钱的不是某个因子，而是**一套"怎么想、怎么自我证伪、从哪些论文/repo 借力"的方法论**。
单看 commit 看不出思想从哪来、为什么转向。本 wiki 把每一次"思想吸纳"显式化并**强制标注出处**，目的：

1. **溯源** — 任何一个设计决策都能追到它的论文/repo 来源。
2. **移植** — 换个数据集/市场/项目时，照着条目就能复现这套思维，而不只是抄代码。
3. **防自欺** — 把"为什么相信它"写下来，未来能检验当时的推理是否成立。

## 怎么写一篇（硬规则）
- 复制 `_TEMPLATE.md` → `wiki/<YYYY-MM-DD>_<slug>.md`。
- **必须有引用出处**（论文标题+arXiv/DOI / GitHub URL / 作者 / 链接）。无出处不写。
- Karpathy 风格：第一性原理、讲直觉和"为什么"、叙事清楚、可被另一个 Agent 照着复现。
- 用 `[[条目名]]` 互链思想谱系。
- 结尾必须有"移植提示"。
- 写完在下面索引表加一行。

## 索引 (思想谱系, 倒序)

| 日期 | 条目 | 思想来源 (出处) | 裁决 | 一句话 |
|------|------|----------------|------|--------|
| 2026-10-01 | [[2026-10-01_announcement-sentiment-and-revisions]] | Chan-Jegadeesh-Lakonishok 1996 / Loughran-McDonald 2011 / 本仓四次文本方向验证 | ❌ | 2 万条公告 LLM 情绪 AUC 0.50，利好优先/剔除利空都不如随机；研报 EPS 修正覆盖仅 6-11%、AUC 0.39；公开文本在候选池内已定价 |
| 2026-10-01 | [[2026-10-01_risk-gates]] | Frazzini-Lamont 2007 / Barber et al. JFE 2013 财报溢价 / Field-Hanka JF 2001 解禁 | ❌/⏸ | 解禁、减持、负预告、财报窗口四个排雷门均未过（对照随机剔除）；财报窗口两段一致为加分项，登记为 v1.2 候选 |
| 2026-10-01 | [[2026-10-01_turnover-amount-ablation]] | DGTW 1997 / Liu-Stambaugh-Yuan 2019 / 本仓朴素消融纪律 | 🧩/⏸ | 去掉 16 个量能特征开发期变差（每笔 1.34→1.07）但换手偏向不变：偏向是振幅选择副产品；成交额≥5亿仍有效，容量非约束；两个候选待预登记 |
| 2026-10-01 | [[2026-10-01_style-audit-not-small-cap]] | DGTW 1997 特征匹配基准 / Liu-Stambaugh-Yuan 2019 | ❌/🧩 | S20 进攻版市值分位 0.66、微盘仅 16%，不是小盘；偏高换手/高成交/低 EP/高波动；扣风格后剩 +0.50pp/笔（区间含 0） |
| 2026-09-30 | [[2026-09-30_theory-audit-and-roadmap]] | Gu-Kelly-Xiu 2020 / Leippold 2022 / Liu-Stambaugh-Yuan 2019 / Ang 2006 / Bali 2011 / Frazzini-Pedersen 2014 / Moreira-Muir 2017 / Daniel-Moskowitz 2016 / Kronos / AIPM / AlphaAgent / RD-Agent-Q | 🧩 | 振幅上限、稳健版、阀门都有成熟异象支撑；方向无理论撑腰；缺口=风格暴露未审计、阀门应连续调仓并区分恐慌阶段 |
| 2026-09-30 | [[2026-09-30_r20-feature-store-rebuild]] | 本仓生产因子流程 / Peng, Science 2011 可复现研究 | ✅ | 因子库补到 09-28，R20 重放与生产发布逐只一致（21 天 Jaccard 1.0）；公平窗口 95 天 S20 进攻版每笔 +2.0pp [+0.2,+3.8]，但优势集中在 4–6 月，7–8 月无优势 |
| 2026-09-29 | [[2026-09-29_market-valve-monitor]] | Engle 1982 / Bollerslev 1986 / De Bondt–Thaler 1985 / arXiv:2603.13252 | ✅监测 | 近 5 日跌停合计作普跌阀门（绿/黄/橙/红），全样本 AUC 0.59–0.64；最高警报常是恐慌尾声，开发期无动作优于不动 → 监测模式 + 3 个预登记动作；09-30 用户启用 B（红色日进攻版换稳健版） |
| 2026-09-29 | [[2026-09-29_jev-market-selloff-probe]] | TypeSafe Jev API / Engle 1982 ARCH / Bollerslev 1986 GARCH / AKShare 新闻联播 | ❌/🧩 | Jev 判 5 日普跌 AUC 0.73、无背答案，但错配新闻不掉分，免费梯度提升同数字 0.72；日级普跌可预测（波动聚集），Jev 无增量 |
| 2026-09-29 | [[2026-09-29_s20-pure-r13-vol-band]]（R11–R13） | Ang et al. JF 2006 / Frazzini–Pedersen JFE 2014 / AFML 区间障碍 / 本仓朴素消融纪律 | ✅影子 | 目标改为安全且上涨、止盈为区间；stage1 按安全口径比全市场更危险；学习式安全模型输给低波动朴素规则；全市场低波动 40% + stage1 排序 + 行业扩容 = v1.1 稳健版 |
| 2026-09-29 | [[2026-09-29_s20-pure-v1-freeze]] | 本仓 S20-Pure 十轮链 / 反过拟合脚手架 / TA-Lib matype | ✅影子 | stage1 锚模型补存并逐行复现；冻结先于测试；ta-lib 默认 EMA 漂移已修；新窗口 17 天振幅上限相对 stage1 +1.9pp/笔、纯跌 −14pp，但跑输全市场 |
| 2026-09-28 | [[2026-09-28_s20-pure-chain]]（R01–R10） | AFML 三重障碍/元标签 / Joubert JFDS 2022 / Jin&Candès JMLR 2023 / Gibbs&Candès NeurIPS 2021 / KDD 2025 PDU / arXiv:2603.13252 / Ang et al. JF 2006 / Grinold–Kahn | 🧩框架 | 纯涨/纯跌/震荡参数化；剔除震荡训练有效但靠降振幅；反向减值 = 池内振幅上限（一行规则追平模型）；胜率主方差在日级：好日子 70%、坏日子 35% |
| 2026-09-27 | [[2026-09-27_s20-two-axis-decomposition]] | EJOR 2024 首次触线 / arXiv:2603.13252 两层不确定性 / KDD 2025 PDU / 本仓 Stage1 审计 | 🧩框架 | 正+反拆成幅度轴×方向轴；方向含量看 up/(up+down) 斜率；方向轴只加信息，禁双二分类 |
| 2026-09-27 | [[2026-09-27_s20-uns20-downside-mirror]] | EJOR 2024 首次触线 / KDD 2025 PDU / arXiv:2603.13252 / TOIS 2024 SVAT | 🧩框架 | 下跌标签要单独按先触 −10% 训练；镜像不是自动 SOTA，stage1 也不该换成深度排序器 |
| 2026-09-26 | [[2026-09-26_s20-amplitude-not-direction]] | TypeSafe/Jev System One 文档 + 本仓 v4 打乱对照与 09-20 诊断 | 🧩框架 | S20 的可用信息是振幅不是方向；语义模型无文本则无增量，且错配对照必须与标签独立 |
| 2026-09-20 | [[2026-09-20_s20-causal-screen-specified-rise]] | Dream-RSI / Harness-RSI / Cunningham / Selective classification / Conformal Risk Control / 本仓 v4 诊断 | 🧩框架 | 入池看指定 +15% 止盈；沉默是任务支持不是质检；因果冷却未迁移，发运仍 π0 |
| 2026-09-13 | [[2026-09-13_s20-three-track-evidence-gate]] | Selective classification / Conformal Risk Control / DLinear / DoubleAdapt / MASTER / 本仓 v3·B1 负结果 | 🧩框架 | S20-Safe：O/P/T 三轨分开；TopN 可空缺；先修数据与联合风险再放行复杂模型 |
| 2026-08-29 | [[2026-08-29_pool-e-published-contract]] | Pact Consumer-Driven Contracts / POSIX atomic rename / stock_benchmark 稳定导出契约 | ✅落地 | 池E只消费权威完整快照，下游复验100只/配额/15策略并用真实signal_date守住最近良好版本 |
| 2026-06-25 | [[2026-06-25_worldquant-brain-pipeline]] | WorldQuant BRAIN platform / FASTEXPR / 内生 anti-overfitting | 🧩 框架 | 跨到未枯竭美股空间, 管道 live, 借平台checks当gate; analyst4三批: EPS修正动量峰Sharpe0.93@120d<1.25门槛=优质building block非独立alpha |
| 2026-06-24 | [[2026-06-24_improvement-loop-methodology]] | Ralph Wiggum (Carson) / Sakana AI Scientist / 内生 anti-overfitting | 🧩 框架 | loop 在枯竭空间只跑工程+累积+复检, 不跑挖矿; 每轮选→执行→评估→总结→提下一步 |
| 2026-06-24 | [[2026-06-24_anti-overfitting-scaffold]] | ICBINB / Sakana AI Scientist / Bailey PBO·CSCV / ARA paper | 🧩 框架 | 用"reviewer+gate+walk-forward+PBO"的反过拟合脚手架做研究，吞吐不是目标、证伪才是 |

---
*操作记录看 `git log` / `memory/`；这里只看思想。*
