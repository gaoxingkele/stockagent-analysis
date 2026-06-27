# 对照 Sakana AI Scientist (Nature 2026) 的研究方向反思

> 2026-06-15。用户分享 Sakana AI Scientist (Nature 651:914-919, DOI 10.1038/s41586-026-10265-5;
> AI 全自动跑完 ML 科研流程, ICLR ICBINB workshop 盲审 6.33 过), 要求跳出框框学其思路,
> 重思"如何提高选股(评分导向)效果"。本文记录分析结论 (论文素材 / 研究记录)。

## 1. 我们已在跑一个原始版 AI Scientist

| AI Scientist v2 | 本项目 |
|---|---|
| Agentic tree search 跑实验 | ralph-loop(prd=实验, verdict=节点) |
| Automated Reviewer(5 副本 o4-mini + area chair) | **codex CLI 5 轮交叉审计**(F1 比人高、本 session 抓到 2 次硬伤: lookahead 自信 / embargo 缺失) |
| 4 阶段 + 停止条件 + budget | prd 分 task + 预注册 gate + max-iterations |
| 6 种节点(buggy/refine/hyperparameter/ablation/replication/aggregation) | 有 ablation; **缺 replication/aggregation(多 seed + 误差条 = 已 defer 的 DIAG-003)** |

→ 这篇 Nature 验证的是**我们正在用的研究架构**; "学它"= 升级已有的环, 非从零。

## 2. 对"提高选股"的真教训是反直觉的

AI Scientist 解决"**做研究的流程效率**", 不是"**在有效市场挖新 alpha**"。媒体讲自主度, 对我们有用的是它的
**Reviewer + Limitations** 两节——它的边界(naive ideas / 1-3 过 / 人还在筛 / 多重检验)**就是我们的处境**。

**本 session 刚 live 演完这条教训**: RETRO-003 搜出 within-concept 动量残差 IC **+0.0287** 像"首个正向",
CL 阶段用**全因子消融 + 非重叠 t + 分 regime + 概念/PE桶交叉 + placebo** 把它**完整证伪** (regime_overlay 不落地)。
→ **照搬高吞吐自主搜索, 在枯竭价量空间只会更快量产我们还得回头杀掉的假阳性。更自主 = 更多假阳性。**

**瓶颈不是流程(已很好), 是 AI Scientist 修不了的两条**:
1. **信号在我们数据/universe 基本枯竭**(21 假设全否, 龙虎榜席位是唯一扛消融真信号但 long-only 不可落地);
2. **21 月样本功率太低**(剔 0.005% 因子能晃 Sharpe 0.46; codex#4: V12.31 自身是长研究路径的幸存者, CI 没惩罚搜索过程)。
**再强的搜索引擎也变不出数据里没有的 alpha, 也修不了 21 月功率。**

## 3. 该学 / 不该学

**✅ 该学(都在 reviewer 一侧, 因我们失败模式 = 假阳性)**:
1. **replication/aggregation 默认化**——候选先过多 seed + bootstrap 误差条才许激动(治噪声病);
2. **Reviewer 集成 + 多重检验校正**——codex 单审 → codex + claude + 第三模型独立 panel + meta-judge, 强制 Deflated Sharpe / PBO / family-wise(搜得越多 gate 越须狠);
3. **并行 tree-search 用于对抗性验证(群 skeptic 杀候选), 非生成更多候选**。

**❌ 不该建**: 价量空间自主搜假设机——本 session 已证它量产假阳性。

## 4. AI-Scientist 直接启发的诚实正面产出

它那篇过审论文是 **ICBINB(negative results)workshop**。**本项目这轮 21 个诚实否决 + LLM 交叉审计抓假阳性,
本身 = 一个 ICBINB 式的可发表方法论贡献**(反过拟合脚手架 + LLM cross-audit 在金融时序上抓 false positive)——
这才是这套自动化在我们这儿**真正的产出**, 而非"又一个 alpha"。

## 5. 下一步(三选一)

- **(a) 升级研究环**: 多模型 reviewer panel + replication 默认 + PBO 校正(最 AI-Scientist; RETRO→CL 那两轮, 强 gate 一轮就能抓)—— **推荐当地基**;
- **(b) 强化环指向新信息**(另类数据/微结构/盘口)—— 唯一可能出新 selection alpha, 低先验 + 需新数据管线;
- **(c) 接受选股近最优**, 增量在 book/风控/部署, leak-free V12.31 前向 paper-trade —— **推荐与 (a) 并行**。

**结论: (a) 地基 + (c) 并行; (b) 留给愿投新数据管线再开。单纯照搬 AI Scientist 去价量空间狂搜, 不建议——本 session 已亲眼看到那条路通向哪里。**

关联: `[[feedback_ai_scientist_lesson_gate_not_throughput_0615]]` · `[[feedback_quant_system_meta_lessons_0524]]` ·
`[[project_retro_within_concept_momentum_candidate_0615]]` · `[[project_book_backtester_attribution_0614]]`。
