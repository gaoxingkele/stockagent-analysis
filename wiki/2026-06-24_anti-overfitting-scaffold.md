# 反过拟合研究脚手架 — 吞吐不是目标，证伪才是

- **日期**: 2026-06-24
- **provenance**: ai-executed (元结论由多会话累积, 用户多次确认方向)
- **裁决**: 🧩框架 (这是贯穿全项目的元方法, 不是单个因子)
- **关联代码/实验**: research/* 各分支; memory 中 20+ 条 REJECT 记录; `feedback_ai_scientist_lesson_gate_not_throughput_0615` / `feedback_quant_system_meta_lessons_0524`

## 一句话
在一个信号已近枯竭的价量空间里，自主搜索的瓶颈**不是产生假设的吞吐量**，而是**剔除假阳性的能力**；
所以研究环的核心是 reviewer + gate + walk-forward + PBO 这套"对抗式证伪脚手架"，而不是更快地生成更多因子。

## 出处 (必填)
- **ICBINB**: "I Can't Believe It's Not Better" workshop 系列 (NeurIPS 2020+), https://i-cant-believe-its-not-better.github.io/ —
  把"诚实的负结果 + 方法论教训"当成一等贡献。本项目 20+ 否决即按 ICBINB 范式当成贡献而非失败。
- **Sakana AI Scientist**: Lu, Lu, Lange, Foerster, Clune, Ha, *"The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery"*, arXiv:2408.06292 (2024), https://github.com/SakanaAI/AI-Scientist —
  自主"生成假设→实验→写论文"流水线。我们的 ralph(实验)=worker / codex(reviewer) / scaffold(gate) 即其原始版；但**反向教训**：高吞吐自主搜索在枯竭空间只量产假阳性。
- **Bailey, Borwein, López de Prado, Zhu**, *"The Probability of Backtest Overfitting (PBO)"*, Journal of Computational Finance 20(4), 2016；及 *"Pseudo-Mathematics and Financial Charlatanism"*, Notices of the AMS 61(5), 2014 —
  CSCV (Combinatorially Symmetric Cross-Validation) 算 PBO；"在同一 OOS 上反复调参=反向过拟合"的形式化根据。
- **López de Prado**, *"Advances in Financial Machine Learning"* (Wiley, 2018) —
  walk-forward / **embargo** / purged CV / **triple-barrier labeling**。我们的 r20 triple-barrier 目标、月度 walk-forward 重训、embargo 收尾全部源此。
- **ARA paper**: *"The Last Human-Written Paper: Agent-Native Research Artifacts"*, arXiv:2604.24658 —
  把研究做成"含死胡同的探索图"，死胡同留痕。本 wiki + memory 的 REJECT 留痕即此。

## 为什么引入 (第一性原理)
公开的价量因子，市场效率会把 alpha 追平 → 我们反复观察到"新 OOS 调出来的 +1pp，扩窗/walk-forward 后塌成 ≈0 甚至反指"
(`feedback_quant_system_meta_lessons_0524`)。根本问题不是"想不出新因子"，而是**没有一个机制能在我们自欺前拦住自己**。
人手研究最大的失败模式是：在同一段 OOS 上小修小补堆叠，每步都"看起来更好"，合起来是对噪声过拟合。Bailey 的 PBO 把这件事量化成概率。
所以正确的投入方向是建"证伪基础设施"，不是建"更快的因子生成器"。

## 核心思想 (讲直觉)
1. **Gate 在前，吞吐在后**：先用廉价筛查 (Phase-1: 因子自身 IC / 正交残差 IC / no_residual) 砍掉 90% 候选，再花算力做 walk-forward。
2. **对抗式 reviewer**：第二个独立 Agent (codex CLI) 专门挑"这个改进是不是成交假设乐观 / 缺 embargo / gate 太松 / 是 stratification artifact"。多次把看似 +1.36 的胜利逐步证伪到 0 (`project_wfe_embargo_tp_dead_0614`)。
3. **walk-forward + embargo 才算数**：固定 OOS 的数字一律打折读；in-sample IC +0.44 在 true-OOS 常只剩 +0.18 (`project_wf001_delookahead_real_pnl_0614`)。引用任何绝对 Sharpe/年化前先问"去 lookahead 了吗"。
4. **placebo / 随机对照**：任何 overlay 都和"随机选同样数量的股"比；比随机还差就是零技能 (`project_market_state_exposure_reject_0618`)。
5. **REJECT 是产出**：每个否决写成带 provenance 的死胡同，避免未来重复踩 (ARA 范式)。

## 我们怎么吸纳 / 改造
- López de Prado 的 triple-barrier/embargo/purged-CV：照搬思想，落到 A 股 (T+1、涨跌停、ST 源头排除) 的具体实现。
- Sakana 的自主流水线：**只取"多 Agent 分工"骨架，砍掉"高吞吐自主搜索"**——因为我们的信号空间枯竭，吞吐反而有害 (`feedback_ai_scientist_lesson_gate_not_throughput_0615`)。
- ICBINB：把"诚实 REJECT、不 p-hack"制度化——ratio 相位研究跑完老老实实 REJECT 未 p-hack，是脚手架首次实战自证 (`project_ratio_phase_dynamic_window_reject_0603`)。
- Bailey PBO：作为"同一 OOS 反复调参"的红线告警，触发了从固定 OOS → walk-forward 的范式迁移。

## 结果与裁决
- 整轮 20+ 假设否决，但**脚手架本身是真正的资产**：它多次在"看似胜利"处抓出 stratification artifact / lookahead / 乐观成交 (Hybrid +54%→walk-forward -15%；TP +1.36→embargo 后含 0)。
- 元结论：选股层已挖到底，增量在 book/风控/部署或真新数据；这个判断之所以可信，正因为它是被这套脚手架反复证伪后才下的，不是凭感觉。

## 思想谱系 (演化)
- 取代了: 早期"固定 OOS 上小修小补"的研究方式 (V12 v3-v8 同窗调参)。
- 被取代 / 下一步: 前向 paper-trade (唯一能解"真实 Sharpe 宽带"的办法, 见 SESSION_HANDOFF 下一步#1)；WorldQuant BRAIN (换到未枯竭的美股信号空间)。
- 同源 / 对照: ARA 死胡同留痕 ↔ 本项目 memory 的 REJECT 链。

## 移植提示 (必填)
搬到任何"信号可能枯竭 + 容易自欺"的预测/策略项目都成立 (量化、推荐、A/B 实验挖掘)：
1. 先建 gate（廉价正交/残差筛查）再谈吞吐；
2. 上独立 reviewer Agent，专职找"这个改进是不是假象"（成交乐观/缺 embargo/分层 artifact/gate 太松/placebo 没做）；
3. 任何固定 OOS 的数字默认打折，以 walk-forward + embargo + 随机 placebo 为准；
4. 把否决当一等产出留痕（出处 + 为什么否）。
**本项目特有不可移植**：A 股 T+1/涨跌停/ST 的具体实现细节、triple-barrier 的 r20 参数——这些是实现，不是思想。
