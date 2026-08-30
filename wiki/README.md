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
| 2026-08-29 | [[2026-08-29_pool-e-published-contract]] | Pact Consumer-Driven Contracts / POSIX atomic rename / stock_benchmark 稳定导出契约 | ✅落地 | 池E只消费权威完整快照，下游复验100只/配额/15策略并用真实signal_date守住最近良好版本 |
| 2026-06-25 | [[2026-06-25_worldquant-brain-pipeline]] | WorldQuant BRAIN platform / FASTEXPR / 内生 anti-overfitting | 🧩 框架 | 跨到未枯竭美股空间, 管道 live, 借平台checks当gate; analyst4三批: EPS修正动量峰Sharpe0.93@120d<1.25门槛=优质building block非独立alpha |
| 2026-06-24 | [[2026-06-24_improvement-loop-methodology]] | Ralph Wiggum (Carson) / Sakana AI Scientist / 内生 anti-overfitting | 🧩 框架 | loop 在枯竭空间只跑工程+累积+复检, 不跑挖矿; 每轮选→执行→评估→总结→提下一步 |
| 2026-06-24 | [[2026-06-24_anti-overfitting-scaffold]] | ICBINB / Sakana AI Scientist / Bailey PBO·CSCV / ARA paper | 🧩 框架 | 用"reviewer+gate+walk-forward+PBO"的反过拟合脚手架做研究，吞吐不是目标、证伪才是 |

---
*操作记录看 `git log` / `memory/`；这里只看思想。*
