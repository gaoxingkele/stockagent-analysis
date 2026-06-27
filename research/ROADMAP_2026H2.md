# V12.31 研究 / 部署 完善 Roadmap (2026 H2)

> 2026-06-15。整合本轮全部收敛结论 (21 实验 + 5 轮 codex 交叉审计 + book 层 BT/EX/WF/WFE/FIN/DEP/DIAG)
> + 两篇外部论文借鉴 (Sakana AI Scientist / Nature 2026; QuantML TreeSHAP+消融 A股 XGBoost)。

## 0. 指导性结论 (一切计划的前提, 别再推翻)

1. **V12.31 是均值回归框架内的强局部最优**: 21 个改进假设 (价量/形态/正交/事件/止盈/triple-barrier/within-concept动量) 全否; 龙虎榜席位是唯一扛消融真信号但 long-only 不可落地。
2. **真实可交易基线 = 净 Sharpe ~1.31 (年化~35%, leak-free)**, 但 **CI [0.11, 2.60] 宽**, P(>0)=99% / P(>1)=66% → **正 alpha 方向可信, 幅度不可信, 1.31 不是可承诺下限, 只是 paper-trade 锚点**。
3. **瓶颈是信号枯竭 + 21 月样本功率, 非研究流程**。AI Scientist 变不出数据里没有的 alpha, 也修不了功率; 高吞吐自主搜索在枯竭价量空间只量产假阳性 (本轮 RETRO+0.0287 → CL 杀掉已 live 演)。
4. **增量不在选股, 在 book/风控/部署 + 研究流程可信度 + (低先验) 新数据**。
5. **生产线 V12.31 (v12_scoring/v12_dual_track) 全程只读冻结**; PIT-clean r20 (剔 holder_pct) 是可部署变体。

## Track A — 现在做 (便宜 / 高价值 / book+部署层)

**A1. 低换手 / 稳定化组合构建** ★ 唯一真实 book 层杠杆
- 动机: DIAG-001 证 picks 对小扰动重排 106% (选股不稳/隐性高换手); QuantML 论文证高换手吃 alpha (0.6% 成本 Sharpe 2.23→1.67) 须加换手惩罚; EX-002 风格暴露。
- 做法 (预注册, 在 BT 引擎): 给 book 构建加 ① 换手惩罚 / 持仓粘性 (conviction 变化超阈才换) ② 增 N / 降集中 三档对照; embargo + close-based, apples-to-apples vs PIT-clean 基线。
- gate: 换手↓ + 稳定性↑ (扰动重排率↓) 同时 Sharpe 不降 (CI 不更差) → book 层真改进 (非选股 alpha, 是成本/稳定性)。
- 预期: 中等。这是治"刀尖选股 + 成本"的对路杠杆, 且不依赖新信号。

**A2. TreeSHAP 可解释层** (部署/审计, 非 alpha)
- 给 PIT-clean r20 加逐只票 TreeSHAP 加性归因 → 每日推荐附"为什么打高分"。
- 用途: 前向 paper-trade / 向自己/出资人解释"赚的什么钱" (QuantML 论文核心用例); 提高可审计性, 不提高选股。

**A3. 冻结 PIT-clean V12.31 + 前向 paper-trade 上线** ★ 部署地基
- FIN-002 已起草协议 (V12.31_BASELINE.md): append-only 落每日 picks → 满 20/40d 算实现 P&L → 分 regime 对照 1.31±CI → 红线=前向数据不回流调参。
- 现在做实: 用 PIT-clean r20 (非带 holder 的), 接 daily_review 落库, 攒真实前向数据 (唯一能解"幅度宽带"的途径)。

## Track B — 研究流程升级 (一次性投资, 让未来每个候选骗不过)

> AI Scientist 的真教训在 reviewer 一侧; 我们失败模式 = 假阳性 (RETRO→CL 用了两轮才证伪)。强 gate 一轮就能抓。

**B1. Reviewer 集成 panel**: codex + claude + (可选第三模型) 独立审 + meta-judge, 替代单 codex 审。
**B2. Replication 默认化**: 任何候选先过多 seed + per-cohort bootstrap 误差条才许"激动" (AI Scientist replication/aggregation 节点 = 已 defer 的 DIAG-003)。
**B3. 多重检验校正**: 候选若来自搜索 (如 RETRO 从 ~10 特征选出), 强制 Deflated Sharpe / PBO / family-wise, 惩罚试验次数。
→ 产物 = 一个"骗不过的 verify harness"升级, 非研究结果。是 Track C 任何新探索的前置。

## Track C — 延后 / 条件触发

**C1. r20 特征 block-ablation 剪枝** (卫生, 低 alpha 预期): 像 QuantML 剔估值因子那样, 分块消融找净负块剪掉 + 揪剩余泄漏风险特征。⚠ 效应量 (ΔAUC ±0.001-0.009) 大概率淹没在 21 月噪声里, 当卫生不当 alpha。
**C2. 新数据探索** (唯一可能出新选股 alpha): 另类数据/微结构/盘口/分析师修正 (限频未测)。低先验 + 需新数据管线; 只在愿投管线时开, 且必须走升级后的 Track B gate。
**C3. 7 月血洗窗复检** (dated, ~2026-07): 0508-0603 满 20d 前向数据后, 重跑 FIN-001 误差条 + audit_live_recs 复检 v3c 实盘动量血洗窗真实表现。

## 明确排除 (本轮已证, 别再开)

- ❌ 价量空间自主搜假设机 (枯竭, 量产假阳性);
- ❌ 止盈 / triple-barrier / within-concept 动量 复活 (已严格否);
- ❌ 把 1.31 当稳定 alpha 宣称 / 不修 holder 直接真钱上 (codex 红线)。

## 执行优先级 (claude 建议)

1. **A3 (paper-trade 上线) + A1 (低换手稳定化) 并行** ← 现在做, 一个攒数据一个治稳定性;
2. **A2 (TreeSHAP) 顺带** (A3 落库时附解释);
3. **B (研究环升级) 当地基**, 在开任何 C 之前完成;
4. **C2/C3 条件触发** (新数据意愿 / 7 月数据到)。

关联: `[[feedback_ai_scientist_lesson_gate_not_throughput_0615]]` · `[[project_fin_baseline_frozen_triplebarrier_reject_0614]]` ·
`[[project_diag_stability_global_reshuffle_0615]]` · `[[project_wfe_embargo_tp_dead_0614]]` · `[[feedback_quant_system_meta_lessons_0524]]`。
