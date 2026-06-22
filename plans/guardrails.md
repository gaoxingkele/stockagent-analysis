# Ralph Guardrails — research/ratio-phase

研究型循环的"signs"。核心与机械型 backlog 不同:**完成 = 产出工件 + 写下裁决(正/负都算完成)**,
不是"把 α 做大"。这些 SIGN 直接编码本项目的反过拟合血泪教训。

> "Progress should persist. Failures should evaporate." — Ralph 哲学
> 但在量化研究里还要加一句:**"宣布胜利的冲动必须被 walk-forward 和事前注册的 gate 中和。"**

---

## 反过拟合纪律 (本循环的灵魂)

### SIGN-R01: 不许移门柱
**Trigger:** 任何想修改 prd.json 里 `preRegisteredGate` 的时刻
**Instruction:** gate 数值在循环启动时写死, 任何迭代**绝不**修改阈值/指标/协议。需要改 = 立刻停手交人。
**Reason:** 研究循环若能调 gate, ralph 会一直磨到撞上目标 = 反向过拟合 OOS。见 feedback_quant_system_meta_lessons_0524。

### SIGN-R02: 负结果 = 合法完成
**Trigger:** T-005 walk-forward 没达到 gate
**Instruction:** 写 `verdict=REJECT` + 四项指标对照, task 置 `passes:true` 然后停。**禁止**在同一段 OOS 上反复重调特征/超参再试。
**Reason:** "再调一版试试"正是反向过拟合的引擎。一次诚实的 REJECT 比十次 p-hack 的 PASS 值钱。

### SIGN-R03: 中间指标 ≠ 落地
**Trigger:** 看到 IC / precision / lead-lag 曲线变好, 想据此宣布成功或落地
**Instruction:** 一律不作数。只有 T-005 的 walk-forward α 决定 ship。中间指标只写进 verdict 作描述。
**Reason:** feedback_train_label_over_inference_hack: 中间指标改善 ≠ 实战 α 改善。v3c+B 教训 -0.21pp。

### SIGN-R04: 泄漏自查是前置闸
**Trigger:** 任何产出 IC / 回测 / 特征 parquet 之前
**Instruction:** 先跑 `python research/verify.py` 的 leakage guard。撞 forward-field 黑名单 (r5/r10/r20/r30/r40/dd5-dd40 等) 即停。拐点 label 必须 lagged-confirmed。见到 IC>0.5 / Sharpe>10 先查泄漏和 ST。
**Reason:** feedback_forward_label_assistant_fields + project_gate_1_fail: 研究首跑 RankIC 0.56 就是 forward 字段泄漏的假突破。

---

## 生产安全 (绝对红线)

### SIGN-R05: 生产线 V12.31 冻结
**Trigger:** 任何迭代想动 src/stockagent_analysis/v12_scoring.py、生产模型文件、daily_top20_*.py
**Instruction:** 一律禁止 (T-006 落地除外, 且必须 opt-in 默认关、生产路径逐位一致)。所有工作只在 research/ 下 + research/ratio-phase 分支。verify.py 会校验生产文件 hash。
**Reason:** 生产线在跑实盘推荐, 任何意外改动 = 直接事故。

### SIGN-R06: ST 源头排除 + 全程分层
**Trigger:** 加载任何训练/回测/推理数据
**Instruction:** ST 在数据加载时就 filter (训练/回测/推理三处), 不是事后过滤。所有评估必带分层 (regime / 市值 / PE)。
**Reason:** feedback_st_exclude_at_source + feedback_stratified_analysis: ST 偏见制造 IC 0.77 假信号; 单因子全市场 IC≈0 但分层后翻 5-10 倍。

---

## 执行纪律

### SIGN-R07: 每次只改一处 + 改完重跑泄漏自查
**Trigger:** 单次迭代
**Instruction:** 保持改动小而聚焦, 一次推进一个 task。任何特征/label 改动后立即重跑 leakage guard。verify 通过才 commit。
**Reason:** 大改难定位; 泄漏会在不经意的 join 里溜进来。

### SIGN-R08: 长任务先 checkpoint
**Trigger:** 特征构建 / 回测预计 >5 分钟
**Instruction:** 第一版就要断点续跑 (parquet 分块 + 已完成跳过), 不能事后补。
**Reason:** feedback_long_task_persistence: 超 5 分钟的第一版就要有 checkpoint。

### SIGN-R09: 每迭代更新 progress.md
**Trigger:** 结束任一迭代前
**Instruction:** 写清本轮裁决、产出工件路径、下一步。fresh-context 每轮干净重启, 全靠 progress.md 续命。
**Reason:** 否则下一轮重新发现同样的东西, 浪费迭代。

### SIGN-R10: 完成前清点全部 task
**Trigger:** 准备输出完成 promise
**Instruction:** 重读 prd.json, 数剩余 `passes:false` 且非 skip 的 task。全部有 verdict 才 promise。
**Reason:** 防止提前退出循环留下未完成的研究步骤。

---

### SIGN-R11: 评估必须按 regime 分层, 禁止只看全期平均
**Trigger:** 任何 walk-forward / gate / α 评估
**Instruction:** 全期平均 α 只作参考, **决策必须看分 regime(动量/反转)的表现**。gate 条件必须含"动量月改善 + 反转月不伤"。
**Reason:** 2026-06-03 实盘审计教训: V12.31 全期 +2.2pp/月 是动量/反转混合平均, 掩盖了动量月被血洗 (优秀基金+11.4% 而 picks-10%)。回测全期平均正是漏掉实盘 regime 错配的原因。见 project_v3c_momentum_regime_mismatch_0603 + [[feedback_stratified_analysis]]。
**Added after:** regime-overlay 循环启动 / 2026-06-03

### SIGN-R12: 异类/奇异特征必须存活朴素消融
**Trigger:** 引入梅花易数 / 玄学 / 任何"奇异"编码特征作预测
**Instruction:** 任何此类特征的增益, 必须**扣除朴素对照**(如 公历月历季节性 + 板块/代码 cohort)后**仍独立存在**才算数。先跑廉价 IC+消融筛查 (Phase1), 无残差则廉价 REJECT, 不进 walk-forward。
**Reason:** 梅花组合卦的动态部分≈日历+价格末位编码, 静态卦≈股票ID; "信号"极可能是世俗来源被卦象重新打包。不做消融就会把日历/板块效应误归功于卦象 = 自欺。
**Added after:** meihua 循环启动 / 2026-06-04

---

## 项目专属 signs (循环中遇到失败再追加, append-only)

### SIGN-R13: 受控 Δ 须在去-lookahead 的真实样本上复核 (负控包)
**Trigger:** 用"受控差分"(同 picks/两臂相消)下结论, 或任何改变持仓暴露/持有期/现金比例的 overlay (止盈/止损/sizing/风控)
**Instruction:** (codex 交叉审计 0614) ① Δ 算术相消 ≠ 干净: 若候选池由含 lookahead 的模型选出, Δ 仍在"理想路径样本"上, 真实 walk-forward picks 一换会缩水 → **任何 Δ 结论必须在 de-lookahead 真实 WF picks 上复核**。② 改暴露的 overlay 必带 **placebo 负控**: 随机阈值 + 固定降暴露/现金化对照, 排除"少持仓/降波动在数学上抬 Sharpe"的假象。③ 比较两臂时**持有期/时间 backstop 必须对齐**(baseline-20d vs TP-40d 是混淆, 须同时跑 baseline-40d)。④ 每个改进在最终 holdout(不可再调窗)只许进**一次**。
**Reason:** EX-001 报"分批止盈 +1.36 Sharpe", 但 baseline=20d 而 TP臂=40d backstop, 混入了时间暴露+现金路径; 且 Δ 算在 r20-lookahead 选出的理想样本上。codex 指出: 方向可能真, 量级不可信, 须 de-lookahead + placebo 才能定。
**Added after:** codex 交叉审计 / 2026-06-14

### SIGN-R14: 警惕"框架锁死", 不止"特征增量"
**Trigger:** 连续多个同类假设被否 (如价格/形态特征)
**Instruction:** (codex 0614) N 连否常意味着"在当前 label/universe/评测口径内同类增量已尽", 不等于"无增量"。周期性自问: 是否一直在**同一 r20/pump label + past_r5<0 小盘均值回归 universe + 同 book 引擎**内打转? 真增量可能在**换目标函数 (triple-barrier) / 换 universe (风格) / 换风险约束**, 而非再加一个特征。但换框架的改进**同样**要过 walk-forward + 负控 (不是借"换框架"放松纪律)。
**Reason:** 19 连否; codex 审计指出研究机器擅长否定同框架特征增量, 盲点是从未认真换框架。但换框架自由度更高, 更需 SIGN-R13 负控。
**Added after:** codex 交叉审计 / 2026-06-14

### SIGN-R15: 多重检验必须含 researcher 自由度 (正交因子 campaign)
**Trigger:** 任何正交因子族出存活候选进 walk-forward, 算 DSR / 宣布"找到正交 alpha"
**Instruction:** (codex 0618 #2) DSR 的 n_trials = oa_manifest 累计因子变体总数 **× DoF 乘子(=3)**, 覆盖 named variant 之外的研究者自由度: sign 选择 / winsorization / lag / 中性化变体 / universe 过滤 / A-B 边缘提升 / 1H 对比 / "选最强 survivor" 进 walk-forward。campaign-wide(非 per-family)校正, 因结论是"在某处找到了正交 alpha"。
**Reason:** 钓 5 族 × 多变体, best-of-N 必蒙到假阳性; 只数 named variant 会低估试验数 → DSR 虚高。
**Added after:** codex 交叉审计 / 2026-06-18

### SIGN-R16: 廉价闸先杀 + 正交≠有α + selection-on-selection caveat
**Trigger:** 任何正交因子族的 Phase-1 / walk-forward 裁决
**Instruction:** ① 先跑 oa_screen 廉价 IC 筛查, C/D 即廉价 REJECT 不进 walk-forward (省算力)。② gate t 用 Newey-West HAC (lag=2) 非朴素 t (重叠 r20 label 序列自相关, codex #1)。③ Phase-1 在全 54 月(含 walk-forward 的 19 月)选因子 = 因子选择层泄漏 (codex #4); 任何 PASS 候选 verdict 必须显式标注此 caveat, 落地前需独立 OOS / 前向 paper-trade 复核, 不凭单次 walk-forward 直接真钱。
**Reason:** 0605 元结论"正交≠有α"(即时可拉正交信号多=动量投影); codex 0618 指出最大假阳性=selection-on-selection + 重叠 label 膨胀 t + 单 regime 持续蒙过冻结 gate。
**Added after:** orthogonal-alpha campaign 启动 + codex 审计 / 2026-06-18

<!--
### SIGN-RXX: [名称]
**Trigger:** ...
**Instruction:** ...
**Reason:** ...
**Added after:** [迭代 N / 日期]
-->
