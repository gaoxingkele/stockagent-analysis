# Reviewer Panel Checklist (DLV-003 / Track B1)

> 2026-06-16。AI Scientist (Sakana / Nature 2026) 的真教训在 **reviewer 一侧, 不在吞吐**。
> 我们本轮失败模式 = **假阳性** (RETRO +0.0287 用了 CL 两轮才证伪; Hybrid +54% 是 stratification
> artifact)。一个**强 reviewer panel + 事前注册 gate** 一轮就能抓。本文件把"单 codex 审"升级为
> **多模型独立审 + meta-judge** 的固定流程, 与 `research_env.py` 的三件套 (replication / PBO-DSR)
> 配套, 构成"骗不过的 verify harness"。关联: `[[feedback_ai_scientist_lesson_gate_not_throughput_0615]]`。

任何候选在宣称"可落地 / 可激动"前, **必须**走完本 checklist。负结果 = 合法完成 (SIGN-R02);
中间指标 (IC/precision) 改善 ≠ 落地 (SIGN-R03)。

---

## 0. 触发条件

走 full panel 的场景:
- 任何**搜索出来**的候选 (从 N≥3 个特征/窗口/阈值里选出来的) — 必带多重检验校正 (§3)。
- 任何宣称 ΔSharpe / Δα / IC 过冻结 gate 的候选。
- 任何要动 book 暴露 / 持仓 / 现金比例的 overlay (止盈/止损/sizing) — 必带 placebo 负控 (SIGN-R13)。

跳过 panel (仅记录): 纯描述性诊断 / 纯基础设施交付物 (status=built, 无 alpha 宣称)。

---

## 1. 自审闸 (claude, 提交 panel 前)

提交给 reviewer 之前, 作者 (claude) 先自查, 任一 FAIL 直接打回不浪费 panel:

- [ ] **泄漏自查 (SIGN-R04)**: 跑过 `python research/verify.py` leakage guard? 无 forward-field
      (r5/r10/r20/dd*) 混入特征? label 是 lagged-confirmed? 见到 IC>0.5 / Sharpe>10 先查泄漏 + ST。
- [ ] **ST 源头排除 (SIGN-R06)**: 训练/回测/推理三处都在**数据加载时** filter ST, 非事后?
- [ ] **门柱未动 (SIGN-R01)**: gate 数值/指标/协议与循环启动时**逐字一致**, 没在结果上回调?
- [ ] **同口径对照 (SIGN-R13)**: 持有期 / 时间 backstop / universe / 成本 与 baseline **逐位对齐**?
      (baseline-20d vs candidate-40d = 混淆, 须同时跑 baseline-40d)。
- [ ] **de-lookahead (SIGN-R13)**: 若候选池由含 lookahead 的模型选出, Δ 在"理想路径样本"上 →
      已在真实 walk-forward picks 上复核?

---

## 2. 统计严谨闸 (research_env.py, 机器跑)

用 `research/research_env.py` 跑, 不靠肉眼看曲线:

- [ ] **replication / 多 seed (B2)**: `multi_seed_replicate` — 指标跨 seed `sign_stable` 且
      `cv ≤ 1`? (随机子采样/初始化/split 敏感 = 不可信)。
- [ ] **block bootstrap 误差条 (B2)**: `block_bootstrap_ci` — per-cohort (=持有期块) CI 报了?
      `P(Sharpe>0) ≥ 0.95` 且 CI 下界 > 0? (点估计无 CI = 不许激动)。
- [ ] **非重叠 t (B2)**: 20d 持有期的日级 IC t 统计被重叠膨胀, 必报 NW-t (lag=持有期) 或非重叠 t。
- [ ] **regime 分层 (SIGN-R11)**: 全期平均只作参考; 分动量/反转 regime 都报了? 动量月改善 +
      反转月不伤? (V12.31 实盘血洗在动量态, 全期平均会掩盖)。

## 3. 多重检验校正闸 (B3, 仅搜索出的候选)

候选若来自搜索 (从 N 个里选的), 必跑下列至少一项, 惩罚试验次数:

- [ ] **Deflated Sharpe (DSR)**: `deflated_sharpe_ratio(sr_po, n_trials, n_obs, sr_variance)` —
      扣掉"试了 n_trials 次"的选择偏差后 `dsr ≥ 0.95`? (n_trials = 真实搜索过的配置数, 不是 1!)。
- [ ] **PBO via CSCV**: `pbo_cscv(returns_matrix)` — N 个配置的 T×N 收益矩阵, `pbo ≤ 0.5`?
      (PBO 越接近 0.5+ = IS 选优纯过拟合, OOS 无延续)。
- [ ] **placebo 负控 (SIGN-R13)**: 改暴露的 overlay 跑随机阈值 placebo + 固定降暴露/现金化对照?
      候选须 > 随机 placebo 的 p90 (非均值) 才算择价技能 (WFE-002 教训: +0.809 塌到 +0.117)。

---

## 4. Panel 流程 (多模型独立审 + meta-judge)

> 单 reviewer 有盲区且会被作者的叙事带跑。**独立** + **对抗** + **仲裁** 三段。

### 4.1 独立审 (并行, 互不可见对方意见)

每个 reviewer 拿到**同一份**材料 (候选代码 + verdict 草稿 + research_env 输出 + §1-3 checklist 勾选),
独立产出 {verdict: ACCEPT/REVISE/REJECT, 致命问题 list, 严谨性评分}。**reviewer 不可见作者的"希望结论"**。

- **Reviewer A = codex CLI** (`/codex-review-loop` 或 codex exec): 强在抓口径混淆 / lookahead /
      算术相消假象 (本轮 EX→WF→WFE 三轮把 +1.36 止盈逐步证伪到 0 的就是 codex)。
- **Reviewer B = claude (独立实例/Agent)**: 强在因果/regime/framing 盲点 ("是否一直在同一
      label/universe 内打转", SIGN-R14)。**不是**写候选的那个 claude。
- **Reviewer C = (可选) 第三模型 / meta-judge 兼任**: 仅当 A/B 分歧时介入。

### 4.2 对抗规则

- 每个致命问题必须可证伪 (指向具体一行代码 / 一个数 / 一个负控)。
- "看起来对" / "方向应该真" 不算通过, 要么给出过 §2-3 的数, 要么 REJECT。
- 默认怀疑: 首个名义过闸正向 = 过拟合最爱藏处, reviewer 默认假设它是假阳性, 由作者举证。

### 4.3 meta-judge 仲裁

- **全 ACCEPT** → 候选进 book apples-to-apples gate (仍要过 walk-forward, ACCEPT≠ship)。
- **任一 REJECT 且问题未被反驳** → REJECT, 文档化裁决 (SIGN-R02), **不在死信号上硬跑 book 凑 PASS**。
- **REVISE** → 作者按问题改**一处**, 重跑 §2-3, 回到 4.1 (不可借"改了"放松纪律)。
- meta-judge 只裁"问题是否被有效反驳", **不**自己当第四个 reviewer 引入新论点。

---

## 5. 终判落库

- [ ] verdict JSON 落 `research/verdicts/<id>.json`: status + conclusion + metrics + panel 各 reviewer
      结论 + meta-judge 裁决。
- [ ] 负结果同样落库 (REJECT 的四项指标对照) — 一次诚实 REJECT 比十次 p-hack PASS 值钱 (SIGN-R02)。
- [ ] 生产线 V12.31 (`v12_scoring.py` / `v12_dual_track.py`) 全程只读冻结 (SIGN-R05), `verify.py` 校验指纹。

---

## 附: 与 research_env.py 的对应

| Checklist 项 | 函数 | gate |
|---|---|---|
| §2 多 seed | `multi_seed_replicate` | sign_stable & cv≤1 |
| §2 bootstrap 误差条 | `block_bootstrap_ci` | P(Sharpe>0)≥0.95 & CI 下界>0 |
| §3 Deflated Sharpe | `deflated_sharpe_ratio` | dsr≥0.95 |
| §3 PBO | `pbo_cscv` | pbo≤0.5 |
| 一站式 | `skeptic_report` | should_get_excited = 全适用 gate 过 |

`skeptic_report(...)` 把 §2-3 机器闸打成一个 `should_get_excited` 布尔, 但 **§1 (作者自审) 与 §4
(panel) 仍是人/模型流程, 不可被单个布尔替代** — 统计闸过 ≠ 没口径混淆/没 framing 盲点。
