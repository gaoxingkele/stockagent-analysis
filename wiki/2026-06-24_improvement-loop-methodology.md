# 改进 Loop 方法论 — 在枯竭空间里，loop 跑工程与累积，不跑挖矿

- **日期**: 2026-06-24
- **provenance**: user (用户要求把候选任务规划成 loop 模式) / ai-executed (设计)
- **裁决**: 🧩框架
- **关联代码/实验**: `research/IMPROVEMENT_LOOP.md` (backlog+协议); ralph-loop skill

## 一句话
把"下一步候选"组织成一个每轮"选→执行→评估→总结→提下一步"的改进 loop；
但**因为选股层信号已枯竭**，这个 loop 的合法对象是**工程 / 前向数据累积 / 定时复检**，
**不是**再挖因子——否则就是上一课警告的"高吞吐自主搜索量产假阳性"。

## 出处 (必填)
- **Ralph Wiggum technique** (Ryan Carson): 自主 TDD 开发 loop——每轮 worker 做一步、gate 检查、循环到通过。
  本项目已装 `ralph-loop` / `ralph-loop-setup` skill。loop 的"每轮一步 + gate"骨架借自此。
- **Sakana AI Scientist**: Lu et al., arXiv:2408.06292, https://github.com/SakanaAI/AI-Scientist —
  "生成→实验→评审→迭代"自主科研 loop。**反向取用**: 只取迭代骨架, 拒绝其"高吞吐自主搜索"内核。
- **本项目内生**: [[2026-06-24_anti-overfitting-scaffold]] —
  "吞吐不是目标、证伪才是"；`feedback_ai_scientist_lesson_gate_not_throughput_0615`。

## 为什么引入 (第一性原理)
用户想要一个能自我推进的改进 loop。但一个朴素的"不停产生改进"的 loop 在我们的处境里有毒：
20+ 假设已证选股层挖到底，再让 loop 自由挖因子，每轮都会"找到"一个看起来 +1pp 的东西，
合起来是对噪声过拟合（PBO 高）。所以 loop 的**对象选择**必须先过门控：
只把"真有增量空间"的工作（部署、风控、前向数据、真新数据源）放进队列，挖矿类一律不进。
loop 的价值在**纪律化的评估与留痕**，不在产出速度。

## 核心思想 (讲直觉)
1. **就绪门控在前**: 每个候选先判 READY / GATED(时间) / BLOCKED(凭证) / 需确认(不可逆)。只在 READY 里选。
2. **每轮一个最小可交付步**: 不贪大, 一步一评估, 防止"攒一大坨再发现全错"。
3. **评估带反过拟合纪律**: 数据类用 walk-forward+embargo+placebo、绝对数打折; 工程类要真跑通+幂等+append-only 不回改。
4. **双层留痕**: 操作→git log; 思维进化步→wiki(带出处)。不是每轮都写 wiki, 只有发生"思想吸纳/方法论转向"才写。
5. **节奏匹配机制**: 工程项会话内顺序迭代; 累积/复检项是日历节奏(/schedule), 不硬塞进紧 loop。

## 我们怎么吸纳 / 改造
- 取 Ralph 的"每轮 worker+gate"骨架, 把 gate 从"测试通过"升级成"反过拟合纪律 + append-only 红线"。
- 取 AI Scientist 的迭代外壳, **砍掉自主因子搜索内核**(我们的教训证它在枯竭空间有害)。
- 把 ARA 死胡同留痕接到 loop 的 wiki 步: REJECT 也是一轮合法产出。

## 结果与裁决
框架就位 (IMPROVEMENT_LOOP.md)。首轮候选: I1/I2 (paper-trade 落库+对锚) 价值最高且 READY。
真实价值判断待逐轮兑现, 但**设计本身已遵守元教训**(不挖矿、带门控、双层留痕)。

## 思想谱系 (演化)
- 取代了: 无结构的"想到什么做什么"。
- 同源 / 对照: [[2026-06-24_anti-overfitting-scaffold]] (loop 的评估纪律全部继承自它)。
- 被取代 / 下一步: 若 BRAIN(美股未枯竭空间)接入, loop 对象可重新纳入"挖矿"——因为那里门控会判 READY。

## 移植提示 (必填)
任何"主空间已近枯竭、但仍想持续改进"的项目通用:
1. 先给候选打 READY/GATED/BLOCKED/需确认 四态门控, 只在 READY 选;
2. 每轮最小步 + 带纪律的评估(别让 loop 自评"更好"就过);
3. 操作/思维双层留痕, wiki 只记思想吸纳;
4. 节奏匹配——累积/复检走调度, 别硬塞紧 loop。
**本项目特有**: paper-trade 的 append-only 红线、A 股冻结口径是实现细节, 思想可移植、参数不可。
