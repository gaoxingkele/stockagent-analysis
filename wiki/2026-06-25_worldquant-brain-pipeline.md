# WorldQuant BRAIN 挖矿管道 — 跨到未枯竭的美股空间，借平台自带的 gate

- **日期**: 2026-06-25
- **provenance**: user (定 BRAIN 为新矿 + 给凭证) / ai-executed (搭管道)
- **裁决**: 🧩框架 (新数据空间入口, 管道 live) + analyst4 三批收口 (修正动量峰Sharpe0.93<门槛=building block非独立alpha)
- **关联代码/实验**: `research/brain/wq_auth.py`(认证) + `wq_simulate.py`(单) + `wq_mine{,2,3}.py`(批量); 账号 FL48329

## 一句话
A 股价量空间已被 V12.31 挖到强局部最优 (整轮 25+ 假设全 REJECT)；BRAIN 提供一个**正交的、未被我们挖过的美股信号空间** + 一套**平台级回测 gate**，是元结论指向的"唯一新矿"。本条记录入口管道打通。

## 出处 (必填)
- **WorldQuant BRAIN**: https://platform.worldquantbrain.com (API: https://api.worldquantbrain.com) —
  众包 alpha 平台。FASTEXPR 表达式语言、IS/OS 回测、内置提交 checks。文档 https://platform.worldquantbrain.com/learn
- **认证方式来源**: BRAIN API 无静态 key, 用账号邮箱+密码 Basic Auth `POST /authentication` 换 session cookie (社区 `wqb` 库 / 官方 forum 一致)。
- **思想根**: [[2026-06-24_anti-overfitting-scaffold]] — BRAIN 的内置 checks 正是该脚手架"gate not throughput"的平台外化。

## 为什么引入 (第一性原理)
我们反复证明: 公开 A 股价量因子, 市场效率追平 alpha, 我们已挖到底。要有真增量, 要么换**目标函数/组合层**(book/风控/止盈, 已在做), 要么换**信号空间**。BRAIN 的美股 TOP3000 + 海量数据字段 (基本面/分析师/另类) 是一个我们从未碰过、且不受 A 股 T+1/涨跌停/小盘高 PE 风格束缚的空间。元结论 (`project_session_state_0624`) 早把它列为唯一未测强候选。

## 核心思想 (讲直觉)
1. **管道三段**: `POST /simulations {settings, regular: expr}` → 202+Location → 轮询进度 (Retry-After 控速) → `GET /alphas/<id>` 读 IS 指标。
2. **IS 指标**: Sharpe / fitness / turnover / returns / drawdown / margin / 多空数。
3. **平台自带 gate (关键)**: 每个 alpha 回来带 checks — LOW_SHARPE / LOW_FITNESS / HIGH_TURNOVER / **SELF_CORRELATION** / LOW_SUB_UNIVERSE_SHARPE / MATCHES_COMPETITION。这等于平台替我们做了一部分反过拟合把关, 尤其 SELF_CORRELATION 防止重复发现相关 alpha = 我们脚手架里"去重 vs seen"的平台版。
4. **session 复用**: 登录换 cookie 缓存 `.wq_session.json` (gitignored), 过期自动重登。

## 我们怎么吸纳 / 改造
- 把 BRAIN 管道接到我们的 loop: 候选表达式 → simulate → 用 checks + IS 指标做 gate → 存活的再人工/对抗复核。
- **第一个测试用我们自己的 thesis 验证管道**: `rank(-returns)` (短期反转 = A 股抄底回调的美股镜像)。

## 结果与裁决
- 管道 live: 认证 HTTP 201 (账号 FL48329), simulate 端到端 176s 出结果。
- 首测 `rank(-returns)` USA TOP3000 delay1 industry-neutral: **Sharpe 1.32 (PASS)**, 但 **fitness 0.43 / turnover 1.34 双 FAIL** (decay=0 太快)。
- 观察: **短期反转族在美股仍有方向性 Sharpe** (与 A 股 reversal regime 扛收益呼应), 但裸表达式过不了 fitness/turnover——需 decay/中性化/更慢信号。这正是挖矿要解的工程问题, 管道已能量化它。
- **暂未立 alpha**: 这是入口, 不是发现。真挖矿规模 (跑多少 sim / 哪些 dataset) 待定 (quota 感知, 避免 [[2026-06-24_improvement-loop-methodology]] 警告的"枯竭空间高吞吐"——但 BRAIN 空间未枯竭, readiness 判 READY)。

## analyst4 实证 (三批, 0625-0626, ~38 sims) — 收口裁决
**靶子**: 分析师 EPS 修正信号在美股 TOP3000 能否成 BRAIN 可提交 alpha (Sharpe≥1.25 + fitness≥1.0 + turnover≤0.7)。
- **批1 验证 (15候选/11跑通)**: EPS 修正动量族一致 +Sharpe 0.55-0.60, turnover 0.05, dd 6%; 离散度/惊喜裸形噪声/反向。0/11 过 checks。
- **批2 精化 (18候选, 滚动调度0漏跑)**: **修正动量随窗口单调变强 40/60/90/120 → Sharpe 0.47/0.59/0.70/0.93**; 但组合(rev+conv/cov/winsor)全减分、z-score 归一化伤、SUE/加速度/偏度负 → **纯长窗修正动量最干净, 加任何花活都伤**。中性化(SUB/IND/SECTOR)无差。0/18 过 checks。
- **批3 窗口扫尾 (150-300)**: **见顶非单调** — 120→0.93峰, 150→0.83回落, 180/220 退化无 IS(长窗样本太稀), 250/300 提交失败 → **修正动量 ~120d 封顶 Sharpe 0.93**。

**★裁决 (诚实, 不oversell)**: 分析师 EPS 修正动量是**真的、干净的、低换手的慢信号** (峰 Sharpe 0.93 @120d, turnover 0.03, dd 6%, 方向稳健: revision↑→outperform), **但作单字段 alpha 封顶 <1.25 门槛, 过不了 BRAIN 提交线**。窗口/归一化/组合调参都推不过去。
- 这**不是 REJECT** — 它是**正交、未枯竭空间里的优质 building block** (不像 A 股价量已吃干; 0.93 单因子在公开信号里相当高)。
- 但**也不是独立可提交 alpha**。要过门槛需做**多信号组合 alpha** (修正动量 + 其它正交族如 option/fundamental, 用 BRAIN 的去相关/组合, 是大工程) 或换 region/universe。
- **印证全项目元结论**: 单因子处处弱, edge 在组合 — 连最新鲜的正交空间也封顶 0.93。差异只在: 这里的 building block 是真正交真新鲜, 值得未来组合, A 股价量则已无料。

**miner 演进**: v1 全提交后轮询 (并发限~3 漏 4/15) → v2/v3 **滚动调度** (≤3 在飞, 完成即补, 0 漏跑)。代码 wq_mine.py / wq_mine2.py / wq_mine3.py。

## 多信号组合大 build (Phase A/B, 0626, +25 sims) — 二次收口
用户撑"大 build"冲 1.25, 走多信号组合路。
- **Phase A 正交族侦察 (13)**: option 波动率 + model16 预制评分。**正交族普遍弱/负**: m16_value +0.44(干净低换手) / volprem_60 +0.41(高换手) 是仅有正向; 质量/盈利/现金流/成长/确定性**全负 -0.35~-0.86** (美股 TOP3000 此期质量股反跑输, 翻号=junk/风险溢价不可追)。
- **理论上限**: 最佳积木 rev_mom 0.93 + value 0.44 + volprem 0.41, 无相关 quadrature 上界 √(0.93²+0.44²+0.41²)≈**1.11 < 1.25**。纸面就够不着。
- **Phase B 组合实测 (12)**: rank-sum 组合 × 权重 × 中性化(SUB/IND/SEC/MKT)。**全部 < 单信号 0.93**: best rev_valz 0.82, rev_val_mkt 0.70, rev_val 0.56, 加权 rev2/rev3_val 0.24-0.38, 加 vp/lv 0.26~负。**加任何正交族都稀释修正动量**——naive 等权 rank-sum 把强信号往弱信号拉, 达不到 quadrature 上界, 实测天花板 0.82 < 单信号。

**★二次裁决 (大 build 收口)**: **多信号组合不仅过不了 1.25 门槛, 反而比单信号修正动量(0.93)更差。** 可及的美股正交族(value/vol/quality)单字段太弱(≤0.44), naive 组合稀释而非叠加。冲过门槛需 (a)远强于这些的正交信号(这些数据集单字段给不了) 或 (b)协方差最优加权/去相关机器(本身一个研究项目, 且输入这么弱理论上界仍 1.11)。**大 build 也没掏出 V12.31 挑战者。**
- 三度印证元结论 (A 股价量 → analyst4 单字段 → 跨族组合): **edge 极难, 单信号处处封顶 <1, 组合稀释; 公开可及信号里没有免费午餐**。BRAIN 真出 alpha 是 months-scale 海量搜索 + 最优组合工程, 非几十 sims 能得。
- **未测长尾**: news12/socialmedia12(情绪, 噪声大短horizon) / option9 analytics / pv13 relationship / 协方差最优加权。期望 EV 低(Phase A 已证"显眼"正交族都弱), 但未穷尽。

## 思想谱系 (演化)
- 取代了: "只在 A 股价量空间打转"; "多信号组合能轻松冲过门槛"的乐观。
- 同源 / 对照: [[2026-06-24_anti-overfitting-scaffold]] (BRAIN checks = gate 外化); [[2026-06-24_improvement-loop-methodology]] (BRAIN 是唯一 readiness 判 READY 的挖矿项)。
- 下一步: 数据字段探索 (基本面/分析师/另类正交字段) → 表达式搜索循环 (带 SELF_CORRELATION 去重) → 存活 alpha 对抗复核。

## 移植提示 (必填)
任何想接 BRAIN 程序化挖矿的项目:
1. 认证 = 账号 Basic Auth 换 session, **无静态 key**; session 缓存复用。
2. 管道 = submit→poll(Retry-After)→read alpha IS; 别漏轮询节奏头。
3. **用平台自带 checks 当第一层 gate** (尤其 SELF_CORRELATION), 在它之上再叠自己的对抗复核。
4. quota 感知: 每 sim ~3min + 并发/日限, 先小批验证再放量; 别在管道未验证时就放大搜索。
**本项目特有**: 默认 settings (USA/TOP3000/delay1/industry) 是起点选择, 不同 region/universe 结果不可直接比。
