# 改进 Loop — backlog + 每轮协议

> 把"下一步候选"组织成可 loop 执行的改进队列。每轮: **选 → 执行 → 评估(反过拟合纪律) → 总结 → 提下一步**，
> 思维进化步骤写 `wiki/`、操作过程写 git log。设计依据见 `wiki/2026-06-24_improvement-loop-methodology.md`。
>
> 北极星: **这不是因子挖矿 loop**(选股层已挖到底, 枯竭空间高吞吐=量产假阳性)。是 **工程 + 前向数据累积 + 定时复检** loop。
> 增量在 book/风控/部署/真新数据, 不在再挖价量因子。

## 就绪门控 (readiness gate — 每轮选项前先过)
| ID | 任务 | 价值 | 就绪 | 风险 | 状态 |
|----|------|------|------|------|------|
| I1 | T1 paper-trade 日级落库 runner (pt001_gen_today 跑最新交易日 append) | ★★★ 唯一真 alpha 解 | ✅ READY | 低(只读+append) | ✅ DONE (0616-0622, log 387日) |
| I2 | T1 满期 cohort 实现 P&L + 首次 vs 锚1.31 对照 (realize_cohorts) | ★★★ 首个前向信号 | ✅ READY | 低 | ✅ DONE (实现Sharpe+1.07 CI下半) |
| I3 | T4 池B自选 web 持久化 (WATCHLISTS → 可编辑 JSON/DB) | ★★ 体验 | ✅ READY | 低 | ✅ DONE-全 (JSON+REST+前端加/删按钮+即时打分注入, 端到端HTTP测试过) |
| I4 | T2 7月血洗窗复检 (0508-0603 实盘窗 v3c 动量) | ★★★ 验证元结论 | ⏳ GATED 数据~7月中满 | 低 | ⏸ PARKED (触发+命令见下) |
| I5 | T3 WorldQuant BRAIN 挖矿 (美股未枯竭空间) | ★★★ 唯一新矿 | ✅ READY (账号FL48329已认证) | 中 | ⏸ PARKED-收档(大build完结) (单信号峰0.93+组合反更差0.82, 无V12.31挑战者; harness全套留research/brain/随时可挖长尾) |
| I6 | T5 publish/web-fourpool → main + 清大文件历史 | ★ 卫生 | ✅ READY | **高(push/改历史不可逆)** | 待确认 |

## 每轮协议 (loop unit)
1. **SELECT** — 从 READY 项里选 (价值×就绪) 最高的最小可交付步。
2. **EXECUTE** — 做最小可 ship 的一步 (不贪大)。
3. **EVALUATE (反过拟合纪律)** —
   - 数据/评估类: walk-forward / embargo / 随机 placebo 对照, 绝对 Sharpe 默认打折 (见 anti-overfitting-scaffold)。
   - 工程类: 真能跑通 + 幂等 + append-only 不回改 (SIGN-R01/R02 红线)。
   - 永远问: 这个"改进"是真的, 还是 artifact / 乐观成交 / lookahead?
4. **SUMMARIZE** — 一段结论 + 更新本表状态 + commit (操作→git log)。
5. **PROPOSE NEXT** — 重排 backlog, 点名下一项 + 为什么。
6. **WIKI (条件触发)** — 若本轮含**思维进化步**(新思想/新出处/方法论转向/策略框架迁移)→ 写 `wiki/<date>_<slug>.md`(强制标注出处)。纯机械改动不写 wiki, 只 log。

## 执行模式
- **工程项 (I3/I6)**: 本会话内顺序迭代。
- **累积/复检项 (I1/I2/I4)**: 日历节奏 → /schedule 或长间隔 /loop, 非紧 loop。
- **阻塞项 (I5)**: 出注册指引, 凭证到位再排。

## PARKED 任务的触发条件 + 现成命令

### I4 — 7月血洗窗复检 (数据门控)
- **触发**: 0508-0603 实盘窗每只 picks 满 20 交易日前向数据可得 → 约 **2026-07-14** (0603+20td)。0701 起部分 cohort 可读, 0714 全窗满。
- **现成命令** (数据到位后跑): `python research/backtest/paper_trade_harness.py` 重算 → 看 0508-0603 entry 的 cohort 在 momentum regime 的实现收益, 对 [[project_v3c_momentum_regime_mismatch_0603]] 的"动量血洗"做前向验证。
- **判据**: 若该窗 momentum cohort 实现收益显著负 → 证实尾部动量延续 episode (作 tail-risk 管理, 不可因果 gate); 若回正 → RG-002 因果尸检的"动量态后均值回归"成立。
- **当前已知**: I2 全样本 momentum regime = -0.22%/胜率43% (n=7), 已显著弱于 reversal +3.78%/86%; 7月复检看血洗窗是否更极端。

### I5 — WorldQuant BRAIN 挖矿 (凭证阻塞)
- **阻塞**: 需用户在 https://platform.worldquantbrain.com 注册免费账号。
- **用户要做**: 注册 → 把凭证写入项目 `.env`: `WQ_EMAIL=...` / `WQ_PASSWORD=...`。
- **凭证到位后我做**: 搭 BRAIN API harness (登录→拉数据字段→提交 alpha 表达式→读 IS/OS 指标), 在**美股未枯竭信号空间**挖矿 (这是唯一 readiness 会判 READY 的"挖矿"项, 见 [[2026-06-24_improvement-loop-methodology]])。

## 轮次日志 (append-only, 每轮一行)
| 轮 | 日期 | 选了 | 结果 | 裁决 | wiki? | 下一步 |
|----|------|------|------|------|-------|--------|
| 1 | 2026-06-24 | I1 paper-trade append 0616-0622 | 4日 written, log 387日/7041行, append-only幂等 | ✅ 工程过 | 否(机械) | I2 |
| 2 | 2026-06-24 | I2 realize+对锚 | 实现Sharpe+1.07 落CI下半; reversal+3.78%/86% momentum-0.22%/43%; 20 cohort触发复检; 但+1.07=历史去注水非live(真前向cohort~7月中熟) | ✅ 基线确认(非新alpha) | 否(再确认既有结论) | I3 |
| 3 | 2026-06-24 | I3 池B web持久化 | JSON持久化(config/watchlist_b.json)+REST(GET/POST/DELETE)验证通过; 向后兼容seed原12只; 剩前端按钮 | ✅ 工程过 | 否(纯工程) | I4/I5 |
| 4 | 2026-06-24 | I4 7月血洗窗复检 | 数据门控~7/14满, 触发+命令+判据已记; 当前momentum regime已-0.22%/43%弱于reversal | ⏸ PARKED | 否 | 等7月数据 |
| 5 | 2026-06-24 | I5 BRAIN | 凭证阻塞, 注册指引+到位后步骤已记 | ⏸ PARKED | 否 | 等用户凭证 |
| 6 | 2026-06-25 | I5 BRAIN 管道 | 凭证到位→认证成功(账号FL48329)+simulate管道live; 测试 rank(-returns) 美股TOP3000 Sharpe1.32但fitness0.43/turnover1.34 FAIL; BRAIN自带checks=gate外化 | ✅ 管道通 | 是([[2026-06-25_worldquant-brain-pipeline]]) | 定挖矿规模(quota感知) |
| 7 | 2026-06-25 | I5 analyst4验证批(15候选) | 11/15跑通(并发限~3, 4个退避超时); 0/11过全部checks; EPS修正动量族一致+Sharpe0.55-0.60 turnover0.05极低 dd6% 但<1.25门槛; 离散度/惊喜裸形噪声/反向 | ✅ 方向活(弱+)但裸形不过gate | 是(更新brain-pipeline wiki) | 定: 精化批(组合,quota)/停 |
| 8 | 2026-06-25 | I5 精化批(18候选,滚动调度) | 18/18全跑通0漏跑; 仍0/18过checks; **关键: 修正动量随窗口单调变强 40/60/90/120→Sharpe0.47/0.59/0.70/0.93**; 组合(rev+conv/cov/winsor)全减分; z-score归一化伤; SUE/加速度/偏度负 → 纯长窗修正动量最干净 | ✅ 规律清晰(纯长窗最强,组合伤),仍<门槛 | 待(微批扫更长窗) | 窗口扫尾150-300 |
| 9 | 2026-06-26 | I5 窗口扫尾(150-300) | **见顶非单调**: 120→0.93峰, 150→0.83回落, 180/220退化无IS(长窗样本太稀), 250/300提交失败 → 修正动量~120d封顶0.93过不了1.25 | ✅ **裁决: 单字段修正动量是干净低换手真信号(峰Sharpe0.93/turn0.03/dd6%)但封顶<门槛, 是building block非独立alpha** | 是(终版brain wiki) | 定: 多信号大build/换数据族/park |
| 10 | 2026-06-26 | I5 决策 | 用户定 PARKED-收档: analyst4诚实收口, 不投多信号大build也不换族(单挖别族同模式也封顶~1), harness留存随时可挖 | ✅ 收档 | 否(决策已在brain wiki裁决段) | loop全部READY项完结 |
| 11 | 2026-06-26 | I5 大build重启(用户撑) Phase A正交族侦察(13) | option/model16侦察: 正交族弱—value+0.44/volprem+0.41仅有正向, 质量类全负(美股质量股反跑输); 理论无相关上限√(.93²+.44²+.41²)≈1.11<1.25 | ✅ 正交族太弱 | 待(Phase B实测) | Phase B组合 |
| 12 | 2026-06-26 | I5 Phase B多信号组合(12) | **组合全<单信号0.93**: best rev_valz 0.82, rev_val_mkt 0.70, 其余0.24-0.59; 加value/vol任何族都稀释(naive rank-sum等权拉低强信号); 实测天花板0.82≪1.25 | ✅ **大build裁决: 多信号组合不仅过不了1.25,反比单信号差; 可及正交族太弱+naive组合稀释; 无V12.31挑战者** | 是(brain wiki终版) | 定: 长尾族(news/关系)+最优加权/park |
