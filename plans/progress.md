# Progress — research/backtest (book 级组合回测引擎 + 因子归因)

> fresh-context 每轮干净重启, 跨迭代唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星 (实验19, 由 18 个选股因子全否后转 book/风控层)
我们从没建过真实组合回测器 (一直只测等权 Top-N 选股 α, 见 t005)。本轮建**参数化 book 级回测引擎**
(持仓 carryover + A股真实成本 + T+1 + 双轨 sizing), 用它回答: ①V12.31 扣成本真该不该真金白银跑
②混合因子/铁律里哪些在**实盘 P&L 层面拖后腿**。设计冻结于 research/backtest/DESIGN.md;
成本模型/book基线/归因法预注册于 prd.preRegisteredGate (SIGN-R01 不在结果上调)。
**生产线 V12.31 (v12_scoring.py/v12_dual_track.py) 全程只读冻结, 消融全在 research 侧。**

## 事前注册 (冻结 R01, 见 prd.preRegisteredGate + DESIGN.md)
- **成本模型**: 佣金0.025%双向 + 印花税0.05%仅卖 + 过户费0.001%双向 + 滑点基线0.10%/边
  (敏感档 0/0.05/0.10/0.20); T+1 买入当日不可卖; |pct_chg|>=9.8% 当日不可成交; 整手忽略。round-trip≈0.30%。
- **book 基线**: 复刻 v12_dual_track 生产 book (双轨 70/20/10, 行业 cap, 持有≈r20≈20d 再平衡)。
- **归因**: 对评分组件(r20池/pump_up/pump_down/行业cap/双轨sizing)+V7c 6铁律逐个 leave-one-out 过引擎,
  报 book 层 Δ(净Sharpe/年化/最大回撤) 分 regime, 标正/负贡献。
- **BT-004 (skip 待开)**: 事件上下文 meta-label 单表示残差测 + Deflated Sharpe/PBO, gate 已冻结防钓鱼。

## 任务台账
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| BT-001 | 参数化组合回测引擎 (持仓+A股成本+T+1+双轨sizing) | **built** | engine.py 解析自检全过; picks 复刻 t005 选股α **corr +0.999 / MAE 0.66pp**(输入忠实 V12.31 baseline); 等权成本关 book 每期(≈20d)超额 **+5.45%** vs t005 +3.75% 同量级 = 引擎无 bug |
| BT-002 | V12.31 扣成本真实净 P&L + 基准/基金对照 | **built** | 基线0.10%滑点 年化**+140.5%** 净Sharpe **+2.78** maxDD -22.0% 月换手1.55 胜率85%; 滑点0→0.20% Sharpe仅衰减**0.12**=**成本非杀手**; vs hs300+6.6%/CSI1000+18.5%/动量+12.1%; regime: 动量月超额hs300+4.44pp(67%胜, 绝对未跑输)但是**相对最弱regime**(vs反转+6.23/混合+9.62)=用户"动量月吃亏"部分印证; **caveat: +140%绝对量级含r20池共模lookahead非可交易声明** |
| BT-003 | 因子/铁律 leave-one-out 归因 (谁拖后腿) | **built** | 7 组件 leave-one-out 过引擎. 基线 净Sharpe **+2.10**/年化+94.0%/maxDD-23.6%. **r20池=绝对核心** (去掉 ΔSharpe **-1.37**, 年化 +94%→+19%); pump_up排序+0.16 / pyr_velocity+0.06 次正; **pump_down 完全惰性** (Δ0.00, 不binding); 行业cap+0.03/双轨+0.04 近中性 (双轨增收益但增回撤); **最拖后腿=行业动量排除铁律** (去掉反 ΔSharpe**+0.10**). book α 几乎全靠 r20池, 行业动量排除是 future opt-in 放松候选 |
| BT-004 | 事件上下文 meta-label 残差测 (单表示+PBO) | **REJECT_reskin** | ctx=(MA20前5日内已上拐)×(MA5上穿MA20金叉), 不搜窗口/MA组合(R01), 事件率1.48%. 扣[pump+MA20斜率+pyr+ADX]残差对T+20: 非重叠headline IC **+0.00019** t=0.007(n=19)≈零; raw金叉IC微负(A股均值回归). PSR 0.31. **三闸全FAIL** → ctx被现有TA吃掉(印证先验), 不作meta-label |
| EX-001 | 出场策略测 (分批止盈TP+回撤止损SL, 同picks受控Δ) | **built** | 同V12.31 picks(20 cohort)套预注册出场网格, 入场逐位一致共模相消. baseline 净Sharpe+2.40/年化+90.8%/maxDD-16.1%. **TP分批止盈 ΔSharpe+1.358**(回撤砍半-16→-10.5%, 收益小让-13pp)=出场层唯一Sharpe增益候选; **SL回撤止损两档都毁灭年化**(-51~-73pp, 印证均值回归砍在反弹前)弃. 指向EX-003重标用上屏障为主下屏障谨慎 |
| WF-001 | de-lookahead r20 真实 walk-forward → V12.31 可交易 P&L | **built** | r20 月度重训(24m,固定120树)去 lookahead: 真实年化**+65.9%**/净Sharpe**+1.84**/maxDD-18%, vs 注水版+140%/2.78 **缩水 ΔSharpe-0.94/年化腰斩**; smoking gun=生产r20 IC in-sample +0.44→true-OOS +0.18, WF同段+0.18≈生产(无lookahead优势, +0.44全记忆); 仍净正跑赢全基准→V12.31真可交易但绝对预期按真实数重设, 相对Δ结论不受影响; de-lookahead后momentum月最弱(超额+0.79pp/胜率44%) |
| WFE-001 | label-embargo r20 真·真实 walk-forward (吃 codex review 第1条) | **built** | 加 label-availability embargo (训练截止 ≤ P_start−21交易日, 保证 r20 前向20日 label 预测前可知) 重跑: 真·真实 年化**+34.9%**/净Sharpe**+1.31**/maxDD-15.7%/月胜率70%; vs WF-001 无embargo 1.84 **额外缩水 ΔSharpe-0.53/Δ年化-31pp**, vs BT-002 注水版 2.78 累计-1.47; **仍跑赢全基准** (csi1000 0.81/动量0.51); **关键诊断: r20 IC 几乎不变 (全期 +0.0979 vs WF-001 +0.1007, Δ-0.0028; true-OOS 反升 +0.2049)→近截止 label 泄漏在信号层很小, book Sharpe 0.53 的额外缩水主要是 picks 路径敏感性非大块泄漏**; V12.31 真·真实数定案 ≈Sharpe 1.3/年化+35% |
| WF-002 | 修正版止盈复测 (真实WF picks + baseline-40d + placebo负控) | **TP真改进*** | 在 WF-001 de-lookahead 真实 OOS picks 上, apples-to-apples ΔSharpe(TP40-base40)=**+0.809** (EX-001 混淆口径+1.358 缩到此); bootstrap CI[+0.15,+1.33]不含0/LOO符号稳定/动量月+0.76pp. **但非择价技能**: 随机阈值 placebo(+0.769)与真实 TP(+0.809)统计无法区分 → 边际全来自'结构性 de-risk'(任意阈值都行)非'卖在真高点'; 静态降暴露 placebo≈-0.09 被击败=排除'少持仓数学抬升'. Δ年化≈0 但 maxDD 砍14.7pp=保收益降波动. **(注: 此 +0.809 用日内 high 乐观成交 + WF-001 无embargo picks; WFE-002 在 embargo picks + close-based 保守口径 + 随机p90 下塌到 +0.117/CI含0 = REJECT)** |
| WFE-002 | 强化版止盈复测 (embargo picks + close-based + 随机p90 + 分解负控) | **结构也不成立** | 吃 codex 第二次 review 第2+5条: 主口径=**close-based 保守成交**(非日内high)/强化gate=TP须>随机阈值**p90**(非均值)/扩展4控分解edge. 在 WFE-001 embargo 真·真实 OOS picks 上: baseline40 Sharpe+1.26, TP40_close+1.38 → **ΔSharpe仅+0.117** (日内high乐观口径+0.279, EX-001混淆口径+0.357); **bootstrap CI[-0.242,+0.523]含0**+LOO符号不稳→**结构性减仓改进不成立**; TP+0.117<**随机p90+0.279**→无择价alpha; 静态降暴露placebo≈0(增益非少持仓); regime下TP**两态都伤**(动量-0.45/反转-0.19pp). edge分解(各控ΔvsbaseTP越近越主源): 盈利条件+0.247/缩短持有+0.227/收益分位档位+0.116/随机+0.155→主源=纯档位(任意分批减仓都行). **WF-002的'+0.809真改进'被证是日内high乐观成交+无embargo picks双重抬升的假象→弃止盈** |
| EX-002 | 基金/风格对照 (基金更强是选股还是风格) | **built** | V12.31 picks风格=**小盘高PE成长**(市值百分位32%/PE 66%/波动49%/换手50%在全市场—**波动换手≈市场中位非高波高换手**,纠正先验). 行业over-weight机械设备+3.9/铸货+2/铜+2,under-weight中药/化学药. 基金(池C 7基金真历史7季报)持仓市值百分位**90%**=大盘质量,vs我们32%. book Sharpe: V12.31 +2.78(lookahead注水)/hs300+0.46/CSI1000+0.81/基金共识篮+0.64. **归因=风格差**(responsePlaybook['风格差']): 与基金差异主导来自风格/universe暴露(小盘成长vs大盘质量)→增量在风格tilt/风控非选股重标; 真实选股α须EX-003去lookahead定 |

## FIN 阶段台账 (吃进 codex 第三次 review)
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| FIN-001 | V12.31 真实基线 1.31 补误差条 (bootstrap/LOO/regime) | **built** | 在 WFE-001 embargo 真·真实 book (净Sharpe 1.31/年化+35%) 上: **block bootstrap(1000×, 21个20日cohort) Sharpe 95%CI=[+0.11,+2.60]** 中位+1.27 SE0.62 P(>0)99% **P(>1)仅66%**; **LOO逐cohort Sharpe∈[+1.02,+1.53]符号恒正=对删单期稳健**; **集中度中等(非极端)**: top-3赢月(202512/202508/202601)占正收益46% HHI0.11→有效~9.3独立赢月(14/21正月), 删最重单月202508 Sharpe仅降到+1.05; regime均衡(momentum贡献48%/reversal50%/mixed3%, 两态≈50/50). **核心结论: 1.31对LOO稳健+跨月跨regime均衡, 真实瓶颈是样本短/功率低(CI宽)而非过拟合到某几月; 非可承诺下限, 接7月血洗窗前向paper-trade复检** |
| FIN-003 | triple-barrier V12.32 挑战者 (双屏障label + embargo WF + book gate) | **REJECT** | 双屏障(+15%/-8%/40d backstop, 参数预注册冻结) TB-score 替代 r20 作池 filter, embargo 41交易日 WF, 同引擎/双轨/cap/成本 apples-to-apples vs WFE-001 基线1.31: **TB 净Sharpe +0.72 vs 基线 +1.31 → ΔSharpe -0.59** (maxDD 反升 -15.7→-18.7%). gate_tb **五条件全 False**: bootstrap CI[-1.36,+0.28]含0 P(Δ>0)10% / 剔最利TB月202506后ΔSharpe -0.85仍负 / regime 动量Δ-2.11pp+反转Δ-1.26pp **两态都伤** (仅 mixed n=2 噪声 +3.92pp). TB-score 对 r20_fresh IC≈+0.126 (描述性, 但 book 不兑现). **用户 'triple-barrier/限幅度+回撤' 直觉在选股 label 层不兑现** (出场层 WFE-002 已否止盈, 选股层 FIN-003 再否) → 均值回归选股已近最优, 第 20 个被反过拟合脚手架诚实判的假设; 生产线 V12.31 冻结不动 |
| FIN-002 | 冻结 V12.31 真实基线 + 前向 paper-trade 协议 | **built** | 产 `research/backtest/V12.31_BASELINE.md`: 冻结配置(V7c池/ratio_s5/r20 embargo WF/双轨70-20-10/行业cap4/20d再平衡/close-based/A股成本~0.30%/T+1/ST源头排除)逐字钉死 + 真实期望净Sharpe**1.31**±CI[+0.11,+2.60](P(>0)99%/P(>1)66%/非可承诺下限) + 前向paper-trade协议(每日append-only落picks→满20/40d算实现P&L→分regime对照→累计≥6cohort或过7月血洗窗后重跑FIN-001复检, 红线前向数据不回流调参) + 注水链(2.78→1.84→1.31)钉清. **部署前checklist可执行部分已执行**(run_fin002.py): r20池模型235特征**仅6基本面**(229全价量/技术无披露滞后); 5/6(total_mv/pe/pe_ttm/pb 日频daily_basic + winner_rate 日频cyq)=**PIT安全(LOW)**; **唯一MEDIUM=holder_pct**(季频股东户数, factor_lab合并源未断言ann_date对齐, 若按end_date则~1季度前视)→部署前须核as-of或替换FU-001 PIT面板. codex#3残留盲点定位且影响有限(单字段, BT-003证r20信号主导来自整池非单因子) |

## DEP 阶段台账 (剔泄漏因子, 清 codex 唯一硬红线)
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| DEP-001 | 剔 holder_pct 重训 PIT-clean r20 + 确认 1.31 | **clean确认** | 唯一改动=r20 训练特征集严格剔 holder_pct (247→246, PIT 前视: stk_holdernumber 季频按 end_date<=target 选数未取 ann_date → ~1季度前视; gain 占 0.005% 极低), 余口径逐字同 WFE-001 (24m/embargo P_start-21/固定120树/双轨/cap4/20d再平衡/close-based/T+1/ST源头排除) → apples-to-apples。PIT-clean book 净Sharpe **+0.84**/年化+19.9%/maxDD-26.8%; vs WFE-001 基线 1.31 **ΔSharpe -0.464**。但 **per-cohort 配对 block bootstrap (1000次) ΔSharpe 95%CI=[-1.25,+0.31] 含 0** (中位-0.48 P(Δ<0)89%) → 点估计跌幅落在配对噪声内。**smoking gun: 剔 holder 后 r20 全期 IC +0.0952 vs 含 holder +0.0979 (变化 -0.0027≈0) = holder_pct 在信号层无实质贡献** (印证 0.005% gain + BT-003 r20 α 主导来自整池排序非单因子)。点 Sharpe 跌主要是 picks-path 敏感性 (同 WFE-001 诊断, book 方差由 picks 路径驱动非信号), 非材料性 edge 损失。仍跑赢全基准 (hs300 0.46/CSI1000 0.81/动量 0.51)。**裁决 clean确认: codex 唯一硬红线清除** (零 ann_date 工程隐患彻底消除), PIT-clean r20 = deployable 变体, 真实基线维持 ≈1.31 (FIN-001 CI[+0.11,+2.60] 内)。生产线 V12.31 全程只读冻结。 |

## DIAG 阶段台账 (codex#5 第五次 review — 只读稳定性诊断)
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| DIAG-001 | pick turnover/overlap — 0.46 摆动来自少数票还是全局重排 | **built [全局重排]** | 复用 wfe001(1.31)/dep001(0.84) 已缓存 OOS picks+book+价表只读诊断, 20 个 20d 再平衡 cohort 比两臂 V12.31 实际入账持仓。**overlap 低**: 平均持仓 18 只, Jaccard **0.327**, 每 cohort 被替换 **19.2 只 (≈半数名单换掉)**, 但共同票权重 Spearman **0.86** (留存票排名稳)。**Δ=Σ(w_dep−w_wfe)·ret 归因** (权重相同票贡献恒0→Δ全来自 swing 票): 每 cohort 平均 24 只 swing 票, 前3只占 cohort 毛|Δ| **54%** (top1 28%) = within-cohort 中度集中, **但跨 cohort 是不同的票** (唯一 swing 票 **431 只**, 池化前10事件仅占全期毛|Δ| 20%, HHI 0.008 有效 ~120 独立事件)。信号侧累计净Δ -27.97pp ≈ book 实际 -28.12pp (同号, 归因忠实)。**裁决 [全局重排]**: high_overlap=False (0.327<0.60) → 剔 1 个 0.005% 因子 (holder_pct) 重排了整批入选 (r20 作池 filter, 小扰动在池边界翻动约半数成员), 非个别刀尖票 → **模型对小扰动整体敏感**, 部署须降集中/增 N 稳健性 (strategy 层, 另议; responsePlaybook['全局重排'])。机制印证 DEP-001 'clean确认' (IC 几乎不变, Sharpe 跌=picks-path 敏感性非信号损失)。 |
| DIAG-002 | cohort jackknife — 0.84 vs 1.31 是否少数 cohort 主导 | **built [少数cohort主导]** | 复用 wfe001/dep001 已缓存 OOS book NAV (同 FIN-001 口径, 两臂日期逐位对齐 407日/21个20d cohort), 配对 leave-one-cohort-out 只读。点估计: WFE(含holder) Sharpe **+1.308** / DEP(剔holder) **+0.844** / **gap +0.464**。**① cohort LOO**: 删每 cohort 后 gap(WFE−DEP) 范围 **[+0.270, +0.631] 恒正** (删任一 cohort 含holder 始终 ≥ 剔holder); 删后最接近 = cohort **#14 (20251201~20251226, reversal)** gap→**+0.270**, 最远 = cohort #6 (20250407~20250507) gap→+0.631。**gap 归因**: 前1 cohort #14 占 gap **42%**, 前3 占 **104%** (另18个近相消), HHI 0.182 → **有效 ~5.5 个 cohort** 拉开两臂。**② 月层 LOO**: 删后最接近月 202512 (占 gap 51%) / 最远月 202603, 前3月占 124%。**裁决 [少数cohort主导]** (few_top3≥0.80 命中): gap 集中在 ~3-5 个 20d 实现路径 → **短样本噪声 (非真实信息损失)**, 1.31/0.84 都是宽带抽样的两个实现。**诚实 nuance**: 单 cohort 未独占 (前1仅 42%, 删它 gap 仍 +0.27), 且 LOO gap 恒正 → 不是纯掷硬币噪声, 是"集中于少数路径的弱系统偏移", 量级落 DEP-001 配对噪声带内 → 不改 clean确认。与 DIAG-001 互补 (票层全局重排 + cohort 层少数主导共同把 0.46 讲清)。需多 seed DIAG-003 (部署前) 最终定差是否真实。 |

## RETRO 阶段台账 (0501+ 真实表现 + 概念内漏赢家对比学习可行性)
> 北极星换轨: 回顾 2025-05-01 起 V12.31 picks 真实表现, 与同概念桶内"漏掉的赢家"对比, 判对比学习可行性。
> 主桶=merged 概念库; winner 预注册=桶内后续20d max_gain TOP20%。陷阱: "差异"严格分 ex-ante特征 vs realized收益(hindsight)。
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| RETRO-001 | 0501+ 真实 picks 表现回顾 (vs 全市场/概念均值) | **built** | 复用 bt001 picks_v1231_daily 截 0501+ (4434 (date,stock)观测/242日/1317股, 全 full-20d). **真实涨**: 20d 均值 **+6.90%** 胜率 65%, 相对全市场超额 **+4.66%**(胜率57%), 相对所属概念超额 **+3.77%**. **但 vintage 分裂是关键**: in_sample_r20 月(202505~09, r20训练窗内)ret20 **+9.87%**/win77%/概念超额+5.29% vs true_oos(202510+)ret20 **+4.09%**/win**54%**/概念超额+2.34% **win_vs_concept 仅47%(<50%)** → 真实 OOS 下 picks 多数(53%)在概念内并不跑赢, 概念超额是右尾少数赢家拉起 (hindsight 陷阱前奏, 直指 RETRO-002/003). |
| RETRO-002 | 概念内漏赢家画像 (ex-ante 特征 vs hindsight 收益) | **built** | pick_concepts=1886, 对比宇宙 1.25M (date,stock): winner(概念20d max_gain TOP20%)=513503, **missed_winner(未入池)=511169**, **picks 自身概念 winner 命中率仅 52.6%** (一半 picks 不是其概念涨最猛 20%). **关键发现 = 漏赢家的差异主要在动量 (ex-ante 可观测!)**: 按 \|SMD\| 排非模型 ex-ante 特征前三 = **mom_20 (SMD -0.366)** / mom_60 (-0.283) / adx (-0.277) → **missed_winner 的近期动量系统性更高** (mom_20 picks -0.45% vs missed +4.56%; mom_60 +4.5% vs +11.6%; past_r5 picks 更低), 印证 v3c 是均值回归策略 (买 past_r5<0) 故系统性避开了概念里后来领涨的动量股 (呼应 [[project_v3c_momentum_regime_mismatch_0603]]). 模型分差异 (ratio_s5 +2.71 vs +1.95 SMD+0.49 / pump_down picks 更低 / pyr_velocity picks 更低) = **套套逻辑** (picks 正是被这些分选出). realized(hindsight, 不可交易): missed_winner max_gain +23.4% vs picks +16.6% (定义性 gap, winner 按 max_gain 选). **真问题 (这个动量残差扣桶 beta+扣现有因子后是否可学) 留 RETRO-003 gate**. total_mv 仅 factor_lab 覆盖(到~20260126, 部分). |
| RETRO-003 | 对比学习可行性 gate (扣桶beta+扣因子残差 + PE市值桶交叉验证) | **对比学习候选 (脆弱)** | 主候选=桶内动量 composite z(mom_20)+z(mom_60), 目标=可交易 ret_20, 控制=[ratio_s5,pred_r20], 单遍 streaming 残差 IC (扣桶 beta + 扣因子, NW-t lag20, 概念 member 桶 34.5M 行). **概念桶 full universe: 残差 IC=+0.0286 NW_t=+2.68 名义过冻结 gate (\|IC\|>=0.01&\|t\|>=2) → 19+ 连否后首个正向, 列对比学习候选 (需 walk-forward, 非 ship)**. 但**三处脆弱性**: ①**符号翻转(suppressor)** 桶内 raw 动量 IC=**-0.041(负=均值回归)**, 仅扣 ratio_s5/pred_r20 后残差才转 +0.029 → 信号是'动量正交于现有因子的分量'非动量本身; ②**apples-to-apples 失效** PE/市值可得子样本概念桶 IC 跌到 +0.0201 **t+1.49 不过**, 非重叠 t+1.15 不过 → full 过闸有赖 202602-04(交叉验证缺这 3 月无 PE); ③**动量 regime 归零** momentum 月 IC=+0.0018 t+0.11(≈零), 正值全来自 reversal(+0.044 t3.02)/mixed(+0.038 t3.36) → **不修 V12.31 实盘动量态痛点** ([[project_v3c_momentum_regime_mismatch_0603]]/SIGN-R11). 概念特异性: 概念桶 +0.0201 vs PE市值桶 +0.0135(同子样本)勉强判概念特异(gap 0.0066)但两者都不过 t = 主要泛截面弱动量残差. **净=对比学习 LTR 是 19 否后唯一名义过闸正向但脆弱/regime条件/suppressor, 须独立 walk-forward 且证明修动量态才谈落地; V12.31 不动**. 产 run_retro003.py / cache/retro003/{mvpe_panel,retro003_results,run.log}. |

## CL 阶段台账 (RETRO-003 首个正向的严格证伪验证)
> 北极星: 把 RETRO-003 "within-concept 动量 re-rank +0.0286 NW_t+2.68 名义过闸" 当**首个正向 = 过拟合最爱藏处**, 设计成能主动证伪。
> CL-001 全因子消融+walk-forward+分regime(杀或活信号); CL-002 book apples-to-apples gate(净赚不赚, 仅 CL-001 存活才跑); CL-003 regime/稳健分解。动量预注册单假设(不再搜特征 R01)。
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| CL-001 | 全因子消融 + walk-forward + 分regime 残差验证 (杀或活信号) | **死于消融** | within-concept 动量 composite z(mom_20)+z(mom_60) 概念内 demean 后, 把消融从 RETRO-003 的 2 因子 [ratio_s5,pred_r20] 扩到**全因子集** [past_r5(多窗动量)+pyr_velocity+log_mv(size)+pe(value)+adx+rsi+ma20_slope(形态)+ratio_s5+pred_r20] 逐日截面正交化: headline full(9因子,PE可得,n=181) 残差 IC **+0.0198 NW_t +1.77 未过冻结 gate** (|IC|≥0.01&|t|≥2); **非重叠 t +0.68** (core 同 ~+0.69) = 去 20d 重叠膨胀后无显著性; core(7因子,full universe,n=242) NW_t +2.05 仅擦边但非重叠 t 同样 +0.69 不过 → 整体不稳健。full IC 缩到 RETRO-003 +0.0286 的 **69%** (core 66%) 且显著性蒸发 → +0.0286 约 1/3 被更全因子吃掉余下不过闸。★**诚实 nuance(与 RETRO-003 反转, 未据此移门柱 R01)**: 全消融后残差**不是均匀归零, 而集中到动量 regime** — momentum IC **+0.0393 t+5.41**(强) / reversal +0.0111 t+0.70 / mixed +0.0151 t+0.98(后两者不显著) → **仅动量 regime 显著**, 揭示 RETRO-003 "动量月≈0" 是其 2 因子控制被 regime 混淆的 artifact。桶内 raw 动量 IC -0.055(负=均值回归 suppressor 仍在); by_month 不稳(202505/06 负→202508~202601 正→202602 又负)。**裁决死于消融**: 未过存活闸 → **不进 CL-002** (REJECT 文档化); 残差性质属**动量 regime overlay**(恰在 V12.31 实盘血洗的动量态, [[project_v3c_momentum_regime_mismatch_0603]]+SIGN-R11), 若追须按 CL-003 当 overlay 验(RG-002 已证 tricky), 非稳健 cross-sectional alpha。生产线 V12.31 冻结 |
| CL-002 | book apples-to-apples gate (blend vs PIT-clean基线) | **skipped (前提不满足)** | isGate; 仅 CL-001 存活才跑。CL-001 死于消融 → 按预注册 cl002_book_gate 协议 **skip 并文档化** (R02), **不在死信号上硬跑 book 凑 PASS** (R01)。blend=0.5*rank(ratio_s5)+0.5*rank(within_concept_mom)/λ=0.5/引擎/基线(DEP-001 0.84 或 WFE-001 1.31)/5 条 gate 均为冻结配置, 仅记录不实例化。**纪律意义**: 反过拟合脚手架在 CL-001 处即终止 gate 链, 不下探 book p-hack——首个名义过闸正向 (RETRO-003 +0.0286 t+2.68) 经 CL-001 全因子消融+非重叠去膨胀诚实证伪后, book gate 前提不成立故不产生任何 P&L 数字。残差动量 regime 集中性 (CL-001 nuance) 属 overlay, book gate 不适用 → 留 CL-003。产 `research/verdicts/CL-002.json`。生产线 V12.31 只读冻结 |
| CL-003 | regime/稳健分解 (overlay判定 + PE市值桶 + placebo) | **built [regime_overlay]** | CL-002 无 book ΔSharpe (CL-001 死→skip), 故 regime 分解在**残差 IC 层** (复用 CL-001 full 块, R02 诚实降不动口径)。三路一致定性 **regime_overlay**: ① **regime 分解** 仅动量 regime 显著 (mom IC**+0.0393 t+5.41**; reversal +0.0111 t+0.70 / mixed +0.0151 t+0.98 不显著) = overlay 签名; ② **概念特异性证伪** 概念桶 full IC +0.0198 ≈ **PE×市值分位桶 full IC +0.0205** (gap **−0.0007**, PE/mv NW_t+1.63 同不过闸) → 残差**非概念特异**, 是泛截面弱动量残差被 PE/市值横截面结构吸收 (不是"概念内"alpha); ③ **placebo** 随机 within-concept re-rank (K=200) real +0.0198 vs **null mean +0.0037 sd 0.0002 → z+67.87 p<0.001 落 null 100% 分位** → 残差是**真信号非方法 artifact** (但 null 非 0=残差化几何有微正偏, real 仍远在右尾)。**综合**: 残差是**真但微弱**的动量残差, **恰且仅在动量 regime 显著** (V12.31 实盘被血洗的动量态, [[project_v3c_momentum_regime_mismatch_0603]]+SIGN-R11), 非概念特异/非全 regime 兑现 → **动量择时 overlay** (RG-002 已证 tricky, 需滞后 regime detection), 非稳健 cross-section alpha。**CL 阶段终判**: within-concept 相对动量 re-rank **不落地** — RETRO-003 +0.0286 名义过闸 → 1/3 被全因子吃 (CL-001 死于消融) → 余下是动量 regime overlay (CL-003); 19+ 否后首个正向**未能成为可交易 alpha**。漏概念动量赢家是 hindsight + 我们均值回归身份的必然代价 (RETRO-002), 不引入动量择时不可交易修。产 run_cl003.py / cache/cl003/{blocks,cl003_results.json}。生产线 V12.31 只读冻结 |

## STAB 阶段台账 (A1 Track: 低换手/稳定化组合构建 — book 层杠杆)
> 北极星换轨 (roadmap Track A1): DIAG-001 证 V12.31 picks 对小扰动整批重排 (隐性高换手/刀尖选股) + QuantML 证高换手吃 α。
> 在 DEP-001 PIT-clean 分数 + BT 引擎上, **只改组合构建** (选股分数/池构造逐字不动), 预注册网格测换手↓+稳定↑同时 Sharpe 不降。
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| STAB-001 | 低换手/稳定化构建网格 (baseline/V_stick/V_N30/V_combo) | **built [无差异]** | 复用 stab001_gen_pool 重建每日 V7c 候选池 (NO retrain, 缓存 dep001 剔holder r20 + t005 s5), 仅改构建规则跑 4 变体过引擎 (407 日/21 cohort)。**双验证**: baseline 净Sharpe **+0.843 == DEP-001 0.843** (忠实复刻) 且 baseline 扰动Jaccard **0.3265 == DIAG-001 0.327** (复刻扰动机制)。**核心负发现 = 构建层无法降换手**: ①**V_stick** (已持有名 pool_rank≤30 给 bonus 留任) 月换手仅 1.54→1.50 (**-2%**, 远不及 gate 的 -20%), Sharpe 不变 (ΔCI[-0.093,+0.131] 含0); ②**V_N30** (MAX_A/B 8/15→10/22 降集中, 持仓 18→26) 换手 1.53 (**-0%**), Sharpe 0.84→**0.77** (ΔCI[-0.365,+0.252] 含0), maxDD 微改善 -26.8→-25.8%; ③**V_combo** Sharpe 材料性降 **0.57** (ΔCI[**-0.515,-0.019**] 整段<0)。无变体过 gate_stab (4 条件无一变体全满足)。**机制 = 换手是结构性而非边际**: 连续再平衡日持仓 **Jaccard ≈0.02** (每 20d 期~98% 换名) → V7c 池本身 (r20 top5% × ratio_s5 排序) 每期近乎全量重排, **没有可留任的"粘性名"** (滞回带罕触发)。粘性/N30 这类构建杠杆触不到根因; 真换手在**选股池层** (需改 horizon/平滑分数 = 改选股逻辑, 超 A1 范围)。N30 降集中是收益/回撤微权衡 (maxDD↓但 Sharpe↓), 非换手解。**裁决 [无差异]** (responsePlaybook): 构建变化对 21 月 book 无过 gate 改进, 文档化; **稳定性/换手不能靠粘性或单纯增 N 修, 须选股层**。生产线 V12.31 只读冻结 (R05)。产 stab001_gen_pool.py / run_stab001.py / cache/stab001/{candidate_pool_clean,_holder, book_*, stab001_results.json}。 |
| STAB-002 | 最佳变体深验 (换手-成本敏感+稳定性+分regime) | **skipped (前提不满足)** | STAB-001 无改进变体 (best_variant=null, 4 变体无一过 gate_stab) → 按 acceptanceCriteria '若无改进变体则 skip 并文档化' (R02 documented skip)。**不在不存在的最佳变体上硬跑**成本敏感 (0/0.2/0.6/1.0% 双边同 QuantML) + 稳定性 (扰动重排率) + 分regime 深验网格 (R01: 冻结网格仅记录不实例化)。根因复述 (STAB-001): 换手是**选股池结构性** (连续再平衡 Jaccard ≈0.02 = 每 20d 期 ~98% 换名), 构建层粘性/增N 杠杆触不到根因, 须改选股层 (horizon/分数平滑 = 改选股逻辑, 超 A1 范围)。**纪律意义**: A1 Track gate 链在 STAB-001 '无差异' 处诚实终止, 不下探到空集合上的成本-regime p-hack。生产线 V12.31 只读冻结 (R05)。产 verdicts/STAB-002.json。 |

## DLV 阶段台账 (Roadmap 剩余交付物: A2/A3 + B + C1, 非研究 gate, 每条产 verdict=built)
> 北极星换轨 (ROADMAP_2026H2.md): A1 低换手已否 (换手是选股池结构性)。剩余 = 部署/基础设施/卫生交付物。
> 生产线 V12.31 只读冻结; PIT-clean r20 (dep001 剔 holder) 为部署变体。
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| DLV-001 | A2 TreeSHAP 可解释层 (逐只票加性归因) | **built** | `research/backtest/treeshap_explain.py`: 用 LightGBM 内置 `booster.predict(pred_contrib=True)` (TreeSHAP, **无需 shap 库**) 对 dep001 PIT-clean r20 (剔 holder_pct, embargo WF) 的 **202604 全月 311 只 picks / 21 再平衡日** 算逐只票加性归因。**加性自检 PASS**: max\|Σfeature_contrib + base − pred_r20\|=**5.3e-14** ≪ 1e-4, base_value=**+2.4171** (全样本均分锚, ≈r20 预测分单位)。**全期 top3 驱动因子** (\|mean SHAP\|): `cyb_rel_strength`(2.80, 创业板相对强度, signed +2.80 主推高) / `industry_id`(1.41, 行业, +1.30) / `mkt_ret_20d`(0.92, 大盘 20d 动量, signed −0.26)。每 pick 输出 top5 正/负贡献因子表 (`pred_r20 = base + Σcontrib` 完整可拆) → 可向 paper-trade/出资人解释"为什么打高分"。**纯解释层, 不改选股** (R05 仅 load 缓存模型, 0 生产改动)。faithful: 同 dep001 load_window(20220801-20260601, ST 源头排除 R06) 重建 202604 特征, booster.feature_name() 逐位对齐, 缺源因子 NaN (非 0, LGBM 原生处理)。产 output/treeshap_dep001_202604_{contribs.csv, demo.md}。 |
| DLV-002 | A3 前向 paper-trade 落库 harness (PIT-clean) | **built** | `research/backtest/paper_trade_harness.py`: 落地 V12.31_BASELINE.md §3 协议为可执行 3 段式 harness。**①append-only 落库** `log_picks_for_date` 逐日落 PIT-clean picks, existing 文件**拒绝覆盖** (协议红线 R01/R02: 前向数据不可回改/不回流调参); 源 = dep001 PIT-clean r20 (剔 holder_pct embargo WF) deployable 变体, **非生产 in-sample 注水模型**。**②满持有期实现 P&L** `realize_book` 过冻结成本引擎 (close-based + A股成本 0.10%/边 round-trip≈0.30% + T+1 + 涨跌停 + ST 源头排除) → book 连续 nav + per-cohort 20d 实现收益 + regime。**③对锚** `compare_to_anchor` vs FIN-001 锚 (净 Sharpe **1.31** CI[+0.11,+2.60]) 判落 CI 何处 + 分 regime (R11) + ≥6 完整 cohort 触发 FIN-001 复检。**demo 闭环** (dep001 已有 381 日 OOS picks 回填): book 实现净Sharpe **+0.84** 年化 +19.9% maxDD -26.8% 落 **[CI_lower_half]** (复刻 dep001 0.84 = 引擎口径忠实); regime: momentum 最弱 (cohort 均 -0.22% 胜率 43%) / reversal +2.38% 71% / mixed +2.88% 67% (印证 V12.31 动量月痛点 [[project_v3c_momentum_regime_mismatch_0603]])。二次运行 381 日全 kept_existing = append-only 红线生效。纯落库/复盘基础设施, 不改选股 (R05)。上线时 backfill 换生产每日推理即可。 |
| DLV-003 | B 研究环升级 (replication+PBO+reviewer panel) | **built** | `research/research_env.py` (三件套 util) + `research/REVIEWER_PANEL_CHECKLIST.md` (115 行) + `research/backtest/run_dlv003.py` (demo)。**①replication (B2)**: `block_bootstrap_ci`(per-cohort 块 bootstrap 误差条)+`multi_seed_replicate`(跨 seed 稳定性); 跑 WFE-001 真实 book → Sharpe 95% CI=[0.114,2.605] P(>0)=0.988 **精确复刻 FIN-001 已发布 CI[0.11,2.60]** (零口径漂移=util 忠实)。**②多重检验 (B3)**: PSR/DSR(Bailey-LdP 惩罚试验数 N)/`pbo_cscv`(CSCV); DSR 演示同样 Sharpe 1.31 若搜自 N 候选 N=1→0.87/N=10→0.53/N=200→0.13 (单 PSR=0.949 本就擦边<0.95); PBO 纯噪声族 0.139 vs 含真 alpha 族 0.000。**③reviewer panel (B1)**: 单 codex 审→多模型独立审 (codex 抓口径/lookahead + 独立 claude 抓 framing/regime) + meta-judge 仲裁, 配 §1 自审闸 + §2-3 机器统计闸 (映射 research_env 函数)。一站式 `skeptic_report()`→should_get_excited 布尔。research_env 内置 7 项解析 self-test 全过。纯流程基础设施非 alpha (status=built); 生产只读 (R05), 锚 1.31/CI 来自 FIN-001 (R01)。 |
| DLV-004 | C1 r20 特征 block-ablation 剪枝 (卫生) | **built [全块正/噪声内]** | `run_dlv004.py`: dep001 PIT-clean r20 **246 特征切 12 语义块**(估值市值4/基本面筹码1/趋势均线25/动量震荡19/波动6/量能流动性20/K线形态70/突破位置22/资金流主力45/金字塔10/市场环境12/横截面相对12, 启动 assert 互斥覆盖全集), 逐块剔除 **embargo-WF 重训**(同 dep001 口径: 24m/embargo P_start-21/固定120树/SEED, N_train 缩 300k 提速 baseline 同步重算→Δ内部一致), 19 月 OOS RankIC. **裁决 [全块正/噪声内]: 无净负块** (无块剔除后 IC 显著上升). ΔRankIC=剔块IC-baseline (>0=净负). 最"拖后腿"3块仍噪声内: trend_ma +0.0027 CI[-0.008,+0.014] / volume_liquidity +0.0007 / valuation_size +0.0005 (全 CI 含 0). 其余全负(剔除伤 IC=净正贡献), 2 块 CI 整段<0(显著净正): **market_context ΔRankIC -0.0878 CI[-0.154,-0.024]=最强单块**(剔除 IC 崩 0.088, regime/大盘上下文不可省) + moneyflow -0.0136 CI[-0.026,-0.002]. baseline(300k) IC +0.1091 (dep001 900k 参照 +0.0952). **结论: r20 特征集已干净, 无净负块可剪, 剪枝无实质收益** (印证 prd 预期"效应大概率噪声内"); 卫生消融完成, 生产线 V12.31 只读冻结 |

## 已有资产 (复用)
- **研究环 util ("骗不过的"候选筛查)**: `research/research_env.py` — `block_bootstrap_ci`(per-cohort 误差条) /
  `multi_seed_replicate`(跨 seed) / `probabilistic_sharpe_ratio` / `expected_max_sharpe` /
  `deflated_sharpe_ratio`(惩罚试验数 N) / `pbo_cscv`(PBO via CSCV) / `skeptic_report`(一站式→should_get_excited)。
  PSR/DSR 用 per-obs Sharpe (`annualized_to_per_obs`)。`python research_env.py` 跑 7 项 self-test。
  reviewer 流程见 `research/REVIEWER_PANEL_CHECKLIST.md`。**任何搜索出的候选 gate 前必过此 util** (Track B 前置)。
- **paper-trade harness**: `research/backtest/paper_trade_harness.py` — 3 段式 (`log_picks_for_date`
  append-only 落库 / `realize_book` 过冻结引擎算实现 P&L+regime / `compare_to_anchor` vs 1.31±CI)。
  `backfill_demo` 用 dep001 PIT-clean picks 回填; 上线时换生产每日推理。落库 `research/cache/paper_trade/`
  (picks/{date}.parquet append-only + realized_vs_expected.csv + paper_trade_results.json)。锚=FIN-001。
- **TreeSHAP 解释器**: `research/backtest/treeshap_explain.py` — load dep001 月度 r20 模型 + load_window 重建当月
  特征 → `pred_contrib=True` 逐只票加性归因 → output/ CSV(长表)+MD(人读)。改 DEMO_MONTH 可跑任意已缓存月。
- **回测引擎**: `research/backtest/engine.py` — `PortfolioBacktester` + `CostModel`(冻结)。日频 A股, 因果
  执行 (目标在 D 收盘决定, D+1 开盘成交), T+1/涨跌停不可成交/carryover/换手/成本。`python engine.py`
  跑 5 项解析型自检 (买入持有复刻/成本核算 round-trip 0.30%/涨跌停 n_pos=0/T+1/等权两股 carryover) 全过。
- **V12.31 基线每日持仓 (foundational, BT-002/003 复用)**: `research/cache/bt001/picks_v1231_daily.parquet`
  (trade_date, ts_code, industry, alloc_pct, r20_fresh, month; 7012 行 / 380 日 / 19 月 / 1798 股)。
  由 `research/backtest/gen_picks.py` 复用 t005 缓存月度 s5 模型 + 生产 r20 池模型生成 (**不重训**, 1.5min)。
  baseline 臂 = V7c dual-track 池内按 ratio_s5 排序 = V12.31 等价。
- **运行/裁决**: `research/backtest/run_bt001.py`; book/市场时序 `research/cache/bt001/bt001_*_costoff.parquet`;
  `research/cache/bt001/bt001_results.json`; `research/verdicts/BT-001.json`。
- **t005 选股 α 基准**: `research/cache/t005_monthly.csv` (19 月 base_alpha 均值 3.754pp); 缓存月度模型
  `research/cache/t005_wf_models/{月}/pump_scale_{3,5,10}/` (全 19 月已缓存, 选股侧无需重训)。
- 价表: `output/tushare_cache/daily/*.parquet` (open/high/low/close/pre_close/pct_chg, 1073 股文件)。
- regime: `research/features/regime_timeline.parquet` (逐日 momentum/reversal/mixed)。

## 关键约束
- 成本模型/book基线/归因法冻结禁在结果上调 (R01); 描述性归因/负发现=合法完成 (R02);
  中间指标≠ship (R03); 泄漏闸 (R04, 前向 r20_fresh 只留 research/cache/ 非 features/, 已验证)。
- 生产 hash 冻结 (R05, verify 校验 v12_scoring.py); ST 源头排除 (R06, gen_picks/run 均 ST 排除);
  大缓存 research/cache/ gitignored; 分 regime (R11, BT-002/003 归因必带)。
- verify.py: 每 task 必有 verdict, 全 task 有 verdict 才 exit0。本轮 BT-001 done, BT-002/003 缺 verdict
  → exit1 (未完成非违规, 与历轮同)。

## 迭代日志
<!-- 每轮 append -->
- (PT iter, 0616) **PT-002 built — 数据已到 0615 + 生成并落库 0615 V12.31-clean picks (前向 paper-trade 第二笔)**。
  **数据刷新 (criterion 1) 已由 daily_review.py 自愈管线完成**: 复核 5 路特征源 (mfk/pyramid_v2/v7_extras/
  amount_features/moneyflow 全 `_ext_0615.parquet`) + factor_lab_3y/factor_groups_extension (52 组全
  `_ext_review` 到 0615, 5126 行/0615) + regime/regime_extra 到 0615, 全部就绪 → 无需再跑增量 (PT-001 时
  滞后到 0611, 现已自愈到 0615)。**未动 daily_review/update_features 脚本** (数据已 fresh)。
  - **生成**: `python pt001_gen_today.py 20260615 PT-002` (脚本已参数化, criterion 2)。load_window 到 0615,
    target=20260615, universe 5048 股 (ST 源头排除 266 只/64675 行 R06)。**当期 PIT-clean r20**: embargo cut=
    **20260514** (0615−21 交易日), train_window 20240501~20260514, 246 特征 (剔 holder_pct), n_tr 900k,
    120 树。s5 排序=生产 v3c pump (forward 日 label 不存在, 因果无泄漏)。口径逐字同 V12.31-clean (V7c 池→
    ratio_s5→双轨 70/20/10→行业 cap4→close-based)。
  - **产出**: 20260615 V12.31-clean **8 持仓 / 4 行业 / 总 alloc 0.900** (90% 投资 10% 现金=双轨)。
    A 轨 5 只各 14% (正丹股份/华光新材/安纳达/国科天成/芯动联科), B 轨 3 只各 6.67% (鹏欣资源/腾亚精工/博深股份)。
    行业偏化工原料/元器件/机械基件/铜 (vs 0611 偏元器件/通信设备/航空 = 自然漂移)。ratio_s5 top=正丹股份 5.463。
  - **落库**: log_picks_for_date(20260615, source=v12.31-clean-live) **append-only**, 状态=**written**,
    读回确认 8 行; log 总计 6980 行/383 日。**0611 那笔 14 行保留不可回改** (复核 intact, append-only 红线
    SIGN-R01/R02 生效)。
  - **下一步**: 前向 paper-trade 已有 2 笔 (0611/0615)。后续每交易日续落 (改 TARGET_END 重跑或上线生产每日推理);
    满 20/40d 用 harness realize_book 算实现 P&L → compare_to_anchor 对锚 FIN-001 净 Sharpe 1.31±CI[0.11,2.60]
    分 regime (R11); ≥6 完整 cohort 触发 FIN-001 复检; 过 7 月血洗窗复检 (前向数据不回流调参 R02)。
  - 纪律: 0 违规 (verify exit0, 全 task 有 verdict)。运维落库非研究 gate (preRegisteredGate.frozen=true)。
    R04 全因果无泄漏 (embargo cut=0615−21=0514; forward 日 pump label 尚不存在; 大缓存 pt001+paper_trade
    gitignored); R05 生产线只读冻结 (PIT-clean 变体, v12_scoring hash 未变); R06 ST 源头排除; R08 r20 checkpoint。
  - **产物**: `research/cache/pt001/{pt001_results.json, r20_model/r20_clean_20260615.txt}` (gitignored R04) /
    `research/cache/paper_trade/picks/20260615.parquet` (append-only) / `research/verdicts/PT-002.json`。
- (PT iter, 0616) **PT-001 built — 生成今日 V12.31-clean picks + append-only 落库, 启动前向 paper-trade**。
  新建 `research/backtest/pt001_gen_today.py` (dep001 的**单日 live 扩展**, 补 dep001 walk-forward 批量器无单日入口的缺口)。
  最新可用交易日 **target=20260611** (本地价表/特征到 0611, Tushare 0615 未刷 → responsePlaybook 数据滞后回退, 文档化)。
  - **模型**: r20 池 filter = **当期 PIT-clean r20** (剔 holder_pct + embargo cut=预测日−21交易日=20260513,
    train_window 20240501~20260513, 246 特征, 24m/120 树/同 dep001 超参; checkpoint 落 r20_model/); s5 排序 =
    **生产 v3c pump** (r5_pump_3way_lgbm_v3c, class{0:neutral,1:down,2:up}, ratio=P_up/P_down, pump_down=P_down)。
    **关键判断**: 对真正向 (forward) 交易日, 生产模型因果无泄漏 (label 尚不存在无从泄漏), 生产 v3c = V12.31-clean
    上线后每日 live 推理的 s5 来源; dep001 用 t005 walk-forward s5 仅因它在回测历史月 (用生产 v3c 会共模)。
  - **口径逐字同 V12.31-clean**: V7c 池 (r20 top5%×pyr_velocity<q35×pump_down<0.6) → ratio_s5 排序 → 双轨
    build_dual 70/20/10 → 行业 cap (CAP_IN/CROSS 0.2, MAX_A8/MAX_B15) → close-based → ST 源头排除 (R06)。
  - **产出**: 20260611 V12.31-clean **14 持仓 / 9 行业 / 总 alloc 0.900** (90% 投资 10% 现金=双轨设计)。
    A 轨 7 只各 10% (方邦/金宏气体/金太阳/万祥/大为/信科移动/上海瀚讯), B 轨 7 只各 2.86%。行业偏元器件/通信设备/航空。
  - **落库**: paper_trade_harness.log_picks_for_date(20260611, hold, source='v12.31-clean-live') **append-only**,
    状态=**written**, 读回确认 14 行; log 总计 6972 行/382 日 (含 DLV-002 dep001 backfill demo + 今日 1 笔 live)。
    existing 拒覆盖红线生效 (SIGN-R01/R02 前向数据不回改不回流调参)。
  - **下一步**: 前向 paper-trade 第一笔真实记录就位。后续每交易日续落 (改 TARGET_END 重跑, 或上线生产每日推理);
    满 20/40d 用 harness realize_book 算实现 P&L → compare_to_anchor 对锚 FIN-001 净 Sharpe 1.31±CI[0.11,2.60]
    分 regime (R11); ≥6 完整 cohort 触发 FIN-001 复检; 过 7 月血洗窗复检 (前向数据不回流调参 R02)。
  - 纪律: 0 违规 (verify exit0, 全 task 有 verdict)。运维落库非研究 gate (preRegisteredGate.frozen=true)。
    R04 全因果无泄漏 (embargo cut=预测日−21; forward 日 pump label 尚不存在; load_window 自动删 8 个内嵌 forward
    label; 大缓存 research/cache/pt001 + paper_trade gitignored); R05 生产线只读冻结 (v12_scoring hash 未变,
    用 PIT-clean 变体非生产 in-sample); R06 ST 源头排除 (排 266 只/64597 行); R08 r20 模型 checkpoint 断点续跑。
  - **产物**: `research/backtest/pt001_gen_today.py` / `research/cache/pt001/{pt001_results.json, r20_model/}`
    (gitignored R04) / `research/cache/paper_trade/picks/20260611.parquet` (append-only) / `research/verdicts/PT-001.json`。
- (DLV iter, 0616) **DLV-004 built [全块正/噪声内] — r20 246 特征 12 块 block-ablation 剪枝: 无净负块, 特征集已干净**。
  Roadmap C1 卫生交付物收尾 (DLV 阶段全 built)。`research/backtest/run_dlv004.py`: 把 dep001 PIT-clean r20
  (剔 holder_pct) 的 **246 特征按语义切 12 块** → 逐块剔除 **embargo-WF 重训** (同 dep001 口径逐字: 24m
  lookback / embargo P_start-21 / 固定 120 树 / SEED; N_train 缩 300k 提速, **baseline 同步在 300k 重算 →
  Δ 内部 apples-to-apples**, 另报 dep001 900k 基线 IC 作量级参照) → 19 月 OOS RankIC (r20 是回归器无 AUC,
  RankIC 作 ΔAUC 天然类比)。启动 assert 12 块互斥覆盖全 246。
  - **裁决 [全块正/噪声内]** (responsePlaybook 命中): **无净负块** (无块剔除后 IC 显著上升 = 无可剪)。
    ΔRankIC = 剔块IC − baseline (>0=该块净拖后腿)。per-month 配对 bootstrap (2000×, 19 月) 95%CI 判超噪声:
    - **3 个"最拖后腿"块 Δ 微正但全噪声内** (CI 含 0, 不可剪): trend_ma +0.0027 [-0.0081,+0.0137] /
      volume_liquidity +0.0007 / valuation_size +0.0005。
    - **9 块 Δ 负** (剔除伤 IC = 净正贡献)，其中 **2 块 CI 整段 < 0 (显著净正)**: **market_context
      ΔRankIC -0.0878 CI[-0.154,-0.024] = 最强单块** (剔除 IC 从 +0.109 崩 0.088 ≈ 腰斩, regime/大盘/相对强度
      上下文是 r20 的支柱, 绝不可省) + moneyflow -0.0136 CI[-0.026,-0.002] (主力资金流次强)。
    - baseline(300k) 全期 IC **+0.1091** (dep001 900k 参照 +0.0952, 量级一致 = 300k 缩样保真)。
  - **核心结论**: r20 246 特征集**已干净, 无净负块**, 剪枝无实质收益 → **印证 prd 预期"效应大概率 21 月噪声内"**;
    与 BT-003 (book α 主导来自 r20 池整池排序非单因子) 一致 = r20 的价值是整池协同非个别块。市场环境块是 r20
    单块最大贡献者 (呼应 [[project_market_context]] 市场环境感知系统的设计价值)。
  - 纪律: 0 违规 (verify exit0, 全 task 有 verdict); 块/阈值/口径冻结未上调 (R01); 描述性卫生消融 = 合法完成
    (R02); 中间 IC≠ship 且本就无 ship 主张 (R03); 仅 load dep001 缓存 feature_meta + 重训研究侧模型, 0 生产
    改动 (R05); ST 源头排除 (R06, load_window exclude_st); 月度 checkpoint ic_by_month.csv 断点续跑 (R08);
    大缓存 research/cache/dlv004/ gitignored 无 features 写入 (R04); regime 分层 (R11, 各块 ΔRankIC 分 momentum/
    reversal/mixed)。修了首跑 ✓(U+2713) GBK 编码崩溃 → stdout reconfigure utf-8。
  - **产物**: `research/backtest/run_dlv004.py` / `research/cache/dlv004/{ic_by_month.csv, dlv004_results.json}`
    (gitignored R04) / `research/verdicts/DLV-004.json`。
  - **下一步**: **DLV 阶段全 built** (DLV-001 TreeSHAP / DLV-002 paper-trade harness / DLV-003 研究环升级 /
    DLV-004 特征剪枝)。Roadmap 剩余交付物全交付, prd 全 task passes → 可输出 COMPLETE。
- (DLV iter, 0616) **DLV-003 built — B 研究环升级 ("骗不过的" 候选筛查工具箱), util 复刻 FIN-001 CI 证零口径漂移**。
  Track B 一次性流程投资 (非研究结果): 让未来每个搜索出的候选在"激动"前过三关, 一轮抓假阳性
  (本轮 RETRO+0.0287 / Hybrid+54% 都是 CL/walk-forward 多轮才证伪的)。三件套:
  - **`research/research_env.py`**: ①**replication (B2)** `block_bootstrap_ci` (per-cohort 块 bootstrap,
    从 FIN-001 抽出为可复用函数) + `multi_seed_replicate` (跨 seed 稳定性, sign_stable & cv≤1)。
    ②**多重检验 (B3)** `probabilistic_sharpe_ratio`(PSR) / `expected_max_sharpe` / `deflated_sharpe_ratio`
    (DSR, Bailey-LdP 2014, 扣"试了 N 次"的选择偏差) / `pbo_cscv` (PBO via CSCV, Bailey 2017)。
    ③一站式 `skeptic_report()` 把机器闸打成 should_get_excited 布尔。★单位约定: PSR/DSR 全用 per-obs
    Sharpe (`annualized_to_per_obs` 转), bootstrap 报年化 (FIN-001 口径)。scipy.norm + Acklam 退路。
  - **`research/REVIEWER_PANEL_CHECKLIST.md`** (115 行, B1): 单 codex 审→**多模型独立审 + meta-judge**
    (Reviewer A=codex 抓口径混淆/lookahead/算术相消; B=独立 claude 抓因果/regime/framing 盲点 SIGN-R14;
    meta-judge 只裁问题是否被有效反驳)。§0 触发条件 / §1 作者自审闸 / §2-3 机器统计闸 (映射 research_env
    各函数 + gate) / §4 panel 对抗流程 / §5 终判落库。
  - **demo (`run_dlv003.py`) 真实工件验证**: `block_bootstrap_ci` 跑 WFE-001 真实 book → Sharpe 95% CI
    **[0.114, 2.605] 精确复刻 FIN-001 已发布 [0.11, 2.60]** P(>0)=0.988 (零口径漂移=util 忠实, 同 DLV-002
    复刻 0.84 的手法)。DSR 演示: 同 Sharpe 1.31 若搜自 N 候选 → N=1 DSR 0.87 / N=10 0.53 / N=200 0.13
    (单次 PSR=0.949 本就擦边 <0.95 — 诚实暴露 1.31 单测都不稳超 1); PBO 纯噪声族 0.139 vs 含真 alpha 族 0.000。
    research_env 内置 7 项解析 self-test (PSR/E[max]单调/DSR 惩罚/PBO 噪声vs真alpha/bootstrap/multi_seed/
    skeptic_report 端到端) 全过。
  - **产物**: `research/research_env.py` / `research/REVIEWER_PANEL_CHECKLIST.md` / `run_dlv003.py` /
    `research/cache/dlv003/dlv003_results.json` (gitignored R04) + `verdicts/DLV-003.json`。
  - **纪律**: 纯流程基础设施非 alpha (status=built, R03 不适用); R05 生产只读 (仅 load WFE-001 缓存 book,
    0 生产改动); R01 锚 1.31/CI 来自 FIN-001 不在结果上调 (util 复刻 CI 证未动门柱)。verify 本轮 DLV-004
    缺 verdict → exit1 (未完成非违规, 0 违规)。下一步 DLV-004 (r20 特征 block-ablation 剪枝, 卫生)。
- (DLV iter, 0616) **DLV-002 built — A3 前向 paper-trade 落库 harness (PIT-clean), demo 闭环复刻 0.84 = 引擎口径忠实**。
  落地 V12.31_BASELINE.md §3 协议为**可执行 3 段式 harness** `paper_trade_harness.py`:
  - **①append-only 落库** `log_picks_for_date(date, picks, source)`: existing `{date}.parquet` **拒绝覆盖**
    (除非显式 overwrite) = 协议红线落码 (SIGN-R01/R02: 前向数据不可回改/不回流调参)。二次运行 381 日全
    kept_existing 验证不可变性。picks 源 = dep001 PIT-clean r20 (剔 holder_pct embargo WF) **deployable 变体**,
    刻意不用生产 v12_scoring 的 in-sample 注水模型 (那是 +0.44 IC 记忆来源)。
  - **②满持有期实现 P&L** `realize_book`: 落库 picks 过**冻结成本引擎** (close-based 成交 D→D+1 开盘 +
    A股成本基线 0.10%/边 round-trip≈0.30% + T+1 + 涨跌停不可成交 + ST 源头排除 R06) → book 连续 nav (复用
    engine.summary 全量统计) + per-cohort 20d 实现收益切片 + regime 标注。
  - **③对锚** `compare_to_anchor`: book 实现 Sharpe vs FIN-001 真实基线锚 (净 Sharpe **1.31** CI[+0.11,+2.60],
    冻结 R01 不在结果上调) → 判落 CI 何处 (below/lower_half/upper_half/above; <1.0=重评估, 上半=向可部署收敛) +
    分 regime (R11) + ≥6 完整 cohort 触发 FIN-001 复检标志。
  - **demo 闭环** (dep001 已有 381 日 OOS picks 回填): book 实现净Sharpe **+0.84** 年化 +19.9% maxDD -26.8%
    月换手 1.54 月胜率 55% 落 **[CI_lower_half]** → **精确复刻 dep001 PIT-clean 0.84 = harness 引擎口径忠实**
    (零口径漂移); regime 分层: momentum **最弱** (cohort 均 -0.22% 胜率 43%) / reversal +2.38% 71% /
    mixed +2.88% 67% → 再次印证 V12.31 动量月痛点 ([[project_v3c_momentum_regime_mismatch_0603]] + SIGN-R11)。
  - **产物**: `research/cache/paper_trade/{picks/{date}.parquet, log.jsonl, realized_vs_expected.csv (20 cohort),
    paper_trade_book.parquet, paper_trade_results.json}` (gitignored R04) + `verdicts/DLV-002.json`。
  - **纪律**: 纯落库/复盘基础设施非 alpha (status=built, R03 不适用); R05 生产只读 (仅 load 缓存 picks/价表,
    0 生产改动); R01 口径逐字同 V12.31_BASELINE/dep001/WFE-001; R02 append-only 红线。verify 本轮 DLV-003/004
    缺 verdict → exit1 (未完成非违规, 0 违规)。
- (DLV iter, 0616) **DLV-001 built — A2 TreeSHAP 可解释层落地 (逐只票加性归因), 加性自检 5.3e-14 PASS**。
  北极星换轨到 Roadmap 剩余交付物 (A1 低换手已否)。`treeshap_explain.py` 用 **LightGBM 内置 pred_contrib**
  (TreeSHAP, 无需第三方 shap 库) 对 dep001 PIT-clean r20 (剔 holder_pct, label-embargo WF) 的 **202604
  全月 311 只 picks** 算逐特征加性贡献。
  - **faithful 重建**: 同 dep001 load_window(20220801-20260601, ST 源头排除 R06) + Categorical industry_id +
    booster.feature_name() 246 特征逐位对齐; 缺源长horizon因子 (504d/252d 等 11 个, factor_lab 覆盖到 ~20260126)
    设 NaN (非 0, 与 full-window dep001 一致, LGBM 原生处理 NaN) → 202604 特征值与 dep001 预测口径一致。
  - **加性性质验证**: 每行 `pred_r20 = base_value(+2.4171) + Σ(246 因子 SHAP)`, max\|Σcontrib+base−pred\|=
    **5.33e-14** ≪ 1e-4 (回归 booster TreeSHAP 加性恒等式成立) → 分数可完整拆成"各因子推高/拉低多少分"。
  - **全期 top3 驱动** (\|mean SHAP\|, 审计概览): cyb_rel_strength 2.80 (创业板相对强度, 主推高) /
    industry_id 1.41 (行业, +1.30) / mkt_ret_20d 0.92 (大盘动量, signed −0.26 拉低)。
  - **产物**: output/treeshap_dep001_202604_contribs.csv (逐 date,stock,feature top±贡献长表, utf-8-sig) +
    treeshap_dep001_202604_demo.md (最新再平衡日 20260430 的 14 只 picks 人读 top5 正/负归因表 + 全期 top15 驱动)。
  - **纪律**: 纯解释层非 alpha (无落地宣称, R03 不适用); R05 生产只读 (仅 load dep001 缓存模型, 0 生产改动);
    R06 ST 源头排除; R04 大缓存 gitignored。verify 本轮 DLV-002/003/004 缺 verdict → exit1 (未完成非违规, 0 违规)。
- (STAB iter, 0616) **STAB-002 skipped (前提不满足) — A1 Track 收尾: STAB-001 无改进变体 → documented skip (R02)**。
  STAB-002 是预注册的'最佳低换手变体深验'(换手-成本敏感 0/0.2/0.6/1.0% 双边 + 稳定性扰动重排率 + 分regime 净Sharpe),
  acceptanceCriteria 明确 '仅 STAB-001 找到改进变体才跑, 否则 skip 并文档化'。STAB-001 已裁决 **无差异**
  (best_variant=null, winners=[], 4 变体 baseline/V_stick/V_N30/V_combo **无一过 gate_stab**: 最大换手降幅仅 -2.5%
  远不及 gate 的 -20%, V_N30/V_combo Sharpe 反降) → 深验前提不满足。**故 STAB-002 不运行** (R01: 不在不存在的
  最佳变体上硬跑成本-regime 网格凑数, 冻结网格仅记录不实例化; R02: documented skip = 合法完成; R03: STAB-001
  换手/Sharpe 是描述性中间指标不据此 ship)。机制根因复述: 换手是**选股池结构性** (连续再平衡 Jaccard ≈0.02 =
  每 20d 期 ~98% 换名), 构建层粘性/增N 触不到根因。**纪律意义**: A1 Track gate 链在 STAB-001 '无差异' 处诚实终止,
  不下探到空集合上的 p-hack——同 CL-002 (CL-001 死于消融→skip) 的反过拟合脚手架行为。verify exit0 (全 task 有 verdict)。
  产 verdicts/STAB-002.json。生产线 V12.31 全程只读冻结 (R05, 未碰任何生产文件)。0 违规。
- (STAB iter, 0616) **STAB-001 built [无差异] — A1 低换手/稳定化构建网格: 构建层无法降换手 (换手是结构性)**。
  2 脚本: `stab001_gen_pool.py` (重建每日 V7c 候选池, NO retrain 复用 dep001 剔holder r20 + wfe001 含holder r20 + t005 s5
  缓存模型, clean+holder 双臂供扰动重排率) / `run_stab001.py` (4 变体构建过引擎 + 配对 bootstrap + gate + regime)。
  - **双验证 baseline 忠实**: 净Sharpe **+0.843 == DEP-001 0.843** (size_dual 逐行复刻 build_dual) 且扰动Jaccard
    **0.3265 == DIAG-001 0.327** (clean vs holder 同构建 pick 重叠) → 构建层与 V12.31 逐位一致, 变体 Δ 干净。
  - **4 变体** (同 PIT-clean 分数/双轨权重/embargo/成本, 仅构建规则不同, R01 冻结): baseline (8/15 top20) /
    V_stick (已持有名 pool_rank≤30 给 ratio_s5 bonus 留任) / V_N30 (10/22 降集中, 持仓 26) / V_combo (粘性+N30)。
  - **核心负发现**: 无变体过 gate_stab。换手 baseline 1.54 → V_stick 1.50 (**-2%**) / V_N30 1.53 (-0%) / V_combo 1.51 (-2%),
    **全部远不及 gate 要求的 -20%**。V_N30 Sharpe 0.84→0.77 (CI 含0)、V_combo 材料性降 0.57 (CI[-0.515,-0.019])。
  - **机制 (为何粘性无效)**: 连续再平衡日持仓 **Jaccard ≈0.02** = 每 20d 期 ~98% 换名 → V7c 候选池 (r20 top5% × ratio_s5
    排序) 本身每期近乎全量重排, 已持有名极少仍在 top30 → **滞回带罕触发, 没有可留任的"粘性名"**。换手是**选股池结构性**,
    非构建可修。N30 降集中只是收益/回撤微权衡 (maxDD -26.8→-25.8% 但 Sharpe ↓), 不解换手。
  - **裁决 [无差异]**: 稳定性/换手不能靠粘性或单纯增 N, 须改选股层 (horizon/分数平滑 = 改选股逻辑, 超 A1 范围)。
    STAB-002 前提 (找到改进变体) 不满足 → 下一迭代 skip 文档化。生产线 V12.31 只读冻结 (R05, 缓存模型直接 load 未碰生产)。
  - 纪律: 0 违规; baseline 复刻 0.843 + 扰动 0.327 双锚证构建忠实 (apples-to-apples); 配对 bootstrap 消共模 (R01 阈值冻结);
    ST 源头排除 (R06); regime 分层 (R11, regime_by_variant); 负结果=合法完成 (R02); 中间指标≠ship (R03);
    大缓存 research/cache/stab001/ gitignored 无 features 写入 (R04); verify 本轮 STAB-002 缺 verdict → exit1 (未完成非违规)。
- (init) 由 18 个选股因子全否转入 book/风控层。从没建过真实组合回测器, 本轮建引擎 + V12.31 净 P&L + 因子归因。
- (iter1, 0614) **BT-001 built — 参数化 book 级回测引擎落地 + 双重 sanity 通过 (输入忠实 + 引擎无 bug)**。
  3 脚本: engine.py (引擎) / gen_picks.py (生成 V12.31 基线持仓, 复用缓存模型不重训) / run_bt001.py (运行+sanity)。
  - **引擎** (engine.py): `PortfolioBacktester` 日频 A股模拟器。因果执行 (目标 D 收盘决定→D+1 开盘成交,
    避免前视), 成本模型冻结 (佣金/印花/过户/滑点, buy 0.126% sell 0.176% round-trip 0.302% ≈ DESIGN),
    T+1 (当日买入不可卖, 跟 last_buy_date), 涨跌停 |pct_chg|>=9.8 不可成交, 停牌沿用上一收盘, 持仓
    carryover + 换手 + 现金。输出逐日 nav/ret/exposure/n_pos/turnover/cost + summary (年化/Sharpe/maxDD/月换手)。
  - **解析型自检 (R04 前置闸 + 无 bug 证明)**: 5 项全过 — ①买入持有 nav==标的累计收益 ②成本核算总成本==buy+sell率
    ③涨跌停日 n_pos=0 ④T+1 跨日清仓 ⑤等权两股 carryover nav==等权平均累计。`python engine.py` 全 assert 通过。
  - **picks 生成** (gen_picks.py, 1.5min): load_window(20240901-20260601, ST 排除) → 用 t005 **已缓存**
    19 月 s5 pump 模型 (ratio_s5) + 生产 r20 池模型 (pred_r20) 预测 → t005 `build_dual` 构 V7c dual-track 池
    (池内按 ratio_s5 排序 = V12.31 baseline 臂) → 每月落盘 checkpoint。**全程不重训** (SIGN-R05 生产只读)。
    产出 picks_v1231_daily.parquet (7012 行/380 日/19 月/1798 股, alloc_pct 日和≈0.90)。
  - **sanity (run_bt001.py, 0.2min)**: **(A) t005 选股 α 复刻** — 用 picks 重算 t005 month_metrics 口径选股 α,
    19 月 vs t005_monthly base_alpha: **corr +0.999 / MAE 0.66pp** (repl 均值 +4.42 vs t005 +3.75, ~0.66pp 常数
    偏移来自市场基准宇宙差异: 我用 ST 排除全日宇宙均值, t005 用特征可得 df_test 宇宙均值) → **喂引擎的 picks
    忠实于 V12.31 baseline 臂**。**(B) 引擎 book** — 等权 20d 再平衡成本关 book 过引擎: 年化 +126.5% Sharpe 2.79
    maxDD -21.3% 月换手 1.52 暴露 0.92 持仓 20; 等权全市场基准年化 +16.7% Sharpe 0.86; 引擎每期(≈20d)超额
    均值 **+5.45%** 中位 +5.24% (20 期) vs t005 选股 α +3.75% → **同量级 = 引擎无 bug**。
    (注: +126.5% 是成本关 + 含 r20 池共模 lookahead + 等权集中的 sanity 配置, **非可交易声明**; BT-002 开真实成本+滑点敏感+基准/基金对照才判该不该跑。)
  - 纪律: 0 违规 (verify exit1 仅因 BT-002/003 未做=未完成非违规); 成本/基线/口径冻结逐字未上调 (R01);
    sanity 是描述性非 ship (R03); 前向 r20_fresh 只在 research/cache/bt001/ (gitignored), 非 features/ (R04, verify 无泄漏);
    生产 hash 一致 (R05, gen_picks 复用缓存模型未碰生产文件); ST 源头排除 (R06); checkpoint 每月落盘 (R08)。
  - **下一步 BT-002**: 全 history 跑 V12.31 生产 book (用 picks_v1231_daily, **alloc 加权双轨非等权**) 过引擎,
    开真实成本 + 滑点敏感 (0/0.05/0.10/0.20), 报年化/净Sharpe/最大回撤/月换手/胜率 vs hs300+CSI1000+动量基准,
    分 regime (R11) 看是否动量月系统性跑输 (= 用户"基金赢我们输"的 book 层检验)。run_bt001 已留 `build_rebalance_targets(equal=False)` alloc 加权钩子 + period_excess 框架可复用。
- (iter3, 0614) **BT-003 built — 因子/铁律 leave-one-out 归因 (谁在 book 层拖后腿)**。新增 `research/backtest/run_bt003.py`。
  - **设计 (冻结 R01)**: baseline arm = 完整 V12.31 研究 book (t005 `build_dual` 全过滤+双轨+ratio_s5排序); 每个消融 arm = baseline 去**单一**组件 (leave-one-out), 其余逐字一致 → Δ 是**受控差分** (各 arm 共享同一 r20 池 lookahead + 同基线成本0.10%/边 + 同引擎), 共模偏差相消 → 归因 (谁正/谁拖) 比 BT-002 绝对量级**更可信** (SIGN-R03 只看相对 Δ)。评分复用缓存 s5+r20 模型**不重训** (R05), 评分按月 checkpoint (R08)。
  - **7 个消融 arm** (本研究 book 实际实例化): 评分组件 r20池filter/pump_up排序/pump_down过滤/行业cap/双轨sizing + V7c 铁律 pyr_velocity力竭/行业60d动量排除。诚实标注: 生产推理铁律**双静默(±1%)/非僵尸不在 t005 研究池 harness** (推理时过滤非池构造), 不在本 task 范围, 不杜撰消融。
  - **基线**: 净Sharpe **+2.10** 年化 +94.0% maxDD -23.6% 月换手 1.53 持仓 23。
  - **归因表 (ΔSharpe vs baseline, 负=该组件正贡献)**:
    | 组件 | 类型 | ΔSharpe | Δ年化 | 裁决 |
    |---|---|---|---|---|
    | **r20池filter** | 评分 | **-1.37** | **-74.9%** | **绝对核心** (去掉年化崩 +94%→+19%) |
    | pump_up排序(ratio_s5) | 评分 | -0.16 | -9.6% | 正贡献 |
    | pyr_velocity力竭 | 铁律 | -0.06 | +3.8% | 弱正贡献 |
    | pump_down过滤 | 评分 | **0.00** | 0.0% | **完全惰性** (r20池候选几乎无 s5≥0.60, filter 不 binding) |
    | 行业cap | 评分 | +0.03 | +2.7% | 近中性/边际拖 |
    | 双轨sizing | 评分 | +0.04 | -15.0% | 近中性 (双轨增**收益**但增**回撤** maxDD -23.6 vs 单轨 -18.9) |
    | **行业动量排除** | 铁律 | **+0.10** | +5.3% | **最拖后腿** (去掉反而改善, mixed regime +1.29pp) |
  - **核心结论**: book α **几乎全靠 r20 池** (单组件占 Sharpe 的 ~65%); pump_up 排序与 pyr 铁律次正; **pump_down 在 book 层是死规则** (与 r20 池正交度低, 候选已不含跌启动子); **行业动量排除铁律在 book 层净拖后腿** (尤其 mixed regime) → **future opt-in 放松候选** (生产仍冻结, SIGN-R05, 仅作依据)。双轨 sizing 是收益/回撤权衡 (单轨 Sharpe 微高但回撤更小, dual 是"集中博收益"取向)。
  - 纪律: 0 违规 (verify exit0, 全 task 有 verdict, BT-004 skip); 池参数/成本逐字冻结复刻 t005 build_dual (R01); 描述性归因非 ship (R02/R03); ST 源头排除 (R06, load_prices); regime 分层必带 (R11, 各 arm Δ 分 regime); 复用缓存模型未碰生产 (R05); 评分月度 checkpoint research/cache/bt003/scored_by_month/ (R08); 大缓存 gitignored, 无 features 写入 (R04)。
  - **产出**: `research/backtest/run_bt003.py` / `research/cache/bt003/{scored_daily.parquet, scored_by_month/, ind_mom.parquet, bt003_results.json}` / `research/verdicts/BT-003.json`。
  - **下一步**: BT 主体 (BT-001/002/003) 全 built。BT-004 (skip, 预注册) = 事件上下文 meta-label 单表示残差测 (ctx=MA20上拐×MA5上穿, 测 T+20 残差 RankIC 扣[pump+MA20斜率+pyr_velocity+ADX] + Deflated Sharpe/PBO)。可在用户开启时启动 (回测器+归因已就绪, BT-003 已揭示 r20 池主导/行业动量排除拖后腿, informs meta-label 落点)。
- (iter4, 0614) **BT-004 REJECT_reskin — 事件上下文 meta-label 单表示残差测**。新增 `research/backtest/run_bt004.py`。
  - **单一表示 (冻结 R01, 防钓鱼)**: `ctx = (MA20 在事件前 K=5 日内已上拐) × (MA5 上穿 MA20 金叉)`。K=5 固定 (= MA5 短窗), **严格不搜窗口/不搜 MA 组合** (López de Prado 组合过拟合防护)。事件率 1.48% (28126 例 / 19 月)。
  - **数据 (全因果)**: 价表 daily → close/high/low → MA5/MA20/MA20斜率(5d归一)/ADX(Wilder-14)/金叉/MA20上拐 (全 backward, checkpoint `research/cache/bt004/px_feats.parquet`); pump(ratio_s5)/pyr_velocity/T+20(r20_fresh) 复用 BT-003 `scored_daily` (零重算)。ST 源头排除 (R06), 前向 r20_fresh 只在 cache (R04)。
  - **残差 RankIC (扣 [pump+MA20斜率+pyr_velocity+ADX])**: 每截面 ctx 对 controls (rank 空间) OLS 正交 → 残差 vs T+20 Spearman。
    - **非重叠 (每 20 交易日取样, headline 防 T+20 重叠致 t 膨胀)**: 残差 IC **+0.00019  t=0.007 (n=19)** ≈ 纯零。
    - 全日重叠 (参考): 残差 IC -0.00159 t=-0.25 (n=372)。
    - raw (不扣 controls) 非重叠 IC **-0.00287** → 金叉本身对 T+20 **微负** (A股"确认翻多→均值回归"，与 [[project_duokongk_reskin_reject_0614]] 一致)。
  - **Deflated Sharpe = PSR(SR*=0, 单表示 trials=1)**: ctx=1 篮子超额 spread (非重叠 20d 周期, n=19) per-period Sharpe **-0.12** 年化 -0.42, PSR(SR>0) **0.31**。PBO (CSCV) 单配置退化不适用 (无多策略试验集), 以单表示 DSR=PSR 替代多重检验扣减。
  - **regime (R11)**: reversal 月残差 IC +0.0197 (t=1.84) 微正, momentum -0.0180 (t=-1.67)/mixed -0.0211 (t=-2.30) 微负 → **无稳定方向** (符号随 regime 翻转, 非真信号)。
  - **Gate (冻结阈值 |残差IC|>=0.01 & |t|>=2 & PSR>=0.95, 三闸)**: IC FAIL (0.0002) / t FAIL (0.007) / deflation FAIL (0.31) → **REJECT_reskin**。
  - **核心结论**: ctx 增益被现有 [MA斜率/pump/pyr/ADX] **完全吃掉** (印证 DESIGN §5 先验: MA 事件序列 ≈ 现有 TA 重打包, 同关系张量 TCN ≈0)。**第 19 个被反过拟合脚手架诚实否的假设** — 单表示一次定调, 未 p-hack/未搜窗口再试 (R02)。生产线 V12.31 全程冻结 (R05)。
  - 纪律: 0 违规 (verify exit0, 全 task 有 verdict); 单表示+阈值逐字冻结未上调 (R01); 中间 IC≠ship, 三闸共判 (R03); 非重叠取样防 t 膨胀; 描述性 REJECT=合法完成 (R02); ST 源头排除 (R06); regime 分层必带 (R11); px_feats checkpoint (R08); 大缓存 gitignored 无 features 写入 (R04); 复用缓存评分未碰生产 (R05)。
  - **产出**: `research/backtest/run_bt004.py` / `research/cache/bt004/{px_feats.parquet, bt004_results.json}` / `research/verdicts/BT-004.json`。
  - **下一步**: **BT 全 task 完结** (BT-001/002/003 built + BT-004 REJECT_reskin)。book 级回测器 + 因子归因 + meta-label 残差测 北极星全部交付。可输出 COMPLETE。
- (iter5, 0614) **EX-001 built — 出场策略测 (分批止盈 TP + 回撤止损 SL), 同 picks 受控 Δ**。新增 `research/backtest/run_ex001.py`。
  - **EX 阶段开张** (承用户 triple-barrier 直觉, 19× 选股信号穷尽后转目标函数/出场/universe)。对**同一批 V12.31 picks** (= BT-002 入场, 20 cohort/每20交易日/alloc 加权双轨) 套预注册出场网格, 仅出场逻辑随 arm 变 → **受控 Δ, 入场逐位一致, 共模 (r20 池 lookahead/集中) 相消** (R03)。book = cohort 子账户逐日等权聚合 (止盈后转现金, 无活 cohort 当日收益=0 = 闲置拖累的真实计入)。
  - **出场网格 (冻结 R01, 不搜不调)**: baseline 固定 20d 收盘平; TP 三档 +10/+20/+30% 各卖原仓 1/3 (日内 high 达标价, 缺口越过→开盘, 涨停顺延); SL 自入场峰值回撤 {-8%,-12%} 卖全部 (日内 low, 缺口跌破→开盘, 跌停顺延); 组合 TP+SL; 40d backstop。close-based 保守界另算。
  - **baseline**: 年化 +90.8% 净Sharpe +2.40 maxDD -16.1% (cohort 等权聚合口径, 故与 BT-002 +2.78 略异, 此处只看受控 Δ)。
  - **受控 Δ vs baseline (ΔSharpe / Δ年化 / ΔmaxDD)**:
    | arm | ΔSharpe | Δ年化 | ΔmaxDD | 读 |
    |---|---|---|---|---|
    | **TP_only** | **+1.358** | -13.31pp | +5.64pp | **强帮 Sharpe** (顺均值回归 edge 减仓, 回撤砍半 -16→-10.5%, 收益仅小让) |
    | SL_8 | -0.071 | **-73.37pp** | +12.57pp | Sharpe 微伤, 收益毁灭 (砍在反弹前) |
    | SL_12 | +0.249 | -51.53pp | +5.78pp | Sharpe 微帮但年化腰斩 |
    | TP_SL_8 | -0.187 | -76.82pp | +12.86pp | SL 主导拖垮 |
    | TP_SL_12 | +0.766 | -53.93pp | +10.56pp | TP 救回部分 Sharpe 但 SL 仍杀收益 |
  - **先验验证**: 分批止盈帮 Sharpe=**True** (强, ΔSharpe +1.358, 最优 arm); 回撤止损伤 Sharpe=**False** (两档不同号: -8% 微伤 / -12% 微帮) → 但**两档 SL 都毁灭年化** (-51~-73pp), 印证均值回归先验"止损砍在反弹前"在**收益维度**成立, Sharpe 维度因 vol 同比下降而被掩盖。close-based 保守界同向 (TP_only close ΔSharpe 仍正)。
  - **regime (R11)**: TP_only 跨 regime 近中性 (momentum +0.16 / reversal -0.44 / mixed -4.40 月pp), SL 各 regime 全负 (mixed 最伤 -7.6~-11.7pp)。
  - **核心结论**: **分批止盈 (TP) 是出场层唯一 Sharpe 增益候选** (回撤砍半, 收益小让) → 进 EX-003 上屏障设计为主; **回撤止损 (SL) 弃** (毁灭收益, 印证均值回归反指, 见 [[project_three_rejects_meanreversion_meta_0603]])。指向 EX-003 r20 triple-barrier 重标用**上屏障(止盈)为主, 下屏障(止损)谨慎/放宽**。
  - 纪律: 0 违规 (verify exit1 仅 EX-002 未做=未完成非违规); 出场网格逐字冻结未上调 (R01); 入场=BT-002 同 picks 共模相消 (R03); 描述性受控对照=合法完成 (R02); 中间指标≠ship, ship 由 EX-003 walk-forward 定 (R03); ST 源头排除 (R06); regime 分层必带 (R11); 绝对量级含 r20 池 lookahead 非可交易声明; 大缓存 research/cache/ex001/ gitignored 无 features 写入 (R04); 复用 picks 未碰生产 (R05)。
  - **产出**: `research/backtest/run_ex001.py` / `research/cache/ex001/{ex001_results.json, ex001_nav_*.parquet}` / `research/verdicts/EX-001.json`。
  - **下一步 EX-002**: V12.31 picks 风格画像 (市值/波动/换手/行业 vs 全市场) + book Sharpe vs 大盘质量基准 + 池C基金重仓静态篮 (标注幸存者偏差), 尝试 Tushare fund_portfolio 真共识篮否则降级; 归因"基金更强"是风格还是选股。EX-001 已示出场层 TP 有增益但非主战场 → EX-002 诊断风格差距, 共同 informs EX-003 重标。
- (iter6, 0614) **EX-002 built — 基金/风格对照 (基金更强是选股还是风格)**。新增 `research/backtest/run_ex002.py`。
  - **三部分 (冻结 R01, DESIGN §7)**: [A] picks 风格画像 (20 再平衡日, picks vs 全市场) [B] 基金持仓风格 + book Sharpe 对照 [C] regime 分层。daily_basic (total_mv/turnover/pe/pb) 从 Tushare 拉 20 个再平衡日 (per-date checkpoint, R08); 波动率从价表算 (过去20d年化std); 基金共识篮用 fund_portfolio **真历史 7 季报** (20240930~20260331, 7 基金 742 行) ann_date 因果可得 (非单一静态快照外推)。
  - **[A] 风格画像 (picks 中位数在全市场百分位, 跨日均值)**: 市值(log万元) **32.5%** / 波动(年化) 49.4% / 换手 50.1% / **PE(ttm) 65.6%** / PB 50.9%。→ **小盘 + 高PE成长**, 但**波动换手≈市场中位 (非高波高换手)** — **纠正北极星"小盘高波"先验** (高波高换手不成立, 真画像是小盘成长)。行业 over-weight: 机械设备 +3.9pp / 铸货 +2.0 / 铜 +2.0 / 小金属 +1.8; under-weight: 中药 -0.9 / 化学药 -1.1 / 半导体设备 -1.6 / 电池 -1.6。
  - **[B] 基金持仓风格**: 7 基金最新季报 (20260331, 57 股) 持仓市值中位 **401.6 亿元 = 全市场 90.3% 百分位 = 大盘质量**; vs 我们 picks 32.5% → **基金显著更大盘** (fund_holds_larger_cap=True)。
  - **[B] book Sharpe 对照** (20241008~20260611): V12.31 **+2.78** (lookahead 注水) / hs300 **+0.46** / CSI1000 +0.81 / **基金共识篮 +0.64** (真历史季报 ann_date 因果, 等权 top15 过同引擎同基线成本 0.10%)。
  - **[C] regime (R11, book vs hs300 月超额)**: mixed +9.62pp / momentum +4.44pp (相对最弱, 同 BT-002) / reversal +6.23pp; book vs 基金篮 momentum +2.12pp (动量月基金篮月收益 +3.28% 远高于反转/混合 ≈0, 印证基金大盘在动量行情更顺)。
  - **归因 = 风格差** (responsePlaybook['风格差']): 与基金的差异**主导来自风格/universe 暴露** (我们小盘高PE成长, 基金大盘质量, 市值百分位 32% vs 90%) → **增量在风格 tilt / 风险管理, 非单纯选股重标**; 但 V12.31 book 绝对 Sharpe 受 r20 池 lookahead 注水 (同 BT-001/002), 真实选股 α 须 EX-003 去 lookahead 定。基金篮含 fund-selection survivorship (7 基金今日存在) 但持仓真历史季报 (比纯静态篮诚实)。
  - 纪律: 0 违规 (verify exit0, 全非 skip task 有 verdict, EX-003 skip); 对照法/基准/基金篮构造冻结未上调 (R01); 描述性风格画像+对照=合法完成 (R02); 中间指标≠ship, ship 由 EX-003 walk-forward 定 (R03); ST 源头排除 (R06); regime 分层必带 (R11); daily_basic per-date checkpoint (R08); 复用 BT-002 baseline book + 指数缓存未碰生产 (R05); 大缓存 research/cache/ex002/ gitignored 无 features 写入 (R04); 绝对量级含 lookahead 非可交易声明。
  - **产出**: `research/backtest/run_ex002.py` / `research/cache/ex002/{ex002_results.json, ex002_style_profile.parquet, ex002_fund_basket.parquet, ex002_monthly_regime.parquet, fund_portfolio_hist.parquet, daily_basic/*.parquet}` / `research/verdicts/EX-002.json`。
  - **下一步**: EX-001 (出场) + EX-002 (风格) 双双完结 → EX-003 (skip, 预注册) r20 triple-barrier 重标 + walk-forward 已被 informs: 出场层 TP 有增益 (上屏障为主), 风格差是主因 (小盘成长 vs 大盘质量 → 重标可瞄风格中性的干净路径)。EX-003 是 isGate 重活, 待用户开启 (un-skip) 后启动。本轮非 skip task (EX-001/EX-002) 全有 verdict, verify exit0。
- (WF iter, 0614) **WF-001 built — de-lookahead r20 真实 walk-forward → V12.31 可交易 P&L (吃进 codex 审计, 地基级)**。新增 `research/backtest/wf001_gen_picks.py` (月度重训+生成OOS持仓) + `run_wf001.py` (过引擎+对照+verdict)。
  - **唯一注水源诊断**: BT-001/002/003/EX 的绝对量级 (+140%年化/Sharpe2.78) 注水 = **r20池排序模型 r20_v16_long_nost 是单一生产模型 (训练窗<20250930) 却跨整个测试期 202410-202604 预测 → 19测试月里 12月 (202410-202509) 在其训练窗内 = 共模 memorization lookahead**。(pump s5 排序模型在 t005 已月度 walk-forward, 无 lookahead; 故唯一 de-lookahead 对象 = r20。)
  - **方法 (R01 前置冻结)**: r20 池模型做**严格月度 walk-forward 重训** (复刻 train_daily_long_oos 的 LGBMRegressor 配置, 24m lookback, 每预测月只用该月之前数据)。**树数固定 120 棵无早停** (踩坑后定: 生产自选 best_iter=87 是该信号恰当复杂度; walk-forward 下两种 val 都退化 — 单一2.4m时间切片 IC 噪声大早停到2棵 degenerate高估缩水, 随机holdout与train共享日期 val IC单调升训到2000cap过拟合低估缩水; 固定120棵+强正则是最忠实的"只用过去数据生产风格r20"复刻, 看book前定 = 非tune-to-target)。生成 OOS pred_r20 → 重建 V12.31 dual-track (池内仍按已 walk-forward 的 ratio_s5 排序) → OOS 每日持仓 → 过同 BT-002 引擎/成本/再平衡。
  - **结果**: OOS de-lookahead book 真实可交易: **年化 +65.9% 净Sharpe +1.84 maxDD -18.0% 月换手 1.51 月胜率 65%**; vs BT-002 注水版 (+140.5%/2.78): **缩水 ΔSharpe -0.94 / Δ年化 -74.6pp (年化约腰斩, Sharpe 留 66%)**。仍**净正且跑赢全基准** (hs300 Sharpe 0.46 / CSI1000 0.81 / 动量 0.51)。
  - **smoking gun (因果自检, r20 截面 rank-IC vs r20_fresh)**: 生产单一模型 in-sample段(≤202509) IC **+0.4357** → true-OOS段(≥202510) **+0.1814** (暴跌, lookahead 痕迹); **WF 月度重训在同一 true-OOS 段 IC +0.1844 ≈ 生产 true-OOS +0.1814 (几乎相等)** → 生产模型在真 OOS 上**无 lookahead 优势, 其 +0.44 全是记忆**; 真实 r20 OOS IC ≈ WF 全期 **+0.10**。**这正是 +140% 注水来源** (12/19 月在训练窗内 memorize)。WF 训练**未坏** (202501-04 月 OOS IC +0.15~+0.37 真正向, 早期 202410-12 负 IC 因晚2024动量暴涨 vs 均值回归训练的模型错配)。
  - **regime (R11, de-lookahead后)**: momentum月(n=9) 超额 hs300 **+0.79pp 胜率仅44%** (<50%!) = **真OOS下动量月确是最弱 regime** (vs reversal +4.33pp/78%, mixed +5.60pp/100%) → 强化用户"动量月吃亏"/RG-002 regime错配关切 (注水版掩盖了胜率<50%)。
  - **playbook 命中 [大幅缩水]**: 真实 Sharpe 远低于注水版 → **之前所有绝对数 (BT-001/002/EX, 含 r20 池 lookahead) 不可信, 实盘预期须按真实数 (~Sharpe 1.84/年化+66%, 非+140%) 重设**。但 V12.31 仍真可交易 (Sharpe 1.84 跑赢全基准)。**关键: 相对 Δ/受控对照结论 (BT-003 归因 / EX-001 止盈Δ / EX-002 风格) 因两臂相消不受 lookahead 影响, 仍成立** — 这正是 WF-002 修正版止盈复测的前提 (但 SIGN-R13: Δ 须在这批 de-lookahead 真实 OOS picks 上复核)。
  - 纪律: 0 违规 (verify exit1 仅 WF-002 未做=未完成非违规); de-lookahead 唯一改动=r20 单模型换月度 walk-forward, 引擎/成本/再平衡/双轨逐字同 BT-002 (R01); 树数固定看 book 前定非 tune (R01/R02); ST 源头排除 (R06); regime 分层必带 (R11); 月度 checkpoint r20_models/+picks_by_month/ (R08); 前向 r20_fresh 只在 research/cache/wf001/ gitignored 无 features 写入 (R04); 复用缓存 s5 模型+生产 r20 仅作对照未碰生产文件 (R05)。
  - **产出**: `research/backtest/{wf001_gen_picks.py, run_wf001.py}` / `research/cache/wf001/{picks_oos_daily.parquet, r20_models/, r20_oos_diagnostics.csv, wf001_book_oos.parquet, wf001_monthly_regime.parquet, wf001_results.json}` / `research/verdicts/WF-001.json`。
  - **下一步 WF-002**: 在**这批 WF-001 真实 OOS picks** 上跑修正版止盈复测 (SIGN-R13 负控包: baseline-20d/40d 对齐 + TP-40d/close + placebo 随机阈值/固定降暴露现金化 + per-cohort block bootstrap CI + leave-one-out + 分regime), 按 gate_tp 判 EX-001 的 "+1.36 Sharpe" 在去混淆(40d)+去暴露(placebo)+去lookahead(真OOS picks)后是否仍提 Sharpe。WF-001 已交付可交易地基, picks_oos_daily 即 WF-002 输入。
- (WF iter, 0614) **WF-002 TP真改进(frozen gate)/但非择价技能 — 修正版止盈复测吃进 codex SIGN-R13 负控包**。新增 `research/backtest/run_wf002.py`。
  - **三处去混淆 (codex SIGN-R13)**: ① picks 换成 **WF-001 de-lookahead 真实 OOS picks** (非含 r20-lookahead 的 bt001 picks); ② 加 **baseline-40d** 臂与 TP 的 40d backstop 同持有期 (gate 用 TP40-base40, 去 EX-001 的 baseline-20d/TP-40d 持有期错配); ③ 两个 **placebo 负控**。网格全冻结 (R01, 不搜不调)。
  - **核心受控 Δ (apples-to-apples)**: baseline_40d 净Sharpe +1.42 / TP_40d +2.23 → **ΔSharpe +0.809** (Δ年化 **+0.03pp** ≈不变, ΔmaxDD **+14.71pp** = -25.4→-10.7%). vs **EX-001 混淆口径** (TP40-base20) ΔSharpe +0.556; EX-001 原报 **+1.358** = baseline-20d/TP-40d 持有期差 + r20-lookahead 理想样本双重注水, 此处全去后缩到 **+0.809**。
  - **双 placebo (关键裁决)**: (1) **静态降暴露** (f=0.677 匹配 TP 平均暴露 0.527/baseline40 0.778, 余现金) ΔSharpe **-0.09** ≈0 → 印证零息现金静态混合不抬 Sharpe, TP 增益**非'少持仓数学抬升'** (codex 该担忧**排除**); (2) **随机阈值 TP** (30 seed, uniform[0.05,0.35] 排序, 同'卖1/3三次'机制) ΔSharpe **+0.769** [p5,p95]=[+0.521,+1.085] → **真实 TP +0.809 落在随机分布内, 统计无法区分** → **没有'卖在真高点'的择价技能, 边际全部来自'均值回归宇宙里把赢家随持有期系统性减仓'的结构性 de-risk (任意阈值都行)**。
  - **统计稳健**: block bootstrap (cohort 重采样 1000次, 同集重建两臂) ΔSharpe 95%CI=**[+0.146, +1.328]** 不含0, P(Δ>0)=1.00; leave-one-out [+0.380,+1.088] 符号稳定; regime (R11) **momentum +0.76pp/月** (最大, n=9) / reversal +0.16 (n=9) / mixed -0.43 (n=2, 噪声)。
  - **gate_tp 四条件全 True** (d_sharpe_40>0 / 胜过 placebo bar (取 max=随机TP +0.769, TP +0.809 险胜) / bootstrap CI 不含0 / regime 不伤) → **status=TP真改进** (frozen gate, R01 未移门柱)。但诚实 nuance (SIGN-R03): 险胜随机 placebo (+0.04 在其 CI 内) = **结构性 de-risk 真实稳健 ≠ 择价 alpha**。
  - **用户直觉兑现**: (a) **保收益降波动** (Δ年化≈0 但 maxDD 砍 14.7pp) = 用户'拿波动换 Sharpe'的止盈直觉成立; (b) **动量月增益最大** (+0.76pp/月) 正打在 WF-001 揭示的 book 最弱 regime (动量月超额仅+0.79pp/胜率44%) → **直接缓解用户'动量月吃亏'痛点**。
  - **净结论 / playbook**: TP 结构性 de-risk 是出场层真实稳健 Sharpe 改进 (**ship 候选, 仍需 EX-003 最终 holdout**), 但**非择价技能** (随机阈值同效) → EX-003 r20 triple-barrier 上屏障应作'结构性减仓/路径管理'设计, 不必精调阈值 level。
  - 纪律: 0 违规 (verify exit0, 全非 skip task 有 verdict, WF-003 skip); 网格逐字冻结未上调 (R01, 用更严的随机 placebo 作 bar 仍过, 非挑软柿子); picks=de-lookahead 真 OOS (R13①); baseline-40d 去持有期混淆 (R13③); placebo 负控 (R13②); 受控 Δ 两臂相消 (R03); ST 源头排除 (R06); regime 分层 (R11); 描述性受控对照=合法完成 (R02); 大缓存 research/cache/wf002/ gitignored 无 features 写入 (R04); 复用 WF-001 picks 未碰生产 (R05)。
  - **产出**: `research/backtest/run_wf002.py` / `research/cache/wf002/{wf002_results.json, wf002_nav_*.parquet}` / `research/verdicts/WF-002.json`。
  - **下一步**: WF 主体 (WF-001 de-lookahead 地基 + WF-002 止盈负控复测) 全完结。WF-003 (skip, isGate, 预注册) = r20 triple-barrier 重标, 待用户 un-skip (现已有: ① WF-001 真实可交易基线 ② WF-002 出场层 TP 结构性 de-risk 真实但非择价 → triple-barrier 上屏障作结构减仓设计依据)。本轮非 skip task 全有 verdict, verify exit0。
- (iter2, 0614) **BT-002 built — V12.31 扣成本真实净 P&L + 基准/基金对照 + regime 分解**。新增 `research/backtest/run_bt002.py`。
  - **配置 (冻结 R01)**: V12.31 book = picks_v1231_daily **alloc 加权双轨** (非 BT-001 等权 sanity), 20d 再平衡, 真实成本 ON, 滑点敏感 0/0.05/0.10/0.20%/边。基准: hs300(000300.SH)/CSI1000(000852.SH) 指数净值 (Tushare 拉取缓存 `research/cache/bt002/index_benchmarks.parquet`) + 动量基准 (past-20d top50 等权过同引擎同基线成本)。regime 来自 `research/features/regime_timeline.parquet` 按月取众数。
  - **(A) 滑点敏感**: 基线 0.10%/边 → 年化 **+140.5%** 净Sharpe **+2.78** maxDD -22.0% 月换手 1.55 月胜率 85% 总成本 0.130。0→0.20% 全档: Sharpe +2.84→+2.72 (**衰减仅 0.12**), 年化 +145.0%→+136.1%。**成本/滑点不构成杀手** (换手仅 ~1.5/月, 净正不脆弱)。
  - **(B) 基准**: hs300 年化 +6.6% Sharpe +0.46; CSI1000 +18.5% Sharpe +0.81; 动量基准 +12.1% Sharpe +0.51 maxDD -25.3%。book 绝对量级远超 (但见 caveat)。
  - **(C) regime 分解 (R11, 关键诊断)**: 动量月(n=9) book 超额 hs300 **+4.44pp/月** 胜率 67% (**绝对未系统性跑输** hs300, 反驳"动量月绝对吃亏"的强版本); 但动量月是 book **相对最弱** regime (+4.44pp vs 反转 +6.23 / 混合 +9.62), 且动量月跑输纯动量基准 +6.29pp 方向 → **用户"基金赢我们输/动量月吃亏"在 book 层得部分印证** (相对最弱非绝对跑输) → 指向 RL-lite 风控 / 动量暴露管理。
  - **关键 caveat (诚实标注, R03)**: book 绝对年化 +140% 含 **r20 池模型 (r20_v16_long_nost 单一生产模型跨全测试期) 共模 lookahead + 集中持仓抬高** (同 BT-001 caveat), **非可交易声明**; 真实增量须待 BT-003 leave-one-out + BT-004 残差 gate 在受控对照下定。本 task 可信结论 = (a) 成本层无杀手 (b) regime 相对结构不受共模 lookahead 等比抬高影响。
  - 纪律: 0 硬违规 (verify exit1 仅 BT-003 未做=未完成非违规); 成本/滑点档逐字冻结未上调 (R01); 描述性净 P&L 非 ship gate (R02/R03); ST 源头排除 (R06, load_prices); regime 分层必带 (R11); 大缓存 research/cache/bt002/ gitignored, 无 features 写入 (R04); 生产 hash 一致 (R05, 未碰生产文件)。
  - **下一步 BT-003**: 对评分组件 (r20池/pump_up/pump_down/行业cap/双轨sizing) + V7c 6 铁律逐个 leave-one-out, 消融 arm 过引擎, 报 book 层 Δ(净Sharpe/年化/maxDD) 分 regime, 标正/负贡献。复用 run_bt002 的 nav_summary / monthly_returns / regime 分组 / 引擎封装。leave-one-out 的 Δ 是受控差分 (对照含同样 r20 池 lookahead), 可部分抵消绝对量级 caveat。
- (WFE iter, 0614) **WFE-001 built — label-embargo r20 真·真实 walk-forward (吃进 codex 第二次 review 第1条)**。新增 `research/backtest/{wfe001_gen_picks.py, run_wfe001.py}`。
  - **codex 抓到的残留泄漏**: WF-001 的 r20 月度重训按 `trade_date < train_end`(特征日) 切, 但 r20 标签是前向20日 (next_open→close_20d)。靠近截止日的样本, 其标签要到预测月才实现 → 模型没吃预测月**特征**, 但吃了预测月之后才可知的**标签** = label-availability lookahead。故 WF-001 的 1.84 仍偏高。
  - **修复 (R01 前置冻结)**: 唯一改动 = r20 训练截止从"特征日 < train_end"收紧到"**label 可知**": `trade_date <= cal[idx(P_start) − 21交易日]` (R20_HORIZON 20 + 1)。保证训练样本 T 的 r20 标签 (实现于 T+20 交易日) 在预测窗起始前一交易日已全部可知。24m lookback 起点/固定120树/引擎/成本/再平衡/双轨配置逐字同 WF-001 → embargo 仅削掉最近 ~21 交易日 (恰是 label 最易泄漏的近截止段) 训练样本。新缓存 `research/cache/wfe001/` (19 月独立重训 r20, checkpoint, ~19min)。
  - **结果**: embargo 后**真·真实可交易**: 年化 **+34.9%** 净Sharpe **+1.31** maxDD -15.7% 月换手 1.52 月胜率 **70%**。vs WF-001 无embargo (年化+65.9%/1.84): **额外缩水 ΔSharpe -0.53 / Δ年化 -31.1%**; vs BT-002 注水版 (2.78): 累计-1.47/-105.7pp。**仍跑赢全基准** (hs300 0.46 / CSI1000 0.81 / 动量 0.51)。playbook=**[embargo后大缩]** (额外缩水 -0.53 < -0.5)。
  - **关键诊断 (诚实 nuance, R03)**: **embargo 后 r20 IC 几乎不变** — 全期 +0.0979 vs WF-001 无embargo +0.1007 (Δ **-0.0028**), true-OOS 段甚至微升 +0.2049 vs +0.1844。→ **近截止 label 泄漏在信号 (IC) 层很小** (codex 担忧方向对但量级小); book Sharpe 的 0.53 额外缩水**主要是 19 月 picks 路径敏感性** (embargo 改了哪 ~20 只入选 → 实现路径不同) 而非大块泄漏被切。真·真实 V12.31 数 ≈ Sharpe **1.3 / 年化 +35%** (替代 WF-001 的 1.84 与 BT-002 的 2.78 注水)。
  - **regime (R11, embargo后 book vs hs300)**: momentum(n=9) 超额 **+1.96pp 胜率 67%** (注: embargo 后动量月超额反高于 WF-001 +0.79pp / 胜率 44%→67%, 但绝对量级整体下移) / reversal +1.78pp 胜率56% / mixed -0.54pp(n=2 噪声)。
  - **playbook 命中 [embargo后大缩]**: 真实优势比 WF-001 想象更小 → 实盘绝对预期再下修至 Sharpe~1.3 / 年化~+35%。但 **V12.31 仍真·可交易 (净正且跑赢全基准)**, **相对 Δ/受控对照结论 (BT-003 归因/EX/WF-002 止盈Δ) 因两臂相消不受影响** — 但 WFE-002 须在**这批 embargo OOS picks** 上重跑止盈 (SIGN-R13)。
  - 纪律: 0 违规 (verify exit1 仅 WFE-002 未做=未完成非违规); 唯一改动=训练截止加 embargo, 余逐字同 WF-001 (R01); ST 源头排除 (R06); regime 分层 (R11); 月度 r20_models/+picks_by_month/ checkpoint (R08); 前向 r20_fresh 只在 research/cache/wfe001/ gitignored 无 features 写入 (R04); 复用缓存 s5 模型未碰生产 (R05)。
  - **产出**: `research/backtest/{wfe001_gen_picks.py, run_wfe001.py}` / `research/cache/wfe001/{picks_oos_daily.parquet, r20_models/, r20_oos_diagnostics.csv, wfe001_book_oos.parquet, wfe001_monthly_regime.parquet, wfe001_results.json}` / `research/verdicts/WFE-001.json`。
  - **下一步 WFE-002**: 在**这批 WFE-001 embargo OOS picks** 上跑强化版止盈复测 — 主口径 close-based 保守成交; 臂 baseline-40d/TP-40d/placebo静态降暴露/placebo随机阈值(50seed); **强化 gate: TP ΔSharpe 须 > 随机阈值分布 p90** (非均值, codex 第2条); 扩展负控分解 edge (盈利条件控/缩短持有控/收益分位控, codex 第5条)。picks_oos_daily 即输入。
- (WFE iter, 0614) **WFE-002 结构也不成立 — 强化版止盈复测 (吃进 codex 第二次 review 第2+5条) → 止盈被否**。新增 `research/backtest/run_wfe002.py`。
  - **codex 第二次 review 抓到的 gate 太松 (WF-002 的 'TP真改进' 三处水分)**: ① WF-002 ΔSharpe +0.809 用的是**日内 high 乐观成交** (止盈触发即按当日最高价附近成交) + **WF-001 无embargo picks**; ② gate 用"TP > 随机阈值 placebo **均值** (+0.769)"判, TP +0.809 仅多 +0.04 且落在随机分布内 = 把"结构性 de-risk"过度叙述成"TP真改进"; ③ 未分解 edge 来源。
  - **三处全强化 (R01 冻结, = prd.tp_retest_v2)**: ① picks 换 **WFE-001 label-embargo 真·真实 OOS picks**; ② **主口径 = close-based 保守成交** (TP 触发后按当日 close 成交, 非日内 high), gate/分解全在 close-based 上做, 日内 high 仅留乐观参考; ③ 强化 gate = TP 须 > 随机阈值 placebo 分布 **p90** (非均值); ④ 扩展 4 控分解 edge (各控 ΔSharpe 越接近 TP_close 越是主源): (a) 盈利条件控 cond_profit (首3个上涨且盈利日卖1/3, 无阈值) (b) 缩短持有控 short_hold_random (50seed 随机卖出日+盈利门控) (c) 收益分位控 ret_quantile (阈值由各仓 within-window max-cumret 的 33/67/90 分位派生, 非固定+10/20/30%档) + 已有随机阈值控。
  - **结果 (20 cohort, 20241008~20260611)**: baseline_40d 净Sharpe **+1.26**; TP_40d_close (主口径) **+1.38** → **★close-based ΔSharpe(TP40c−base40) 仅 +0.117** (Δ年化 **-3.60pp** 反降, ΔmaxDD +1.99pp); 对比日内 high 乐观口径 +0.279 / EX-001 混淆口径 (TP−base20) +0.357 / **WF-002 报的 +0.809** → **逐口径收紧暴露假象: 0.809 → (embargo picks + high) 0.279 → (+close-based) 0.117**。
  - **强化 gate 全 False → status=结构也不成立**: 结构性减仓改进=**False** (close-based ΔSharpe+0.117>0 ✓ 但 **bootstrap CI [-0.242,+0.523] 含0** ✗ / LOO [-0.055,+0.271] 符号不稳); 择价 alpha=**False** (TP +0.117 < **随机阈值 p90 +0.279**); 静态降暴露 placebo ≈0 (-0.008, 增益非少持仓数学抬升). regime (R11): TP **动量月 -0.445pp / 反转月 -0.193pp 两态都伤** (mixed +0.149, n=2 噪声) — 与 WF-002 "动量月 +0.76pp" 反号 (那是 high+无embargo 假象)。
  - **edge 分解 (各控 ΔSharpe vs base40)**: 盈利条件 +0.247 / 缩短持有 +0.227 / **收益分位档位 +0.116 (最接近 TP_close +0.117)** / 随机阈值 +0.155 → **edge 主源 = 纯档位/收益分位 (任意分批减仓都行, 无择价技能)**, 印证 codex 预期。
  - **核心结论**: **WF-002 的 "分批止盈 +0.809 TP真改进" 在 (embargo 真picks + close-based 保守成交 + 随机p90 强 gate) 下塌为 +0.117/CI含0 = 弃止盈**。揭示链: EX-001 +1.358 → WF-002 (去混淆+真OOS picks) +0.809 → WFE-002 (去日内high乐观+去无embargo+强gate) +0.117 不显著。**止盈增益主要是日内 high 乐观成交假设 + WF-001 无embargo picks 路径的双重抬升, 非真实出场 alpha** (与 SIGN-R13 codex 警告"方向可能真量级不可信"一致, 此处量级塌到不显著)。这是反过拟合脚手架又一次诚实 REJECT (未 p-hack 未移门柱)。
  - 纪律: 0 违规 (verify exit0, 全非 skip task 有 verdict, WF-003 skip); 网格/gate 逐字冻结未上调 (R01, gate 用更严的 close-based+随机p90, 非挑软柿子); picks=embargo 真·真实 OOS (codex 第1条+R13①); 主口径 close-based 保守 (codex 第2条); 随机 p90 强 gate (codex 第2条); 4 控分解 (codex 第5条); 静态降暴露 placebo (R13②); baseline-40d 对齐去持有期混淆 (R13③); 受控 Δ 两臂入场逐位一致共模相消 (R03); block bootstrap CI + LOO + 分regime (R11); 描述性 REJECT=合法完成 (R02); ST 源头排除 (R06); 大缓存 research/cache/wfe002/ gitignored 无 features 写入 (R04); 复用 WFE-001 picks 未碰生产 (R05)。
  - **产出**: `research/backtest/run_wfe002.py` / `research/cache/wfe002/{wfe002_results.json, wfe002_nav_*.parquet}` / `research/verdicts/WFE-002.json`。
  - **下一步**: WFE 阶段 (WFE-001 embargo 真·真实地基 + WFE-002 止盈被否) 全完结。**embargo 干净基线已稳固** → WF-003 (skip, isGate, 预注册) = r20 triple-barrier 重标可在用户 un-skip 后开 (现已有: ① WFE-001 真·真实可交易基线 Sharpe~1.3 ② WFE-002 证止盈 (固定档位出场) 在干净口径下无 alpha → triple-barrier 上屏障**不应指望择价**, 仅作结构减仓/路径管理设计)。本轮非 skip task 全有 verdict, verify exit0。
- (FIN iter, 0614) **FIN-001 built — V12.31 真实基线 1.31 补误差条 (吃进 codex 第三次 review①)**。新增 `research/backtest/run_fin001.py`。
  - **FIN 阶段开张** (codex 第三次 review 收尾建议): ①给 1.31 补误差条 (本 task) ②冻结基线+前向 paper-trade 协议 (FIN-002) ③triple-barrier 独立挑战者 (FIN-003)。**不改策略** (R01 不动门柱), 在 WFE-001 embargo 真·真实 book (`wfe001_book_oos.parquet`, 点估计 净Sharpe 1.31/年化+35%/maxDD-15.7%) 上做不确定性 + 集中度画像。
  - **方法 (冻结 R01)**: cohort = **20 交易日再平衡/持有期** (= WFE-001 REBAL_EVERY, 自然依赖块, 保块内自相关), 21 个 cohort / 21 月 / 407 日。① per-cohort **block bootstrap 1000 次** (重采样 cohort 有放回拼接日收益→年化 Sharpe/年化收益 CI); ② leave-one-out 逐 cohort; ③ 月度 (LOO-by-month + top-k 占比 + HHI 集中度) / regime (日级按月度 regime 分组日 Sharpe + 收益贡献占比) 归因。固定种子 20260614 复现。
  - **① bootstrap**: **Sharpe 95% CI = [+0.11, +2.60]** 中位 +1.27 SE 0.62 **P(>0)=99% / P(>1)=66%**; 年化 95% CI [-1%, +80%]。→ **CI 很宽是核心结论**: 仅 ~21 cohort/21 月样本统计功率有限, 真实 Sharpe 几乎必正但"稳超1"信心仅 66% → 1.31 是点估计非可承诺下限。
  - **② LOO 逐 cohort**: Sharpe ∈ [+1.02, +1.53] **符号恒正 → 对删任一单期稳健** (不靠某孤立 cohort); 最撑 cohort #15 (20251229~20260127) 删后仍 +1.02。
  - **③ 集中度 = 中等 (非极端)**: top-3 赢月 (202512/202508/202601) 占正收益 **46%**, HHI **0.11 → 有效 ~9.3 个独立赢月** (14/21 正月); LOO-by-month 删最重的 202508 Sharpe 仅降到 +1.05 = **单月依赖不强**。regime 归因 **均衡**: momentum 组内日Sharpe +1.44 贡献 48% / reversal +1.47 贡献 50% / mixed +0.47 贡献 3% → 动量/反转近 50/50, 非靠单一 regime。
  - **核心结论 (诚实)**: 1.31 是最干净真实基线点估计, **对 LOO (删单 cohort/月) 稳健 + 收益跨月跨 regime 中等均衡** → 真实瓶颈是 **样本短/功率低 (CI 宽)** 而非过拟合到某几个月; 非最终可部署期望, 主要不确定性来自样本量 → 接 7 月血洗窗前向 paper-trade 复检 (FIN-002 冻结协议)。
  - 纪律: 0 违规 (verify exit1 仅 FIN-002/003 未做=未完成非违规); 不改策略不动门柱 (R01); 复用 WFE-001 embargo 真·真实 OOS book, ST 已源头排除 (R06); regime 分层必带 (R11); 描述性补误差条=合法完成 (R02); 中间指标≠ship (R03); 复用缓存未碰生产 (R05); 大缓存 research/cache/fin001/ gitignored 无 features 写入 (R04)。
  - **产出**: `research/backtest/run_fin001.py` / `research/cache/fin001/{fin001_results.json, fin001_loo_cohort.parquet, fin001_monthly.parquet}` / `research/verdicts/FIN-001.json`。
  - **下一步 FIN-002**: 产 `research/backtest/V12.31_BASELINE.md` 按 prd.freeze_protocol (冻结配置 V7c池/ratio/行业cap4/双轨/成本/close-based/embargo + 真实期望 Sharpe 1.31±CI[来自本 task]=[+0.11,+2.60] + 前向 paper-trade 追踪计划 + 部署前 checklist 含 r20 特征基本面 point-in-time 风险核查)。本 task 的 CI/集中度数已就绪供 FIN-002 引用。
- (FIN iter, 0614) **FIN-002 built — 冻结 V12.31 真实基线 + 前向 paper-trade 协议 + 部署前 checklist (吃进 codex 第三次 review②)**。新增 `research/backtest/{V12.31_BASELINE.md, run_fin002.py}`。
  - **① 基线冻结文档** (`V12.31_BASELINE.md`, FROZEN 契约): (a) **冻结配置**逐字钉死 (V7c dual-track 池 / 池内 ratio_s5 排序 / r20 label-embargo 月度 walk-forward (训练截止 ≤ P_start−21交易日) / pump s5 月度 WF / 双轨 70-20-10 / 行业 cap 4 / 每 20 交易日再平衡 / close-based 保守成交 / A股成本 round-trip≈0.30% / T+1 / 涨跌停 / ST 源头排除), 任何改动=新策略另测; (b) **真实期望** 净Sharpe **1.31** (年化+35%/maxDD-16%) ± bootstrap 95%CI **[+0.11,+2.60]** (来自 FIN-001), P(>0)=99%/P(>1)=66%/LOO稳健/regime均衡, **明确 1.31 是点估计非可承诺下限** (CI宽=样本短非过拟合); (c) **前向 paper-trade 协议** (每日 append-only 落 picks→满20/40d 按冻结口径算实现P&L→分regime 对照 1.31 (SIGN-R11, 重点盯动量月)→累计≥6 cohort 或过 2026-07 血洗窗后重跑 FIN-001 误差条复检; **红线: 前向数据不回流调参** SIGN-R01/R02); (d) **注水链** (BT-002 2.78 → WF-001 1.84 → WFE-001 +embargo **1.31**) 钉清, 凡引用旧绝对 Sharpe 须腰斩重读, 止盈已弃 (WFE-002)。
  - **② 部署前 checklist 可执行部分已执行** (`run_fin002.py` → `research/cache/fin002/fin002_checklist.json`): 程序化核对 r20 池排序模型 `r20_v16_long_nost` 的 **235 特征** (除 industry_id) 里基本面 PIT 风险字段 — **仅 6 个基本面** (229 全是价量/技术/资金流/形态/市场环境, 无披露滞后)。分级 (依据 tushare_enrich.py / fu001_build_pit.py provenance): **5/6 LOW** (`total_mv/pe/pe_ttm/pb` 来自 Tushare daily_basic 日频快照 as-of 正确; `winner_rate` 来自 cyq_perf 日频筹码 as-of 正确); **唯一 MEDIUM = `holder_pct`** (Tushare stk_holdernumber **季频股东户数**, factor_lab 合并源**未在代码层断言** ann_date(公告日) 对齐 → 若按 end_date(报告期末) 则有 ~1季度前视, 如 Q1 报告 end_date 0331 vs ann_date ~0430)。
  - **核心结论**: codex#3 残留盲点 (r20 含基本面 PIT 风险?) **已定位且影响有限** — r20 基本面占比极低 (6/235), 5/6 日频快照天然 PIT 安全; 唯 `holder_pct` 须**部署前**核 as-of 对齐或替换为 FU-001 已建的 ann_date PIT 面板 (`research/features/fundamental_pit.parquet`, 见 [[project_fundamental_growth_reject_0605]])。单字段且 BT-003 已证 r20 信号主导来自整池排序非单因子, 不动当前基线。其余部署前项 (embargo 口径重训确认 / 成本滑点实测 / ST 实时过滤) 文档化待落地。
  - 纪律: 0 违规 (verify exit1 仅 FIN-003 未做=未完成非违规); 不改策略不动门柱 (R01); 只读冻结基线 + checklist, 复用 WFE-001/FIN-001 缓存 + 生产 feature_meta 未碰生产文件 (R05); 描述性冻结+核查=合法完成 (R02); 中间指标≠ship (R03); ST 已源头排除 (R06); regime 分层写进前向协议 (R11); 大缓存 research/cache/fin002/ gitignored 无 features 写入 (R04)。
  - **产出**: `research/backtest/{V12.31_BASELINE.md, run_fin002.py}` / `research/cache/fin002/fin002_checklist.json` / `research/verdicts/FIN-002.json`。
  - **下一步 FIN-003** (isGate, 收尾): triple-barrier V12.32 独立挑战者 — 按 prd.triple_barrier 双屏障 (+15%/-8%/40d backstop, 参数预注册冻结) 重训 r20 替代 score, embargo walk-forward (P_start−41交易日, 因屏障 horizon 40d), score 排序替代 r20 重建 V12.31 book 过引擎 (close-based+成本+embargo) apples-to-apples vs 本基线 1.31, 按 gate_tb 断言 (净Sharpe>1.31 + maxDD不升 + bootstrap CI不含0 + 单月outlier + 分regime)。本基线 1.31 + 冻结口径已就绪供 FIN-003 apples-to-apples 对照。
- (FIN iter, 0614) **FIN-003 REJECT — triple-barrier V12.32 挑战者 (吃进 codex 第三次 review③, isGate 收尾)**。续跑上轮已断点的 `research/backtest/{fin003_gen_picks.py, run_fin003.py}` (gen 从 11/19 月 checkpoint 续到 19 月全完成, ~12min)。
  - **设计 (冻结 R01, 独立 V12.32 挑战者非基线补丁)**: 唯一改动 = r20 回归 label → **triple-barrier label** (entry=next_open; 持有窗 D+1..D+40; **上屏障 X=+15% / 下屏障 Y=-8% / 40d 时间 backstop**, 同日上下都触保守判先触下; 参数预注册冻结**不搜**)。embargo 收紧到 **P_start−41交易日** (屏障 horizon 40d → label 全可知, R04); TB-score 替代 build_dual 的 pred_r20 作池 filter, **池内仍按已 walk-forward 的 ratio_s5 排序** (= V12.31 口径); 24m lookback/固定120树/双轨70-20-10/行业cap4/20d再平衡/close-based 成本/T+1 逐字同 WFE-001。tb_label 全市场 4.0M 行 checkpoint, 月度模型+picks checkpoint (R08); ST 源头排除 (R06)。
  - **结果 (407 日受控对照 vs WFE-001 基线 1.31, 共模成本/集中相消)**: TB book 净Sharpe **+0.715** (年化+17.4%/maxDD-18.7%/月胜率60%) vs 基线 **+1.308** (FIN-001 口径一致) → **ΔSharpe -0.593**; maxDD **反升** (-15.7→-18.7%)。
  - **gate_tb 五条件全 False** (冻结阈值, R01 未移门柱): ① net_sharpe>baseline **False** (TB 低 0.59); ② maxDD_not_up **False** (升 3pp); ③ bootstrap ΔSharpe 95%CI **[-1.36,+0.28] 含0** P(Δ>0)仅 10% **False**; ④ 单月outlier: 剔最利TB月 202506 (Δ收益+9.76pp) 后 ΔSharpe **-0.845 仍负** **False**; ⑤ regime (R11) **动量 Δ-2.11pp / 反转 Δ-1.26pp 两态都伤** (仅 mixed n=2 噪声 +3.92pp) **False** → **status=REJECT**。
  - **诚实 nuance (SIGN-R03)**: TB-score 对 r20_fresh 截面 IC≈**+0.126** (描述性, 与 r20 信号同向中度相关) 但**book 层完全不兑现** — 即 triple-barrier label 重训出的排序信号在 OOS 持仓 P&L 上劣于 r20 回归 label。triple-barrier 的"先到为准结清"把 +15%/-8% 内的路径压平 (上屏障截断了 r20 能捕到的大涨右尾, 下屏障在均值回归宇宙里又常被瞬时下影线误触) → 选股层信息量 < 固定 20d 收益回归。
  - **核心结论 / playbook[REJECT]**: **用户 'triple-barrier / 不限天数限幅度+回撤' 直觉在选股 label 层不兑现** (出场层 WFE-002 已否分批止盈, 选股层 FIN-003 再否双屏障 label) → **均值回归选股已近最优, 文档化** (第 20 个被反过拟合脚手架诚实判的假设, 未 p-hack 未移门柱 R01/R02)。生产线 V12.31 (Sharpe~1.31) 冻结不动, 仍是干净真实基线。增量方向 (codex SIGN-R14): 换 universe (风格 tilt 大盘质量) / 换风险约束 / 接 7 月血洗窗前向 paper-trade, 非再在 r20/pump label + 小盘均值回归 universe 内换 label。
  - 纪律: 0 违规 (verify exit0, 全非 skip task 有 verdict); 双屏障参数+gate阈值逐字冻结**不搜不调** (R01); embargo 41交易日 label 可知 (R04); 受控对照两臂相消 (R03/R13); bootstrap CI+单月outlier+分regime 共判 (R11); 描述性 REJECT=合法完成 (R02); 中间IC≠ship (R03); ST 源头排除 (R06); 复用缓存 s5+生产 r20 仅对照未碰生产文件 (R05); 大缓存 research/cache/fin003/ gitignored 无 features 写入 (R04)。
  - **产出**: `research/backtest/{fin003_gen_picks.py, run_fin003.py}` / `research/cache/fin003/{tb_label.parquet, picks_oos_daily.parquet, r20_models/, picks_by_month/, tb_oos_diagnostics.csv, fin003_book_oos.parquet, fin003_results.json}` / `research/verdicts/FIN-003.json`。
  - **下一步**: **FIN 阶段全完结** (FIN-001 误差条 + FIN-002 冻结协议 built + FIN-003 triple-barrier REJECT)。prd 全 task passes=true, verify exit0 → 可输出 COMPLETE。北极星: V12.31 真实可交易基线 Sharpe~1.31±CI[+0.11,+2.60] 已冻结 + 前向 paper-trade 协议就绪 + triple-barrier 独立挑战者诚实否决 (选股层均值回归近最优)。
- (DEP iter, 0614) **DEP-001 clean确认 — 剔 holder_pct 重训 PIT-clean r20, 清 codex 唯一硬红线**。续跑已断点的 `research/backtest/{dep001_gen_picks.py, run_dep001.py}` (gen 从 15/19 月 checkpoint 续到 19 月全完成, ~9.8min)。
  - **背景**: FIN-002 部署前 checklist 定位到 r20 池排序模型唯一 MEDIUM PIT 风险 = `holder_pct` (Tushare stk_holdernumber 季频股东户数, `compute_holder_pct_at` 按 end_date<=target 选数未取 ann_date → Q1 报告 end_date 0331 在 ~0430 公告前即被用 = 真前视), 但重要度极低 (gain 排 118/236 占 0.005%)。用户决策 = 低影响则剔最干净 (零 ann_date 工程)。
  - **唯一改动 (R01 冻结)**: r20 训练特征集严格剔 `holder_pct` (247→246), 其余口径 (24m lookback / embargo P_start-21交易日 / 固定120树 / 双轨70-20-10 / 行业cap4 / 20d再平衡 / close-based 成本 / T+1 / ST 源头排除) 逐字同 WFE-001 → apples-to-apples, 唯一差异 = 去 holder_pct。月度 r20 模型 + picks checkpoint (R08)。
  - **结果 (407 日受控对照 vs WFE-001 基线 1.31)**: PIT-clean book 净Sharpe **+0.84** (年化+19.9%/maxDD-26.8%/月胜率55%) → 点估计 **ΔSharpe -0.464 / Δ年化 -15.0%**。但 **per-cohort 配对 block bootstrap (1000次, 21个20日cohort 配对重采样消共模) ΔSharpe 95%CI = [-1.246, +0.312] 含 0** (中位-0.477, P(Δ<0)=89%)。
  - **smoking gun (因果自检, R03)**: **剔 holder 后 r20 全期 IC +0.0952 vs 含 holder +0.0979 (变化 -0.0027 ≈ 0)** → holder_pct 在**信号 (IC) 层无实质贡献** (印证其 0.005% gain + BT-003 "r20 book α 主导来自整池排序非单因子")。点 Sharpe 跌 0.46 主要是 **picks-path 敏感性** (同 WFE-001 诊断: book Sharpe 方差由 19 月 picks 路径驱动非信号; 剔 1 个低重要度特征改了哪 ~20 只入选 → 实现路径不同), 非材料性 edge 损失 → 落在配对 bootstrap 噪声内 (CI 含 0)。
  - **regime (R11, PIT-clean book vs hs300)**: momentum(n=9) +0.97pp/胜率33% (最弱, 同历轮动量月痛点) / reversal +0.12pp/56% / mixed -0.04pp(n=2)。仍跑赢全基准 (hs300 0.46/CSI1000 0.81/动量 0.51)。
  - **裁决 [clean确认] (gate_dep, ci_contains_0)**: |ΔSharpe| 0.464 落在配对 bootstrap 噪声内 + IC 几乎不变 → **holder_pct 泄漏对 book 无实质影响, codex 唯一硬红线清除** (零 ann_date 工程, 隐患彻底消除), PIT-clean r20 = 干净可部署变体, 真实基线维持 ≈1.31 (FIN-001 CI[+0.11,+2.60] 内, 0.84 落其内)。命中 responsePlaybook['clean确认']。**反而说明早该剔**。
  - 纪律: 0 违规 (verify exit0, 全非 skip task 有 verdict); 唯一改动=剔 holder_pct, gate 阈值逐字冻结未上调 (R01); 配对 bootstrap 消共模只测 holder 净效应 (R03/R13); 因果自检无泄漏 (R04, 前向 r20_fresh 只在 research/cache/dep001/ gitignored 无 features 写入); ST 源头排除 (R06); regime 分层 (R11); 描述性=合法完成 (R02); 月度 checkpoint (R08); 复用缓存 s5 模型 + 生产 r20 仅对照, 未碰生产文件 (R05, verify 指纹一致)。(注: 裁决 prose 模板写"234特征"是占位, 实际 247→246, 不影响 metrics/gate。)
  - **产出**: `research/backtest/{dep001_gen_picks.py, run_dep001.py}` / `research/cache/dep001/{picks_oos_daily.parquet, r20_models/, picks_by_month/, r20_oos_diagnostics.csv, dep001_book_oos.parquet, dep001_monthly_regime.parquet, dep001_results.json}` / `research/verdicts/DEP-001.json`。
  - **下一步**: **DEP 阶段完结, prd 全 task passes=true, verify exit0 → COMPLETE**。北极星全交付: V12.31 真实可交易基线 Sharpe~1.31 冻结 + 前向 paper-trade 协议 + triple-barrier 诚实否决 + **PIT-clean r20 (剔 holder_pct) 确认 codex 唯一硬红线清除, deployable 变体就绪**。
- (DIAG iter, 0615) **DIAG-001 built [全局重排] — pick turnover/overlap: 0.46 摆动来自全局重排非少数票 (纯只读)**。新增 `research/backtest/run_diag001.py`。
  - **背景 (codex#5)**: WFE-001 (含 holder, 1.31) vs DEP-001 (剔 holder, 0.84) 仅差一个 0.005% 重要度因子却晃 0.46 book Sharpe。DEP-001 已判 clean确认 (配对 bootstrap CI 含 0, IC 几乎不变 → Sharpe 跌是 picks-path 敏感性)。本 task 把 "picks-path 敏感性" 拆开: 0.46 是 ① 少数边际 swing 票 (刀尖选股) 还是 ② 全局重排。
  - **方法 (纯只读 R05, 描述性 R02/R03)**: 复用 dep001/wfe001 已缓存 picks_oos_daily + book NAV + 价表, **不重训不改策略**。复刻引擎 `build_dual_alloc_targets` 口径取 20 个 20d 再平衡 cohort 两臂"实际入账权重", 逐 cohort 算 Jaccard/被替换票数/共同票权重 Spearman/差异权重, 并把 cohort 收益差 **Δ=Σ(w_dep−w_wfe)·ret** 归因到单票 (权重相同票贡献恒 0 → Δ 全部来自 swing 票), 看前 k 只 swing 票占 cohort 毛|Δ| 比例 + 池化集中度。阈值冻结 (R01): HIGH_OVERLAP=0.60 / CONCENTRATED=0.50 / TOPK=3。
  - **结果**: 平均持仓 18 只, **Jaccard 0.327** (每 cohort 被替换 ≈19 只 = 半数名单换掉), 但共同票权重 **Spearman 0.86** (留存票排名稳), 差异权重 57%。Δ 归因: 每 cohort 平均 24 只 swing 票, 前3只占 cohort 毛|Δ| **54%** (top1 28%) = within-cohort 中度集中; **但跨 cohort 是不同的票** — 唯一 swing 票 **431 只**, 池化前 10 个单票-cohort 事件仅占全期毛|Δ| **20%** (前20 占 32%), HHI 0.008 → **有效 ~120 个独立事件**。信号侧累计净Δ(DEP−WFE) -27.97pp ≈ book 实际累计净Δ -28.12pp (同号, 归因口径忠实)。
  - **裁决 [全局重排]** (responsePlaybook['DIAG-001'], 阈值冻结 R01): high_overlap=False (0.327<0.60) ∧ concentrated=True → **few_stocks_dominant=False**。机制: r20 作池 filter, 剔 1 个 0.005% 因子 (holder_pct) 把 r20 score 微扰 → 在池边界翻动约半数成员 → 整批入选重排 (非个别刀尖票)。**0.46 是模型对小扰动整体敏感** (picks-path 全局敏感), 非单票/容量风险型刀尖选股。**部署含义**: 须降集中/增 N (或 ensemble/多 seed) 提稳健性 (strategy 层, 另议)。within-cohort 前3占 54% 只是任意 20d 窗内少数票主导收益离散度的正常现象, 因每 cohort swing 票不同 (431 唯一码), 不构成对特定少数票的依赖。
  - **诚实 nuance**: 此结论与 DIAG-002 (cohort jackknife) 互补 — DIAG-001 判"票层"摆动是全局重排, DIAG-002 将判"cohort/月层" 0.84 vs 1.31 差是否少数月主导。两者共同把 0.46 摆动讲清 (codex#5 部署稳健性判断)。机制印证 DEP-001 clean确认 (IC 不变, Sharpe 跌纯 picks-path)。
  - 纪律: 0 违规 (verify exit1 仅 DIAG-002 未做=未完成非违规); 纯只读复用缓存未碰生产 (R05, 指纹一致); 阈值逐字冻结未上调 (R01); 描述性定位=合法完成 (R02); 中间指标≠ship (R03); ST 源头排除 (R06, load_prices); regime 已标注每 cohort (R11); 大缓存 research/cache/diag001/ gitignored 无 features 写入 (R04)。
  - **产出**: `research/backtest/run_diag001.py` / `research/cache/diag001/{diag001_cohort_turnover.parquet, diag001_results.json}` / `research/verdicts/DIAG-001.json`。
  - **下一步 DIAG-002**: 对 dep001(0.84) 与 wfe001(1.31) 各做 leave-one-cohort-out 重算 Sharpe, 看 0.84 vs 1.31 的差是否被少数 cohort/月主导 (短样本噪声 vs 系统性), 报删哪个 cohort 后两者最接近/最远。复用本 task 的 cohort 划分 + FIN-001 的 LOO 框架。完成后 DIAG-003 (skip, 部署前多 seed 重训) → prd 全非 skip task 有 verdict → COMPLETE。
- (DIAG iter, 0615) **DIAG-002 built [少数cohort主导] — cohort jackknife: 0.84 vs 1.31 gap 集中在少数 20d 路径 (纯只读)**。新增 `research/backtest/run_diag002.py`。
  - **背景 (codex#5)**: DIAG-001 已在"票层"判 0.46 摆动 = 全局重排 (Jaccard 0.327, 跨 cohort swing 票各异)。本 task 在"cohort/月层"补刀: 对两臂各 leave-one-cohort-out, 判 gap 是 ① 少数 cohort/月主导 (短样本噪声) 还是 ② 普遍 (系统性小降需多 seed)。
  - **方法 (纯只读 R05, 描述性 R02/R03)**: 复用 wfe001/dep001 已缓存 OOS book NAV (= FIN-001 口径), 不重训不改策略。两臂 book 日期逐位对齐 (407日/21个20交易日 cohort) → 配对 LOO 干净。删每 cohort 后各臂重算全期 Sharpe + gap; gap 归因到单 cohort (前1/前3占比 + HHI 有效数); 月层 LOO 补充。阈值冻结 (R01): FEW_TOP1=0.50 / FEW_TOP3=0.80。
  - **结果**: 点估计 WFE(含holder) Sharpe **+1.308** / DEP(剔holder) **+0.844** / **gap +0.464** (= 0.46 摆动, 与 DIAG-001/DEP-001 一致)。**配对 cohort LOO**: 删每 cohort 后 gap 范围 **[+0.270, +0.631] 全程恒正** (删任一 cohort 含holder 始终 ≥ 剔holder); 删后两臂最接近 = cohort **#14 (20251201~20251226, reversal)** gap→**+0.270**, 最远 = cohort #6 (20250407~20250507) gap→+0.631。**gap 归因**: 前1 cohort #14 占 gap **42%**, 前3 占 **104%** (另18个近相消), HHI 0.182 → **有效 ~5.5 个 cohort** 拉开两臂。**月层 LOO**: 最接近月 202512 (占 gap 51%) / 最远月 202603, 前3月占 124%。
  - **裁决 [少数cohort主导]** (few_top3=1.04≥0.80 命中, R01 阈值冻结): gap 集中在 ~3-5 个 20d 实现路径 → **短样本噪声 (非真实信息损失)**, 1.31 与 0.84 都是宽带抽样的两个实现, gap 由少数 20d 路径驱动 (印证 DEP-001 IC 几乎不变 + 配对 bootstrap CI 含 0)。**诚实 nuance (R03)**: 单 cohort 未独占 (前1仅 42%, 删它 gap 仍 +0.27), 且 LOO gap **恒正** → 不是纯掷硬币噪声, 是"集中于少数路径的弱系统偏移", 量级落 DEP-001 配对噪声带内 → **不改 clean确认**。命中 responsePlaybook['DIAG-002']['少数cohort主导']。
  - **DIAG 阶段两诊断合论 (codex#5 部署稳健性)**: DIAG-001 (票层) = 全局重排 (剔 1 个 0.005% 因子翻动约半数入选, 非刀尖单票) + DIAG-002 (cohort 层) = 少数 20d 路径主导 (短样本噪声, 含/剔 holder 始终同号, gap 落配对噪声带内)。共同结论: **0.46 book Sharpe 摆动 = picks-path 全局敏感 × 短样本 (21 cohort) 抽样方差, 非真实信息损失也非刀尖单票/容量风险** → 1.31/0.84 都是宽带抽样实现, codex#5 "部署预期'正但宽带~0.84非1.31'" 得机制支撑。部署稳健性建议 (strategy 层, 另议): 降集中/增 N/ensemble 多 seed。真实差是否存在留 DIAG-003 (skip, 部署前多 seed 重训) 最终定。
  - 纪律: 0 违规 (verify exit0, 全非 skip task 有 verdict); 纯只读复用缓存未碰生产 (R05, 指纹一致); 阈值逐字冻结未上调 (R01); 描述性定位=合法完成 (R02); 中间指标≠ship (R03); ST 已源头排除 (R06); regime 每 cohort/月标注 (R11); 大缓存 research/cache/diag002/ gitignored 无 features 写入 (R04)。
  - **产出**: `research/backtest/run_diag002.py` / `research/cache/diag002/{diag002_loo_cohort.parquet, diag002_loo_month.parquet, diag002_results.json}` / `research/verdicts/DIAG-002.json`。
  - **下一步**: **DIAG 阶段非 skip task 全完结** (DIAG-001 built [全局重排] + DIAG-002 built [少数cohort主导]); DIAG-003 (skip, 部署前多 seed 重训) 待部署前 un-skip → **prd 全非 skip task passes=true, verify exit0 → COMPLETE**。北极星: 0.46 摆动来源讲清 (全局重排 × 短样本噪声), V12.31 部署预期定为"正但宽带~0.84-1.31", 生产线只读冻结。
- (RETRO iter, 0615) **RETRO-001 built — 0501+ V12.31 picks 真实表现回顾 (vs 全市场/概念均值, 纯只读描述性)**。新增 `research/backtest/run_retro001.py`。
  - **RETRO 阶段开张** (北极星换轨): 回顾 2025-05-01 起 V12.31 每日评分股票真实涨没涨/相对市场+概念跑赢没, 为 RETRO-002 漏赢家画像 + RETRO-003 对比学习可行性 gate 打地基。
  - **picks_source (prd 冻结)**: 复用 BT-001 `picks_v1231_daily.parquet` (V7c dual-track 池, 池内 ratio_s5 排序 = V12.31 baseline 臂) 截 trade_date>=20250501 → **4434 (date,stock) 观测 / 242 交易日 / 1317 股 / [20250506~20260430]**, 全部 full-20d (数据到 20260611, 0 partial)。收益 close[t]->close[t+h] (h=5/10/20); max_gain_20=max(high[t+1..t+20])/close[t]-1 (=winner_def 口径)。
  - **model-vintage caveat (诚实标注)**: 生产 r20 (r20_v16_long_nost) 训练窗<20250930 → 0501+ 早 5 月 (202505~202509) 对 r20 有 in-sample 暴露, by_vintage 拆开报。
  - **结果 (full-20d)**: 总体 20d 均值 **+6.90%** (中位 +4.23%, 胜率 65%), max_gain20 +16.56%; 相对全市场超额 **+4.66%** (胜率 57%), 相对所属概念超额 **+3.77%** (胜率 54%) → **picks 真实涨且双双跑赢市场与概念基准**。
  - **★vintage 分裂 = 关键发现**: in_sample_r20(202505~09) ret20 **+9.87%**/win **77%**/exc_mkt +6.45%/exc_con +5.29% **vs** true_oos(202510+) ret20 **+4.09%**/win **54%**/exc_mkt +2.97%/exc_con +2.34% → 真实 OOS 段腰斩 (与 WF-001/WFE-001 de-lookahead 一致, r20 记忆抬高早期)。**且 true_oos 的 win_vs_concept 仅 47%(<50%)** 但 mean exc_con 仍 +2.34% 为正 → **真实 OOS 下多数 (53%) picks 在概念内并不跑赢, 概念超额由右尾少数赢家拉起** = hindsight 陷阱前奏, 直指 RETRO-002 (漏赢家是 ex-ante 可分还是纯 realized) / RETRO-003 (扣桶 beta+扣因子后还有没有 ex-ante 残差)。
  - **regime (R11)**: by_regime 已落 verdict (momentum/reversal/mixed 各报 ret20/exc_mkt/exc_con)。
  - 纪律: 0 违规 (verify exit1 仅 RETRO-002/003 未做=未完成非违规, 0 硬违规); 纯只读复用缓存 picks 未碰生产 (R05, 指纹一致); 描述性回顾=合法完成 (R02); 中间指标≠ship (R03); ST 源头排除 (R06, load_prices); regime+vintage 分层必带 (R11); 大缓存 research/cache/retro001/ gitignored 无 features 写入 (R04)。
  - **产出**: `research/backtest/run_retro001.py` / `research/cache/retro001/{retro001_picks_perf.parquet, retro001_by_month.parquet, retro001_results.json}` / `research/verdicts/RETRO-001.json`。
  - **下一步 RETRO-002**: 按 winner_def 映射概念、标 winner (概念内20d max_gain TOP20%)、定位 missed_winner (winner 且未入 V12.31 池), 对比 picks vs missed_winners 的 ex-ante 特征分布 (past_r5/动量/pyr_velocity/pump/ratio/市值/换手/ADX/RSI) + realized max_gain 差异, **严格分 ex-ante 可观测差异 vs 纯 realized 收益差异 (hindsight)**。RETRO-001 已揭示真实 OOS 概念超额是右尾驱动 → RETRO-002 须查这右尾赢家与 picks 的 ex-ante 差异在哪。
- (iter6, 0615) **RETRO-002 built — 概念内漏赢家画像: 差异主要在动量 (ex-ante 可观测)**。新增 `research/backtest/run_retro002.py` (纯只读, 0.4min)。
  - **winner_def 冻结 (R01)**: 主桶=merged 概念库映射每股概念; 每 (pick_date, concept) 全成员按后续 20d max_gain TOP20%=winner; missed_winner=winner 且当日未入 V12.31 池; 仅含 ≥1 pick 的概念 (pick_concepts=1886) 且仅 full-20d 的日 (242/242 全可标, 末端不满 20d 的日剔除)。
  - **对比宇宙** 1.25M (date,stock): winner 513503 / **missed_winner 511169** / picks 4434; **picks 自身概念 winner 命中率仅 52.6%** (一半 picks 不是其概念里涨最猛的 20%, 印证 RETRO-001 真实 OOS 概念超额由右尾少数赢家拉起)。
  - **ex-ante 特征源 (全因果, D 收盘可知)**: 模型分 (BT-003 scored_daily 全宇宙) pred_r20/ratio_s5/pump_down_s5/pyr_velocity_20_60; 价量派生 (本脚本 backward 算) past_r5/mom_20/mom_60/rsi_14/turn_amt_20; adx (BT-004 px_feats 全宇宙); total_mv (factor_lab, 覆盖到 ~20260126 部分)。realized(hindsight) max_gain_20/ret_20 严格隔离。
  - **★关键发现 = 漏赢家差异主要在动量 (ex-ante 可观测!)**: 非模型 ex-ante 特征按 |SMD| 排前三 = **mom_20 (SMD -0.366)** / mom_60 (-0.283) / adx (-0.277)。**missed_winner 近期动量系统性更高**: mom_20 picks **-0.45%** vs missed **+4.56%**; mom_60 +4.5% vs +11.6%; past_r5/rsi/adx picks 均更低 → **v3c 均值回归策略 (买 past_r5<0) 系统性避开了概念里后来领涨的动量股** (呼应 [[project_v3c_momentum_regime_mismatch_0603]] 实盘审计: 动量月被血洗)。
  - **模型分差异 = 套套逻辑**: ratio_s5 picks +2.71 vs missed +1.95 (SMD +0.49) / pump_down picks 更低 (SMD -0.55) / pyr_velocity picks 更低 (SMD -0.62) — picks 正是被这些分选出的, 不算独立信息 (诚实标注, 真信号在非模型 ex-ante 残差)。
  - **realized (hindsight, 不可交易)**: missed_winner 20d max_gain **+23.4%** vs picks +16.6% (gap -6.8pp), ret_20 +10.6% vs +6.9% → 这是**定义性 hindsight** (winner 按 max_gain 选), 不可交易 (SLV-002 陷阱), 只作收益差分不混 ex-ante。
  - **ex-ante vs hindsight 分解 (核心交付)**: 差异既有 ex-ante 可观测部分 (动量, 可能可学) 又有纯 realized 部分 (max_gain, hindsight)。**真问题** = 这个动量 ex-ante 残差扣桶 beta (概念 demean) + 扣现有因子 (pump ratio+r20) 后是否还独立存在 → **留 RETRO-003 contrastive_gate 判** (NEW=残差|IC|>=0.01&|t|>=2 才是对比学习候选)。
  - 纪律: 0 违规 (verify exit1 仅 RETRO-003 未做=未完成非违规, 0 硬违规/0 泄漏); winner_def+特征集逐字冻结未上调 (R01); 描述性=合法完成 (R02); 中间 SMD≠ship (R03); ST 源头排除 (R06, load_prices); regime+vintage 分层必带 (R11); 大缓存 research/cache/retro002/ gitignored 无 features 写入 (R04); 复用缓存评分未碰生产 (R05, 指纹一致)。
  - **产出**: `research/backtest/run_retro002.py` / `research/cache/retro002/{retro002_universe.parquet, retro002_results.json}` / `research/verdicts/RETRO-002.json`。
  - **下一步 RETRO-003 (gate)**: 概念主桶 within-bucket 去均值(扣 beta)后 ex-ante 特征 (尤其动量 mom_20/mom_60) 对后续 20d 收益残差 IC, 再扣 [pump ratio+r20] 残差 IC+t; 加 PE+市值 peer 桶交叉验证 (差异概念特异 or 泛截面); 按 contrastive_gate 判 NEW/REJECT_beta_hindsight/真小。RETRO-002 已锁定候选信号=动量残差, RETRO-003 验它扣 beta+扣因子后是否独立。
- (iter7, 0615) **RETRO-003 对比学习候选 (脆弱) — 19 否后首个名义过闸的正向, 但 suppressor/regime条件/子样本失效三重脆弱**。新增 `research/backtest/run_retro003.py` (8.2min)。
  - **方法 (冻结 R01)**: 主候选=桶内动量 composite z(mom_20)+z(mom_60) (来自 RETRO-002 top 差异, 预注册); 目标 y=**ret_20 (可交易前向 close 收益, 非 winner 定义用的 max_gain hindsight)**; 控制 C=[ratio_s5, pred_r20] (= 冻结 'pump ratio + r20')。桶 = 概念 member-level (一股属多概念则多行, 34.5M 行) 或 PE×市值 25 桶。Stage A 桶内 (date,bucket) 去均值扣 beta (桶<5 剔), Stage B 逐日 f_d~C_d OLS 无截距残差 f_res, 残差 IC=逐日 Spearman(f_res,y_d), **Newey-West lag=20 修 20d 重叠** headline + 非重叠 robustness。**单遍 streaming per-date** (避免 34.5M 全表 transform OOM, 第一版全表 transform 已 OOM 改写)。
  - **概念桶 full universe (gate 主指标)**: mom_comp 残差 IC **+0.0286 NW_t +2.68** → **名义过冻结 gate** (|IC|>=0.01 & |t|>=2); mom_20 +0.0326 t+2.19 亦过; mom_60/rsi/past_r5/turn/adx 均不过。**playbook 命中"对比学习候选" (19+ 否后首个正向, 如 northStar/playbook 预期)**。
  - **★三处脆弱性 (诚实记录, walk-forward 前必认清)**:
    ① **符号翻转 (suppressor)**: 桶内 raw 动量 IC=**-0.0414 (负=均值回归, 高动量股在概念内反而回调)**, 仅在扣掉 ratio_s5/pred_r20 后残差才转 +0.0286 → 信号本质是"动量中正交于现有因子的分量"而非动量本身 (扣因子是 gate 设计的本意=测增量, 但 sign-flip 须 walk-forward 验稳定)。
    ② **apples-to-apples 子样本失效**: PE/市值可得子样本 (剔 202602-04 无市值尾部, 181 日) 概念桶 mom_comp 跌到 **+0.0201 t+1.49 不过 t**; 非重叠 t (每20日取样, n=13) **+1.15 不过** → full-universe 过闸有赖交叉验证未覆盖的最后 3 月, 只有满 242 重叠日的 NW-t 过 = 边际过闸。
    ③ **动量 regime 归零 (R11 最关键)**: 残差 IC 在 **momentum 月 = +0.0018 t+0.11 (≈零)**, 正值全来自 reversal (+0.0439 t+3.02)/mixed (+0.0376 t+3.36) → **恰恰不在 V12.31 实盘被血洗的动量态生效** ([[project_v3c_momentum_regime_mismatch_0603]] 实盘审计: 动量月 picks-10% 而优秀基金+11.4%)。即便信号真也不修已知痛点。
  - **概念特异性 (bucket_crosscheck)**: 同 PE/市值可得子样本上 概念桶 +0.0201 vs PE×市值桶 +0.0135 (gap 0.0066, 勉强过 concept_specific 阈 0.005 判"概念特异") **但两桶都不过 t** → 主要是泛截面弱动量残差, 概念结构增量很小。
  - **净结论**: 对比学习/within-bucket LTR 是 19 否后**唯一名义过闸的正向候选**, 但属脆弱、regime 条件 (动量月归零)、suppressor 性质的弱信号; 按 R03 残差 IC≠ship, NEW 仅"值得独立 walk-forward"。**落地前置条件**: 须先过独立 walk-forward + 证明能修动量态 (而非只在 reversal/mixed 生效)。在此之前生产线 V12.31 全程冻结 (R05, 指纹一致)。
  - 纪律: 0 违规 (verify exit0, 全 task 有 verdict); gate/候选/控制/桶定义逐字冻结未在结果上调 (R01); 残差 IC≠ship 即便过阈也只"值得 walk-forward" (R03); 描述性/候选=合法完成 (R02); ST 源头排除 (R06, universe 已排); regime 分层必带且为否定关键 (R11); R12 异类增益须存活朴素消融 (扣桶 beta + 扣现有因子 = 朴素对照, momentum 残差勉强存活但脆弱); 大缓存 research/cache/retro003/ gitignored 无 features 写入 (R04); ex-ante vs hindsight 严格分 (TARGET=可交易 ret_20 非 max_gain)。
  - **产出**: `research/backtest/run_retro003.py` / `research/cache/retro003/{mvpe_panel.parquet, retro003_results.json, run.log}` / `research/verdicts/RETRO-003.json`。
  - **下一步**: **RETRO 阶段全 task 完结** (RETRO-001/002 built + RETRO-003 对比学习候选)。北极星 (0501+ 真实表现 + 概念内漏赢家对比学习可行性) 全部交付。若要推进对比学习候选: 独立设计 within-bucket LTR 的 walk-forward (月度重训 + embargo + 分 regime gate, 重点验动量态能否 >0), 含负控 (placebo / 随机分桶) 防 suppressor 假象 (SIGN-R13)。
- (CL iter, 0615) **CL-001 死于消融 — 全因子消融把 RETRO-003 首个正向 +0.0286 证伪 (杀信号本身)**。新增 `research/backtest/run_cl001.py` (block-checkpoint, 2.3min)。
  - **CL 阶段开张**: RETRO-003 的 within-concept 动量 re-rank 是 19+ 否后**首个名义过闸正向**, 但 prd northStar 把它当**过拟合最爱藏处**, 设计成主动证伪。CL-001 = 把消融从 RETRO-003 的 **2 因子** [ratio_s5,pred_r20] 扩到 prd cl001_ablation 冻结的**全因子集**, 测信号扣全集后是否存活。
  - **方法 (冻结 R01, = RETRO-003 同候选/同桶/同 streaming, 仅扩控制集)**: 信号 = mom_comp z(mom_20)+z(mom_60) 概念内 (date,concept) demean (扣桶 beta, 桶<5 剔); 控制全集 [past_r5(mkt多窗动量, 与信号20/60不同窗非循环)+pyr_velocity_20_60+log_mv(size)+pe_ttm(value)+adx+rsi_14+ma20_slope(形态z)+ratio_s5(pump ratio)+pred_r20(r20)]; 逐日截面 OLS(无截距,已demean)正交化残差→Spearman(f_res,ret_20); NW_t(lag20) headline + 非重叠(每20日)robustness; 月度 by_month(walk-forward视图)+分regime(R11)。两口径: **full**(9因子,含size/value, PE可得子样本 n=181) headline gate + **core**(7因子去size/value, full universe 242日)功率补充。
  - **结果 (headline full)**: 残差 IC **+0.0198 NW_t +1.77 未过冻结 gate** (|IC|≥0.01&|t|≥2.0); **非重叠 t +0.68** (core 同 ~+0.69) → 去 20d 重叠膨胀后**无显著性**; core NW_t +2.05 仅擦边但非重叠 t 同样不过 → **整体不稳健**。full IC 缩到 RETRO-003 +0.0286 的 **69%** (core 66%) 且显著性蒸发 → +0.0286 约 1/3 被更全因子 (多窗动量 past_r5 + 形态 ma20_slope + size/value) 吃掉, 余下整体不过闸。
  - **★诚实 nuance (与 RETRO-003 反转, 未据此移门柱 R01)**: 全消融后残差**不是均匀归零, 而集中到动量 regime** — momentum IC **+0.0393 t+5.41**(强) / reversal +0.0111 t+0.70 / mixed +0.0151 t+0.98 (后两者不显著) → **仅动量 regime 显著** (significant_regimes=['momentum'])。这**揭示 RETRO-003 "动量月≈0" 是其 2 因子控制被 regime 混淆的 artifact** (那两个 mean-reversion 控制 ratio_s5/pred_r20 在反转月吸收了信号, 反在动量月留下残差)。桶内 raw 动量 IC -0.055 (负=均值回归, suppressor 仍在); by_month 不稳 (202505/06 负 → 202508~202601 正 → 202602 又负)。
  - **裁决 [死于消融]** (responsePlaybook['CL-001']['死于消融'], gate 阈值冻结 R01): 按冻结 gate (headline 全消融 |IC|&|t|) ic_t_pass=False → **未通过存活闸 → 不进 CL-002** (REJECT 文档化, R02/R03)。within-concept 相对动量 re-rank **不是干净的 cross-sectional alpha 候选**。但残差的**动量 regime 集中性 ≠ 干净归零**: 性质上属**动量 regime 条件/overlay** (恰在 V12.31 实盘被血洗的动量态, [[project_v3c_momentum_regime_mismatch_0603]] + SIGN-R11), 若后续追须按 **CL-003 当动量 overlay 验** (RG-002 已证 overlay tricky), 而非当稳健 alpha 上 CL-002 book gate。
  - 纪律: 0 违规 (verify exit1 仅 CL-002/003 未做=未完成非违规, 0 硬违规/0 泄漏); 候选/控制集/gate 阈值逐字冻结未在结果上调 (R01, 仅按 prd cl001_ablation 扩控制集, regime nuance 未用于反推门柱); 描述性残差 IC≠ship, 死于消融=合法完成 (R02/R03); ST 源头排除 (R06, universe 已排); regime 分层必带且为裁决关键 (R11); R12 全因子集=最严朴素对照; 大缓存 research/cache/cl001/ gitignored 无 features 写入 (R04); 复用 RETRO-002 universe + 缓存评分未碰生产 (R05, 指纹一致); block-level checkpoint (R08, 8 块各落 blocks/*.json)。
  - **产出**: `research/backtest/run_cl001.py` / `research/cache/cl001/{blocks/*.json, cl001_results.json, run.log}` / `research/verdicts/CL-001.json`。
  - **下一步 CL-002 (isGate)**: CL-001 死于消融 (未存活) → 按 CL-002 acceptanceCriteria "若 CL-001 未存活则 skip 并文档化", 下一迭代写 CL-002 skip/REJECT 裁决 (信号未过存活闸, book apples-to-apples gate 无意义)。CL-003 可在 CL-001 揭示的"残差=动量 regime overlay"nuance 上做 overlay 判定/placebo (但 book gate 前提已随 CL-002 skip 落空)。生产线 V12.31 全程冻结 (R05)。
- (CL iter, 0615) **CL-002 skipped (前提不满足) — gate 链在 CL-001 处诚实终止, 不下探 book p-hack**。无新增脚本 (纯文档化裁决)。
  - **裁决依据**: CL-002 是 isGate, prd.cl002_book_gate 与 acceptanceCriteria 明确"**仅 CL-001 存活才跑**, 否则 skip 并文档化"。CL-001 已裁决**死于消融** (全因子消融残差 IC +0.0198 NW_t +1.77 未过冻结存活闸, 非重叠 t +0.68 无显著性, IC 缩到 RETRO-003 +0.0286 的 69%) → **信号未通过存活闸, book gate 前提不成立**。
  - **为何不硬跑 (R01/R02)**: 不在已被证伪的死信号上实例化 book 凑 PASS——blend=0.5*rank(ratio_s5)+0.5*rank(within_concept_mom)/λ=0.5/引擎(embargo+close-based+成本+双轨+cap+20d再平衡)/基线(DEP-001 PIT-clean 0.84 或 WFE-001 真实 1.31)/5 条 gate(ΔSharpe>0+bootstrap CI不含0+maxDD不升+单月outlier+分regime反转月不伤) 均为**冻结配置, 仅记录不实例化**。
  - **纪律意义 (反过拟合脚手架的设计目的兑现)**: 首个名义过闸正向 (RETRO-003 +0.0286 t+2.68, 19+ 否后唯一) 经 CL-001 全因子消融 + 非重叠去 20d 重叠膨胀**诚实证伪**后, **gate 链即在 CL-001 处终止, 不下探到 book 层 p-hack**。这正是 SIGN-R01/R02/R03 + responsePlaybook 预注册的行为: 死于消融 → REJECT 文档化, 不产生任何 book P&L 数字。
  - **残差性质**: CL-001 揭示残差非干净归零而集中动量 regime (momentum IC +0.0393 t+5.41 强, reversal/mixed 不显著) → 属**动量 regime overlay**, book apples-to-apples gate 不适用; 该 overlay 判定/placebo 对照留 **CL-003** (下一迭代), 但 book gate 前提既因 CL-001 死而落空, CL-002 本身无 book 数字。
  - 纪律: 0 违规 (verify exit1 仅 CL-003 未做=未完成非违规, 0 硬违规/0 泄漏); gate 阈值/blend/λ 冻结未动, 不为凑 PASS 在死信号上硬跑 book (R01); documented skip = 合法完成 (R02); CL-001 残差 IC 是描述性中间指标不据此 ship 也不下探 book (R03); 生产线 V12.31 只读冻结指纹一致 (R05); CL-001 已按 regime 分层揭示 overlay 性质 (R11)。
  - **产出**: `research/verdicts/CL-002.json` (status=skipped, 含冻结但未实例化的 book gate 设计 + CL-001 kill evidence + 4 条 caveat)。
  - **下一步 CL-003**: CL 阶段仅剩 CL-003 (非 gate)。在 CL-001 揭示的"残差=动量 regime overlay"nuance 上做最终定性: 分解 (book gate 前提已落空故无 book ΔSharpe, 改为在 CL-001 残差层面) overlay 判定 + PE×市值桶交叉 (概念特异确认, RETRO-003 已得 gap 0.0066 勉强过) + hindsight placebo (随机 within-concept re-rank 对照), 综合分类 stable_alpha / regime_overlay / artifact 给最终定性。CL-001/RETRO-003 已提供绝大部分素材 (动量 regime 集中性 + 概念特异 gap + suppressor)。生产线 V12.31 全程冻结 (R05)。
- (CL iter, 0615) **CL-003 built [regime_overlay] — CL 阶段收尾定性, RETRO-003 首个正向终判=动量择时 overlay 不落地**。新增 `research/backtest/run_cl003.py` (复用 CL-001 面板/残差机器, block-checkpoint, 12.9min 主要耗在 K=200 placebo)。
  - **设计 (R01 冻结, = CL-001 同候选/控制/桶, 仅换桶维度交叉 + 加随机 null)**: CL-002 因 CL-001 死于消融 skip → **无 book ΔSharpe**, 故 prd 要求的"分解 book ΔSharpe by regime"改在**残差 IC 层** (R02 诚实降——gate 链在 CL-001 已终止, 不为凑指标硬造 book)。三路分解: ① **regime** 复用 CL-001 缓存 full 块 (逐位同口径); ② **PE×市值桶交叉** 把桶从概念换成 5×5 PE×市值分位桶, 同全因子消融, 比残差 IC (概念特异确认); ③ **hindsight placebo** 每日把 mom_comp 在概念桶内**随机置换** (保桶内均值=0, 同 finite mask/demean, 仅信号打乱), K=200 构 null, 真信号须落 null 右尾。
  - **结果三路一致 → regime_overlay**:
    ① **regime 分解**: 仅动量 regime 显著 (momentum IC **+0.0393 NW_t +5.41** 强; reversal +0.0111 t+0.70 / mixed +0.0151 t+0.98 **不显著**) = overlay 签名。
    ② **概念特异性证伪**: 概念桶 full 残差 IC +0.0198 ≈ **PE×市值分位桶 full 残差 IC +0.0205** (gap **−0.0007**, PE/mv NW_t +1.63 同不过闸) → 残差**非概念特异** (概念桶甚至略低于 PE/市值桶), 是**泛截面弱动量残差被 PE/市值横截面结构吸收**, 不是"概念内"alpha。**修正 RETRO-003** 的 "gap 0.0066 勉强判概念特异" (那是 2 因子弱消融; 扩全因子后概念增量蒸发为负)。
    ③ **placebo**: real IC +0.0198 vs 随机 within-concept re-rank **null mean +0.0037 sd 0.0002 → z+67.87 p<0.001 落 null 100% 分位** → 残差是**真信号非方法 artifact** (null 非 0=残差化几何有微正偏, real 仍远在右尾)。
  - **综合定性 [regime_overlay]** (非 artifact 因胜 placebo; 非 stable_alpha 因非概念特异+仅动量 regime): 残差是**真但微弱**的动量残差, **恰且仅在动量 regime 显著** (V12.31 实盘被血洗的动量态, [[project_v3c_momentum_regime_mismatch_0603]]+SIGN-R11), 非概念特异/非全 regime 兑现 → **动量择时 overlay** (RG-002 已证 tricky, 需滞后 regime detection 才能用), 非稳健 cross-section alpha。
  - **CL 阶段终判**: within-concept 相对动量 re-rank **不落地** — RETRO-003 +0.0286 名义过闸 → 1/3 被全因子吃 (CL-001 死于消融) → 余下是动量 regime overlay (CL-003)。**19+ 否后的首个名义过闸正向, 经反过拟合脚手架三关 (CL-001 全因子消融 + CL-003 概念特异证伪 + placebo) 后, 未能成为可交易 alpha**。漏概念动量赢家是 hindsight + 我们均值回归身份 (买 past_r5<0) 的必然代价 (RETRO-002), 不引入动量择时不可交易修。
  - 纪律: 0 违规 (verify exit0, 全 task 有 verdict); 候选/控制集/桶定义/gate 阈值冻结未在结果上调 (R01, 仅换桶维度交叉 + 加随机 null, 不翻 CL-001 的 REJECT); 残差 IC/placebo≠ship, CL-001 已 REJECT CL-003 只定性=合法完成 (R02/R03); ST 源头排除 (R06); regime 分层为核心定性依据 (R11); R12 placebo=随机对照消融, 异类增益须胜 null (本例胜但属动量 overlay 非 alpha); 大缓存 research/cache/cl003/ gitignored 无 features 写入 (R04); 复用缓存评分/CL-001 块未碰生产 (R05, 指纹一致); block-level checkpoint (R08)。
  - **产出**: `research/backtest/run_cl003.py` / `research/cache/cl003/{blocks/*.json, cl003_results.json, run.log}` / `research/verdicts/CL-003.json`。
  - **下一步**: **CL 阶段全 task 完结** (CL-001 死于消融 + CL-002 skipped + CL-003 regime_overlay)。北极星 (RETRO-003 首个正向的严格证伪验证) 全部交付, 结论一致 = within-concept 动量 re-rank 不是可交易 alpha 而是动量 regime overlay。生产线 V12.31 全程只读冻结。

- (OA campaign iter, 0618) **正交因子 campaign 启动 + OA-VOL/OA-VP Phase-1 双双 PROCEED**。Track C2 新数据探索, 用户开闸广搜正交正-alpha (波动率/量价/大宗商品/分析师修正/新闻政策 + 1H对比)。建 research/oa_screen.py 共享Phase-1廉价闸(扣[动量+size+value+roe/margin]残差IC+消融+分regime+A/B/C/D, NW-t HAC校正重叠label)。codex对抗审计纳入冻结gate: NW-t/DSR含DoF乘子3/selection-on-selection caveat。**OA-VOL: 7/10存活=低波动异象, ivol_60 orth IC=-0.0811 t=-6.69(NW)最强**。**OA-VP: 5/8存活, overnight_ret_20 +0.0467 t=8.61(隔夜动量,残差后增强) + on_minus_in_20 +0.0483 t=8.04(隔夜-日内分解); amihud=D被size吃识破reskin**。两族解锁 walk-forward GATE。⚠纪律: Phase-1只正交标准控制非V12.31本身(已含pyr_velocity波动信号), 21连否死法正是GATE, IC≠落地(R03)。下一步: 建通用walk-forward GATE测survivors对V12.31增量 + OA-COM(商品,codex#2)。manifest n_trials=18(VOL10+VP8)。生产线V12.31冻结(R05)。

- (OA campaign iter2, 0618) **OA-COM REJECT(商品渠道塌缩) + OA-RC/OA-NEWS 数据闸关闭 deferred**。探测Tushare权限: ✅fut_daily/index_daily/major_news(市场级); ❌report_rc/news/anns_d 无权限。**OA-COM**: build oa_com_factors.py(10商品settle收益篮子+油+金属). 真供应链渠道因子(com_beta_basket/oil/metal + transmit)全C_塌缩(raw~0.02-0.03被size/industry/动量吸收=商品beta无独立横截面alpha, 应验codex#2 mapping noise); 唯一存活com_betavol_60=残差波动=ivol换皮(Spearman 0.917) → 元判断REJECT, GATE保持skip不重复计。**OA-RC**(codex#1最高prior)无report_rc权限→deferred非失败. **OA-NEWS**仅major_news市场级无ts_code→无法横截面→deferred. **决定性剩余=walk-forward GATE测 ivol_60(VOL)+on_minus_in_20(VP) 对V12.31真增量**=21连否死法之处。manifest n_trials=24(VOL10+VP8+COM6)×DoF3=72 for DSR。生产线冻结(R05)。

- (OA campaign 收尾, 0618) **正交因子campaign完结: 净0可落地新alpha (第22-23否)**。walk-forward GATE双REJECT: ivol_60 Δα=-0.153pp(regime全伤 PBO0.8过拟合), on_minus_in_20 Δα=-0.702pp(全regime负). 真异象(低波动IC t-6.69/隔夜-日内 t8.04)过Phase-1廉价闸(正交标准控制)但对V12.31零/负增量=价量均值回归核已吃(低波动经pyr_velocity, 隔夜经pump/ratio). OA-1H: 1H日内RV(-0.055)弱于日线ivol(-0.065)Spearman0.869→1H对波动因子不增强(区别于1H动量增强). OA-SUMMARY: research/ORTHOGONAL_ALPHA_FINDINGS.md. n_trials=24 DSR基72. 唯一未测强候选=report_rc(分析师修正,codex#1, token无权限). 元结论:正交≠有α再印证, 瓶颈信号枯竭非流程, Tushare token档限制真正交数据(分析师修正/个股新闻/微结构均无权限). 脚手架诚实REJECT未p-hack. 生产线V12.31全程冻结. campaign 4 commit checkpoint. → COMPLETE.

- (market-state iter, 0618) **市场状态暴露overlay REJECT — paper对long-only反指**。用户分享paper'In Good and in Bad Times?'(因子alpha熊市强41%, 75%空头端, 套利限制驱动), /loop测能否改善V12.31。复刻事前GOOD_TIMES(MA200+3年趋势, 沪深300+500), 在de-lookahead WFE-001 book(Sharpe1.31)测暴露缩放5臂+placebo负控。**决定性发现: 我们绝对Sharpe(过渡0.81→偏牛1.26→牛1.65)+超额alpha(过渡+0.35%→牛+2.82%)双双随GOOD_TIMES单调上升=alpha bull-loaded与paper完全反转**。paper规则(熊加牛减)Sharpe1.31→1.21反指(牛市砍仓), trend(牛加熊减)≈base, placebo p=0.588零timing技能, static_dele≡base证maxDD改善纯降暴露。机制: long-only吃不到paper空头端熊市alpha(A股做空约束极致=机制最强但不可收割)+启动子牛市最有效+净多头beta。局限: 诚实book无熊月(walk-forward只到2024-10), 熊市加仓侧不可测不补lookahead。元结论: paper是long-short市场中性+空头端现象不可移植long-only A股净多头book。不实施暴露择时, V12.31保持充分投资。脚手架placebo+static_dele隔离全到位真小未夸大。生产线冻结。findings: research/MARKET_STATE_EXPOSURE_FINDINGS.md。

- (QuantML campaign 收尾, 0619) **QuantML-Qlib 48篇四轴全REJECT — 模型/目标/解耦/搜索非杠杆=数据不行非模型不行**。用户分享合集要求学+规划; 我筛出唯一未测轴=模型class/目标/搜索(23否全feature侧), 用户要Tier1+2+3顺序全做。QML-1排序目标(文章33): 回归OOS RankIC+0.1145 vs lambdarank+0.0744(NW-t-3.66)/xendcg+0.0800(NW-t-2.68)显著劣, MSE错配假设证伪(回归+spearman_ic早停已优化排序). QML-2 NN(LNN/iTransformer等): LGBM+0.0953 vs torch MLP+0.0660(NW-t-1.87)更差更噪, GBDT截面tabular极强基线. QML-3 Alpha/Beta解耦(ABCM/FactorVAE): beta中性选股book Δα-0.381pp/月(14/19月负)去beta反伤=edge与beta耦合(印证市场状态0618). QML-4自动因子挖掘(自建符号搜索80因子): 24存活过正交闸+Bonferroni但全是已walk-forward否决族换皮(最强tsstd_intraday_60=OA-VOL intraday_vol_60已REJECT)→AI Scientist教训(枯竭空间搜索量产已知因子假阳性,gate抓)完美实证. 元结论: PoC IC(0.023-0.087)=我们样本内0.44→OOS0.18同类过不了walk-forward, 与23否+正交camp+市场状态一致收敛=信号枯竭非模型, 增量在book/风控/部署或真新数据(微结构/分析师修正token无权限). 脚手架NW-t/Bonferroni/廉价闸全程, 机械PROCEED如实override为REJECT. 生产线V12.31冻结. findings: research/QUANTML_MODELS_FINDINGS.md.

- (基金抱团分析, 0619) **高收益基金=AI算力主题极端抱团**。用户问近1年高收益平滑基金持仓/抱团/逻辑/找alpha。建 fund_screen.py(排行×fund_basic→top70股票混合型,fund_nav算回撤) + fund_holdings.py(fund_portfolio近5季度→抱团/概念/Jaccard). 发现: top70近1年均+283% maxDD仅-9~-21% Sharpe3.3全平滑(主题单边牛非风控); 全AI/算力/半导体/光通信主题; 中际旭创94%+新易盛93%基金持有,Jaccard0.197(~10x随机)极端抱团,核心抱团股5季度加深(光模块→光纤铜缆PCB扩散); 收益=主题beta+抱团动量非分散选股alpha,与V12.31抄底回调正相反(印证市场状态0619:牛市我们选回调跑输追高). ⚠季度变化有年报全持仓vs季报top10披露口径陷阱. 3方案: ①基金抱团度正交因子(PIT,先验≈动量镜像) ②机构抱团∩回调交叉池(最契合DNA) ③拥挤度风控. 基金持仓=真正交数据(正交camp唯一可能出alpha方向). findings: research/FUND_CROWDING_FINDINGS.md. 用户待选方向. 生产线冻结.

- (fund-crowding campaign 收尾, 0619) **基金抱团→alpha 3方案全REJECT — 基金持仓对我们净0可交易alpha**。用户要3方案全做。纪律=宽基随机498只(业绩无关避幸存者)+PIT(ann_date零前视)+扣动量残差+placebo. FC-1抱团度因子: crowding_level C零信号(动量镜像已priced)/crowding_delta B边缘(t2.12低功率). FC-2机构抱团∩回调池: crowding_delta入V7c回调池Δα-0.224pp/月(13/19负). FC-3拥挤风控: 排拥挤Sharpe3.85<随机placebo3.94(比随机还差,纯去暴露). 机制: 抱团水平=动量镜像, 我们小盘回调vs机构大盘光模块龙头风格正交(重叠27%). 描述性(高收益基金极端抱团光模块,中际旭创94%/新易盛93%,Jaccard0.197)作风格对照有效非可交易. 元结论: 与正交camp+QuantML一致=增量不在选股层(价量/正交/模型/基金持仓都试尽), 在book/风控/部署+微结构数据(token无权限). findings: research/FUND_CROWDING_ALPHA_FINDINGS.md. 生产线冻结.
