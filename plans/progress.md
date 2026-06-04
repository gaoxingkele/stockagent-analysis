# Progress — research/fundamental-orthogonal

> fresh-context 每轮干净重启, 这里是跨迭代的唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星
验证**基本面成长因子(营收增速 or_yoy / 利润增速 netprofit_yoy)** 是否对 V12.31 带来经
walk-forward 验证、且扣 [动量+size+value+其他基本面] 后仍存在的真 alpha。
**这是 8 次价量否决后第一个不塌的正交信号** —— 快测 (202410-202604, 18月): or_yoy 原始 IC +0.033,
**扣动量后还是 +0.033 (不塌!=真正交)**, 但 t=1.39 未显著 (样本短)。**需拉长历史补功率。生产线 V12.31 冻结。**

## 事前注册 gate (冻结 R01 + 分regime R11 + 基本面消融 R12++)
- FU-003 显著性: 拉长≥40月, 扣[动量+size+value+其他基本面]残差 IC |IC|>=0.02 且 |t|>=3 → 显著正交。
- FU-004 gate: Δα(基本面臂−V12.31)>=+0.30pp 且 Sharpe/最差月不降 且 单月outlier剔后>0 且 分regime不伤 且 扣全控制存活。

## ★ 应对措施 (responsePlaybook, 冻结 —— 用户要的"各种结果应对", 防事后找补 R02)
**FU-003 (正交IC验证) 分叉**:
- A 显著正交 (|IC|>=0.02 & |t|>=3) → 进 FU-004 walk-forward
- B 边缘 (|IC|>=0.02, t∈[2,3)) → 进 FU-004 但标低功率; 若 walk-forward 也边缘 → REJECT + promising_unproven (待更多数据, 不强上)
- C 塌缩 (扣控制 |IC|<0.01 或 t<2) → 廉价 REJECT 跳 walk-forward (size/value/动量 reskin 或 artifact), FU-004/005 skip
- D 已知因子 (被 size 或 value 单独吃掉) → REJECT, 记=成长/小盘 reskin 非新 alpha

**FU-004 (walk-forward gate) 分叉**:
- A PASS (Δα>=+0.30 全条件过) → FU-005 opt-in 落地
- B 真小 (Δα∈(0,+0.30) 显著) → 不独立落地, 记 blend 候选, 不强上
- C 子条件挂 (Δα>=+0.30 但 worst/outlier/regime 挂) → REJECT, 记挂哪条+机理
- D 负 (Δα<0) → 干净 REJECT
- E 条件有效 (仅某子域/regime) → 仅当条件可因果识别(RG-002教训)才记条件部署候选, 否则 REJECT

**FU-001 数据**: 重述用首次ann_date/vintage不回填(防泄漏); ann_date缺用first_ann_date兜底; 覆盖<10%档单列评估。

## 任务台账
| id | 任务 | 状态 | 裁决 |
|----|------|------|------|
| FU-001 | Point-in-time 基本面面板(拉长~2019) | ✅ built | 268525行×5435股×54月度网格, or_yoy覆盖98.9% |
| FU-002 | 成长因子构建 + 控制集 | ✅ built | growth_composite 覆盖98.9%, vs roe +0.35/value -0.14/size +0.12/动量近正交 |
| FU-003 | 功率充足正交 IC 验证(≥40月) | ✅ REJECT | **C_塌缩**: or_yoy 47月 orth IC=+0.0055 t=1.08, raw 转负, 短窗 +0.033 不存活 |
| FU-004 | walk-forward 决策 gate | ⏭ skip | precondition skip (FU-003=C) |
| FU-005 | opt-in 落地 (依赖 FU-004=A_PASS) | ⏭ skip | 未 A_PASS 不落地 |

## 已有资产 / 复用
- fina_indicator_vip 一调全市场 (5722股×109列, 含 ann_date 点-in-time) —— 已验证可行
- forecast 缓存 output/tushare_cache/forecast_2025H2.parquet (预告惊喜≈0 已测, 备用)
- daily cache (close + daily_basic 取 total_mv/pe/pb); t005 walk-forward harness; RG-001 regime; verify.py(isGate)
- ⚠ report_rc(分析师修正) 限频 1次/小时 —— 本轮不依赖它 (用 fina_indicator 免费批量)

## 关键约束
- 负结果=合法完成禁同段OOS重调(R02); IC≠落地只认walk-forward α(R03); 泄漏前置闸(R04, 基本面用ann_date不用end_date!)
- 生产hash冻结(R05); ST源头排除(R06); 分regime(R11); 基本面消融扣[动量+size+value+其他基本面](R12++); 单月outlier
- **基本面头号泄漏坑: 必须用 ann_date(公告日)对齐, 不能用 end_date(报告期末); 重述不回填**

## 迭代日志
<!-- 每轮 append: 迭代N | task | 做了什么 | 裁决(命中playbook分支)/产出路径 | 下一步 -->
- (init) 由 relation-tensor 循环(已完结REJECT, 第8个价量否决)转入。快测发现 or_yoy 正交 IC 不塌(+0.033) = 首个非价量换皮信号。等待启动。
- 迭代2 | FU-002 | 建 research/fu002_build_factors.py: 从 FU-001 PIT 面板建成长因子 growth_or/growth_np/growth_composite (逐日 winsorize(1/99)→横截面 z→行业中性减行业均值, stock_basic.industry, NaN行业归 __NA__ 组), 组合=0.5*(or+np)。控制集 R12++: 动量 mom_5/20/60 (从 daily close 全市场 backward, 月末采样, checkpoint research/cache/fu002_momentum.parquet) + size ln_total_mv + value inv_pe/inv_pb (仅正值) + 其他基本面 roe/margin。merge regime_timeline(覆盖79.5%)供 FU-003 分层。产出 research/features/fundamental_factors.parquet (268525行×17列, 键 ts_code,trade_date, 无 forward 字段——label 在 FU-003 另接以免触泄漏闸)。**裁决=built** (构建任务无分叉)。**相关性结构 (逐日横截面 Spearman 均值, 供 R12++ 消融)**: growth_composite vs roe **+0.352** (最强, 与盈利质量同向) / vs inv_pb **-0.137** (与价值反向, 高成长=高估值) / vs ln_total_mv **+0.121** (轻微大盘倾斜) / vs 动量 **近 0** (与动量正交)。growth_or/np 各覆盖98.9%。泄漏自查通过(无 forward 列名)。verify.py 无硬违规(0 violations, leakage+指纹全绿; FAIL 仅因 FU-003/004 未做)。 | 下一步 FU-003: ≥40月(本面板54月 OK)横截面 rank-IC(成长因子, fwd r20)全期+分regime; 残差化扣[动量+size+value+其他基本面(roe/margin)]后 IC+t; 逐步消融(单扣动量/size/value 看塌不塌, **重点查 roe 是否吃掉成长——corr +0.35 最高**); 按 playbook 判 A_显著正交/B_边缘/C_塌缩/D_已知因子reskin。注意 r20 label 不落 features parquet(触泄漏闸), 只在 cache 工作面板内用。
- 迭代3 | FU-003 | 建 research/fu003_orthogonal_ic.py: merge FU-002 成长因子 + 复用 rt004_r20_label.parquet(前向 r20=close[t+20]/close[t]-1, 因果, 不进 features). 47 月度 rebalance(2022-06~2026-04, ≥40 功率达标) × 3 成长因子(growth_or/np/composite) × 6 残差变体(raw / 单扣 mom / 单扣 size / 单扣 value / 单扣 ofund / orth_full=扣全控制). 逐月横截面 OLS 残差化(控制标准化)→ rank-IC, 月度序列→ t=mean/(std/√n), 分 regime(SIGN-R11). **裁决=C_塌缩 (REJECT, playbook FU-003_outcomes::C)**. 头号因子 or_yoy(growth_or): **原始 IC 在 47 月窗口转负 -0.0080(18月快测 +0.033 是短窗 artifact!), 扣[动量+size+value+其他基本面]全控制残差 orth IC=+0.0055 |t|=1.08** — 远未达显著线(需 |IC|>=0.02 & |t|>=3). growth_composite orth IC=+0.0081 t=1.52; 最强 growth_np orth IC=+0.0130 t=2.44 仍不过线. 逐步消融: raw 本就≈0/负(无可被吃掉的信号)→ 归 C 非 D(非 size/value reskin, 是信号本身在长窗口不存在). 分 regime 无任一 cell 过线(最强 growth_np@reversal IC=0.0188 t=2.08, 仍 <0.02/3). **元结论: 这是第 9 个被否假设, 也是"8 次价量否决后第一个不塌的正交信号"最终在功率充足的长窗口下塌缩 —— 短窗 +0.033 是窗口 artifact, 拉长历史(补功率)正是为揭穿它而设, 系统诚实 REJECT 未 p-hack(SIGN-R02)**. 产出 research/cache/fu003_ic_table.parquet(846 行 IC 明细) + research/verdicts/FU-003.json. FU-004 skip:true(廉价省 walk-forward 算力) + FU-004.json status=skipped; FU-005 保持 skip. verify.py 全绿(无泄漏/指纹一致/全 task 有裁决). | 下一步: 全部非 skip task 已有裁决, 循环可完成. 若未来补更多财报季数据可复检 growth_np(唯一全 regime 正号但弱), 但当前证据=成长正交 alpha 不存在/微弱, 生产线 V12.31 不动.
- 迭代1 | FU-001 | 建 research/fu001_build_pit.py: bulk 拉 fina_indicator_vip 22报告期(2019Q4~2025Q3, 首次披露min ann_date去重防回填), 54个月末日 bulk daily_basic 取 size/value 控制, merge_asof(ann_date<=trade_date, by ts_code, backward) 建 PIT 面板。缓存 research/cache/fu001_fina_raw.parquet + fu001_daily_basic.parquet (checkpoint)。产出 research/features/fundamental_pit.parquet (268525行×16列: close+total_mv/circ_mv/pe/pe_ttm/pb/ps_ttm+fina_ann_date/end_date+roe/margin/netprofit_yoy/or_yoy/debt_to_assets), 键(ts_code,trade_date)。**裁决=built** (命中 playbook FU-001_data: 重述用首次ann_date原值不回填)。泄漏自查通过(无行 ann_date>trade_date, 报告滞后中位62d 均≥0), ST源头排除14118行, or_yoy覆盖98.9%(small/mid/large 各>98%, 无<10%低覆盖档)。verify.py 无硬违规。 | 下一步 FU-002: 用此面板建成长因子(or_yoy/netprofit_yoy winsor+rank/z+行业中性+组合)+控制集(mom_5/20/60 从daily cache, ln total_mv, 1/pe 1/pb, roe/margin)+相关性矩阵。
