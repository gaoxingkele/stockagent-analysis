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
| FU-001 | Point-in-time 基本面面板(拉长~2019) | todo | — |
| FU-002 | 成长因子构建 + 控制集 | todo | — |
| FU-003 | 功率充足正交 IC 验证(≥40月) | todo | — |
| FU-004 | walk-forward 决策 gate | todo | — |
| FU-005 | opt-in 落地 (依赖 FU-004=A_PASS) | skip | — |

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
