# metaRSI 账本 — stockagent-analysis

> 由 `metarsi.py` 从 `metarsi.db` 生成，每次写入后重写。不要手改；改记录请用命令。

## 入口

- 总图：`graph.md`
- 节点（窗口、受保护面、影子规则、准入提议）：`nodes.md`
- 试验：`trials.md`
- 否决过的想法：`negatives.md`
- 想法与思考算子：`ideas.md`
- 模型与模型审计：`models.md`
- 问答日志：`log.md`
- 调用手册：`playbooks.md`

## 当前模式

**后台学习**

- 保留窗口 reserved_20260806_plus 已成熟 21 个信号日；5 条影子规则里 0 条到期，最近的一条还差 39 个
- 有待处理的信号，但没有未用过的留出窗口，影子规则也已到上限
- 没有未用过的留出窗口，新规则在已用窗口上最多算 reused_holdout
- 校准：弱证据上显著、又到更强证据复验的结果 3 条，成立 0 条
- 上线模型 s20_frozen_stage1 的审计有 9 条问题（model audit 查看），它的历史成绩要按问题打折
- 审计问题更少的候选模型：s20_unified_v1、s20_unified_v2、pump_v5_shape。要替换先看 model diagnose，再登记成影子在保留窗口上并行验证
- 现在可以做：记录问答、试验和否决项；做描述性分析（只能算 in_sample 或 reused_holdout）；整理 wiki；可以把新想法登记成影子规则
- 现在不做：在已用过的窗口上搜新规则并当作验证；读保留窗口；改受保护面
- 已疲劳的类别：day-gate、in-list-filter/R、in-list-filter/S、model-routing、s20-model、s20-model-audit

## 当前状态

- 证据等级（强到弱）：formal > prospective > holdout_once > reused_holdout > in_sample > judge

| 窗口 | 起 | 止 | 状态 | 试验数 | 同类问题已比较的规则数（需要的 /t/） |
|---|---|---|---|---:|---|
| dev_20250303_20260126 | 20250303 | 20260126 | consumed | 8 | in-list-filter/S: 133 条（3.56）；in-list-filter/R: 11 条（2.84）；s20-model-audit: 4 条（2.50）；s20-model: 2 条（2.24） |
| confirm_20260127_20260805 | 20260127 | 20260805 | consumed | 42 | in-list-filter/S: 148 条（3.58）；in-list-filter/R: 22 条（3.05）；list-turnover/S: 2 条（2.24）；s20-model-audit: 1 条（1.96）；s20-model: 3 条（2.39）；model-routing: 5 条（2.58）；day-gate: 61 条（3.35）；pump-model: 4 条（2.50）；mean-reversion: 3 条（2.39）；intraday: 32 条（3.16）；external-factors: 42 条（3.24）；risk-flag: 1 条（1.96） |
| reserved_20260806_plus | 20260806 |  | reserved | 0 |  |

- 未结的影子规则：5 条；主规则：S0010
- 等人批准的提议：0 条
- 试验 51 条，否决 23 条，问答 49 条
- 校准：在较弱等级上显著、又在更强等级上复验过的结果 3 条，其中成立 0 条
- 账本链：完整

## 不变量

- 模型或评审的意见不能作为改动受保护面的证据。
- 在某段数据上发现的规则，不能用同一段数据确认。
- 保留窗口只用于给读取之前就登记好的规则计分。
- 不自动更改股票冻结合约、交易动作或发出的名单。
- JEV 与任何模型评审只算 judge 级意见。
- 2026-08-06 起的信号日是保留窗口，只用于给事先登记的影子规则计分，且要满 60 个成熟信号日。
- 报每笔收益时写明是名单口径还是账户口径。
