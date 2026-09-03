# S20 checkpoint — 2026-09-03

## 已完成

- 冻结五目标路径标签：+15/-8、+20/-10、+25/-12、+30/-12、+35/-12。
- 实现 D+1 开盘入场、20 日路径、positive/N1/N2/N3/ambiguous 标签。
- 全量生成 3,215,747 行、5,300 只股票，零重复，行情到 2026-09-02。
- 修正标签优先级：较早已止损而目标日又双触的路径确定归 N2，不归 ambiguous。
- 完成 50% 固定样本、三折 purged walk-forward、五目标 × 三种子二分类和五个原因模型。
- 增加机器 gate 评估器与研究报告。

## 最终裁决

状态：`offline_completed_not_shadow_eligible`。

S20-20 汇总每日 Top20 命中率 21.17%，lift 1.321；分折 lift 1.511/1.276/1.260。wf1 相对 B0 退化 3.10pp，三折校准后 Brier Skill 均为负。冻结的五项晋级检查全部失败，因此没有运行全市场确认，没有启动 shadow，没有生产影响。

## 关键文件

- `config/s20_v2_training_contract.json`
- `config/s20_v2_label_audit.json`
- `config/s20_v2_validation_results.json`
- `src/stockagent_analysis/s20.py`
- `scripts/build_s20_v2_labels.py`
- `scripts/train_s20_v2_multitarget.py`
- `scripts/evaluate_s20_v2_gate.py`
- `docs/research/s20_v2_multitarget_report_20260903.md`
- `wiki/2026-09-03_s20-v2-training-result.md`

本地 gitignored 产物：

- `output/experiments/s20_v2_labels/`
- `output/experiments/s20_v2_multitarget/`

## 验证命令

```powershell
python -m pytest tests/test_s20.py tests/test_r20_target_prob.py -q
python scripts/build_s20_v2_labels.py
python scripts/train_s20_v2_multitarget.py --sample-bps 5000
python scripts/evaluate_s20_v2_gate.py
```

gate 脚本在“不合格”时按设计返回非零退出码，报告仍写入 `gate_report.json`。

## 下一研究边界

下一候选预先冻结为“每日截面归一化 + LambdaRank 排序头 + 成熟标签滚动校准”。当前三折测试结果已被查看，禁止继续在同一测试期调参后宣称晋级。更新因子后，以 2026-01-27 至 2026-08-05 为一次性新 holdout。

生产 R20、池 A/G、网页和每日 21:00 更新均未被本次 S20 工作修改。
