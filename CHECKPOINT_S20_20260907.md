# S20 checkpoint — 2026-09-07

## 当前状态

S20-20R R20 锚定研究已完成开发、portable 重训、滚动校准审计和一次性全市场确认。最终状态为 `confirmation_completed_not_shadow_eligible`，生产 R20 不变。

## 关键结果

- portable Stage 1：Top20 22.79%，lift 1.422。
- portable Stage 2 固定混合：Top20 23.59%，lift 1.471。
- 新确认期：Top20 22.50%，R20 参考 16.63%，提升 5.87pp，lift 1.759。
- 确认期正命中日比例 81.75%，低于 90% 门槛。
- 最差月份相对 R20 退化 26.67pp，月度稳定门槛失败。
- 独立 S20 概率头失败；只保留 rank + R20 reference 双输出语义。

## 关键文件

- `config/s20_20r_training_contract.json`
- `config/s20_20r_results.json`
- `scripts/train_s20_20r_residual.py`
- `scripts/train_s20_20r_ranker.py`
- `scripts/calibrate_s20_20r_rolling.py`
- `scripts/build_s20_20r_confirmation_factors.py`
- `scripts/confirm_s20_20r_portable.py`
- `docs/research/s20_20r_r20_anchor_report_20260907.md`
- `wiki/2026-09-07_s20-20r-r20-anchor.md`

本地 gitignored 产物位于：

- `output/experiments/s20_20r_residual_portable/`
- `output/experiments/s20_20r_ranker_portable/`
- `output/experiments/s20_20r_calibration_portable/`
- `output/experiments/s20_20r_confirmation/`

## 后续约束

2026-01-27～2026-08-05 确认窗口已使用一次并关闭。下一候选可以参考确认期诊断，但不得在该窗口重新选择后宣称验证通过。至少积累 2026-08-06 之后 60 个完整成熟交易日，再做新确认。

生产 R20、池 A/G、网页和每日 21:00 更新均未修改。
