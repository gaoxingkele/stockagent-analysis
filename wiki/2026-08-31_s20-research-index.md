# S20 独立研究指数

S20 已确定为 R20 之外的独立探索，不替代当前有效 R20，也不修改池 A 和每日 21:00
生产更新。其核心定义为：预测未来 20 个交易日内“先触达 +15%/+20%/+25%，而不是先
触达 -15%”的校准概率，主指数为 `100 × P(+25% first)`。

相较旧的窗口最高涨幅/最大回撤标签，S20 保留事件顺序和触达时间，能区分“先大跌再反弹”
与“先达标后回撤”。同一天同时碰到上下线时，日线无法判断先后，标记为 ambiguous。

研究顺序固定为：竞争风险 LightGBM → DLinear → TFT → TRA 软路由 → DoubleAdapt 漂移
适配 → MASTER 市场上下文；TACTiS/扩散式联合路径模型仅作为后段挑战者。评估使用严格
时间顺序 walk-forward、Brier Skill、ECE、top-decile lift 和分市场阶段校准，至少影子
运行 60 个交易日后才讨论生产用途。

完整文献矩阵与实验闸门见
[`docs/research/s20_literature_and_algorithm_roadmap_20260831.md`](../docs/research/s20_literature_and_algorithm_roadmap_20260831.md)，
机器契约见 [`config/s20_experiment_v1.json`](../config/s20_experiment_v1.json)。

## B1 进度

全量首次触达标签已经生成并与旧标签口径验证一致。50% 样本的三折实验显示 B1 能提高
PR-AUC 和顶十分位命中率，但静态 Platt 校准在第一折出现负 Brier Skill，因此状态为
`offline_completed_not_shadow_eligible`。测试后发现的 B1+B0 融合只能作为新时期的冻结
验证线索，不能用本次结果晋级。详见
[`docs/research/s20_b1_competing_risk_report_20260831.md`](../docs/research/s20_b1_competing_risk_report_20260831.md)。
