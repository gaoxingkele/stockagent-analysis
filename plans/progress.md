# Progress — research/ratio-phase

> fresh-context 每轮干净重启, 这里是跨迭代的唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星
判定"相位感知 / 动态窗口"对 pump ratio 是否带来经 **walk-forward 验证**、相对生产线 **V12.31**
的真 alpha。有就 opt-in 落地, 没有就把负结论文档化并停手。**生产线 V12.31 全程冻结。**

## 事前注册 gate (冻结, SIGN-R01)
walk-forward 19 月, 同协议同 harness 对照 V12.31:
- Δα(月化) ≥ +0.30pp 且 Sharpe 不降 且 无新增灾难月 且 正α月占比不降 → PASS, 否则 REJECT。

## 任务台账
| id | 状态 | 裁决 |
|----|------|------|
| T-001 事件研究 (ratio vs MA5 拐点, lead/lag) | done | **sync** (ignition τ0 / peak +2d / trough -7d; choppy ignition 漂移 -5d) |
| T-002 窗口非平稳检验 | done | **window_dynamic_needed** (份额平稳 max漂移2.42pp; 但 ignition相位漂移5d 触发; 非平稳=相位非窗口配比) |
| T-003 ratio 轨迹特征 (全因果) | done | **built** (5类11特征落库; 最强 ratio_div_strength RankIC +0.065 多头背离, regime 稳健; ratio_vel_3 +0.050) |
| T-004 多尺度门控启动子 (依赖 T-002) | done | **built** (3 尺度 pump 模型 s3/s5/s10 + regime/run_len 门控; 切换 58% vs 固定s5; choppy→s3 95% / trend→s10) |
| T-005 walk-forward 决策 gate | todo | — |
| T-006 opt-in 落地 (依赖 T-005=PASS) | skip | — |

## 关键约束 (摘自 guardrails)
- 负结果=合法完成, 禁止在同段 OOS 反复重调 (R02)
- 中间指标 (IC/precision/lead-lag) ≠ 落地, 只认 walk-forward α (R03)
- 任何 IC/回测/特征前先跑 leakage guard (R04); 见 IC>0.5/Sharpe>10 先查泄漏+ST
- 生产文件 hash 冻结 (R05); ST 源头排除 + 全程分层 (R06)

## 迭代日志
<!-- 每轮 append: 迭代N | task | 做了什么 | 裁决/产出路径 | 下一步 -->
- (init) 脚手架就绪, 等待启动。web-platform prd 已归档为 plans/prd.web-platform.json。
- 迭代1 | T-001 事件研究 | 19月窗口(20240601-20260101) v3c 逐日 causal 推理出 ratio(190万行,落 research/cache/ratio_series.parquet 做 checkpoint); MA5 斜率滞后确认上拐点(零前视, 19万事件/5026股); 60d MA5/MA60 穿越数分 trend/mid/choppy; ±10日相位均值曲线。
  - **裁决 sync**: ratio 不领先拐点。raw 相位曲线 trough≈τ-7 → ignition(最大单日抬升)τ0 → peak τ+2 → 回落。即"同步/确认型"而非"预测型"信号。
  - **关键 regime 发现**: peak 跨 regime 恒为 +2d (不变); 但 ignition 在 choppy 漂移到 -5d (轻度领先), trend/mid 保持 τ0 → 相位关系随 regime 非平稳。**这是 T-002(窗口非平稳)的直接证据/起点。**
  - 教训复盘: 初版用事件内 z-score 提相位形状被重尾污染 + 边界伪峰, 误报 lead@-10d; 改用 raw 均值曲线 + 内区[-8,8]搜地标后干净。中间指标只描述不作 gate (SIGN-R03)。
  - 产出: research/t001_phase_event_study.py / research/cache/ratio_series.parquet / research/cache/t001_results.json / research/cache/figs/t001_phase_curve.png / research/verdicts/T-001.json
  - 下一步 (T-002): 把 ratio 拆短窗/长窗贡献 + 量化 ignition lead/lag 随 regime(快/慢市)的漂移幅度。T-001 已显示 choppy ignition 提前 5d 是漂移存在的强提示 → 预期 status 偏向 window_dynamic_needed, 但须按数据定。ratio_series.parquet 可复用免重算。
- 迭代2 | T-002 窗口非平稳检验 | v3c 3way 模型跑 TreeSHAP (pred_contrib), 方向信号 signal_shap=shap_up-shap_down; 247 特征按名内最大整数分桶 short<=10/mid11-30/long>=31/cdl=short/无窗静态=context; 135k 分层样本(each regime 45k, seed20260603, ST源头排除); 全样本+分regime 算 |signal_shap| 占比。
  - **裁决 window_dynamic_needed**: 窗口贡献'份额'近似平稳 (短9.6/中56.1/长20.5/上下文13.8%, regime间份额极差仅 max 2.42pp(mid桶) < 5pp 阈值) → 模型在快/慢市用的窗口混合几乎不变。但 T-001 ignition lead/lag 相位漂移 5d (choppy-5d vs trend/mid0d) ≥ 3d → 触发。
  - **关键洞察**: 非平稳来自**相位/时序**(信号何时相对拐点起飞)而非**窗口配比**(模型给各窗口的权重)。long-short 净份额跨 regime 也稳 (11.87/10.04/10.90pp)。含义: T-004 多尺度门控的增量应来自 regime/run-length **相位门控**, 而非给短/长窗特征重配权重 (后者本就平稳, 重配空间小)。T-004 保持 skip:false 执行。
  - 注意 (SIGN-R03): SHAP份额/ignition漂移都是中间指标, 仅描述; ship 与否只由 T-005 walk-forward α 定。window_dynamic_needed 只解除 T-004 条件门, 不保证 T-004 带 α。
  - 产出: research/t002_window_nonstationarity.py / research/cache/t002_sample.parquet (复用免重算 load_window) / research/cache/t002_results.json / research/verdicts/T-002.json
  - 下一步 (T-003): 构建 5 类 ratio 轨迹特征 (Δratio/Δ²ratio/背离象限/自分位/距峰天数), 全 causal, 落 research/features/ratio_traj.parquet。注意 features/ 目录会触发 verify.py 黑名单 guard, 列名勿撞 forward-field。ratio_series.parquet 可复用。
- 迭代3 | T-003 ratio 轨迹特征 | 复用 ratio_series.parquet + daily close, 构建 5 类全 causal 轨迹特征 (191.8 万行/5055 股), 落 research/features/ratio_traj.parquet (14 列, 入库前自检列名不撞 forward-field 黑名单)。特征: ①速度 ratio_vel_1/3/5 ②加速度 ratio_acc_1/3 ③背离象限 ratio_div_quad(-2/-1/1/2)+ratio_div_strength(各自 60d 滚动 std 归一后相减) ④自分位 ratio_selfpct_20/60 (rolling rank) ⑤距峰天数 ratio_days_since_hi/lo_60 (rolling argmax/min)。耗时 35s。
  - **裁决 built**: 描述性 RankIC(fwd5, 仅内存评估前向收益未入库) 最强 = **ratio_div_strength +0.0654** (ratio 升势快于价格的多头背离, regime 间 +0.0548~+0.0752 稳健); ratio_vel_3 +0.0495 / ratio_vel_5 +0.0446 次之; ratio_days_since_hi_60 +0.0242; div_quad -0.0232 (离散象限编码方向, 用连续 div_strength); acc_1/selfpct/days_since_lo 偏弱。量级温和 (最强 0.065 << 0.5 阈值, 无泄漏/ST 红旗)。
  - **正交性印证**: div_strength 跨 regime 稳健 + vel_3 最强, 与 T-002 "非平稳来自相位/时序而非窗口配比" 一致 → T-004 门控应优先纳入轨迹速度 (vel_3) + 背离强度 (div_strength)。
  - 纪律: IC 仅描述非 gate (SIGN-R03); ship 只由 T-005 walk-forward α 定。脚本入库前自检 + verify.py leakage guard 双通过 (0 违规, 生产指纹未变)。控制台需 PYTHONUTF8=1 (Δ²/中文在 GBK 控制台会 UnicodeEncodeError)。
  - 产出: research/t003_ratio_trajectory.py / research/features/ratio_traj.parquet / research/cache/t003_results.json / research/verdicts/T-003.json
  - 下一步 (T-004): 多尺度门控启动子。依 T-002 裁决 (skip:false), 对 K 个窗口尺度各训 pump 模型落 research/models/, regime/run-length 门控集成输出动态窗口版 ratio, 推理路径样本日跑通不碰生产模型。T-003 的轨迹特征 (尤其 vel_3/div_strength) 可作门控输入候选。ratio_series.parquet + ratio_traj.parquet 均可复用。
- 迭代4 | T-004 多尺度门控启动子 | K=3 窗口尺度 = 预测前向窗口 H (s3:H3/g6% s5:H5/g10% s10:H10/g15%), 标签用前向 high/low max_gain/max_dd 3way (0中性/1跌/2涨, 仅训练用不入库), 复用 v3c 247 特征 (load_window causal, ST 源头排除); 各训 LGBM 3way (500k train/120k val 按 20251001 时间切分早停, best_iter 82/45/35, 类分布短窗更均衡 65/15/20→长窗 80/7/14); 逐尺度推理 ratio_sH=P涨/(P跌+eps) 落 research/cache/t004_scale_ratios.parquet (checkpoint); 门控 regime(MA5/MA60穿越数 p20/p80, 同 T-001/2/3 口径)+run_len(连续MA5上斜天数, causal) 硬选择。耗时 94s。
  - **裁决 built**: 3 尺度模型 + 门控全跑通, 产出 ratio_dyn (191.3 万行/5034 股 落 research/features/dynamic_ratio.parquet, 列名自检+verify leakage guard 双过, 生产指纹未变)。
  - **门控行为印证 T-001/T-002**: 规则取自实证非拟合 — choppy→s3 95% (响应 T-001 choppy ignition 漂移 -5d, 需短窗领先), trend→s5 69%+s10 31% (同步且稳, 偏长窗), mid 混合 (s3 67%/s5 27%); 整体动态切换 58% vs 固定 s5; ratio_dyn 与 s5 相关 0.955 / s10 0.818 (长窗在 trend 拉开差异)。设计哲学: 离散切尺度 = 动态有效窗口的最廉价近似, 与 T-002 "非平稳来自相位非窗口配比" 一致 (不重配特征权重, 只切预测尺度)。
  - 纪律: 选尺度分布/IC/相关全是中间指标, 仅描述非 gate (SIGN-R03); window_dynamic_needed 只解除 T-004 条件门, 不保证带 α — ship 唯由 T-005 walk-forward 决定。生产模型/v12_scoring 全程未碰 (SIGN-R05)。
  - 产出: research/t004_multiscale_gate.py / research/models/pump_scale_{3,5,10}/ / research/cache/t004_scale_ratios.parquet / research/features/dynamic_ratio.parquet / research/cache/t004_results.json / research/verdicts/T-004.json
  - 下一步 (T-005): walk-forward 决策 gate。19 月独立训练/测试, 协议同 project_walk_forward_validation_0525, 同 harness 重跑 V12.31 baseline 做 apples-to-apples 对照; 把 ratio_dyn (或其衍生 pump 信号) 接进 V7c 池内排序对照 V12.31 的 pump_ratio 默认。报告 α(月化)/Sharpe/最差月/正α月占比 vs baseline, 断言 preRegisteredGate 四条件 → PASS/REJECT。**REJECT 是合法完成, 禁止在同段 OOS 反复重调 (SIGN-R02)。** dynamic_ratio.parquet + 三尺度模型可复用。注意: ratio_dyn 与 s5(≈生产 v3c 档) 相关 0.955, 增量空间可能有限 — 须诚实按 walk-forward 数字定。
