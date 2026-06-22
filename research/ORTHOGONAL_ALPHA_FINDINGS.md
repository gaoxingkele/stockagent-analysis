# 正交因子 campaign 发现 (Track C2, 2026-06-18)

> 用户开闸广搜正交正-alpha。每族 build→Phase-1 廉价正交IC闸(扣[动量+size+value+roe/margin]残差, NW-t HAC, ≥40月, 分regime)→ 存活进 V12.31 池 19月 walk-forward GATE + skeptic(bootstrap/PBO/DSR)。
> codex 对抗审计纳入冻结gate (NW-t / DSR含DoF乘子3 / selection-on-selection caveat)。生产线 V12.31 全程冻结。

**多重检验基数**: 筛过 24 因子变体, DSR n_trials = 24×3(DoF) = 72。

## Phase-1 廉价闸结果

| 族 | 最强因子 | 裁决 |
|---|---|---|
| OA-VOL 波动率 | ivol_60 orth_ic=-0.0811 | PROCEED |
| OA-VP 量价 | intraday_ret_20 orth_ic=-0.0484 | PROCEED |
| OA-COM 商品供应链 | com_betavol_60 orth_ic=-0.0708 | REJECT |
| OA-RC 分析师修正 | — | deferred (无 report_rc 权限) |
| OA-NEWS 新闻政策 | — | deferred (仅市场级 major_news) |

## walk-forward GATE (存活因子 vs V12.31)

| 候选 | Δα/月 | gate | 裁决 |
|---|---|---|---|
| ivol_60 (OA-VOL-GATE) | -0.1535pp | regime {'mixed': 0.184, 'momentum': -0.093, 'reversal': -0.263} | **REJECT** |
| on_minus_in_20 (OA-VP-GATE) | -0.702pp | regime {'mixed': -1.089, 'momentum': -0.584, 'reversal': -0.707} | **REJECT** |

## 元结论

正交因子 campaign 净结果 = 无可落地新 alpha。两个真异象 (低波动 ivol_60 IC t=-6.69, 隔夜-日内 on_minus_in_20 IC t=8.04) 过 Phase-1 廉价正交闸 (正交于 mom/size/value/quality), 但接 V12.31 池 19月 walk-forward 双双 REJECT (Δα=-0.15/-0.70pp, PBO 0.8 过拟合) — V12.31 价量均值回归核已捕获 (低波动经 pyr_velocity, 隔夜经 pump/ratio)。商品供应链渠道 Phase-1 塌缩 (beta 被 size/industry 吸收)。分析师修正 (理论最强) + 新闻政策因数据闸关闭 deferred。再次印证 0605 元结论: 正交≠有α; 即时可拉信号的可交易部分=V12.31 已吃。第22-23个被否假设。

## 真异象但不可落地 (为什么)
- **低波动 ivol_60** (Ang-Hodrick-Xing-Zhang): orth IC −0.081 t=−6.69 强, 但 V12.31 的 pyr_velocity 过滤已隐含低波动偏好 → Δα −0.15pp, PBO 0.8 过拟合。
- **隔夜-日内分解 on_minus_in_20** (Lou-Polk-Skouras): orth IC +0.048 t=8.04 (残差后增强), 但隔夜动量已被 V12.31 pump/ratio 启动子捕获 → Δα −0.70pp 全regime负。
- **商品供应链 beta**: raw IC ~0.02-0.03 但被 size/industry 完全吸收 (Phase-1 C_塌缩), 商品 beta 无独立横截面 alpha。

## 下一步 (条件触发)
- report_rc 权限开通 → 分析师修正动量 (codex 排第1真正交 prior, 唯一未测的强候选)
- 个股级 news/anns_d 权限 或 投 LLM 实体抽取管线 → 新闻/政策横截面因子
- 微结构/盘口/Level-2 数据 (本 token 不含) → 真正脱离价量空间的正交信息

## 1H vs 日线对比

1H 日内已实现波动 (rv_1h_60 orth IC=-0.0549) vs 日线 ivol_60 (-0.0650) 在同 28 月 1H 覆盖窗对比: 日线更强 (差<0.005=相当)。Spearman 0.869 高(同序=换皮, 1H无新增信息)。低波动信号在两尺度一致 (符号同负)。SIGN-R03: 仅IC对比, 落地仍以 GATE walk-forward 为准。