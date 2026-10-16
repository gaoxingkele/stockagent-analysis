# 证据图

> 由 `metarsi.py` 从 `metarsi.db` 生成，每次写入后重写。不要手改；改记录请用命令。

```mermaid
graph TD
    W0["dev_20250303_20260126<br/>consumed"]
    W0 --> F1["in-list-filter/S<br/>已比较 133 条"]
    W0 --> F2["in-list-filter/R<br/>已比较 11 条"]
    W0 --> F3["s20-model-audit<br/>已比较 4 条"]
    W0 --> F4["s20-model<br/>已比较 2 条"]
    W5["confirm_20260127_20260805<br/>consumed"]
    W5 --> F6["in-list-filter/S<br/>已比较 165 条"]
    W5 --> F7["in-list-filter/R<br/>已比较 24 条"]
    W5 --> F8["list-turnover/S<br/>已比较 2 条"]
    W5 --> F9["s20-model-audit<br/>已比较 1 条"]
    W5 --> F10["s20-model<br/>已比较 3 条"]
    W5 --> F11["model-routing<br/>已比较 5 条"]
    W5 --> F12["day-gate<br/>已比较 61 条"]
    W5 --> F13["pump-model<br/>已比较 4 条"]
    W5 --> F14["mean-reversion<br/>已比较 3 条"]
    W5 --> F15["intraday<br/>已比较 32 条"]
    W5 --> F16["external-factors<br/>已比较 42 条"]
    W5 --> F17["risk-flag<br/>已比较 1 条"]
    W5 --> F18["list-combination<br/>已比较 10 条"]
    W5 --> F19["s20-label<br/>已比较 16 条"]
    W5 --> F20["execution<br/>已比较 36 条"]
    W5 --> F21["exit<br/>已比较 9 条"]
    W5 --> F22["dual-screen<br/>已比较 9 条"]
    W5 --> F23["bottom-pattern<br/>已比较 2529 条"]
    W5 --> F24["in-list-filter<br/>已比较 3282 条"]
    W25["reserved_20260806_plus<br/>reserved"]
    F1 --> T26["T0001 atr10/S<br/>in_sample pass t=+4.58"]
    F6 --> T27["T0002 atr10/S<br/>holdout_once fail t=+0.87"]
    F2 --> T28["T0003 atr10/R<br/>in_sample pass t=+4.95"]
    F7 --> T29["T0004 atr10/R<br/>holdout_once fail t=+0.14"]
    F6 --> T30["T0005 rank_atr10/S<br/>in_sample inconclusive t=+2.00"]
    F7 --> T31["T0006 rank_atr10/R<br/>in_sample inconclusive t=+2.33"]
    F6 --> T32["T0007 r5s5_cross_model10/S<br/>reused_holdout fail t=-1.09"]
    F7 --> T33["T0008 r5s5_cross_model10/R<br/>reused_holdout fail t=+0.79"]
    F6 --> T34["T0009 up10_positive_screen/S<br/>reused_holdout fail t=-0.73"]
    F7 --> T35["T0010 up10_positive_screen/R<br/>reused_holdout fail t=-0.21"]
    F6 --> T36["T0011 top50_to_20_atr/S<br/>reused_holdout fail t=-1.51"]
    F7 --> T37["T0012 top50_to_20_atr/R<br/>reused_holdout fail t=-0.17"]
    F8 --> T38["T0013 hold_zone_top50/S<br/>holdout_once inconclusive t=+0.31"]
    F8 --> T39["T0014 swap_when_dropped_past5…<br/>reused_holdout inconclusive t=-0.71"]
    F1 --> T40["T0015 frozen_natr_cut_vs_raw_…<br/>in_sample pass t=+5.80"]
    F6 --> T41["T0016 frozen_natr_cut_vs_raw_…<br/>reused_holdout fail t=-0.43"]
    F6 --> T42["T0017 second_cut_low_natr_50t…<br/>reused_holdout fail t=-1.08"]
    F6 --> T43["T0018 second_cut_low_turnover…<br/>reused_holdout fail t=-0.98"]
    F6 --> T44["T0019 second_cut_main_money_o…<br/>reused_holdout fail t=-2.16"]
    F6 --> T45["T0020 frozen_first10_places/S<br/>in_sample inconclusive t=+1.94"]
    F6 --> T46["T0021 frozen_stayers_vs_new/S<br/>in_sample inconclusive t=+1.67"]
    F6 --> T47["T0022 second_cut_dn7_model_50…<br/>reused_holdout fail t=-1.18"]
    F1 --> T48["T0023 ma20_rising_first50to20…<br/>reused_holdout fail t=-2.49"]
    F6 --> T49["T0024 ma20_rising_first50to20…<br/>reused_holdout fail t=+0.95"]
    F6 --> T50["T0025 ma20_and_ma99_rising_fi…<br/>in_sample inconclusive t=+2.52"]
    F6 --> T51["T0026 macd_golden_age_11_20_f…<br/>reused_holdout fail t=-0.30"]
    F3 --> T52["T0028 frozen_model_vs_fold_mo…<br/>holdout_once inconclusive"]
    F3 --> T53["T0029 frozen_model_offensive_…<br/>reused_holdout inconclusive"]
    F9 --> T54["T0030 frozen_model_safe_list/…<br/>holdout_once fail"]
    F4 --> T55["T0031 unified_v1_offensive_mi…<br/>holdout_once inconclusive t=+0.72"]
    F10 --> T56["T0032 unified_v1_offensive_mi…<br/>holdout_once fail t=-1.61"]
    F10 --> T57["T0033 unified_v1_safe_minus_f…<br/>reused_holdout inconclusive t=+1.41"]
    F4 --> T58["T0034 unified_v2_offensive_mi…<br/>reused_holdout inconclusive t=+1.71"]
    F10 --> T59["T0035 unified_v2_offensive_mi…<br/>reused_holdout fail t=-1.88"]
    F11 --> T60["T0036 R1_index_up_phase_froze…<br/>holdout_once inconclusive t=+1.01"]
    F11 --> T61["T0037 R2_breadth_ge_50_frozen…<br/>reused_holdout fail t=-0.36"]
    F11 --> T62["T0038 R3_sector_leader_persis…<br/>reused_holdout fail t=-1.01"]
    F12 --> T63["T0039 style_gate_cyb_minus_hs…<br/>holdout_once inconclusive t=+3.82"]
    F12 --> T64["T0040 sector_structure_synerg…<br/>reused_holdout inconclusive t=+1.35"]
    F12 --> T65["T0041 stack_style_gate_plus_c…<br/>reused_holdout inconclusive t=+2.05"]
    F12 --> T66["T0042 trailing_matured_up_sto…<br/>reused_holdout fail t=+0.00"]
    F7 --> T67["T0043 pump_ratio_bands_r20_po…<br/>in_sample inconclusive t=+0.00"]
    F7 --> T68["T0044 s20_pump_up_and_ratio_c…<br/>reused_holdout inconclusive t=+2.45"]
    F7 --> T69["T0045 S0014_pump_up_lowest_th…<br/>reused_holdout inconclusive t=+1.12"]
    F13 --> T70["T0046 pump_v5_shape_and_v5b<br/>holdout_once fail t=+0.00"]
    F14 --> T71["T0047 ma120_oversold_reversion<br/>holdout_once fail t=-4.50"]
    F13 --> T72["T0048 pump_v6_phase_checkpoin…<br/>reused_holdout fail t=+0.00"]
    F15 --> T73["T0049 intraday_16_features_s20<br/>holdout_once inconclusive t=+3.90"]
    F16 --> T74["T0050 semas_21_factors_s20<br/>holdout_once inconclusive t=-4.00"]
    F17 --> T75["T0051 S0016_risk_score_half_s…<br/>holdout_once inconclusive t=+4.50"]
    F18 --> T76["T0052 s20_plus_pool_a_merge<br/>holdout_once fail t=-2.40"]
    F18 --> T77["T0053 nested_A_r20top100_by_s…<br/>in_sample inconclusive t=+0.90"]
    F18 --> T78["T0054 nested_B_s20top100_by_r…<br/>in_sample fail t=-5.70"]
    F19 --> T79["T0055 alt1a_atr_label_top20<br/>in_sample fail t=-1.20"]
    F19 --> T80["T0056 alt1b_atr_label_funnel<br/>in_sample fail t=-0.80"]
    F19 --> T81["T0057 alt2a_exc_label_top20<br/>in_sample fail t=-0.40"]
    F19 --> T82["T0058 alt2b_exc_label_funnel<br/>in_sample fail t=-0.60"]
    F6 --> T83["T0059 alt3_F1_unlock<br/>in_sample inconclusive t=+0.90"]
    F6 --> T84["T0060 alt3_F2_reduce<br/>in_sample fail t=-3.80"]
    F6 --> T85["T0061 alt3_F3_margin<br/>in_sample fail t=-1.50"]
    F6 --> T86["T0062 alt3_F4_toplist<br/>in_sample fail t=+0.20"]
    F20 --> T87["T0063 alt4a_top3_stage1<br/>in_sample inconclusive t=+3.20"]
    F20 --> T88["T0064 alt4a_low_natr3<br/>in_sample inconclusive t=+0.50"]
    F20 --> T89["T0065 alt4b_entry_1000<br/>in_sample fail t=-1.80"]
    F20 --> T90["T0066 alt4b_entry_vwap<br/>in_sample fail t=-2.00"]
    F20 --> T91["T0067 alt4b_skip_gap5<br/>in_sample fail t=-0.90"]
    F20 --> T92["T0068 alt4c_scaled_size<br/>in_sample inconclusive t=+0.70"]
    F21 --> T93["T0069 alt5_x1_time10<br/>in_sample fail t=-0.00"]
    F21 --> T94["T0070 alt5_x2_half10<br/>in_sample fail t=-1.20"]
    F21 --> T95["T0071 alt5_x3_breakeven<br/>in_sample inconclusive t=+0.00"]
    F22 --> T96["T0072 alt6a_pool_then_directi…<br/>in_sample fail t=-2.10"]
    F22 --> T97["T0073 alt6b_natr_band_directi…<br/>in_sample fail t=-1.20"]
    F22 --> T98["T0074 alt6c_ev_mu_sigma<br/>in_sample fail t=-0.80"]
    F23 --> T99["T0075 bottom_B1_s20pool<br/>in_sample fail"]
    F23 --> T100["T0076 bottom_B2c_s20pool<br/>in_sample fail t=+1.00"]
    F23 --> T101["T0077 bottom_B3_s20pool<br/>in_sample fail"]
    F23 --> T102["T0078 bottom_B4_s20pool<br/>in_sample inconclusive t=+1.70"]
    F23 --> T103["T0079 bottom_B2f_s20pool<br/>in_sample fail t=+0.00"]
    F23 --> T104["T0080 bottom_B6_s20pool<br/>in_sample fail"]
    F23 --> T105["T0081 bottom_B7_s20pool<br/>in_sample fail"]
    F23 --> T106["T0082 bottom_B5_s20pool<br/>in_sample inconclusive t=+0.30"]
    F23 --> T107["T0083 bottom_B1_r20pool<br/>in_sample fail t=-1.10"]
    F23 --> T108["T0084 bottom_B2c_r20pool<br/>in_sample fail t=-0.70"]
    F23 --> T109["T0085 bottom_B3_r20pool<br/>in_sample fail"]
    F23 --> T110["T0086 bottom_B4_r20pool<br/>in_sample fail t=-0.10"]
    F23 --> T111["T0087 bottom_B2f_r20pool<br/>in_sample fail t=-3.50"]
    F23 --> T112["T0088 bottom_B6_r20pool<br/>in_sample fail t=-1.10"]
    F23 --> T113["T0089 bottom_B7_r20pool<br/>in_sample fail"]
    F23 --> T114["T0090 bottom_B5_r20pool<br/>in_sample fail t=-1.60"]
    F23 --> T115["T0091 bottom_composite_s20pool<br/>in_sample fail"]
    F23 --> T116["T0092 bottom_composite_r20pool<br/>in_sample fail t=-0.30"]
    F23 --> T117["T0093 wb_base_vs_market<br/>reused_holdout fail t=+2.00"]
    F23 --> T118["T0094 wb_strict_vs_market<br/>reused_holdout fail t=-0.70"]
    F23 --> T119["T0095 wb_top30_vs_market<br/>reused_holdout fail t=+1.70"]
    F23 --> T120["T0096 wb_strict_rank_s20<br/>reused_holdout fail t=+1.40"]
    F23 --> T121["T0097 wb_strict_rank_r20<br/>reused_holdout fail t=+0.30"]
    F23 --> T122["T0098 precursor_any_k20_s20<br/>in_sample fail t=-4.90"]
    F23 --> T123["T0099 precursor_any_k20_r20<br/>in_sample fail"]
    F23 --> T124["T0100 precursor_boll_k5_r20<br/>in_sample inconclusive t=-2.80"]
    F23 --> T125["T0101 precursor_kdj_k20_s20<br/>in_sample fail t=-1.10"]
    F23 --> T126["T0102 precursor_rsi_k20_s20<br/>in_sample fail t=-1.60"]
    F24 --> T127["T0103 ind_boll_pctb_r20<br/>in_sample inconclusive t=+5.20"]
    F24 --> T128["T0104 ind_dev_ma20_r20<br/>in_sample inconclusive t=+4.80"]
    F24 --> T129["T0105 ind_ret10_r20<br/>in_sample inconclusive t=+3.90"]
    F24 --> T130["T0106 ind_strength_family_s20<br/>in_sample fail t=+3.20"]
    F24 --> T131["T0107 zoo_r20_strength_cci<br/>in_sample inconclusive t=+5.80"]
    F24 --> T132["T0108 zoo_s20_kup<br/>in_sample fail t=+2.80"]
    F24 --> T133["T0109 zoo_s20_any<br/>in_sample fail"]
    F7 --> T134["T0110 val_r20_strength_pool<br/>in_sample inconclusive t=+5.20"]
    F7 --> T135["T0111 val_r20_strength_list<br/>in_sample fail t=+2.80"]
    F6 --> T136["T0112 val_s20_strength_list<br/>in_sample fail t=+0.20"]
    S137["S0001 影子<br/>第5日收盘时前5日跌=5%且仍持仓的，不再算作继续…<br/>未结"] -.-> W25
    S138["S0002 影子<br/>每日进攻版Top20里去掉当天跌启动分处于全市场最…<br/>已结 用户 2026-10-06 淘汰：依据只在开发期（ATR/跌启动类否决在确认期 t 0.1~0.9），名额让给风格门；未读保留窗口"] -.-> W25
    S139["S0003 影子<br/>名单20只不减，名单内按名次分位加ATR分位排序<br/>已结 用户 2026-10-06 淘汰：依据最弱（确认期为正、开发期为零、账户口径看不出），名额让给风格门；未读保留窗口"] -.-> W25
    S140["S0004 影子<br/>持仓掉出前50名时换成当天名单最靠前的未持有名字，…<br/>已结 用户 2026-10-05 淘汰：依据最弱（确认期同日配对差 -0.41，t -0.7），名额让给盘面感知规则；未读保留窗口"] -.-> W25
    S141["S0005 影子<br/>S20 进攻版漏斗只改一个参数：stage1 前 …<br/>已结 用户 2026-10-08 结掉，名额让给合成风险分：S0005 为风控型，收益中性（每笔 -0.12），与风险分作用重叠；未读保留窗口"] -.-> W25
    S142["S0007 影子<br/>盘面感知三态路由（信号日收盘后决定当天用哪张名单，…<br/>未结"] -.-> W25
    S143["S0010 影子（主）<br/>风格门：信号日收盘，创业板指(399006.SZ)…<br/>未结"] -.-> W25
    S144["S0012 影子<br/>pump 比值剔除：每个信号日，冻结合约 conf…<br/>已结 用户 2026-10-06 替换为 P(涨) 规则：S20 幸存者上 ratio>5 段 pump 验证期 +3.43、样本外 -3.08 前后反号，对冻结同日差 0.00，126 天仅 35 天起作用（T0044）；只读了信号日 pump 分，未读保留窗口结果"] -.-> W25
    S145["S0014 影子<br/>pump 上涨启动子筛选：每个信号日，冻结合约 c…<br/>未结"] -.-> W25
    S146["S0016 影子<br/>合成风险分（仓位层，名单不变）：每个信号日，在冻结…<br/>未结"] -.-> W25
    N147["N0001 否决<br/>用 R5/S5 模型做正向(5日碰+10%)与负向…"]
    T32 -.-> N147
    T33 -.-> N147
    N148["N0002 否决<br/>按 5 日内碰 +10% 的概率做正向筛选"]
    T34 -.-> N148
    T35 -.-> N148
    N149["N0003 否决<br/>从第 21-50 名(或 21-40 名)里选进 …"]
    T36 -.-> N149
    T37 -.-> N149
    N150["N0004 否决<br/>把「前20去掉ATR最高的10只」用于名单"]
    T26 -.-> N150
    T27 -.-> N150
    T28 -.-> N150
    T29 -.-> N150
    N151["N0005 否决<br/>把名单减到 10 只"]
    N152["N0006 否决<br/>持仓掉出前 20 就卖出换新进的"]
    N153["N0007 否决<br/>把 adx 当模型特征"]
    N154["N0008 否决<br/>在冻结的 S20 进攻版上再剪第二刀（NATR、换…"]
    T42 -.-> N154
    T43 -.-> N154
    T44 -.-> N154
    N155["N0009 否决<br/>按均线方向（MA10/20/30/60/99，单条…"]
    T48 -.-> N155
    T49 -.-> N155
    T50 -.-> N155
    T51 -.-> N155
    N156["N0010 否决<br/>把开发期（2025-03～2026-01）的 S2…"]
    T52 -.-> N156
    N157["N0011 否决<br/>用统一做法 v1（单模型、扩展窗口、固定300棵树…"]
    T56 -.-> N157
    N158["N0012 否决<br/>用统一做法 v2（按日排序 LambdaRank、…"]
    T59 -.-> N158
    N159["N0013 否决<br/>按市场广度（MA20向上占比）或按板块领涨持续性来…"]
    T61 -.-> N159
    T62 -.-> N159
    N160["N0014 否决<br/>名单内按个股所属行业的 20 日强弱或行业 MA2…"]
    N161["N0015 否决<br/>在名单内按个股所属板块是否为当天领涨类别做倾斜（加…"]
    N162["N0016 否决<br/>用板块间协同/抵消（抱团度、成长与周期/防御的20…"]
    N163["N0017 否决<br/>用名单的到位股数/先止损股数比值（ratio2）做…"]
    T66 -.-> N163
    N164["N0018 否决<br/>用 K 线技术/形态特征训练方法 C（ATR 单位…"]
    T70 -.-> N164
    N165["N0019 否决<br/>用超买度（离120日线、RSI24）作为 S20 …"]
    N166["N0020 否决<br/>远低于向上的120日线且RSI超卖（或RSI回升过…"]
    N167["N0021 否决<br/>用 v6（入场后路径+背景模型）给 S20 持仓加…"]
    N168["N0022 否决<br/>分钟级干净度/60分钟RSI/涨跌量比/低点抬高/…"]
    N169["N0023 否决<br/>SEMAS 5 日清单中除以 total_mv 的…"]
    N170["N0024 否决<br/>S20 进攻版与 R20 池 A 合并（并集或资金…"]
    T76 -.-> N170
    N171["N0025 否决<br/>S20 前 100 内改按 R20 r20_pre…"]
    N172["N0026 否决<br/>S20 标签改为 ATR 单位止盈止损（+3ATR…"]
    T79 -.-> N172
    N173["N0027 否决<br/>S20 标签改为相对行业等权指数的超额首达（±10…"]
    T81 -.-> N173
    N174["N0028 否决<br/>方向分数×波动的双重筛选（S20 前100再按方向…"]
    T96 -.-> N174
    N175["N0029 否决<br/>推迟入场（10:00 或次日 VWAP）、高开 5…"]
    T89 -.-> N175
    N176["N0030 否决<br/>出场变体：第 10 日时间止损、+10% 先卖一半…"]
    T93 -.-> N176
    N177["N0031 否决<br/>名单内按减持公告、融资余额急增、龙虎榜剔除"]
    T84 -.-> N177
    N178["N0032 否决<br/>在 S20 / R20 前 100 池内用日线底部…"]
    N179["N0033 否决<br/>用 WorkBuddy/通达信日线双底（W底）规则…"]
    N180["N0034 否决<br/>成功案例之前是否普遍出现双底类形态（W底、缠论背驰…"]
    N181["N0035 否决<br/>在 S20 前100 里用信号日技术指标（RSI、…"]
    N182["N0036 否决<br/>从 TA-Lib 和 Qlib Alpha158 …"]
    N183["N0037 否决<br/>%b/CCI 作为 S20 名单的额外筛选；以及作…"]
```
