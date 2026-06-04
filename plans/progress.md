# Progress — research/meihua-traj

> fresh-context 每轮干净重启, 这里是跨迭代的唯一记忆 (SIGN-R09)。每轮结束前更新。

## 北极星
用户批评单日起卦贫乏(代码=静态ID/日期=当日常数/价格末位=噪声), 提出把'近N日走势'做进梅花。
问题升级为: **梅花对走势的数论分桶能否超过/补充标准技术指标对同一走势的编码**。按顺序检验三方案
**B(体用=趋势vs当日) → A(轨迹数字起卦) → C(累积卦+序列特征)**, 三组对比, **升级 gate**:
残差必须扣除"公历月×板块 **+ 标准动量/趋势因子**"后仍存在。先验=仍偏怀疑但**这次是真问题非结构性必死**
(轨迹是个股特异+随时间变, 有信息可编码)。**生产线 V12.31 冻结。**

## 事前注册 gate (冻结, R01 + 分regime R11 + 升级消融 R12+)
- Phase1 筛查: 残差(扣 月×板块 + 动量因子) |IC|>=0.01 且 |t|>=3 → residual_signal。
- Phase2 (有残差才跑): 19月 walk-forward Δα>=+0.30pp 且 Sharpe/最差月不降 且 消融存活。
- **三组全 no_residual → REJECT (轨迹梅花亦无独立于动量信号), 三组对比本身是交付物。**

## 任务台账 (按顺序 B→A→C→对比→落地)
| id | 方案 | 状态 | 裁决 |
|----|------|------|------|
| MT-001 | B 体用=趋势vs当日 | done | residual_signal (体卦=MA堆叠换皮, 延MT-004) |
| MT-002 | A 轨迹数字起卦 | done | no_residual (数字起卦=动量换皮, 扣后无残差) |
| MT-003 | C 累积卦+序列特征 | done | residual_signal (残差=move_std≈波动率换皮, 延MT-004) |
| MT-004 | 三组对比 + 决策 gate | todo | — |
| MT-005 | opt-in 落地 (依赖 MT-004=PASS) | skip | — |

## 已有资产 / 复用
- 编码器 research/meihua_encoder.py (单日版已建, 本轮扩展轨迹起卦)
- 梅花核心 vendored from ds-oracle-cli; load_daily 给 close/high/low 路径
- RG-001 regime_timeline.parquet (分层); 标准动量因子在 factor_lab / load_window 里 (消融对照用)
- mh002_phase1_screen.py 的 OOF 编码+残差化框架可复用升级 (这次消融多加动量因子)
- t005 walk-forward harness (Phase2 用)

## 关键约束
- 负结果=合法完成, 禁同段OOS重调 (R02); 中间指标(IC)≠落地只认walk-forward α (R03)
- IC/回测/特征前先跑 leakage guard (R04); 生产hash冻结 (R05); ST源头排除 (R06)
- R11 按regime分层; **R12+ 轨迹梅花消融必须扣 月×板块 AND 动量因子 (否则只是动量换皮=自欺)**

## 迭代日志
<!-- 每轮 append: 迭代N | task | 做了什么 | 裁决/产出路径 | 下一步 -->
- (init) 由 meihua 循环 (单日版已 REJECT) 转入。用户批评单日贫乏, 要 B/A/C 轨迹三方案对比。等待启动。
- 迭代1 | MT-001 (方案B) | 扩展 meihua_encoder.py 加 cast_tiyong_B/build_tiyong_B_lookup (体=上卦=多尺度MA堆叠3bit趋势, 用=下卦=当日K结构3bit, 动爻锁下卦1..3 ⇒ 体用生克=今日冲击vs既有趋势)。写 research/mt001_traj_B.py: 全历史5.09M行因果落 features/meihua_traj_B.parquet (仅key+10个mhB_, 零泄漏); 升级Phase-1 消融= resid1(扣月×板块OOF) → resid2(逐日横截面OLS再扣mom_5/mom_20/ma_pos/rsi_14)。**裁决=residual_signal**: 最强 mhB_upper resid2-IC=+0.0214 t=10.32, 三regime均存活(mom t3.93/mix t4.49/rev t8.84)。
  - **关键诚实预警 (SIGN-R12)**: 残差全集中在 体卦族(upper/base/changed/ti_elem)=多尺度MA堆叠三元状态换皮。仅存活**线性**动量残差化; MA堆叠是非线性/多尺度趋势, resid2-IC 极可能就是"非线性趋势超出线性动量"的世俗TA结果, **非占卜**。不作落地依据 (SIGN-R03)。
  - **给 MT-004 的硬约束**: apples-to-apples baseline **必须含 MA堆叠/非线性趋势对照** (如直接用 ti_tri 三元状态或GBDT喂动量因子), 否则会把"非线性趋势"误归功于卦象 = 自欺。
  - 产出: research/features/meihua_traj_B.parquet, research/cache/mt001_panel.parquet(含r20+动量,不进features), research/cache/mt001_results.json, research/verdicts/MT-001.json。verify.py 无违规(无泄漏+生产指纹一致), 剩 MT-002/003/004 未裁决=正常。
  - 下一步: MT-002 (方案A 轨迹数字起卦)。
- 迭代2 | MT-002 (方案A) | 扩展 meihua_encoder.py 加 cast_traj_A/build_traj_A_lookup (经典数字起卦用在走势: 近20日日涨跌幅×100取整=量化整数, 分窗求和→上卦=较早半窗Σ%8 / 下卦=较近半窗Σ%8, 动爻=近20日振幅最大日窗内序号%6, sliding_window argmax 向量化因果)。写 research/mt002_traj_A.py: 全历史5.19M行因果落 features/meihua_traj_A.parquet (仅key+10个mhA_, 零泄漏); 同 MT-001 升级 Phase-1 消融 (resid1 扣月×板块OOF → resid2 逐日横截面扣 mom_5/20/ma_pos/rsi)。**裁决=no_residual**: 最强动态 mhA_moving resid2-IC=-0.0037 t=-2.53, 未过 |IC|>=0.01 且 |t|>=3.0 双闸。原始/OOF IC 几乎全 |t|<3 (唯 mhA_moving OOF t=-2.99 边缘, resid2 后衰减到 t=-2.53)。
  - 解读: 方案A 上/下卦本质是"分窗累积涨跌幅 %8"的余数分桶, 与累积动量 (mom_10/20) 同源; 动爻=振幅最大日序号是路径位置非占卜。扣掉线性动量后无残差 ⇒ **数字起卦只是动量换皮** (SIGN-R12 消融未存活)。与方案B 对比鲜明: B 体卦=多尺度MA堆叠的非线性趋势状态, 仍存活线性动量残差化 (resid2-IC +0.0214 t10.32); A 纯余数分桶则被线性动量吃干净。
  - 产出: research/features/meihua_traj_A.parquet, research/cache/mt002_panel.parquet(含r20+动量,不进features), research/cache/mt002_results.json, research/verdicts/MT-002.json。verify.py 无违规(无泄漏+生产指纹一致), 剩 MT-003/004 未裁决=正常。
  - 给 MT-004 的输入: B=residual_signal(但=MA换皮预警) / A=no_residual。三组对比中 A 是廉价 REJECT 项。
  - 下一步: MT-003 (方案C 累积卦+序列特征)。
- 迭代3 | MT-003 (方案C) | 扩展 meihua_encoder.py 加 cum_gua_from_parity/build_traj_C_lookup (累积卦: 每爻 bit=窗内被翻次数奇偶 parity, 种子=坤; 当日动爻驱动累积本卦→变卦) + build_wuxing_sign_lookup (逐日单日卦体用生克带符号 朝体: 用生体+2/体克用+1/比和0/体生用-1/用克体-2)。写 research/mt003_traj_C.py: 每日动爻 m_t=abs(round(ret*100))%6 逐日翻 base 第m_t爻, 6爻 rolling 奇偶向量化; 序列特征=动爻漂移(近半−早半rolling均值)/动爻std/五行净生克(逐日signed Σ rolling)。全历史5.19M行因果落 features/meihua_traj_C.parquet (仅key+10个mhC_, 零泄漏)。**筛查口径修正**: 名义卦象(5个cat)用OOF target encoding, 连续序列标量(5个cont)用原始rank-IC (OOF对连续值退化)。**裁决=residual_signal**: 最强 mhC_move_std resid2-IC=+0.0463 t=10.87, 三regime均存活(mom t8.52/mix t3.15/rev t6.93); 次强 mhC_wuxing_net +0.0244 t7.87, mhC_wuxing_net_recent +0.0230 t9.03。
  - **关键诚实预警 (SIGN-R12)**: 残差最强落在 mhC_move_std。因动爻 m_t=abs(round(ret*100))%6 是当日涨跌幅幅度的桶, mhC_move_std='动爻位置离散度'≈**近N日已实现波动率的换皮代理** (与累积动量正交 = 正是它存活线性动量残差化的原因)。残差化只扣 mom_5/20/ma_pos/rsi 四个**线性动量/趋势**因子, **未含波动率因子也未含MA堆叠/非线性趋势**。故 resid2-IC 极可能是'波动率+非线性趋势'这些标准TA量超出线性动量残差化的世俗结果, **非占卜**。不作落地依据 (SIGN-R03)。
  - **给 MT-004 的硬约束 (与 MT-001 同类)**: apples-to-apples baseline **必须额外含 已实现波动率 (std(ret_20)/ATR) + MA堆叠/非线性趋势对照**, 否则把波动率/非线性趋势误归功卦象=自欺。
  - 产出: research/features/meihua_traj_C.parquet, research/cache/mt003_panel.parquet(含r20+动量,不进features), research/cache/mt003_results.json, research/verdicts/MT-003.json。verify.py 无违规(无泄漏+生产指纹一致), 剩 MT-004 未裁决=正常。
  - **三组 B/A/C 齐备**: B=residual_signal(MA堆叠换皮) / A=no_residual / C=residual_signal(move_std≈波动率换皮)。两个 residual 都是世俗TA(趋势/波动率)换皮预警, 非占卜独立信号。下一步 MT-004: 三组对比 + 决策 gate (含 MA堆叠+波动率 apples-to-apples baseline)。
