你是一名资深量化研究审计员。请对下面这个 A 股日频选股系统的研究方法和最新结论做**独立交叉审计**,重点找盲点和过拟合风险,并就下一步给出你的优先级建议。请只做分析与讨论,不要修改任何文件。可以只读查看 repo（research/verdicts/*.json, research/backtest/, plans/prd.json）来核实。

## 系统
V12.31: 均值回归策略。V7c 池(6 铁律)→ 池内按 ratio=P_up/(P_down+ε) 排序 → 行业 cap 4 → Top N。核心是两个 LGBM: r20 池排序模型 + pump 三分类启动子(前向5日 max_gain≥10% & max_dd≥-5%)。买 past_r5<0 的回调(抄底)。

## 反过拟合脚手架(每个假设都过)
预注册冻结 gate(R01); 负结果=合法完成、禁同 OOS 重调(R02); 中间指标≠落地、只认 walk-forward(R03); 前视泄漏前置闸(R04); 生产文件 hash 冻结(R05); ST 源头排除(R06); 按 regime 分层(R11); 升级消融扣[动量+市场+size+value+形态](R12); 单月 outlier 检验; combinatorial 钓鱼用预注册单表示 + Deflated Sharpe/PBO 防。

## 已证伪 19 个假设
价量编码 8 否(全归约为标准 TA)、正交信息 4 否(成长/内部人/概念/事件,或动量镜像或被 price-in)、label/形态 3 否、三原型 sleeve 2 否、duokongK 多空K线 1 否、MA 事件序列 meta-label(BT-004)1 否。唯一扛过消融的真信号=龙虎榜席位印记,但 long-only 不可落地(相对信号活在输家池+微容量+需做空)。

## 最新 book 级回测器发现(关键)
- BT-003 leave-one-out 归因: **r20 池=几乎全部 book α**(去掉 ΔSharpe -1.37); pump_down 过滤完全惰性(Δ0); 行业动量排除铁律拖后腿(去掉 +0.10 Sharpe); 成本非杀手(滑点 0→0.2% 仅 -0.12 Sharpe)。
- BT-002: 动量月是相对最弱 regime(我们小盘成长在动量行情吃亏)。
- EX-001 出场策略(同 picks 受控Δ): **分批止盈 TP-only ΔSharpe +1.358**(年化让出13pp换 maxDD降5.6pp); 回撤止损基本中性(SL-8 -0.07/SL-12 +0.25)。
- EX-002: 与基金的差距**主导来自风格**(我们小盘高PE成长 市值百分位32% vs 基金大盘质量90%),不是选股。

## 最大 caveat(必须纳入你的判断)
所有绝对 Sharpe/年化(如 book +2.4~2.78)都被 **r20 池的共模 lookahead + 集中度抬高**注水,**不是可交易数字**。只有**受控 Δ**(同 picks/两臂相消)和**相对结论**可信。真实可交易 P&L 需要把 r20 做成真 walk-forward 去 lookahead 才知道。

## 请回答三个问题
1. 我们这套反过拟合方法有什么**盲点**?19 连否是否暗示方法本身有系统性问题(比如标签/universe/评测口径锁死了结论)?
2. EX-001 的"分批止盈 +1.36 Sharpe"在**绝对量级被 lookahead 注水**的前提下,**这个 Δ 可信吗**?会不会是 lookahead 或路径依赖的 artifact?如何干净验证?
3. **下一步最高价值动作**是哪个,为什么: (a) 先 de-lookahead 拿 r20 真实 walk-forward P&L; (b) EX-003 r20 triple-barrier 重标; (c) 风格 tilt(往大盘质量靠); (d) 止盈做成正式 walk-forward 验证; (e) 其他你想到的。请排序并说明。
