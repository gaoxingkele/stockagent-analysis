你已四次审计这个 A 股均值回归系统 V12.31, 帮我们把注水 Sharpe 2.78 剥到 de-lookahead 1.84 → embargo 1.31, 并否了止盈和 triple-barrier。现在做了最后一步: 剔除你定位的唯一硬红线 holder_pct(确认是真前视: stk_holdernumber 按 end_date<=target 选数未取 ann_date; 但 r20 gain 仅 0.005% 排118/236)。请做**第五次 review**, 专审这一个判断, 不改文件。可只读 research/verdicts/DEP-001.json。

## DEP-001 结果
- 唯一改动=r20 训练特征剔 holder_pct(234特征), 余口径(embargo P_start-21/24m/120树/双轨/cap4/20d再平衡/close-based/ST排除)与 WFE-001(1.31)逐字相同, apples-to-apples。
- PIT-clean book: 年化+19.9% 净Sharpe **+0.84** maxDD-26.8% 月胜率55%。
- vs WFE-001(含holder, Sharpe 1.31): **ΔSharpe -0.464**。
- per-cohort 配对 block bootstrap(1000): ΔSharpe 95%CI=**[-1.246,+0.312]** 中位-0.477 **P(Δ<0)=89%** CI含0。
- r20 全期 IC: 剔后+0.0952 vs 含holder+0.0979 (**-0.0027, 几乎没变**)。
- 仍跑赢 hs300(0.46)/CSI1000(0.81)/动量(0.51)。
- loop 自动裁决=**clean确认**(理由: |ΔSharpe| CI含0 → 不材料性 → 红线清除, 基线维持≈1.31)。

## 我(claude)对 loop 这个裁决的质疑
我认为 loop 判 "clean确认" 在统计上(CI含0)站得住, 但话术过乐观: 点估计从1.31掉到0.84, P(变差)89%。机理上 holder IC贡献-0.0027≈0 说明这0.46不是真丢edge而是噪声; 但**一个0.005%重要度的因子剔掉就能晃0.46**, 这恰恰是"20-cohort book Sharpe被噪声主导、1.31根本不该被当精确数字"的最强证据(呼应你#4说的功率不足)。

## 请审三点
1. loop 的 "clean确认" 对吗, 还是应该更诚实地标为 "统计不可区分但点估计偏差、被噪声主导"? 红线到底清没清?
2. "剔0.005%废因子却晃Sharpe 0.46" —— 你同意这是"1.31是噪声点估计不该当数字宣称"的证据吗? 还是说它反而暴露了别的问题(比如book构造/pick边际不稳/holder其实通过非线性/池过滤有隐性影响)?
3. 最终: PIT-clean r20(剔holder)能否作为可部署变体? 可部署期望该怎么表述(1.31? 0.84? 还是"正但宽带")? 下一步是否就是冻结+前向paper-trade, 还是这个0.46的不稳本身就该再下修预期/补什么检验?
