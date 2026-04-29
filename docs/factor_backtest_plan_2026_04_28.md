# 全因子回测方案（纯量化, 日线 + 周线）

> 日期: 2026-04-28
> 范围: 所有未经过严格回测的批次 1-14 因子, 共 ~95 个独立因子
> 频率: 日线 + 周线两套
> 工程: 5 进程并行, 三层 checkpoint, parquet 存储

---

## 一、动机

经过 1 年 / 3 年 / 924 段三轮回测, 我们已对 v3.1 + 04-26 增强的 8 个因子有结论:
- ✅ holder_pct (跨风格稳健)
- ✅ winner_rate (熊市强)
- ⚠️ adx (风格依赖, 牛市正向)
- ❌ mf_divergence/mf_strength/mf_consecutive/market_score_adj (失效)

但 v1-v2 的批次 1-14 中有大量因子从未做过严格 IC 回测:
- 12 Agent 时代的 trend_momentum / divergence / chanlun 等仅在 146 股 / 54k 样本上回测过 (2026-04-05)
- K 线 12 种形态仅有汇总 IC = -0.029, 各形态单独 IC 未知
- 相对强弱 (rs_industry / rs_leaders / rs_etf) 完全没回测
- 基本面增强后的多维评分没回测
- 批次 8 的 ATR / Fibonacci / 仲裁三方案没单独回测
- 批次 11 的 channel_reversal 8 阶段状态机各阶段单独 IC 未做

加上 Tushare Pro / Qlib / TA-Lib 还有大量未引入的因子, 应该一次性补齐。

**关键约束**: 只用纯计算, 不用 LLM。

---

## 二、未回测因子清单（按 Group 分类）

### Group A：技术指标基础（日 + 周）— 约 18 个

MACD / MA 排列(5/10/20/60/120) / MA 交叉 / RSI(6/12/24) / KDJ / BOLL 轨道位 / 乖离率(5/10/20) / ATR / Fibonacci / ROC / MFI / CCI / WR / TRIX / 等

### Group B：量价因子（日 + 周）— 约 10 个

成交量 vs 量均 / OBV / 量价配合(量增价涨等 5 种) / 换手率 / 振幅 / 量比 / VR / 资金流向

### Group C：K 线形态（日 + 周）— 12 种细分

锤子 / 倒锤子 / 十字星 / 长十字 / 看涨吞没 / 看跌吞没 / 启明星 / 黄昏星 / 三只乌鸦 / 三个白兵 / 上吊线 / 流星
+ 位置权重(顶/底) + 形态连续性

### Group D：趋势线因子（日 + 周）— 约 6 个

上升通道 / 下降通道 / 趋势线突破确认 / 颈线突破 / 通道阶段(channel_reversal 8 阶段单独 IC)

### Group E：缠论因子（日 + 周）⚠️ 最复杂 — 约 8 个

笔方向 / 笔大小 / 段方向 / 段大小 / 中枢位置 / 中枢强弱 / 一买信号 / 二买 / 三买 / 笔背离 / 段背离

### Group F：相对强弱（日 + 周）— 约 6 个

个股 vs 行业(90/250d) / 个股 vs 龙头 TOP3 / 个股 vs 行业 ETF / 行业 vs 大盘 / 板块内排名

### Group G：基本面（季度更新）⚠️ 需新数据 — 约 12 个

PE_TTM 分位 / PB 分位 / ROE / 资产负债率 / 营收 YoY / 净利润 YoY / 营收 QoQ / 净利润 QoQ / 毛利率 / 净利率 / 经营现金流 / 三年复合增长

### Group H：筹码进阶 — 约 5 个

cyq_chips 集中度 / 筹码方差 / 成本分位 / 获利盘 5d 变化 / 获利盘趋势

### Group I：市场环境（横截面）— 约 5 个

所属行业当日涨幅排名 / 板块热度 / 北向持仓变化% / 上证 60d 动量 / 创业板 vs 上证比

### Group J：事件因子 — 约 8 个

突破前高 / 突破前低 / 缺口 / 涨停板首板 / 连板 / 振幅缩量 / 巨量长阴 / 巨量长阳

**合计: ~95 个独立因子 × 2 频率 = 约 190 个因子样本**

---

## 三、可行性细分

| Group | 数据需求 | 实现难度 | 周线可行性 | 现有代码复用 |
|-------|--------|---------|----------|------------|
| A 技术指标 | ✅ 已有 (daily + stk_factor_pro) | 🟢 简单 | ✅ 本地 resample | 有 |
| B 量价 | ✅ 已有 | 🟢 简单 | ✅ | 有 |
| C K 线形态 | ✅ 已有 | 🟡 中等 | ✅ | 部分有 |
| D 趋势线 | ✅ 已有 | 🟡 中等 | ✅ | 部分有 |
| E 缠论 | ✅ 已有 | 🔴 复杂 (要重写) | ⚠️ 慢 | 散落难抽 |
| F 相对强弱 | ⚠️ 要建行业 ETF 映射 | 🟡 中等 | ✅ | 有 |
| G 基本面 | ❌ 要新拉 fina_indicator | 🟡 中等 | N/A 季度 | 有 |
| H 筹码进阶 | ✅ 已有 (cyq_chips) | 🟢 简单 | N/A | 部分有 |
| I 市场环境 | ⚠️ 要拉指数日线 | 🟢 简单 | ✅ | 有 |
| J 事件因子 | ✅ 已有 | 🟡 中等 | ✅ | 部分有 |

---

## 四、周线信号方案

**周线 = 日线 resample**, 纯本地计算, 不打 Tushare API:

```python
def daily_to_weekly(daily_df):
    daily_df['week'] = pd.to_datetime(daily_df['trade_date']).dt.to_period('W')
    weekly = daily_df.groupby('week').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'vol': 'sum'
    })
    return weekly
```

3 年回测期 ≈ 159 周, 5149 股共 81.8 万行周线数据。

**周线相对日线特征**:
- 噪声小, 信号慢, 长期 IC 通常更强
- 适合 D+20/30/40 中长期持有期

---

## 五、设计优化

### 5.1 数据组织 — Parquet 列式

```
output/factor_lab/
├── data/
│   ├── daily_{ts_code}.parquet
│   └── weekly_{ts_code}.parquet
├── factors/
│   ├── ma5_daily.parquet         # 列: ts_code, date, factor, r5..r40
│   ├── ma5_weekly.parquet
│   └── ...
├── ic_reports/
│   └── ma5_daily.json
└── status.json
```

Parquet 比 JSON 快 100x, 压缩 4-10x, pandas 直读。

### 5.2 因子并行 (不是股票并行)

```
Worker 1: 25 个技术指标因子 → 全部 5149 股
Worker 2: 22 个量价/形态因子 → 全部 5149 股
Worker 3: 12 个趋势线/相对强弱 → 全部 5149 股
Worker 4: 22 个基本面/筹码/环境 → 全部 5149 股
Worker 5: 14 个事件因子 → 全部 5149 股
```

因子相互独立 → 完美并行。单因子算完立即出 IC。

### 5.3 缠论分阶段

Group E 缠论太复杂 (1500 行代码), 单独最后做。

### 5.4 单股 cache + 增量

每只股票算完所有因子 → 存 `factor_lab/cache/{ts_code}.parquet`。
新增因子时读 cache → 算新因子 → 追加列。

避免重复计算共享中间变量 (MA, RSI 等)。

---

## 六、Checkpoint 三层设计

### 第 1 层: 股票级原始数据缓存
```
data/daily_{ts_code}.parquet
data/weekly_{ts_code}.parquet
```

### 第 2 层: 因子级输出
```
factors/{factor_name}_{freq}.parquet
ic_reports/{factor_name}_{freq}.json
```

### 第 3 层: 状态总表
```json
// status.json
{
  "ma5_daily": {"status": "done", "ic_5d": 0.012, "samples": 4080000},
  "rsi6_daily": {"status": "running", "progress": "1500/5149"},
  "chanlun_pen_weekly": {"status": "pending"}
}
```

中断恢复: 扫 status.json 找未完成的因子继续。

---

## 七、时间和资源预估

| 阶段 | 数据需求 | 预估时间 |
|------|--------|---------|
| 1. 数据准备 (resample 周线) | 已有 | 10 分钟 |
| 2. 拉 fina_indicator | 新拉 | 35 分钟 (后台并行) |
| 3. 拉指数日线 | 新拉 | 5 分钟 |
| 4. Group A-D + F + H + I + J | 已有 | 60-90 分钟 (5 worker 并行) |
| 5. Group G 基本面计算 | 步骤 2 完成后 | 10-15 分钟 |
| 6. Group E 缠论 | 已有但要重写 | 4-6 小时 (含编码) |
| 7. IC 聚合 + 报告 | 各因子 parquet | 20 分钟 |

**总计**:
- 不含缠论: 约 **2-3 小时**
- 含缠论: 约 **6-9 小时**

---

## 八、推荐执行顺序

```
Stage 1 (最快出结果, ~1.5 小时):
  Group A + B + C + D + H + J (~50 因子 × 2 频率)
  完全用现有数据, 本地纯计算

Stage 2 (并行 1.5 小时):
  Group F (建行业 ETF 映射) + Group I (拉指数)
  Group G (拉 fina_indicator + 算)

Stage 3 (专项 4-6 小时):
  Group E 缠论 (最后做, 含代码重写)
```

---

## 九、Tushare Pro 额外可测因子

### 9.1 已使用接口
- `daily` (OHLC), `daily_basic` (mv/PE/PB), `stk_factor_pro` (60+ 技术因子), `cyq_perf` (winner_rate), `cyq_chips` (筹码集中度), `moneyflow` / `moneyflow_dc` / `moneyflow_mkt_dc` (主力资金), `stk_holdernumber` (股东户数), `index_daily` (指数), `fund_basic` / `fund_nav` / `fund_portfolio` (ETF)

### 9.2 未使用、可挖掘的关键接口（按可信度+独特性排序）

#### 🔥 P1 强信号接口（必测）

| 接口 | 数据 | 可衍生因子 | 推荐度 |
|------|------|----------|------|
| `forecast` | 业绩预告（预增/预减/扭亏）| 预告幅度 / 预告类型 dummy / 预告距今天数 | ⭐⭐⭐ |
| `express` | 业绩快报 | 营收/净利润 vs 前期 / vs 一致预期 | ⭐⭐⭐ |
| `top_list` + `top_inst` | 龙虎榜 + 机构席位 | 机构净买入金额 / 机构净买入比 / 游资 vs 机构 | ⭐⭐⭐ |
| `block_trade` | 大宗交易 | 大宗折价/溢价比 / 近 30d 大宗成交占比 | ⭐⭐ |
| `repurchase` | 股票回购 | 回购公告距今天数 / 回购金额占市值比 | ⭐⭐ |
| `share_float` | 限售解禁 | 解禁日距今天数 / 解禁规模占流通比 | ⭐⭐ |
| `hsgt_top10` | 沪深股通 TOP10 | 北向 TOP10 dummy（被纳入即强信号）| ⭐⭐ |
| `hk_hold` | 港股通持股 | 北向持仓比例 / 北向持仓变化 | ⭐⭐⭐ |

#### 🟡 P2 中等可用（可选测）

| 接口 | 数据 | 可衍生因子 |
|------|------|----------|
| `broker_recommend` | 券商推荐 | 推荐数 / 推荐评级中位数 |
| `pledge_stat` | 股权质押 | 大股东质押比例 |
| `dividend` | 分红送股 | 股息率 / 派息频率 / 送转比 |
| `top10_holders` / `top10_floatholders` | 前十大股东 | 第一大股东持股变化 / 机构股东数 |
| `limit_list` | 涨跌停 | 涨停板首板 dummy / 连板天数 |
| `managers` | 高管 | 高管增减持金额 |
| `mainland_holdings` | 陆股通持股 | 同 hk_hold |

#### 🟢 P3 数据接口（用于横截面）

| 接口 | 用途 |
|------|------|
| `index_dailybasic` | 指数 PE / PB / 总市值（用于宽基板块对比）|
| `sw_member` / `sw_index` | 申万行业成分 / 申万行业日线 |
| `concept` / `concept_detail` | 概念板块成分（题材分类）|
| `fina_indicator` | 财务指标（已计划 Group G）|

### 9.3 新增 Group K：Tushare 增强事件因子

新建 Group K, 覆盖 P1 接口:

| 因子 ID | 来源 | 计算 |
|---------|------|------|
| forecast_grade | forecast | 预告类型 (预增 +3 / 略增 +1 / 预减 -1 / 大降 -3) |
| forecast_pct | forecast | 净利润预告幅度均值 |
| express_yoy_npg | express | 净利润同比 |
| lhb_inst_net | top_inst | 机构净买入(万元) / 流通市值 |
| lhb_seat_concentration | top_inst | 前 5 席位占比（机构主导度）|
| block_trade_premium | block_trade | 近 30d 大宗折价/溢价均值 |
| repurchase_signal | repurchase | 回购公告距今天数（90d 内 = 1）|
| float_release_pressure | share_float | 30d 内解禁规模 / 流通市值 |
| hk_hold_ratio | hk_hold | 北向持仓占流通股本比 |
| hk_hold_change_30d | hk_hold | 北向持股 30d 变化 |

→ Group K **新增 10 个因子**，全部需要新拉 Tushare 数据。

---

## 十、Qlib 因子库适配（基于 Alpha158 设计）

### 10.1 Qlib Alpha158 因子设计

Qlib 是微软开源量化框架，Alpha158 因子集 = **基础 KBAR (9) + 滚动算子 (29) × 多周期 (5/10/20/30/60)** 组合。

### 10.2 KBAR 类（9 个，全部独特, 必测）

| 因子 | 公式 | 含义 |
|------|------|------|
| **KMID** | (close-open)/open | 实体相对开盘比 |
| **KLEN** | (high-low)/open | K 线全长比 |
| **KMID2** | (close-open)/(high-low+1e-12) | 实体占整体比 |
| **KUP** | (high-max(open,close))/open | 上影线 |
| **KUP2** | (high-max(open,close))/(high-low+1e-12) | 上影线占整体 |
| **KLOW** | (min(open,close)-low)/open | 下影线 |
| **KLOW2** | (min(open,close)-low)/(high-low+1e-12) | 下影线占整体 |
| **KSFT** | (2×close-high-low)/open | 收盘相对中位偏移 |
| **KSFT2** | (2×close-high-low)/(high-low+1e-12) | 同上归一化 |

### 10.3 ROLL 算子（29 种 × 多周期 5/10/20/30/60）

#### 价格类（已部分有, 选独特的）

| 算子 | 含义 | 已有? |
|------|------|------|
| ROC(N) | N 日变化率 | ✅ stk_factor_pro |
| MA(N) | N 日均线 | ✅ |
| STD(N) | N 日标准差 | ⚠️ 部分 |
| MAX/MIN(N) | N 日最高/最低 | ✅ rolling_max |
| **QTLU(N)** | N 日 80% 分位 | ❌ 新 |
| **QTLD(N)** | N 日 20% 分位 | ❌ 新 |
| **RANK(N)** | 今天 close 在 N 日内排名分位 | ❌ 新 |
| **RSV(N)** | (close-min)/(max-min) | ⚠️ 类似 KDJ 的 RSV |
| **IMAX(N)** | N 日最高出现位置（距今）| ❌ 新, 重要 |
| **IMIN(N)** | N 日最低出现位置 | ❌ 新, 重要 |
| **IMXD(N)** | IMAX-IMIN | ❌ 新, 趋势特征 |

#### 量价相关性（独特视角，必测）

| 算子 | 含义 |
|------|------|
| **CORR(close, vol, N)** | N 日量价相关性 |
| **CORD(return, vol, N)** | N 日收益率与量的相关性 |

#### 涨跌天数计数（A 股特化）

| 算子 | 含义 |
|------|------|
| **CNTP(N)** | N 日内涨天数 |
| **CNTN(N)** | N 日内跌天数 |
| **CNTD(N)** | CNTP-CNTN |

#### 累计涨跌幅

| 算子 | 含义 |
|------|------|
| **SUMP(N)** | N 日内正涨幅累计 |
| **SUMN(N)** | N 日内负跌幅累计 |
| **SUMD(N)** | SUMP-SUMN（净涨幅）|

#### 量类

| 算子 | 含义 |
|------|------|
| **VMA(N)** | N 日均量 |
| **VSTD(N)** | N 日量标准差 |
| **WVMA(N)** | 加权量 MA（按价格变化加权）|
| **VSUMP/VSUMN/VSUMD(N)** | 涨/跌/差日的量累计 |

#### 回归类（个股 vs 市场，对应相对强弱）

| 算子 | 含义 |
|------|------|
| **BETA(N)** | 个股 vs 上证 N 日 beta |
| **RSQR(N)** | R²（拟合优度）|
| **RESI(N)** | 残差（特异收益）|

### 10.4 Qlib 新增因子总数

精选 ~30 个独特因子 × 5 个周期 (5/10/20/30/60) = **~150 个 Qlib 因子样本**

但实际去重后, 取核心 ~25 个算子 × 3 个周期 (10/20/60) ≈ **75 个 Qlib 因子**。

新增 **Group L: Qlib 因子集**

---

## 十一、TA-Lib 因子适配

### 11.1 TA-Lib 全功能（158 函数）

| 类别 | 数量 | 说明 |
|------|------|------|
| Cycle Indicators | 5 | 希尔伯特变换周期指标 (新颖) |
| Math Operators / Transform | 26 | 数学辅助 (不是因子) |
| Momentum Indicators | 30 | 动量指标 |
| Overlap Studies | 17 | 均线/通道 |
| **Pattern Recognition** | **61** | K 线形态 (大量补充) |
| Price Transform | 4 | 价格变换 |
| Statistic Functions | 9 | 统计 |
| Volatility Indicators | 3 | 波动率 |
| Volume Indicators | 3 | 量能 |

### 11.2 与现有 stk_factor_pro 去重

stk_factor_pro 已有: macd / kdj / rsi / boll / dmi / atr / cci / wr / mtm / obv / mfi / trix / asi / bbi / bias / brar / cr / dfma / dpo / ema / ma / expma / ktn / mass / psy / roc / taq / vr / xsii (TD)

### 11.3 TA-Lib 独有/补充的关键因子

#### 🔥 高价值新增（17 个）

| 函数 | 含义 | 价值 |
|------|------|------|
| **AROON / AROONOSC** | Aroon 指标 (反映高低点位置时间) | ⭐⭐⭐ A 股反转特征 |
| **KAMA** | Kaufman 自适应均线 | ⭐⭐⭐ 自适应避免假突破 |
| **MAMA** | Mesa 自适应均线 | ⭐⭐⭐ 同上 |
| **SAR / SAREXT** | 抛物线指标 | ⭐⭐⭐ 经典反转信号 |
| **ULTOSC** | 终极震荡器 | ⭐⭐ 多周期合成 |
| **CMO** | Chande Momentum | ⭐⭐ 动量改进 |
| **BOP** | Balance of Power | ⭐⭐ 多空力量 |
| **PPO / APO** | 价格百分比/绝对振荡 | ⭐⭐ MACD 改进 |
| **AD / ADOSC** | 累积派发 / Chaikin | ⭐⭐ 量价独立信号 |
| **NATR** | 归一化 ATR | ⭐⭐ ATR 跨股可比 |
| **STOCHRSI** | RSI 的 STOCH | ⭐⭐ 超买超卖增强 |
| **HT_DCPERIOD / HT_TRENDMODE** | 希尔伯特变换 | ⭐ 趋势/震荡判别 |

#### 🟡 中等价值（统计回归类）

| 函数 | 含义 |
|------|------|
| LINEARREG / LINEARREG_SLOPE / LINEARREG_ANGLE | 线性回归（趋势量化）|
| BETA / CORREL | 个股 vs 市场（同 Qlib BETA）|
| TSF | 时间序列预测值 |
| STDDEV / VAR | 标准差/方差 |

#### 🔥 K 线形态（61 种，全部新增！）

我们之前手写 12 种, TA-Lib 提供 **61 种**, 包括:
- 已有: CDLDOJI(十字), CDLENGULFING(吞没), CDLHAMMER(锤), CDLINVERTEDHAMMER(倒锤), CDLEVENINGSTAR(黄昏星), CDLMORNINGSTAR(启明星), CDL3WHITESOLDIERS(三白兵), CDL3BLACKCROWS(三乌鸦), CDLHANGINGMAN(上吊), CDLSHOOTINGSTAR(流星)
- **新增 51 种**: CDL2CROWS, CDL3INSIDE, CDL3LINESTRIKE, CDL3OUTSIDE, CDLABANDONEDBABY(弃婴), CDLADVANCEBLOCK(大敌当前), CDLBELTHOLD(捉腰带), CDLBREAKAWAY(脱离), CDLCONCEALBABYSWALL(藏婴吞没), CDLCOUNTERATTACK(反击), CDLDARKCLOUDCOVER(乌云盖顶), CDLDOJISTAR(十字星), CDLDRAGONFLYDOJI(蜻蜓), CDLEVENINGDOJISTAR(暮星), CDLGAPSIDESIDEWHITE(并列白线), CDLGRAVESTONEDOJI(墓碑), CDLHARAMI(母子), CDLHARAMICROSS(母子十字), CDLHIGHWAVE(长影线), CDLHIKKAKE(陷阱), CDLHOMINGPIGEON(归巢鸽), CDLIDENTICAL3CROWS, CDLINNECK(颈内), CDLKICKING(反冲), CDLLADDERBOTTOM(梯底), CDLLONGLEGGEDDOJI(长腿十字), CDLLONGLINE(长线), CDLMARUBOZU(光头光脚), CDLMATCHINGLOW(相同低价), CDLMATHOLD(铺垫), CDLMORNINGDOJISTAR(晨星十字), CDLONNECK(颈上), CDLPIERCING(刺透), CDLRICKSHAWMAN(人力车), CDLRISEFALL3METHODS(上升下降三法), CDLSEPARATINGLINES(分离线), CDLSHORTLINE(短线), CDLSPINNINGTOP(纺锤), CDLSTALLEDPATTERN(停顿), CDLSTICKSANDWICH(条形三明治), CDLTAKURI(探水竿), CDLTASUKIGAP(跳空并列阴阳), CDLTHRUSTING(插入), CDLTRISTAR(三星), CDLUNIQUE3RIVER(奇特三河床), CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS

→ Group C K 线形态从 12 种扩到 **61 种**

### 11.4 TA-Lib 新增因子总数

- 高价值动量/通道/统计: ~17 个
- 中等价值统计回归: ~5 个
- K 线形态: 61 个 (含已有 10 种和新增 51 种, 全部用 TA-Lib 重写)

新增 **Group M: TA-Lib 增强因子集**, 共约 **22 个非 K 线因子 + 51 个新 K 线形态**

---

## 十二、最终因子总清单（去重后）

### Group 总览

| Group | 描述 | 因子数 | 数据源 | 优先级 |
|-------|------|------|------|------|
| A | 技术指标基础 (日 + 周) | 18 | 已有 | P1 |
| B | 量价 (日 + 周) | 10 | 已有 | P1 |
| **C** | **K 线形态 (TA-Lib 61 种 + 位置 + 连续)** | **63** | 已有 | P1 |
| D | 趋势线 + channel_reversal 8 阶段 | 6 | 已有 | P1 |
| E | 缠论 (单独, 最后做) | 8 | 已有 | P3 |
| F | 相对强弱 | 6 | 部分新拉 | P2 |
| G | 基本面 | 12 | 新拉 fina_indicator | P2 |
| H | 筹码进阶 | 5 | 已有 cyq_chips | P1 |
| I | 市场环境 (横截面) | 5 | 新拉指数 | P2 |
| J | 事件因子 (突破/缺口/涨停) | 8 | 已有 | P1 |
| **K** | **Tushare 事件接口 (新增)** | **10** | 新拉 | P2 |
| **L** | **Qlib KBAR + ROLL** | **75** | 已有 | P1 |
| **M** | **TA-Lib 动量/通道/统计** | **22** | 已有 | P1 |

**合计**: ~250 个独立因子 × 2 频率 (日/周) = **~500 个因子样本**

### 数据 vs 计算 vs 工程 难度对比

| 类型 | 数据 | 计算 | 工程 |
|------|-----|-----|-----|
| Group A/B/C/D/H/J | ✅ 全有 | 🟢 公式 | 🟢 框架已建 |
| Group L (Qlib) | ✅ 全有 | 🟢 简单算子 | 🟡 算子库 |
| Group M (TA-Lib) | ✅ 全有 | 🟢 一行 talib.X | 🟢 现成 API |
| Group F | ⚠️ 行业 ETF 映射 | 🟡 中等 | 🟡 |
| Group I | ⚠️ 拉指数 | 🟢 简单 | 🟢 |
| Group G | ⚠️ 拉财报 | 🟡 中等 | 🟡 |
| Group K | ⚠️ 拉事件接口 | 🟡 中等 | 🟡 |
| Group E | ✅ 已有 | 🔴 复杂 | 🔴 重写 |

---

## 十三、调整后的执行计划

### Stage 1（最快出结果, ~3 小时）— 已有数据 + 简单计算

并行 5 worker:
- Worker 1: Group A (18) + Group L 价格类 (15)
- Worker 2: Group B (10) + Group L 量类 (10)
- Worker 3: **Group C 61 种 K 线形态** (TA-Lib)
- Worker 4: Group L 其余 (50) + Group M (22)
- Worker 5: Group D (6) + Group H (5) + Group J (8)

总因子: ~205 个 × 2 频率 = **410 个因子样本**

### Stage 2（并行 1.5 小时）— 需新数据

后台拉数据 + 算因子:
- 拉 fina_indicator → Group G 算 (12)
- 拉指数日线 → Group I 算 (5)
- 建行业 ETF 映射 → Group F 算 (6)
- 拉事件接口（forecast/express/top_list/hk_hold 等）→ Group K 算 (10)

新增: 33 个因子 × (日 + 周) = 66 个

### Stage 3（专项 4-6 小时）— 缠论

Group E (8) × 2 频率 = 16 个

### 最终产出

- 全部因子: **~250 个**
- 全部样本: **~500 个 (因子 × 频率)**
- IC 报告: 单因子粒度 + 跨风格 + 跨市值/PE 分层

---

## 十四、Renko 砖头线 与 Heikin-Ashi 鬼线 适配性分析

### 14.1 Renko 砖头线分析

**核心设计**: 完全抛弃时间维度, 只看价格变动 (≥box size 画砖)。本质是"趋势跟随"工具。

**4 个致命问题** (在我们体系下):

1. ❌ **时间不一致**: 不同股票每天产生砖头数不同, 与 D+N 回测框架冲突 (大量缺失值)
2. ❌ **A 股反转特性反向**: 我们已验证牛市 D+40 胜率 41.3% (持有越久越亏), ADX 强趋势是顶部信号。Renko 的"连续 10 块绿砖" 在 A 股恰恰是反转信号
3. ❌ **box size 参数敏感**: 茅台 1500 元 vs 普通 5 元, 绝对值不可比, 百分比也敏感
4. ❌ **丢失 OHLCV**: 抛弃成交量/缺口/振幅, 我们已验证量价信号有效

**结论**: 不作为主框架, 但可衍生少量辅助因子。

### 14.2 Heikin-Ashi 平均 K 线分析

**核心设计**: 每根 K 线受前根影响, 平滑波动:
```
HA_Open  = (前根 HA_Open + 前根 HA_Close) / 2
HA_Close = (本根 Open + High + Low + Close) / 4
HA_High  = max(本根 High, HA_Open, HA_Close)
HA_Low   = min(本根 Low,  HA_Open, HA_Close)
```

**优势** (vs Renko):
- ✅ **保留时间维度**, 完全兼容 D+N 框架
- ✅ **作为对照实验**: 把 Group C 的 61 种 TA-Lib 形态 + Group A 技术指标在 HA 上重跑一遍
- ✅ **实现简单**: 4 行公式生成 HA OHLC

**警告**:
- ⚠️ HA_Close 不是真实价格, 仅作信号不作成交价
- ⚠️ A 股反转特性可能让 HA "连续同色"信号失效, 需要回测验证

### 14.3 三者对比

| 维度 | Renko | Heikin-Ashi | 原始 K |
|------|-------|------------|-------|
| 保留时间维度 | ❌ | ✅ | ✅ |
| 兼容 D+N 框架 | ❌ | ✅ | ✅ |
| 价格真实 | ⚠️ 砖头价 | ⚠️ 平滑价 | ✅ |
| 保留 OHLC | ❌ | ⚠️ 平滑 | ✅ |
| 保留 量/缺口 | ❌ | ❌ | ✅ |
| 噪声过滤 | 强 | 中 | 弱 |
| A 股适配 | 不适配 | 待验证 | 已验证 |
| 工程成本 | 高 | 低 | 已建 |
| **优先级** | 🔴 P3 | 🟡 P2 对照 | 🟢 P1 主体 |

### 14.4 Group N: Renko 衍生因子 (5-8 个, P3)

| 因子 | 含义 |
|------|------|
| renko_count_30d | 近 30 交易日产生砖头数 (波动强度替代) |
| renko_direction | 当前砖头方向 (绿/红) |
| renko_streak | 当前同色连续砖头数 |
| renko_reversal_count_60d | 60 天内反转次数 |

box size 取 ATR_14 的倍数 (自适应, 跨股可比)。

### 14.5 Group HA: Heikin-Ashi 对照因子 (30-40 个, P2)

#### HA 上的 TA-Lib 61 种 K 线形态
直接把 HA OHLC 喂给 talib.CDL* 系列, 输出 61 个对照因子。

#### HA 上的 Group A 技术指标
HA 上的 MA / MACD / RSI / KDJ / ATR 等, 共 ~15 个对照因子。

#### HA 独有因子
- ha_streak_up: 当前连续 HA 阳线数
- ha_streak_down: 当前连续 HA 阴线数
- ha_body_ratio: HA 实体长度 / HA 全长
- ha_trend_strength: 近 N 天 HA 阳线占比

### 14.6 关键验证目标

1. **HA 上的"连续同色 K 线"因子 IC 是正是负?**
   - 正 → HA 在 A 股有效, 趋势信号成立
   - 负 → HA 也是反转信号, 跟原始 K 没本质差异

2. **HA 形态 vs 原始 K 形态的 IC 对比**
   - HA 显著更强 → 形态识别受日内噪声严重干扰
   - 接近 → 原始 K 信息已足够
   - HA 弱 → HA 的平滑反而丢失关键信息

3. **Renko 砖头数量 vs ATR 等波动指标的 IC 对比**
   - 验证"价格事件密度"是否比"时间维度波动"更有效

### 14.7 最终因子总数 (含 Renko + HA)

| Group | 因子数 |
|-------|------|
| A-M (前述) | ~250 |
| **N Renko 衍生** | **5-8** |
| **HA Heikin-Ashi 对照** | **30-40** |

**最终合计 ~290 个因子 × 2 频率 = ~580 个因子样本**

### 14.8 执行调整

新增 **Stage 4 (HA/Renko 对照实验, ~1 小时)**, 在 Stage 1-3 完成后做:
- HA 因子: 复用 Group C / Group A 算子, 喂入 HA OHLC
- Renko 因子: 单独模块, 用 ATR 倍数自适应 box size

不影响主路径优先级。


## 十二、关键决策点

| 项 | 推荐值 | 备注 |
|---|------|------|
| Group E 缠论 | 包含, 最后做 | 工作量最大 |
| 基本面 | 包含 | 要拉 fina_indicator (35 min) |
| 周线 K | 本地 resample | 不打 API |
| 数据格式 | Parquet | 比 jsonl 快 100x |
| 老数据 | 保留 | 新数据存 output/factor_lab/ |
