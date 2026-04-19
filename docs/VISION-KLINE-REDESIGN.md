# K 走势专家 · 多模态视觉升级方案

**作者:** gaoxingkele + Claude
**日期:** 2026-04-19
**分支:** feat/llm-role-refactor
**相关 commit 系列:** `f0a6cf7` (首版) → `959dbc1` (纯文本重跑) → 本方案(真视觉)

---

## 背景

### 问题 1: K 走势专家系统性偏严

93 只 v3 批处理发现 K 走势专家(StructureExpert)评分显著偏低:
- 平均 52.6 vs 其他专家 65+
- 最高仅 75, 从未"偏多"异议(16 次异议全是偏空)
- 原 prompt 要求"多周期完美共振"才给 80+, A 股日常基本不可能达到

### 问题 2: 首次视觉升级实际走了文本降级

`f0a6cf7` 提交虽然加了视觉模式, 但 `logs/rerun_structure_expert.py` 加载已有 v3 run_dir 时:
- 旧 run_dir 没有 `data/kline/*.csv` (那时还没 kline 复用机制)
- `generate_expert_charts` 找不到 K 线数据 → 静默降级到纯文本
- **51/52 只实际走了文本模式**, 只有 1 只(000876 新希望今天手动单测时)真用视觉

评分提升 +15 分完全来自 prompt 改写 + FOMC 公式, 并非真视觉分析。

### 问题 3: 单图多指标叠加可能降低 LLM 识别精度

首版绘图把 K + MA5/10/20/60 + 布林带 + Ichimoku 云 + SAR + 成交量**全部叠在主图上**,
虽然一眼看全局, 但:
- 颜色/线条密集, LLM 可能抓不全指标
- Ichimoku 云在 K 线密集区难看清
- SAR 散点被其他元素覆盖

---

## 最终方案 · 6 子图分层 + "主+副"配对

### 核心理念

1. **不要极端拆分**(每指标一张图共 14 张), 保留交叉关系
2. **也不要全叠一张**, 避免信息密度过高
3. **按"研判主题"分组**, 每张子图 = 一个完整判断单元(主图+对应副图)
4. **日线在左列, 周线在右列**, 让 LLM 横向对比多周期

### 6 子图结构(一张大图 2 列 × 3 行)

```
┌─────────────────────────────────┬─────────────────────────────────┐
│ 日线 · 主题 1 · 趋势与动能       │ 周线 · 主题 1 · 趋势与动能       │
│   主图: K + MA5/10/20/60 +       │   主图: K + MA5/10/20/60 +       │
│         布林带 + 成交量          │         布林带 + 成交量          │
│   副图: MACD (DIF/DEA/柱)        │   副图: MACD (DIF/DEA/柱)        │
├─────────────────────────────────┼─────────────────────────────────┤
│ 日线 · 主题 2 · 云图与强度       │ 周线 · 主题 2 · 云图与强度       │
│   主图: K + Ichimoku 云 +        │   主图: K + Ichimoku 云 +        │
│         转换线 + 基准线          │         转换线 + 基准线          │
│   副图: RSI(14) + 超买超卖区     │   副图: RSI(14) + 超买超卖区     │
├─────────────────────────────────┼─────────────────────────────────┤
│ 日线 · 主题 3 · 反转与综合       │ 周线 · 主题 3 · 反转与综合       │
│   主图: K + SAR + 自动趋势线     │   主图: K + SAR + 自动趋势线     │
│   副图: MACD 柱 + RSI 叠加       │   副图: MACD 柱 + RSI 叠加       │
└─────────────────────────────────┴─────────────────────────────────┘

尺寸: 宽 18 × 高 21 英寸, dpi=110, ~500-700 KB
```

### 配对设计思路

| 主题 | 主图 | 副图 | 为什么这样配 |
|------|------|------|-------------|
| 1 | K + MA + 布林 + 成交量 | MACD | 趋势主体 ↔ 趋势动能, 金叉死叉与均线位置协同判断 |
| 2 | K + Ichimoku 云 | RSI | 云图位置(多/空/震荡) ↔ 动量强度(超买/超卖) |
| 3 | K + SAR + 趋势线 | MACD 柱 + RSI | 反转信号(SAR 翻转, 趋势线突破) ↔ 双动量确认 |

**指标出现次数:**
- MACD 出现 2 次(主题 1、3) → 动能信号强化
- RSI 出现 2 次(主题 2、3) → 强度信号强化
- K 线出现 3 次(每张都有) → 形态+突破是核心

### 每张子图的布局比例

```
┌────────────────┐
│                │
│  主图 (65%)    │  ← K 线 + 指标叠加
│                │
├────────────────┤
│  副图 (30%)    │  ← 对应动量指标
└────────────────┘
    共享 X 轴 (日期)
```

### Prompt 核心: 日线服从周线

```
判决优先级(关键!):

1. 周线向上 + 日线回调 → 日线回调属"调整" → 偏多打分 60-75
   (不要因日线短期弱就给 <50)
2. 周线向下 + 日线反弹 → 日线反弹属"反抽" → 偏空打分 35-50
   (不要因日线强势就给 >65)
3. 周线/日线方向一致 → 按一致方向给分, 可到 80+
4. 周线横盘 + 日线有方向 → 按日线给分但降置信度
```

---

## 技术实现

### 文件改动

| 文件 | 改动 |
|------|------|
| `src/stockagent_analysis/agents_v3/kline_charts.py` | 重构, 6 子图合成一张大图 |
| `src/stockagent_analysis/prompts_v3/expert_structure.txt` | 增加布局说明 + 日周优先级 |
| `logs/rerun_structure_expert.py` | 增加 kline csv 复用逻辑(读 stock_pool_94.txt 里的 v2 run_dir) |

### 绘图主函数(新签名)

```python
def generate_merged_expert_chart(
    run_dir: Path,
    symbol: str, name: str,
    daily_n: int = 100, weekly_n: int = 80,
) -> Path:
    """生成单张 2×3 六子图大图,直接 matplotlib 一次出图。"""
    fig = plt.figure(figsize=(18, 21), dpi=110)
    gs = fig.add_gridspec(
        nrows=6, ncols=2,
        height_ratios=[3, 1.2, 3, 1.2, 3, 1.2],
        hspace=0.18, wspace=0.08
    )
    # 加载日周 K 线
    df_day = _load_csv(run_dir / "data/kline/day.csv", daily_n)
    df_week = _load_csv(run_dir / "data/kline/week.csv", weekly_n)

    # 日线左列 (列索引 0) · 3 主题
    _plot_theme1(fig, gs, 0, df_day, "日线")
    _plot_theme2(fig, gs, 0, df_day, "日线")
    _plot_theme3(fig, gs, 0, df_day, "日线")
    # 周线右列 (列索引 1) · 3 主题
    _plot_theme1(fig, gs, 1, df_week, "周线")
    _plot_theme2(fig, gs, 1, df_week, "周线")
    _plot_theme3(fig, gs, 1, df_week, "周线")

    out = run_dir / "charts/expert/merged.png"
    fig.savefig(out, ...)
    return out
```

### Rerun 脚本修复

```python
# 从 stock_pool_94.txt 建立 code → v2_run_dir 映射
v2_map = {}
for ln in pool_file.read_text(encoding="utf-8").splitlines():
    code, name, v2dir = ln.split("|", 2)
    v2_map[code] = v2dir.strip()

# 对每只高分股票
for code, name, v3_dir in gte60_list:
    # 1. 复制 kline csv 到 v3 run_dir
    v2_kline = Path(v2_map[code]) / "data/kline"
    dst = Path(v3_dir) / "data/kline"
    if v2_kline.exists() and not dst.exists():
        shutil.copytree(v2_kline, dst)
    # 2. 跑 StructureExpert(这次会有图)
    ...
```

---

## 验收标准

### 功能验证

- [ ] 52 只重跑完成, 每只生成 `charts/expert/merged.png` (~500-700 KB)
- [ ] 每张图含 6 个子图, 布局符合设计
- [ ] StructureExpert 返回的 key_data 里 `_chart_path` 字段 100% 有值
- [ ] JSON 里的 `analysis` 字段引用图像观察(如"日线突破布林上轨, 周线 MA20 下方")

### 质量验证

- [ ] K 走势平均从 74.6 → 新值, 期待小幅变化 ±5 分(prompt 已改, 视觉主要影响分析质量, 不是打分绝对值)
- [ ] `analysis` 字段长度 ≥ 100 字且有具体指标引用
- [ ] 日周冲突时 Judge 采纳周线方向的比例(从 JSON 抽样人工审核)

### 工程验证

- [ ] 单只耗时 10-20s (含绘图+vision 调用)
- [ ] 总耗时 10-15 分钟
- [ ] 失败率 < 5%
- [ ] 新 PDF 生成成功 (~430 KB)

---

## 历史复盘要点

### 已识别的陷阱

1. **静默降级问题**: StructureExpert 视觉失败会静默降级到文本模式, 日志不明显。
   → 修复: 代码里加 `INFO` 级日志区分 "真视觉 vs 文本降级", 每只跑完后打印模式。

2. **kline 数据复用遗漏**: reuse_context_from 只复用 analysis_context.json, kline csv 没跟上。
   → 修复: orchestrator_v3 已加 kline 复用, 但历史 run_dir 没有, 需 rerun 脚本补。

3. **CRLF 行尾导致 bash read**: 之前 94 只批处理因 stock_pool 文件 CRLF 全部失败。
   → 修复: Python 生成文件时用 POSIX 路径 + LF 行尾。

### 本次方案已避免的坑

- [x] 不再静默降级: rerun 脚本要显式复制 kline
- [x] 不再过度拆分: 6 子图而非 14 子图, 保留指标交叉关系
- [x] 不再信息过密: 每子图只 1 主题, 不全叠
- [x] 日周优先级: 在 prompt 层强化, 不改图

---

## 未来可扩展方向

1. **WaveExpert 也用视觉版**(参考此方案, 波浪理论在图像识别上更吃视觉)
2. **多 provider 交叉**: grok + gemini 各看一次图, 融合判断
3. **图标识别增强**: 在图上标注关键位/突破点, 减少 LLM 识别负担
4. **月线补充**: 超长线判断(暂不做, 月线数据 60 根够用)

## 参考资料

- 原 PlotDriver 参考: `D:\aicoding\stock_selector_agents\chan\Plot\PlotDriver.py`
- 首版视觉实现: commit `f0a6cf7`
- FOMC 点阵图模式: commit `4db5187`
- 当前基线 PDF: `output/portfolio_summary_vision_231028.pdf`
