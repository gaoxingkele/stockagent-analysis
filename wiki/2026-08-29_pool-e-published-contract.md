# 池 E 跨项目发布契约 — 只消费完整快照，失败时守住上一份良好清单

- **日期**: 2026-08-29
- **provenance**: user-revised（明确上游三层保障与稳定 CLI）/ ai-executed（下游契约校验、缓存与 Web 信号日透传）
- **裁决**: ✅落地
- **关联代码/实验**: `daily_dashboard.py`、`tests/test_pool_e_contract.py`、`web/app/services/v12_service.py`、`web/templates/v12.html`；上游 `stock_benchmark/scripts/export_daily_top100_list.py` 与 `docs/operations/daily_pool_e_pipeline.md`

## 一句话

跨项目股票池不能把“最新 CSV 文件存在”当成数据有效；池 E 现在只通过上游稳定 JSON 契约读取权威发布快照，在消费者侧复验完整性，并在任何失败下继续使用带真实 `signal_date` 的上一份已知良好清单。

## 出处（必填，无出处不写）

- **Pact 官方文档 — Contract Testing / Consumer Driven Contracts**: https://docs.pact.io/ 。核心思想是把应用间消息视为共享契约，并让消费者只验证自己实际依赖的行为。
- **The Open Group Base Specifications — `rename()`**: https://pubs.opengroup.org/onlinepubs/9799919799/functions/rename.html 。原子替换发布指针的系统语义来源；同一文件系统内，成功替换不会向读取方暴露中间态。
- **Python 官方文档 — `os.replace()`**: https://docs.python.org/3/library/os.html#os.replace 。上游原子发布在 Python 文件系统接口中的对应实现参考。
- **本项目上游契约**: `D:\aicoding\stock_benchmark\docs\operations\daily_pool_e_pipeline.md`；稳定入口为 `scripts/export_daily_top100_list.py --group strategy --format json`。

## 为什么引入（第一性原理）

池 E 的选股来自另一个项目。旧做法直接定位 `daily_top100_unique_latest.csv`，等于让下游重复猜测“哪个文件最新、它是否完整、manifest 是否与 CSV 同批”。即使生产者已经实现原子发布，下游绕过正式入口直接读文件，仍会重新制造三类风险：

1. 读到文件与发布指针不一致的组合；
2. 上游目录或文件命名变化时静默失效；
3. 数据源停更时，把看板日期误当作信号日期。

真正需要稳定的不是某个路径，而是一个明确承诺：给定 `strategy` 组，返回一份经过发布校验的完整快照；失败则不发布半成品。消费者再验证自己依赖的最小字段，双方才能独立演进。

## 核心思想（讲直觉，不堆公式）

这套接入分为三道门：

1. **生产者发布门**：上游先生成 dated 产物，验证 100 只唯一股票、H5/H10/H20 配额 30/35/35、15 策略 × 100 行，然后才原子替换 `latest_manifest.json` 和 latest 别名。失败时权威指针不动。
2. **稳定接口门**：下游不再自行拼文件路径，而是调用 `export_daily_top100_list.py --group strategy --format json`。导出器先读权威 manifest，再返回它指向的 strategy 清单；strategy 与 SEMAS 始终独立，不融合。
3. **消费者验收门**：StockAgent 再检查 `result_group=strategy`、有效 `signal_date`、100 个唯一合法代码、完整排名 1–100、配额 30/35/35、15 个策略及逐策略权重。任何一项不满足，都不覆盖本地缓存。

这不是重复造一遍上游校验。上游保证“我发布的是完整快照”，下游保证“这份快照满足我真正依赖的契约”。两侧错误域不同，必须各守一层。

## 我们怎么吸纳 / 改造

- **原版**：池 E 直接读取外部 latest CSV，再按排名取前 N。
- **现在**：通过稳定 CLI 获取 JSON；`POOL_E_EXPORT_SCRIPT` 环境变量允许换机时改路径而不改代码。
- **上游 100 → 本项目 30**：上游契约保留完整 100 只清单；池 E 沿用 `POOL_E_TOPN=30`，按 `overall_rank` 取前 30 只进入本项目 V12 打分。这个截断属于消费者策略，不属于上游发布契约。
- **双缓存**：`config/pool_e.json` 保存最近一次通过验收的代码，`config/pool_e_meta.json` 保存其 `signal_date`、发布时间、策略数、权重数和配额。
- **按看板日固化元数据**：每次生成看板时写 `output/daily_pick/dashboard_<date>/pool_E_meta.json`，避免日后查看历史看板时错误套用当前最新信号日。
- **Web 诚实展示**：池 E 面板显示“信号日 + 策略数”。看板日期是本项目评分日，信号日是上游最近成功发布日，两者允许不同。
- **流水线验收**：`generate_pool_e_only()` 跑完上游每日管线后，不再只相信退出码或状态文件，而是立即通过同一稳定 JSON 接口验收；接口不合格就视为失败，并由读取层回退。

## 结果与裁决

- 当前真实接口实测：`signal_date=20260828`、`unique_count=100`、15 个策略、15 份权重、配额 30/35/35。
- StockAgent 成功按排名读取前 30 只；当时前三只为 `002886.SZ`、`601996.SH`、`002482.SZ`。
- 新增 5 项消费者侧契约测试，全部通过：正常快照、重复代码、错误配额、缺策略、稳定 CLI 调用参数。
- `daily_dashboard.py` 与 Web 服务编译检查通过，`git diff --check` 通过。

**裁决：PASS。** 这里的“保证”只覆盖接口稳定、完整快照发布、消费者验收和最近良好版本回退，不保证每日一定产生新信号，也不保证清单内容不变化。若 Tushare 或上游计算失败，接口返回最近一次成功发布，`signal_date` 必须原样展示，不能拿本项目看板日期冒充。

## 思想谱系（演化）

- 取代了：跨项目直接读取“看起来最新”的 CSV；只用进程退出码判断数据成功。
- 同源 / 对照：[[2026-06-24_anti-overfitting-scaffold]]——这里同样是 gate 优先，只有通过契约门的数据才能进入评分。
- 下一步：若其他外部池也跨项目消费，应复用“稳定导出接口 + 消费者契约 + 信号日固化”的结构，不复制池 E 的具体字段。

## 移植提示（必填）

把这套模式移植到别的项目时：

1. 先定义一个稳定的机器接口，不让消费者扫描目录猜 latest；接口必须返回版本/信号时间。
2. 生产者采用“临时 dated 产物 → 全量校验 → 原子切换发布指针”，不要逐个覆盖消费者正在读取的文件。
3. 消费者只验证自己依赖的最小契约，但要覆盖数量、唯一性、排序键、分组标识和时间戳。
4. 缓存代码与元数据必须绑定；只有代码没有 `signal_date` 的回退是不诚实的。
5. 区分“源信号日”和“本地加工日”，尤其在周末、停牌日、数据源故障和补算场景。

本项目特有的部分是 strategy 组 100 只、30/35/35 配额、15 策略以及池 E 只取前 30；移植时应替换这些业务常量，但保留发布与验收结构。
