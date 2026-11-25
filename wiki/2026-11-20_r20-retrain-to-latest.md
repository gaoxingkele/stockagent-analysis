# 生产 R20 模型训练到最新：用户决定、宪法修订与可行性（2026-11-20）

- **性质**: 决定记录 + 可行性核查。**没有训练出新模型**，没有改 `output/production/*` 和池 A 合约。
- **关联**: [[2026-10-31_naming-registry]]（`R20-M-V16` 现行）、`train_v16_full.py`（仓库根目录）、`scripts/rebuild_r20_labels_from_cache.py`（新）。

## 对话

用户：生产模型能否训练到最新的数据。→ 我答：技术上可以，但会用掉保留窗口，且需你明确决定；并（错误地）说训练脚本不在仓库里。
用户：我建议训练到最新。→ 我答：做不了，且需要你明确说接受保留窗口作废。
用户：你为什么不训练到昨天。→ 我答：缺东西 + 门槛。
用户：训练到最新，接受保留窗口作废，修改宪法，由我负责。

## 已做

1. **宪法修订**（用户明确指示，署用户名）：`.metarsi/constitution.json` 里保留窗口那条不变量末尾加了一句 2026-11-20 的修订说明：生产 R20 模型改为训练到最新已成熟标签日，对保留窗口作废；S20 冻结模型未变，S20 相关影子规则仍按该窗口计分，直到用户另行决定。只改了这一行文字，其它条款、受保护面清单没动。`verify-ledger` 仍是 intact。窗口状态（reserved）暂未改：模型还没真正训练，窗口此刻仍没被用过。
2. **纠正**：我之前说"训练脚本不在仓库里"是错的。`train_v16_full.py` 在仓库根目录（我只查了 `scripts/`），训练 `r20_v16_all_nost` 时 `TRAIN_START=20230101`、`TRAIN_END=20260413`，写入 `output/production/`。
3. **标签重建**：生产标签（entry_open、max_gain_20、max_dd_20、r20_close、is_clean）原本从 `D:/tdx` 读取，本机没有。`scripts/rebuild_r20_labels_from_cache.py` 用日线缓存（未复权原价）按同一定义重算，与现有 `max_gain_labels.parquet` 在 2025-01-02～2026-01-26 的 1,312,433 行逐位一致（四列相关 1.0，差异 0，is_clean 一致率 100%）。已写出 2026-01-27～2026-09-22 的 880,480 行到 `output/labels/max_gain_labels_ext_cache.parquet`（20 日标签最迟只到 09-22，因为日线缓存到 10-29）。训练加载器按 `max_gain_labels_ext*.parquet` 通配读取，会自动收进来。

## 为什么还是没能训练

训练表由 `train_v15_refresh.load_window` 拼成，它需要这些文件，本机缺 2023 年起到 2026-01-26 的主表：

| 输入 | 本机状态 |
|---|---|
| `output/factor_lab_3y/factor_groups` + `_extension` | 有（2023-01-03～2026-10-29） |
| `output/labels/max_gain_labels*.parquet` | 有（本次补到 09-22） |
| `output/amount_features/amount_features.parquet`、`moneyflow/features.parquet`、`pyramid_v2/features.parquet`、`v7_extras/features.parquet`、`mfk_features/features.parquet`（主表，2023～2026-01-26） | **目录不存在**，只有 2026-01-27 起的 `*_ext_*` 日文件 |
| `output/cogalpha_features/`（`features.parquet`、`labels_10d*.parquet`） | **目录不存在**，ext 也没有 |

只用本机有的数据（因子表加 2026 年的扩展）训出来的会是另一个模型（缺 5 组特征的 2023～2025 样本），不是 `R20-M-V16` 的延长，不能当作"生产模型训练到最新"。

## 下一步（需要用户在生产机器上做或提供数据）

- 在生产机器上把 `train_v16_full.py` 的 `TRAIN_END` 改成 2026-09-22，并先把上面三类缺口的 ext 标签补齐（`max_gain_labels_ext_cache.parquet` 拷过去即可；`labels_10d_ext*` 要用同样办法补）；训练 `r20_v16_all_nost` 的同时还要重训 `regressor_gain`、`regressor_dd`（池 A 用它们），输出放非受保护目录，比较后再由用户决定是否替换 `output/production/`。
- 或者把缺失的主表拷到本机，我再接手训练、对比。
- 训练完成并使用了保留窗口之后，再把窗口状态改成 consumed 并登记在这里。

## 局限

- 对"训练到最新"，最新可用标签日是 2026-09-22（日线缓存到 10-29，标签要 20 个交易日）。
- 标签重建只验证了 2025-01-02 至 2026-01-26，2026-01-27 之后没有对照基准。

## 交接（2026-11-25，用户要求写入）

### 后续对话

用户：把缺的主表拷到本机，你来训练。→ 我答：本机还没有这些表，请拷过来。
用户：表已拷好，开始训练。→ 我核查：`output/` 下五个特征目录仍只有 `*_ext_*` 分片（目录时间戳 10-29 没变），`output/cogalpha_features/` 不存在；`D:\aicoding\stockagent-analysis` 内容与 C 盘一致；全盘搜索 `features.parquet`、`amount_features.parquet`、`labels_10d.parquet` 无结果。**没训练。**
用户：拷贝的文件应该在哪里？是在我原来的电脑里吗？→ 我答：是，在跑生产模型的原电脑上，路径见下表。
用户：写入交接文档，并备注下一步应该是在当前电脑重新训练；回顾一下最后，缺失了什么。→ 本节。

### 当前状态（截至 2026-11-25）

- 已完成：宪法一行修订、标签重建与延长（见上，commit `917228cfe`）。
- **未完成：没有训练任何模型。** `output/production/*`、`config/pool_a_r20_target_v1.json` 未动。保留窗口状态仍是 reserved（训练真正用到后再改 consumed）。
- 阻塞原因：训练输入没有到位。本机核查时，以下文件都不存在。

### 缺失清单（本机 `C:\aicoding\stockagent-analysis\output\` 下）

| 路径 | 缺的文件 | 说明 |
|---|---|---|
| `amount_features\` | `amount_features.parquet` | 主表，2023～2026-01-26；本机只有 `_ext_*` |
| `moneyflow\` | `features.parquet` | 同上 |
| `pyramid_v2\` | `features.parquet` | 同上 |
| `v7_extras\` | `features.parquet` | 同上 |
| `mfk_features\` | `features.parquet` | 同上 |
| `cogalpha_features\`（整个目录） | `features.parquet`、`labels_10d.parquet`，如有 `labels_10d_ext*.parquet` | 目录不存在；`train_v15_refresh.load_window` 会读 |

这些在原电脑（生产机）上，不是本机。拷到本机同名目录即可，`cogalpha_features` 需新建。如果原电脑上也没有 `labels_10d_ext*`，用日线缓存按同样定义重建，重建后先和旧标签核对是否一致。

### 备注：下一步

**下一步应该是在当前电脑（这台笔记本）重新训练**，前提是上表的文件已拷到位。步骤：

1. 先核对文件齐全（上表六项）和日期覆盖（主表到 2026-01-26 或更晚，`_ext` 接续）。
2. `train_v16_full.py`（仓库根目录）：`TRAIN_END` 改为 `20260922`；`OUT_BASE` 改到非受保护目录，不要写入 `output/production/`。
3. 训练 `r20_v16_all_nost`，并重训 `regressor_gain`、`regressor_dd`（池 A 用）；`r10` 是否需要重训另议。
4. 和旧 `R20-M-V16` 逐项对比；训练到最新之后保留窗口已作废，对比不再是样本外，只能报 in_sample / reused_holdout。
5. 训练用到保留窗口后，把窗口状态改成 consumed 并写入 wiki。
6. 替换 `output/production/` 需要用户明确指示后再 `propose` / `decide`，署用户名。

### 未核实的事

- 主表的列和训练代码期望的是否一致，要等文件到位后才能看。
- 2026-01-27 之后的重建标签没有对照基准。
