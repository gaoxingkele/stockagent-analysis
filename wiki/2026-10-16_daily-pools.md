# 每日推荐池 2026-10-16

- **日期**: 2026-10-16
- **关联代码/实验**: `scripts/daily_best_lists.py --date 20261016`；产物 `output/experiments/daily_best/best_20261016.md`
- **证据等级**: 展示，不是实验；冻结合约与生产规则未改。

## 对话

用户：跑一下今天的推荐股票池

## 过程

`update_daily_caches.py`（到 10-16）→ `update_regimes_tushare.py`（669 天重叠逐列一致，追加 4 天）→ `daily_review.update_data('20261016')` → `build_s20_20r_confirmation_factors.py --confirmation-start 20260727 --confirmation-end 20261016 --output-dir output/experiments/s20_pure_v1_shadow/factor_groups`（重建 S20 影子因子库，27.7 万行）→ `daily_best_lists.py --date 20261016`。因子库没重建时 daily_best_lists 会在 stage1 打分处报 "Input data must be 2 dimensional and non empty"。

## 结果

- 阀门橙；S0010 风格门关（创业板 20 日 −16.96% 对沪深 300 −8.31%，当天不建新仓）；S0007 三态：下行。
- 池 A（R20）2 只：皖通科技、恒实科技（均为软件服务）。
- 池 S（S20）20 只，半导体占大头；两池无交集。

## 后续

用户：S20 影子因子库要先重建到当天（引用上一条回答里的提醒，没有写别的指令；我理解为要把这一步自动化）。

- `scripts/daily_best_lists.py` 现在在 S20 影子因子库的最后一天早于要跑的日子时，自动调用 `build_s20_20r_confirmation_factors.py`（起 20260727、止当天）重建，再继续；缺数据时给出明确提示而不是 LightGBM 的报错。`--no-rebuild` 则只提示不重建。不带 `--date` 时默认用缓存里最新的交易日。
- 前置的 `update_daily_caches.py`、`update_regimes_tushare.py`、`daily_review.update_data` 仍需先跑。
