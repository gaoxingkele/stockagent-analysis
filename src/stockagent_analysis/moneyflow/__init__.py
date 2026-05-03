"""资金分层分析模块 (tushare moneyflow).

独立封装, 可用于未来选股系统. 核心 API:

  extractor.fetch_moneyflow(ts_code, start, end)
      → 单股拉取 raw moneyflow 数据
  extractor.batch_fetch(ts_codes, start, end, cache_dir)
      → 批量拉取 + 文件缓存

  features.compute_features(daily_mf)
      → 从 raw 数据算 12 个资金分层特征
  features.merge_to_parquet(parquet_path, feature_dir)
      → 合并到 factor parquet

特征列表:
  super_lg_net_5d/20d:  超大单 5/20 日累计净流入 (亿元)
  lg_net_5d/20d:        大单 5/20 日累计净流入
  mid_net_5d:           中单 5 日累计
  sm_net_5d:            小单 5 日累计 (散户)
  main_net_5d:          主力 (大+超大) 5 日累计
  main_consec_in:       主力连续净流入天数 (正)
  main_consec_out:      主力连续净流出天数 (正数, 表示连续天数)
  dispersion:           主力 vs 散户方向分歧 (-1 反向 / +1 一致)
  elg_ratio:            超大单占主力比 (大资金占主比)
  buy_sell_imbalance:   买卖盘失衡度 (买盘量减去卖盘量 / 总量)
"""
from .extractor import fetch_moneyflow, batch_fetch
from .features import compute_features, merge_to_parquet

__all__ = ["fetch_moneyflow", "batch_fetch", "compute_features", "merge_to_parquet"]
