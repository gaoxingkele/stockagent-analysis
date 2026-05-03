"""Moneyflow 数据拉取 + 文件缓存."""
from __future__ import annotations
import logging, time
from pathlib import Path
import pandas as pd

logger = logging.getLogger("stockagent.moneyflow")

# 字段
RAW_FIELDS = [
    "ts_code", "trade_date",
    "buy_sm_amount", "sell_sm_amount",
    "buy_md_amount", "sell_md_amount",
    "buy_lg_amount", "sell_lg_amount",
    "buy_elg_amount", "sell_elg_amount",
    "net_mf_amount",
]


def _get_pro():
    """获取 tushare pro 实例."""
    import os
    import tushare as ts
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        # 尝试从 .env 读
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if env_path.exists():
            for enc in ["utf-8", "utf-16", "gb18030"]:
                try:
                    for line in env_path.read_text(encoding=enc).splitlines():
                        if line.startswith("TUSHARE_TOKEN="):
                            token = line.split("=", 1)[1].strip().strip('"\'')
                            break
                    if token: break
                except Exception:
                    continue
    if not token:
        raise RuntimeError("TUSHARE_TOKEN 未配置")
    return ts.pro_api(token)


def fetch_moneyflow(ts_code: str, start_date: str, end_date: str,
                    pro=None) -> pd.DataFrame:
    """拉取单只股票 moneyflow.

    Args:
        ts_code: 600519.SH 格式
        start_date / end_date: YYYYMMDD
    Returns:
        DataFrame, 列见 RAW_FIELDS
    """
    if pro is None: pro = _get_pro()
    df = pro.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if df is None or df.empty:
        return pd.DataFrame(columns=RAW_FIELDS)
    df["trade_date"] = df["trade_date"].astype(str)
    return df.sort_values("trade_date").reset_index(drop=True)


def batch_fetch(ts_codes: list[str], start_date: str, end_date: str,
                cache_dir: Path | str | None = None,
                sleep_sec: float = 0.1, log_every: int = 200) -> pd.DataFrame:
    """批量拉取 + 文件缓存.

    缓存格式: {cache_dir}/{ts_code}.parquet
    """
    pro = _get_pro()
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    fail_count = 0
    t0 = time.time()
    for i, ts in enumerate(ts_codes):
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(ts_codes) - i - 1)
            logger.info("[%d/%d] 已 %ds, 预计剩 %ds, 失败 %d",
                        i + 1, len(ts_codes), int(elapsed), int(eta), fail_count)
        cache_path = cache_dir / f"{ts}.parquet" if cache_dir else None
        if cache_path and cache_path.exists():
            try:
                all_dfs.append(pd.read_parquet(cache_path))
                continue
            except Exception:
                pass

        try:
            df = fetch_moneyflow(ts, start_date, end_date, pro=pro)
            if not df.empty and cache_path:
                df.to_parquet(cache_path, index=False)
            if not df.empty:
                all_dfs.append(df)
            time.sleep(sleep_sec)
        except Exception as e:
            fail_count += 1
            if fail_count <= 5:
                logger.warning("拉 %s 失败: %s", ts, e)
            elif fail_count == 6:
                logger.warning("...更多失败略")
            time.sleep(sleep_sec * 3)

    if not all_dfs:
        return pd.DataFrame(columns=RAW_FIELDS)
    full = pd.concat(all_dfs, ignore_index=True)
    logger.info("批量拉取完成: %d 行, %d 股, 失败 %d, 耗时 %.1fs",
                len(full), full["ts_code"].nunique(), fail_count, time.time() - t0)
    return full
