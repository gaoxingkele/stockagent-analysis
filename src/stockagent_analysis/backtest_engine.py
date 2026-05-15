"""V12 真实回测引擎 (Sprint 2.1, V12.13).

核心: 真实模拟交易 (滑点/手续费/T+1/涨跌停约束)
- 每个交易日跑 V12Scorer + 仓位管理
- 追踪持仓 / 累计净值 / 回撤 / 交易日志

约束:
  - T+1: 买入次日才能卖
  - 滑点: 0.1% (开盘买 + 收盘卖)
  - 手续费: 0.05% 双向
  - 涨跌停: 涨跌停日不能交易 (|pct_chg| ≥ 9.8%)
  - 单股仓位上限: 由 pool 决定 (默认 5%)

入参:
  start_date / end_date: 回测区间
  initial_capital: 初始资金 (默认 1.0)
  rebalance_freq: 调仓频率 ('daily' / 'weekly' / 'pool_holding_days')

输出:
  - 净值曲线 (DataFrame: date, nav, return, drawdown)
  - 交易日志 (entry/exit 明细)
  - 月度收益统计
"""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Optional, Callable
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
DAILY_CACHE = ROOT / "output" / "tushare_cache" / "daily"

SLIPPAGE = 0.001    # 0.1% 单边滑点
FEE = 0.0005        # 0.05% 单边手续费
LIMIT_THRESHOLD = 9.8  # 涨跌停 ≥9.8%


class V12Backtester:
    """V12 端到端回测器."""

    def __init__(self, start_date: str, end_date: str,
                 initial_capital: float = 1.0,
                 use_regime_sizing: bool = True,
                 use_industry_diversification: bool = True,
                 use_zombie_filter: bool = True,
                 rebalance_holding_period: bool = True):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.use_regime_sizing = use_regime_sizing
        self.use_industry_diversification = use_industry_diversification
        self.use_zombie_filter = use_zombie_filter
        self.rebalance_holding_period = rebalance_holding_period

        # 状态
        self.cash: float = initial_capital
        self.holdings: dict[str, dict] = {}  # ts_code -> {entry_date, entry_price, position_size, pool, r20_pred}
        self.nav_curve: list[dict] = []
        self.trade_log: list[dict] = []

    def _load_daily_prices(self, start: str, end: str) -> pd.DataFrame:
        """加载区间内每日 close + pct_chg."""
        files = sorted(DAILY_CACHE.glob("*.parquet"))
        parts = []
        for f in files:
            d = f.stem
            if start <= d <= end:
                df = pd.read_parquet(f)[["ts_code", "trade_date", "close", "open", "pct_chg"]]
                parts.append(df)
        big = pd.concat(parts, ignore_index=True)
        big["trade_date"] = big["trade_date"].astype(str)
        return big

    def _get_trading_days(self, start: str, end: str) -> list[str]:
        """获取区间内的交易日列表."""
        files = sorted(DAILY_CACHE.glob("*.parquet"))
        return [f.stem for f in files if start <= f.stem <= end]

    def run(self, progress_cb: Optional[Callable] = None) -> dict:
        """执行回测."""
        from .v12_scoring import V12Scorer
        from .position_manager import build_portfolio, check_exit_signal
        from .pool_classifier import POOL_CONFIG

        t0 = time.time()
        days = self._get_trading_days(self.start_date, self.end_date)
        print(f"[BT] 回测区间 {self.start_date} → {self.end_date}, {len(days)} 交易日", flush=True)

        # 加载所有价格 (一次性)
        all_prices = self._load_daily_prices(self.start_date, self.end_date)
        prices_by_date = {d: g.set_index("ts_code") for d, g in all_prices.groupby("trade_date")}

        scorer = V12Scorer.get(ROOT)

        for i, date in enumerate(days):
            day_t0 = time.time()
            day_prices = prices_by_date.get(date)
            if day_prices is None:
                continue

            # 1. 跑 V12 评分 (含 pool / position_size)
            try:
                v12 = scorer.score_market(date)
            except Exception as e:
                print(f"[BT] {date} V12 推理失败: {str(e)[:100]}", flush=True)
                continue

            regime_info = v12.attrs.get("regime_info", {})
            regime_ratio = regime_info.get("position_ratio", 1.0) if self.use_regime_sizing else 1.0

            # 2. 检查持仓出场
            exits = []
            for ts, h in list(self.holdings.items()):
                cur = day_prices.loc[ts] if ts in day_prices.index else None
                if cur is None: continue
                close = float(cur["close"])
                pct_chg = float(cur.get("pct_chg", 0))
                # 涨跌停日不卖
                if abs(pct_chg) >= LIMIT_THRESHOLD:
                    continue
                days_held = sum(1 for d in days[:i+1] if d > h["entry_date"])
                # 是否还在 V12 输出
                v12_row = v12[v12["ts_code"] == ts]
                is_zombie = bool(v12_row.iloc[0].get("is_zombie", False)) if len(v12_row) else False
                sig = check_exit_signal(
                    holding={**h, "days_held": days_held},
                    current={"close": close, "is_zombie": is_zombie,
                             "regime_position_ratio": regime_ratio},
                )
                if sig["action"] == "exit_all":
                    exits.append((ts, "full", close, sig["reason"]))
                elif sig["action"] in ("reduce_half", "reduce_to_regime"):
                    new_size = sig.get("new_size", h["position_size"] * 0.5)
                    exits.append((ts, new_size, close, sig["reason"]))

            for ts, target_size, close, reason in exits:
                self._execute_sell(ts, target_size, close, date, reason)

            # 3. 调仓 (按池子分配新资金)
            #    简化: 每日调仓 (实战可降频)
            if self.rebalance_holding_period or len(self.holdings) < 10:
                # 取目标组合
                target_portfolio = build_portfolio(
                    v12, total_capital=1.0, regime_position_ratio=regime_ratio
                )
                # 加 price 列
                if not target_portfolio.empty:
                    target_portfolio = target_portfolio.merge(
                        day_prices[["close", "pct_chg"]].reset_index(),
                        on="ts_code", how="left"
                    )
                # 当前持仓 ts_codes
                holding_codes = set(self.holdings.keys())
                target_codes = set(target_portfolio["ts_code"].tolist()) if not target_portfolio.empty else set()

                # 买入新进入目标组合的股 (优先 r20_pred 高的)
                to_buy = target_portfolio[~target_portfolio["ts_code"].isin(holding_codes)].sort_values(
                    "r20_pred", ascending=False
                ) if not target_portfolio.empty else pd.DataFrame()
                for _, row in to_buy.iterrows():
                    ts = row["ts_code"]
                    target_size = row["position_size"]
                    pct_chg = float(row.get("pct_chg", 0))
                    if abs(pct_chg) >= LIMIT_THRESHOLD:
                        continue
                    if self.cash < target_size:
                        break  # 资金用尽
                    close = float(row.get("close", 0))
                    if close <= 0: continue
                    self._execute_buy(ts, target_size, close, date, row)

            # 4. 计算当日净值
            nav = self._calc_nav(day_prices)
            self.nav_curve.append({
                "date": date,
                "nav": nav,
                "cash": self.cash,
                "n_holdings": len(self.holdings),
                "regime_ratio": regime_ratio,
            })

            if progress_cb and i % 20 == 0:
                progress_cb(i, len(days), date, nav, time.time() - day_t0)

        print(f"[BT] 回测完成, 总耗时 {time.time()-t0:.0f}s", flush=True)
        return self._summarize()

    def _execute_buy(self, ts: str, size: float, close: float, date: str, row: pd.Series):
        """买入 (close 价 + 滑点 + 手续费). T+1 简化: 当日 close 入场."""
        if self.cash < size * 1.01: return  # 不够付手续费
        cost = size * (1 + SLIPPAGE + FEE)
        self.cash -= cost
        self.holdings[ts] = {
            "entry_date": date,
            "entry_price": close * (1 + SLIPPAGE),  # 滑点后成本
            "position_size": size,
            "pool": row.get("pool", "unknown"),
            "r20_pred_at_entry": float(row.get("r20_pred", 0)),
            "buy_score": float(row.get("buy_score", 0)),
        }
        self.trade_log.append({
            "date": date, "action": "buy", "ts_code": ts,
            "price": close, "size": size, "cost": cost,
            "pool": row.get("pool"), "cash_after": self.cash,
        })

    def _execute_sell(self, ts: str, target_size, close: float, date: str, reason: str):
        """卖出 (close 价 - 滑点 - 手续费). target_size: 'full' 或 数值."""
        h = self.holdings.get(ts)
        if not h: return
        if target_size == "full" or target_size <= 0.005:
            sold_size = h["position_size"]
            del self.holdings[ts]
        else:
            sold_size = h["position_size"] - target_size
            if sold_size <= 0.005: return
            h["position_size"] = target_size

        gross = sold_size * (close / h["entry_price"]) * (1 - SLIPPAGE - FEE)
        self.cash += gross
        ret = (close / h["entry_price"] - 1) * 100
        self.trade_log.append({
            "date": date, "action": "sell", "ts_code": ts,
            "price": close, "size_sold": sold_size, "gross": gross,
            "ret_pct": round(ret, 2), "reason": reason, "cash_after": self.cash,
        })

    def _calc_nav(self, day_prices: pd.DataFrame) -> float:
        """当日总资产 = 现金 + 持仓市值."""
        nav = self.cash
        for ts, h in self.holdings.items():
            if ts in day_prices.index:
                close = float(day_prices.loc[ts, "close"])
                # 持仓市值 = (close / entry_price) × position_size
                nav += (close / h["entry_price"]) * h["position_size"]
        return nav

    def _summarize(self) -> dict:
        nav_df = pd.DataFrame(self.nav_curve)
        if nav_df.empty:
            return {"error": "no nav data"}
        nav_df["return_pct"] = (nav_df["nav"] - 1) * 100
        nav_df["drawdown_pct"] = (nav_df["nav"] / nav_df["nav"].cummax() - 1) * 100

        trade_df = pd.DataFrame(self.trade_log)
        sells = trade_df[trade_df["action"] == "sell"] if not trade_df.empty else pd.DataFrame()
        win_rate = (sells["ret_pct"] > 0).mean() * 100 if len(sells) > 0 else 0

        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_days": len(nav_df),
            "final_nav": round(float(nav_df["nav"].iloc[-1]), 4),
            "total_return_pct": round(float(nav_df["return_pct"].iloc[-1]), 2),
            "max_drawdown_pct": round(float(nav_df["drawdown_pct"].min()), 2),
            "n_trades": len(trade_df),
            "n_sells": len(sells),
            "win_rate_pct": round(win_rate, 1),
            "avg_holding_n": round(nav_df["n_holdings"].mean(), 1),
            "nav_df": nav_df,
            "trade_df": trade_df,
        }


def run_backtest_demo(start: str = "20260214", end: str = "20260413") -> dict:
    """简单 demo (OOS 期 60 日)."""
    bt = V12Backtester(start, end, initial_capital=1.0)
    return bt.run(progress_cb=lambda i, n, d, nav, dt: print(
        f"[{i}/{n}] {d}: nav={nav:.4f}, holdings={dt:.0f}s", flush=True
    ))
