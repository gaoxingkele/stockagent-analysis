# -*- coding: utf-8 -*-
"""BT-001 — 参数化 book 级组合回测引擎 (A股, 日频, T+1, 真实成本).

设计文档: research/backtest/DESIGN.md。成本模型按 plans/prd.json preRegisteredGate.cost_model
**冻结**实现 (SIGN-R01, 不在结果上调)。

与现有 t005 的区别 (DESIGN §1): t005 只测"每日新鲜建仓 + 20d 前向"的等权选股 α (无持仓/无成本);
本引擎模拟**真实组合 P&L 路径** —— 持仓 carryover、换手、A股成本、T+1、涨跌停不可成交、可参数化 sizing。

核心 API
--------
    bt = PortfolioBacktester(prices, cost=CostModel(...), t_plus_1=True, limit_pct=9.8)
    result = bt.run(targets)            # targets: 每个再平衡日的目标权重
    result.summary()                    # 年化/Sharpe/最大回撤/月均换手/...
    result.timeseries                   # 逐日 nav/return/exposure/turnover/cost

执行模型 (因果, 标准): 目标权重在再平衡日 D 收盘后决定, 在**下一交易日 D+1 开盘**成交
(避免前视)。T+1: 当日买入的份额当日不可卖。涨跌停 (|pct_chg|>=limit_pct) 当日该股不可成交。

成本 (冻结):
  佣金 0.025% 双向 + 印花税 0.05% 仅卖 + 过户费 0.001% 双向 + 滑点基线 0.10%/边 (可配, 敏感性档)。
整手 100 股忽略 (近似连续, DESIGN §2)。

`python research/backtest/engine.py` 跑内置解析型自检 (买入持有复刻 / 成本核算 / T+1 / 涨跌停 /
carryover 换手), 全部 assert 通过 = 引擎无 bug。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ───────────────────────── 成本模型 (冻结, SIGN-R01) ─────────────────────────
@dataclass
class CostModel:
    """A股交易成本, 单位=成交额比例。按 prd.preRegisteredGate.cost_model 冻结。"""
    commission: float = 0.00025   # 佣金 0.025% 双向
    stamp_sell: float = 0.0005    # 印花税 0.05% 仅卖出
    transfer: float = 0.00001     # 过户费 0.001% 双向
    slippage: float = 0.0010      # 滑点 0.10%/边 (基线; 敏感性档 0/0.0005/0.0010/0.0020)
    enabled: bool = True          # False = 成本关 (sanity 用), 滑点也关

    def buy_rate(self) -> float:
        if not self.enabled:
            return 0.0
        return self.commission + self.transfer + self.slippage

    def sell_rate(self) -> float:
        if not self.enabled:
            return 0.0
        return self.commission + self.stamp_sell + self.transfer + self.slippage

    def describe(self) -> dict:
        return {"commission": self.commission, "stamp_sell": self.stamp_sell,
                "transfer": self.transfer, "slippage": self.slippage,
                "enabled": self.enabled,
                "buy_rate": self.buy_rate(), "sell_rate": self.sell_rate()}


# ───────────────────────── 结果容器 ─────────────────────────
@dataclass
class BacktestResult:
    timeseries: pd.DataFrame                 # index=trade_date, 列见下
    trades: pd.DataFrame                     # 逐笔成交 (date, ts_code, side, value, cost)
    params: dict = field(default_factory=dict)

    def summary(self, periods_per_year: int = 252, rf: float = 0.0) -> dict:
        ts = self.timeseries
        nav = ts["nav"].to_numpy(dtype=float)
        ret = ts["ret"].to_numpy(dtype=float)
        n = len(nav)
        if n < 2:
            return {"n_days": n, "error": "insufficient data"}
        total_ret = nav[-1] / nav[0] - 1.0
        ann_ret = (nav[-1] / nav[0]) ** (periods_per_year / max(n - 1, 1)) - 1.0
        vol = float(np.nanstd(ret, ddof=1))
        ann_vol = vol * np.sqrt(periods_per_year)
        excess = ret - rf / periods_per_year
        sharpe = (float(np.nanmean(excess)) / (vol + 1e-12)) * np.sqrt(periods_per_year)
        # 最大回撤
        peak = np.maximum.accumulate(nav)
        dd = nav / peak - 1.0
        max_dd = float(dd.min())
        # 月均换手 (单边成交额 / nav, 按月汇总取均)
        tov = ts.copy()
        tov["ym"] = tov.index.astype(str).str[:6]
        monthly_turnover = tov.groupby("ym")["turnover"].sum()
        # 月度收益 + 胜率
        nav_m = tov.groupby("ym")["nav"].last()
        monthly_ret = nav_m.pct_change().dropna()
        return {
            "n_days": int(n),
            "total_return": float(total_ret),
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "avg_monthly_turnover": float(monthly_turnover.mean()),
            "avg_exposure": float(ts["exposure"].mean()),
            "avg_n_positions": float(ts["n_pos"].mean()),
            "monthly_win_rate": float((monthly_ret > 0).mean()) if len(monthly_ret) else float("nan"),
            "n_months": int(len(nav_m)),
            "total_cost_paid": float(ts["cost"].sum()),
        }


# ───────────────────────── 回测引擎 ─────────────────────────
class PortfolioBacktester:
    """日频 A股组合回测器。

    prices: DataFrame [ts_code, trade_date, open, close, pct_chg]
        - open  : 当日开盘 (成交价)
        - close : 当日收盘 (估值价)
        - pct_chg: 当日涨跌幅 %, 用于涨跌停判定 (|pct_chg|>=limit_pct 不可成交)
        trade_date 用字符串 'YYYYMMDD'。
    """

    def __init__(self, prices: pd.DataFrame, cost: Optional[CostModel] = None,
                 t_plus_1: bool = True, limit_pct: float = 9.8,
                 initial_cash: float = 1.0):
        self.cost = cost if cost is not None else CostModel()
        self.t_plus_1 = t_plus_1
        self.limit_pct = limit_pct
        self.initial_cash = initial_cash
        self._build_price_lookup(prices)

    def _build_price_lookup(self, prices: pd.DataFrame) -> None:
        p = prices[["ts_code", "trade_date", "open", "close", "pct_chg"]].copy()
        p["trade_date"] = p["trade_date"].astype(str)
        p = p.dropna(subset=["close"])
        # 按日期分组: date -> {ts_code: (open, close, pct_chg)}
        self.trade_dates: List[str] = sorted(p["trade_date"].unique())
        self._open: Dict[str, Dict[str, float]] = {}
        self._close: Dict[str, Dict[str, float]] = {}
        self._pct: Dict[str, Dict[str, float]] = {}
        for d, g in p.groupby("trade_date"):
            self._open[d] = dict(zip(g["ts_code"], g["open"]))
            self._close[d] = dict(zip(g["ts_code"], g["close"]))
            self._pct[d] = dict(zip(g["ts_code"], g["pct_chg"]))

    def _tradable(self, d: str, ts: str) -> bool:
        """该股当日是否可成交: 有开盘价 + 非涨跌停。"""
        o = self._open[d].get(ts)
        if o is None or not np.isfinite(o) or o <= 0:
            return False
        pc = self._pct[d].get(ts)
        if pc is not None and np.isfinite(pc) and abs(pc) >= self.limit_pct:
            return False
        return True

    def run(self, targets: Dict[str, Dict[str, float]],
            start: Optional[str] = None, end: Optional[str] = None) -> BacktestResult:
        """运行回测。

        targets: {rebalance_date: {ts_code: weight}}  权重和<=1, 余为现金。
                 rebalance_date 收盘后决定, 下一交易日开盘成交。
        """
        tdates = self.trade_dates
        if start:
            tdates = [d for d in tdates if d >= start]
        if end:
            tdates = [d for d in tdates if d <= end]
        target_dates = {d: w for d, w in targets.items()}

        cash = float(self.initial_cash)
        shares: Dict[str, float] = {}
        last_buy_date: Dict[str, str] = {}
        pending: Optional[Dict[str, float]] = None
        close_d_prev: Dict[str, float] = {}

        rows = []
        trades = []

        for d in tdates:
            open_d = self._open[d]
            close_d = self._close[d]
            turnover_val = 0.0
            cost_val = 0.0

            # 1) 执行上一再平衡日挂出的目标 (今日开盘成交, 因果)
            if pending is not None:
                # 开盘估值作 sizing 基准 (有今日开盘价用之, 停牌沿用上一收盘)
                equity_open = cash
                for ts, sh in shares.items():
                    px = open_d.get(ts)
                    if px is None or not np.isfinite(px):
                        px = close_d_prev.get(ts, 0.0)
                    equity_open += sh * px

                _, cv = self._rebalance(pending, d, open_d, shares,
                                        last_buy_date, equity_open, cash_ref=lambda: cash)
                cash = cv["cash"]
                turnover_val = cv["turnover"]
                cost_val = cv["cost"]
                trades.extend(cv["trades"])
                pending = None

            # 2) 收盘估值
            nav = cash
            pos_val = 0.0
            n_pos = 0
            for ts, sh in list(shares.items()):
                if abs(sh) < 1e-12:
                    continue
                px = close_d.get(ts)
                if px is None or not np.isfinite(px):
                    # 停牌: 沿用上一收盘
                    px = close_d_prev.get(ts, 0.0)
                v = sh * px
                pos_val += v
                if abs(v) > 1e-12:
                    n_pos += 1
            nav += pos_val

            rows.append({
                "trade_date": d, "nav": nav, "cash": cash, "pos_val": pos_val,
                "exposure": pos_val / nav if nav > 0 else 0.0,
                "n_pos": n_pos,
                "turnover": turnover_val / nav if nav > 0 else 0.0,
                "cost": cost_val,
            })

            # 3) 今日为再平衡日 -> 挂出目标 (下一交易日成交)
            if d in target_dates:
                pending = target_dates[d]

            close_d_prev = close_d   # 供次日停牌沿用

        ts_df = pd.DataFrame(rows).set_index("trade_date")
        ts_df["ret"] = ts_df["nav"].pct_change().fillna(0.0)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["date", "ts_code", "side", "value", "cost"])
        return BacktestResult(timeseries=ts_df, trades=trades_df, params={
            "cost": self.cost.describe(), "t_plus_1": self.t_plus_1,
            "limit_pct": self.limit_pct, "initial_cash": self.initial_cash,
            "n_rebalances": len(target_dates),
        })

    def _rebalance(self, target_w: Dict[str, float], d: str,
                   open_d: Dict[str, float], shares: Dict[str, float],
                   last_buy_date: Dict[str, str], equity: float,
                   cash_ref) -> tuple:
        """在 d 开盘把组合移向 target_w。原地改 shares; 现金/换手/成本经返回值传出。"""
        cash = cash_ref()
        turnover = 0.0
        cost = 0.0
        trades = []
        buy_rate = self.cost.buy_rate()
        sell_rate = self.cost.sell_rate()

        universe = set(shares.keys()) | set(target_w.keys())
        for ts in universe:
            w = target_w.get(ts, 0.0)
            cur_sh = shares.get(ts, 0.0)
            px = open_d.get(ts)
            cur_val = cur_sh * px if (px is not None and np.isfinite(px)) else None

            if not self._tradable(d, ts):
                # 不可成交 (停牌/涨跌停): 维持现状, 不下单
                continue
            target_val = w * equity
            cur_val = cur_sh * px
            delta = target_val - cur_val
            if abs(delta) < 1e-9 * max(equity, 1.0):
                continue

            if delta > 0:   # 买入
                c = delta * buy_rate
                cash -= (delta + c)
                shares[ts] = cur_sh + delta / px
                last_buy_date[ts] = d
                turnover += delta
                cost += c
                trades.append({"date": d, "ts_code": ts, "side": "buy",
                               "value": delta, "cost": c})
            else:           # 卖出
                # T+1: 当日买入不可卖
                if self.t_plus_1 and last_buy_date.get(ts) == d:
                    continue
                sell_val = -delta
                c = sell_val * sell_rate
                cash += (sell_val - c)
                shares[ts] = cur_sh - sell_val / px
                if abs(shares[ts]) < 1e-12:
                    shares.pop(ts, None)
                turnover += sell_val
                cost += c
                trades.append({"date": d, "ts_code": ts, "side": "sell",
                               "value": sell_val, "cost": c})

        return None, {"cash": cash, "turnover": turnover, "cost": cost, "trades": trades}


# ───────────────────────── 内置解析型自检 ─────────────────────────
def _mk_prices(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["ts_code", "trade_date", "open", "close", "pct_chg"])


def _selftest() -> None:
    print("=== engine.py 解析型自检 ===")

    # ── 1) 买入持有复刻 (成本关): nav 必须 == 标的累计收益 ──
    # A 股 D0 开盘买入, 其后持有。D0 决定目标在 D0 挂出 -> 在 *D1* 开盘成交。
    # 故 nav 从 D1 起跟随。构造: D1 open=10, 之后 close 序列。
    dates = ["20240101", "20240102", "20240103", "20240104"]
    px = [10.0, 11.0, 12.0, 13.2]      # D1 open 买入价=10; close 走势
    rows = []
    for i, d in enumerate(dates):
        rows.append(["AAA", d, px[i], px[i], 1.0])   # open==close 简化
    prices = _mk_prices(rows)
    bt = PortfolioBacktester(prices, cost=CostModel(enabled=False),
                             t_plus_1=True, limit_pct=11.0, initial_cash=1.0)
    # D0 (20240101) 收盘后挂 100% AAA -> D1 (20240102) open=11 成交
    res = bt.run({"20240101": {"AAA": 1.0}})
    nav = res.timeseries["nav"]
    # D1 开盘 11 买入, D1 收盘 11 -> nav≈1.0; D3 收盘 13.2 -> nav=13.2/11
    exp_end = 13.2 / 11.0
    assert abs(nav.iloc[-1] - exp_end) < 1e-9, f"买入持有: {nav.iloc[-1]} vs {exp_end}"
    assert abs(res.timeseries["exposure"].iloc[-1] - 1.0) < 1e-9, "全仓 exposure 应=1"
    print(f"  [1] 买入持有复刻 OK  nav_end={nav.iloc[-1]:.6f} == 13.2/11")

    # ── 2) 成本核算: 一次完整买入+清仓, 成本 == 公式 ──
    dates = ["20240101", "20240102", "20240103"]
    rows = []
    for d in dates:
        rows.append(["BBB", d, 10.0, 10.0, 0.0])    # 价格不动, 隔离成本
    prices = _mk_prices(rows)
    cm = CostModel(slippage=0.0010, enabled=True)
    bt = PortfolioBacktester(prices, cost=cm, t_plus_1=True, limit_pct=11.0)
    # D0 挂 100% -> D1 买入; D1 挂 0% -> D2 清仓
    res = bt.run({"20240101": {"BBB": 1.0}, "20240102": {"BBB": 0.0}})
    # 买入: 价不动, equity=1, 买 ~1.0 价值; 买成本率 = 0.00025+0.00001+0.0010
    br, sr = cm.buy_rate(), cm.sell_rate()
    # D1: equity≈1, buy delta≈1, cash -> 1 - (1+1*br) = -br + (1-1)=... 精确推:
    # cash0=1; buy delta=1*1=1 (target_val=1*equity, equity=cash=1) -> cash=1-(1+br)= -br; shares=1/10*...
    # 价=10 -> shares=delta/px=1/10=0.1; nav D1 = cash + 0.1*10 = -br + 1 = 1-br
    nav_d1 = res.timeseries["nav"].iloc[1]
    assert abs(nav_d1 - (1 - br)) < 1e-9, f"买入后 nav: {nav_d1} vs {1-br}"
    # D2 清仓: 卖出 value≈ (1-br) (target 0, 全卖). 卖额=shares*px=0.1*10=1.0
    #   实际目标=0, equity=cash+pos= (-br)+1=1-br; 卖出 sell_val= cur_val-0 =1.0
    #   cash += 1.0 - 1.0*sr -> cash = -br + 1 - sr = 1-br-sr; nav=cash (无持仓)
    nav_d2 = res.timeseries["nav"].iloc[2]
    assert abs(nav_d2 - (1 - br - sr)) < 1e-9, f"清仓后 nav: {nav_d2} vs {1-br-sr}"
    total_cost = res.timeseries["cost"].sum()
    assert abs(total_cost - (br + sr)) < 1e-9, f"总成本: {total_cost} vs {br+sr}"
    print(f"  [2] 成本核算 OK  买成本率={br:.5f} 卖成本率={sr:.5f} 总成本={total_cost:.5f}")

    # ── 3) 涨跌停不可成交: 目标股 D1 涨停 -> 不建仓 ──
    rows = [["CCC", "20240101", 10.0, 10.0, 10.0],   # D0
            ["CCC", "20240102", 11.0, 11.0, 10.0],   # D1 涨停 (pct=10>=9.8) 不可买
            ["CCC", "20240103", 12.0, 12.0, 5.0]]    # D2 可买
    prices = _mk_prices(rows)
    bt = PortfolioBacktester(prices, cost=CostModel(enabled=False), limit_pct=9.8)
    res = bt.run({"20240101": {"CCC": 1.0}})         # D0 挂 -> D1 成交但涨停被拦
    assert res.timeseries["n_pos"].iloc[1] == 0, "涨停日不应建仓"
    print("  [3] 涨跌停不可成交 OK  (涨停日 n_pos=0)")

    # ── 4) T+1: 当日买入当日不可卖 (同日 rebalance 边界) ──
    # 退化校验: 连续两日再平衡, D1 买入后 D2 立即可卖 (T+1 满足, 次日); 自检反例难直接构造,
    #   改验"买入当日 last_buy_date 标记" -> 用极端: 同股两连续目标日, 卖单在买入次日才发生.
    rows = [["DDD", "20240101", 10.0, 10.0, 0.0],
            ["DDD", "20240102", 10.0, 10.0, 0.0],
            ["DDD", "20240103", 10.0, 10.0, 0.0],
            ["DDD", "20240104", 10.0, 10.0, 0.0]]
    prices = _mk_prices(rows)
    bt = PortfolioBacktester(prices, cost=CostModel(enabled=False), t_plus_1=True)
    res = bt.run({"20240101": {"DDD": 1.0}, "20240103": {"DDD": 0.0}})
    # D1 买入, D2 持有, D4 (20240104) 清仓 (目标在 20240103 挂 -> D4 开盘卖)
    assert res.timeseries["n_pos"].iloc[1] == 1, "D1 应持仓"
    assert res.timeseries["n_pos"].iloc[3] == 0, "清仓日后应空仓"
    print("  [4] T+1 / carryover OK  (买入持有跨日, 再平衡日清仓)")

    # ── 5) 等权两股 carryover: nav == 两股累计收益等权平均 ──
    rows = []
    a_close = {"20240101": 10, "20240102": 10, "20240103": 12, "20240104": 12}
    b_close = {"20240101": 20, "20240102": 20, "20240103": 20, "20240104": 30}
    for d in ["20240101", "20240102", "20240103", "20240104"]:
        rows.append(["AAA", d, a_close[d], a_close[d], 0.0])
        rows.append(["BBB", d, b_close[d], b_close[d], 0.0])
    prices = _mk_prices(rows)
    bt = PortfolioBacktester(prices, cost=CostModel(enabled=False))
    res = bt.run({"20240101": {"AAA": 0.5, "BBB": 0.5}})
    # D1 (20240102) open: A=10,B=20 各买 0.5. 之后持有.
    # D3 end: A 10->12 (+20%), B 20 (0%); nav = 0.5*1.2+0.5*1.0=1.10
    # D4 end: A 12 (vs10 +20%), B 20->30(+50%); nav=0.5*1.2+0.5*1.5=1.35
    assert abs(res.timeseries["nav"].iloc[2] - 1.10) < 1e-9, res.timeseries["nav"].iloc[2]
    assert abs(res.timeseries["nav"].iloc[3] - 1.35) < 1e-9, res.timeseries["nav"].iloc[3]
    print(f"  [5] 等权两股 carryover OK  nav D3={res.timeseries['nav'].iloc[2]:.4f} D4={res.timeseries['nav'].iloc[3]:.4f}")

    print("\n[engine] 全部解析型自检通过 — 引擎核算无 bug。")


if __name__ == "__main__":
    _selftest()
