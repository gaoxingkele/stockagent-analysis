# K 线数据未能获取的原因分析

> 参考 [AKShare 官方文档](https://akshare.akfamily.xyz/introduction.html) 与 [股票数据接口](https://akshare.akfamily.xyz/data/stock/stock.html)

## 一、从运行结果看到的直接原因

### 1. 当前环境主要失败原因：**代理/网络连接**

最近一次运行（如 `20260227_084203_300617`）的 `data/kline/*.meta.json` 中，**日线、周线、2h、1h、30m** 全部失败，且错误信息一致：

```text
HTTPSConnectionPool(host='push2his.eastmoney.com', port=443): Max retries exceeded with url: ... 
(Caused by ProxyError('Cannot connect to proxy.', RemoteDisconnected('Remote end closed connection without response')))
```

说明：

- **数据源**：AKShare 的日线/周线（`stock_zh_a_hist`）和分钟线（`stock_zh_a_hist_min_em`）都请求 **东方财富 push2his.eastmoney.com**。
- **失败点**：连接在**代理层**就断了——要么代理不可达，要么代理主动断开，导致从未成功连上东方财富服务器。

因此，在你当前网络/代理环境下，**无法获取 K 线的主要原因就是：经代理访问 push2his.eastmoney.com 失败**。

---

### 2. 其他可能原因（在其他运行或环境下会出现）

| 原因 | 说明 | 如何确认 |
|------|------|----------|
| **代理未配置或错误** | 系统/环境变量设置了 HTTP(S)_PROXY，但代理不可用或地址错误 | 看错误是否为 `ProxyError` / `Cannot connect to proxy` |
| **代理或防火墙拦截** | 公司/学校网络禁止访问东方财富或 443 端口 | 同一错误；可尝试手机热点或换网络 |
| **Tushare 未配置或失败** | 未设置 `TUSHARE_TOKEN` 或 Tushare 接口限频/报错 | 若最后一轮是 tushare，meta 里会看到 `missing TUSHARE_TOKEN` 或 tushare 相关异常 |
| **2h 周期行数不足** | 2h 由 1h 数据每 2 根合成，若 1h 返回不足 200 根会报 `akshare_2h_rows_not_enough` | 在 meta 中 error 为该字符串（曾出现在 20260227_083607 的 2h） |
| **东方财富限频/不稳定** | 东财对访问频率和连接数有限制，易出现 RemoteDisconnected | 错误里多为 `RemoteDisconnected` 或连接重置 |
| **必须满 100 根才判成功** | 代码里要求每个周期至少 `limit=100` 根，不足则视为失败 | meta 中 `ok: false` 且 `rows` 小于 100 或 error 为 `*_rows_not_enough` |

---

## 二、数据流与判定逻辑（便于你对症排查）

1. **K 线抓取顺序**  
   每个周期（week / day / 2h / 1h / 30m）依次用：
   - 第 1 次：AKShare  
   - 第 2 次：Tushare（若配置了 token）  
   - 第 3 次：再 AKShare  

2. **成功条件**  
   - 接口未抛异常，且  
   - 返回数据行数 `>= 100`（2h 为 1h 数据聚合，需至少 200 根 1h 再合成 100 根 2h）。

3. **AKShare 实际请求**  
   - 日线/周线：`stock_zh_a_hist(period="daily"|"weekly")` → **push2his.eastmoney.com**  
   - 分钟线：`stock_zh_a_hist_min_em(period="60"|"30")` → 同上东财接口  

因此，只要代理或到东财的网络不通，**所有周期**都会在同一处失败。

---

## 三、建议处理方式

1. **取消或修正代理（推荐先试）**  
   - 若不需要代理，在运行前取消代理环境变量再跑：
     - Windows PowerShell：`$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''`  
     - 然后：`python run.py analyze --symbol 300617 --name 安靠智电`  
   - 若必须用代理，请确认代理地址、端口、账号密码正确，且允许访问 `push2his.eastmoney.com:443`。

2. **换网络**  
   - 用手机热点或家庭网络再跑一次，排除公司/校园网对东财的拦截。

3. **配置 Tushare 作为备用**  
   - 在 `.env` 中设置有效的 `TUSHARE_TOKEN`，这样在 AKShare 因代理失败时，Tushare 仍可能拉取到日线/周线等（取决于 Tushare 权限和限频）。

4. **降低“必须 100 根”的约束（可选）**  
   - 若你接受“有多少根用多少根”，可修改 `data_backend.py` 中 `_fetch_multi_timeframe_klines` 的成功条件，把 `len(data) >= limit` 改为更小的阈值或仅判断非空；2h 的 `limit * 2` 也可按同样思路放宽。这样在数据偏少时不会因“行数不足”直接判失败。

---

## 四、AKShare 接口与数据源说明（基于官方文档）

| 接口 | 数据源 | 用途 | 限制 |
|------|--------|------|------|
| `stock_zh_a_hist` | 东方财富 push2his.eastmoney.com | 日/周/月线 | 默认 start_date=19700101, end_date=20500101 |
| `stock_zh_a_hist_min_em` | 东方财富 push2his.eastmoney.com | 1/5/15/30/60 分钟 | 1 分钟仅约 5 个交易日；30/60 分钟约 3 个月；易 RemoteDisconnected |
| `stock_zh_a_minute` | 新浪 finance.sina.com.cn | 1/5/15/30/60 分钟 | 分钟线备选，数据源不同可规避东财限频 |
| `stock_zh_a_hist_tx` | 腾讯 gu.qq.com | 日线 | 需带市场标识（sz/sh+代码），无周线 |
| `stock_zh_a_daily` | 新浪 finance.sina.com.cn | 日线 | 易封 IP，不推荐 |

当前项目仅使用东财接口，一旦代理/网络异常，所有 K 线均会失败。

---

## 五、代码级修改建议

### 5.1 分钟线：东财失败时用新浪备选 + 重试延迟

- **原因**：`stock_zh_a_hist_min_em` 请求东财 push2his.eastmoney.com，连续请求易触发限频或 RemoteDisconnected。
- **方案**：① 东财请求增加 3 次重试，每次间隔 2+ 秒；② 失败时调用 `stock_zh_a_minute`（新浪 finance.sina.com.cn）作为备选。
- **已实现**：见 `data_backend._fetch_kline_akshare` 分钟线逻辑。

### 5.2 增加腾讯数据源作为日/周线备选

在 `_fetch_kline_akshare` 中，当东财 `stock_zh_a_hist` 失败时，可尝试 `stock_zh_a_hist_tx`（腾讯）作为日线备选。腾讯数据源走 gu.qq.com，与东财不同，可规避代理对东财的拦截。

### 5.3 请求时绕过代理（NO_PROXY）

在发起 AKShare 请求前，可临时将 `push2his.eastmoney.com` 加入 `NO_PROXY`，使东财请求直连：

```python
import os
os.environ.setdefault("NO_PROXY", "")
no_proxy = os.environ.get("NO_PROXY", "")
if "eastmoney.com" not in no_proxy:
    os.environ["NO_PROXY"] = f"{no_proxy},push2his.eastmoney.com,quote.eastmoney.com".strip(",")
```

### 5.4 放宽行数要求（降级策略）

将 `len(data) >= limit` 改为 `len(data) >= min(limit, 30)` 或 `len(data) > 0`，在数据不足时仍可部分分析，并在 meta 中标注 `partial: true`。

### 5.5 显式传入日期范围（可选）

`stock_zh_a_hist` 虽默认 19700101–20500101，但显式传入近一年范围可减少单次请求数据量，有时更稳定：

```python
from datetime import datetime, timedelta
end = datetime.now().strftime("%Y%m%d")
start = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
raw = ak.stock_zh_a_hist(symbol=symbol, period=period, start_date=start, end_date=end, adjust="")
```

---

## 六、小结

- **本次你遇到的情况**：K 线全部拿不到，是因为**经代理访问 push2his.eastmoney.com 失败**（ProxyError + RemoteDisconnected）。  
- **处理顺序建议**：先取消或修正代理、换网络重试；再确保 Tushare 配置正确作为备用；若仍有个别周期失败，再根据 meta 中的 `error` 和 `rows` 判断是行数不足还是接口/限频问题。
- **代码增强**：已实现腾讯日/周线备选、新浪分钟线备选、NO_PROXY、重试延迟、放宽行数要求，提高在不同网络环境下的成功率。
