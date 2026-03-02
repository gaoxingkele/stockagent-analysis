# AKShare 接口测试分析报告

> 测试时间：2025-02-27  
> 测试股票：300617 安靠智电

---

## 一、测试结果汇总

### 1.1  standalone 直接调用（未走项目 pipeline）

| 接口 | 数据源 | 结果 | 错误 |
|------|--------|------|------|
| `stock_zh_a_hist` (日线) | 东财 push2his.eastmoney.com | ❌ 失败 | ProxyError |
| `stock_zh_a_hist` (周线) | 东财 push2his.eastmoney.com | ❌ 失败 | ProxyError |
| `stock_zh_a_hist_min_em` (30分钟) | 东财 push2his.eastmoney.com | ❌ 失败 | ProxyError |
| `stock_zh_a_hist_min_em` (60分钟) | 东财 push2his.eastmoney.com | ❌ 失败 | ProxyError |
| `stock_zh_a_minute` (30/60分钟) | 新浪 finance.sina.com.cn | ✅ 成功 | 1970 行 |
| `stock_zh_a_hist_tx` (日线) | 腾讯 gu.qq.com | ✅ 成功 | 2184 行 |

### 1.2 通过项目 pipeline 运行（`run.py analyze`）

| 周期 | 结果 | 实际数据源 |
|------|------|------------|
| week | ✅ 100 行 | 东财 或 腾讯（东财失败时） |
| day | ✅ 100 行 | 东财 或 腾讯（东财失败时） |
| 2h | ✅ 100 行 | 东财 或 新浪（东财失败时） |
| 1h | ✅ 100 行 | 东财 或 新浪（东财失败时） |
| 30m | ✅ 100 行 | 东财 或 新浪（东财失败时） |

**结论**：项目 pipeline 中因有 NO_PROXY 设置 + 备选数据源，即使东财不可达，仍能通过腾讯/新浪完成数据获取。

---

## 二、问题分析

### 2.1 东财接口失败原因

| 原因 | 说明 |
|------|------|
| **代理干扰** | 系统/环境变量设置了 HTTP(S)_PROXY，请求经代理转发时失败（ProxyError） |
| **NO_PROXY 未生效** | standalone 测试时未设置 NO_PROXY，东财域名仍走代理 |
| **连接被重置** | 即使设置 NO_PROXY 直连，部分网络下仍出现 RemoteDisconnected（防火墙/运营商/东财限频） |

### 2.2 数据源可用性对比

| 数据源 | 域名 | 稳定性 | 需代理环境 |
|--------|------|--------|------------|
| 东财 | push2his.eastmoney.com | 易受代理/限频影响 | 建议 NO_PROXY |
| 腾讯 | gu.qq.com | 较稳定 | 一般无需特殊配置 |
| 新浪 | finance.sina.com.cn | 较稳定 | 一般无需特殊配置 |

---

## 三、当前项目已实现的解决方案

### 3.1 日线/周线

```
东财 stock_zh_a_hist → 失败 → 腾讯 stock_zh_a_hist_tx（备选）
```

- 腾讯日线可重采样为周线
- 腾讯返回无 volume，需在 normalize 中补 0

### 3.2 分钟线（30m/1h/2h）

```
东财 stock_zh_a_hist_min_em（3 次重试，间隔 2+ 秒）→ 失败 → 新浪 stock_zh_a_minute（备选）
```

- 2h 由 1h 数据每 2 根聚合
- 新浪返回 day/open/high/low/close/volume/amount，需映射 day→ts

### 3.3 代理与直连

- 在 `collect_and_save_context` 开始时，将 `push2his.eastmoney.com`、`quote.eastmoney.com` 加入 `NO_PROXY`
- 减少东财请求被代理拦截的概率

---

## 四、建议与操作指引

### 4.1 运行前环境检查

```powershell
# 取消代理（若不需要）
$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''

# 东财直连（若必须保留其他代理）
$env:NO_PROXY='push2his.eastmoney.com,quote.eastmoney.com'

# 运行分析
python run.py analyze --symbol 300617 --name 安靠智电
```

### 4.2 若东财持续不可用

项目已内置备选逻辑，无需额外配置：

- 日/周线 → 自动切换腾讯
- 分钟线 → 自动切换新浪

### 4.3 可选增强

| 方案 | 说明 |
|------|------|
| 配置 Tushare | 在 `.env` 设置 `TUSHARE_TOKEN`，作为日线/分钟线主源 |
| 调整数据源优先级 | 在 `project.json` 中将 `tushare` 置于 `akshare` 之前 |
| 换网络 | 使用手机热点或家庭网络，排除公司/校园网限制 |

---

## 五、小结

- **东财**：在代理或网络受限环境下易失败，需 NO_PROXY 或备选源。
- **腾讯/新浪**：作为备选可保证日线和分钟线获取成功率。
- **项目 pipeline**：已实现多源 + NO_PROXY，在当前环境下可正常完成分析。
