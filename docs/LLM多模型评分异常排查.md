# 多模型评分异常排查（Grok/Perplexity/Kimi 等）

## 现象：某模型对所有智能体评分都是 50

当某一大模型（如 Grok、Perplexity）对每个 Agent 的独立评分都显示为 50 时，通常是 **API 未正常返回** 或 **返回内容解析失败**，程序会回退到默认分 50。

## 常见原因与处理

### 1. 请求超时（Read timed out）

- **表现**：日志出现 `score_agent_analysis: exception provider=grok … Read timed out. (read timeout=8.0)` 或类似。
- **原因**：Grok / Kimi 等接口响应较慢，默认 8 秒内未返回即超时。
- **处理**：
  - 项目已把默认超时改为 **25 秒**（`configs/project.json` 中 `llm.request_timeout_sec`）。
  - 若仍超时，可把该值调大（如 40），或设置环境变量覆盖（若代码支持）。

### 2. Perplexity 返回 400 Bad Request

- **表现**：日志出现 `score_agent_analysis: exception provider=perplexity … 400 Client Error: Bad Request`。
- **原因**：请求体不符合 Perplexity 校验（模型名、参数范围、消息格式等）。
- **处理**：
  - 查看日志中是否打印了 `Perplexity API 400 … response: …`，根据返回内容里的错误说明调整。
  - 确认环境变量 `PERPLEXITY_MODEL` 为官方支持名称之一：`sonar`、`sonar-pro`、`sonar-deep-research`、`sonar-reasoning-pro`。
  - 确认 `temperature` 在 0–2、`max_tokens` 在允许范围内。

### 3. 解析失败（返回了内容但未识别为分数）

- **表现**：无超时/400，但日志出现 `score_agent_analysis: parse failed (provider=… agent=…) raw snippet: …`。
- **原因**：模型返回了文字说明而非单一数字，或格式与当前解析规则不一致。
- **处理**：根据日志中的 `raw snippet` 调整提示词或解析逻辑（见 `llm_client._parse_score_from_response`）。

## 运行并保存日志

便于排查时查看上述异常信息：

```bash
python run.py analyze --symbol 301221 --name 光庭信息 --providers grok,perplexity,kimi 2>&1 | tee run.log
```

Windows PowerShell：

```powershell
python run.py analyze --symbol 301221 --name 光庭信息 --providers grok,perplexity,kimi 2>&1 | Tee-Object -FilePath run.log
```

完成后在 `run.log` 中搜索 `score_agent_analysis` 或 `exception` 即可定位是哪家模型、哪个 Agent 出错。
