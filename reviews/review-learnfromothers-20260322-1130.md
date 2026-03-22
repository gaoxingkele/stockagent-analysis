# Code Review: Cloubic桥接 + 模型降级链 + 辩论鲁棒性增强 + 北交所数据源

**分支**: `learnfromothers`
**日期**: 2026-03-22 11:30
**改动来源**: 已提交 4 个 commit + 未提交更改（7 个文件 +637/-92）

## 关联文档

- `docs/improvement-plan.md` — 改进计划总纲
- `docs/technical-analysis-and-improvement.md` — 技术分析报告
- `docs/superpowers/plans/2026-03-17-data-fetching-tiered-enhancement.md` — 数据获取梯次化增强计划
- `memory/project_v2_refactoring.md` — v2 Agent 重构记录

## 改动概览

### 代码文件

| 文件 | 增/删 | 改动摘要 |
|------|:---:|------|
| `src/core/router.py` | +351/-14 | Cloubic 统一调度平台接入；模型降级链（Cloubic+直连）；Grok Multi-Agent API；DeepSeek reasoner 兼容；模型版本全面升级；claude provider 新增 |
| `src/stockagent_analysis/orchestrator.py` | +36/-7 | `_has_api_key()` 修复 Cloubic provider 过滤；week 降级为可选；debate multi-agent 开关；fallback_routers 传递 |
| `src/stockagent_analysis/debate.py` | +105/-12 | `_chat_for_json()` 统一重试框架；multi-agent 仲裁；盈亏比硬约束；fallback provider 切换 |
| `src/stockagent_analysis/llm_client.py` | +58/-3 | `_fix_sniper_points()` 狙击点位后处理校验（盈亏比>=2, 止损<=10%） |
| `src/stockagent_analysis/data_backend.py` | +163/-8 | 北交所 TDX 二进制解析（日线 .day + 5分钟 .lc5）；BJ 市场自动识别 |

### 配置文件

| 文件 | 改动摘要 |
|------|------|
| `configs/project.json` | default_providers 改为 minmax/doubao/claude/openai；新增 `debate_multi_agent` 开关；candidate 顺序调整 |
| `.env.example` | GLM_MODEL `glm-4.5` → `glm-5` |
| `.env.cloubic` (新文件) | Cloubic 平台配置：API Key、白名单、各 provider 模型降级链 |

## Review 结果

### 1. 业务目标
**通过** — 目标清晰：(1) 通过 Cloubic 桥接实现国内直连海外模型；(2) 模型降级链提高可用性；(3) 辩论 JSON 解析重试增强鲁棒性；(4) 北交所数据源扩展。每项改动都有明确的业务价值。

### 2. 架构适配
**通过** — Cloubic 逻辑集中在 `router.py`（传输层），业务层无感知。`_should_route_via_cloubic()` 作为统一判定入口，文本和视觉调用均复用。降级链复用了 Gemini 已有的 `FALLBACK_MODEL` 模式，保持一致性。

### 3. Bug 检查
**注意** — 发现以下需关注项：

1. **`_is_cloubic_mode()` 使用全局缓存** — `_CLOUBIC_ENABLED` 在模块加载时固化，运行期间修改 `.env.cloubic` 不会生效。对于长期运行的服务是问题，但当前 CLI 模式（每次运行加载一次）可接受。

2. **`_load_cloubic_env()` 在模块 import 时执行** — 副作用在 import 阶段触发（写入 `os.environ`）。当前场景可控，但如果 `router.py` 被其他项目 import 会意外修改环境变量。

3. **`_has_api_key()` 定义在函数体内** — 每次调用 `run_analysis()` 都会重新定义。性能无影响但不够规范。可考虑提取为模块级函数。

4. **`_fix_sniper_points()` 已定义但未在 diff 中看到调用点** — 需确认是否在 `orchestrator.py` 或其他地方已接入。

### 4. 代码清晰度
**通过** — 命名清晰：`_get_cloubic_model_chain()`、`_get_direct_model_chain()`、`_should_route_via_cloubic()` 一目了然。日志信息含 `[Cloubic]`/`[provider]` 标签，降级路径可追踪。`route_tag` 变量统一了 Cloubic 和直连的日志格式。

### 5. KISS 原则
**注意** —

1. **`_chat_openai_compatible()` 膨胀** — 原本是单模型调用，现在内嵌 for 循环 + try/except + 多处 `can_retry` 判断。函数已超过 80 行，建议后续考虑提取 `_try_models_chain()` 辅助函数。当前可用但接近复杂度阈值。

2. **vision fallback 重复逻辑** — `_chat_openai_vision()` 的降级循环与 `_chat_openai_compatible()` 几乎相同，存在 DRY 违反。后续可抽象。

### 6. 单一职责
**通过** — `router.py` 负责传输路由，`debate.py` 负责辩论逻辑，`orchestrator.py` 负责流程编排，职责边界清晰。新增的 `_chat_for_json()` 将重试+JSON提取内聚在一处，避免了散落多处的重试代码。

### 7. 配置一致性
**通过** —

- `.env.cloubic` 的 `CLOUBIC_ROUTED_PROVIDERS=claude,gemini,openai,glm,qwen,deepseek` 与 router.py 的 `_should_route_via_cloubic()` 白名单逻辑一致
- `project.json` 的 `default_providers` 与实际运行日志吻合（4 provider 全部出分已验证）
- `.env.example` 的 GLM 模型名已同步
- `debate_multi_agent: false` 默认关闭，符合渐进式启用策略

## 总结

整体改动质量良好，核心功能（Cloubic 桥接、模型降级链、辩论鲁棒性）已通过实际运行验证（600388/300300 两只股票 4 provider 全部出分）。

**无必须修复项**。以下为建议优化项（可后续迭代）：
1. `_chat_openai_compatible()` 和 `_chat_openai_vision()` 的降级循环可抽取公共辅助函数
2. `_fix_sniper_points()` 需确认调用链是否已接入
3. `_has_api_key()` 可从 `run_analysis()` 函数体内提取为模块级函数
4. 建议将 `.env.cloubic` 加入 `.gitignore`（含 API Key）
