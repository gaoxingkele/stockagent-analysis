# StockAgent Web 平台实施方案

> 创建日期: 2026-04-25
> 版本: v1.0 (最终版, 所有决策已锁定)
> 总工期: 14-16 天
> 部署目标: Win10 办公电脑 + ngrok 公网映射

---

## 一、所有决策点(已锁定)

| # | 决策点 | 选定方案 |
|:-:|------|------|
| 1 | 注册赠送积分 | 100 积分(邀请注册再 +50,共 150) |
| 2 | 邀请人奖励 | +100 积分,无限层级 |
| 3 | 邀请码格式 | 1 字母 + 6 数字 (例 `A123456`) |
| 4 | 邀请方式 | 邀请码 / QR 海报 / 分享链接 三选一 |
| 5 | 海报内容 | 股票分析浓缩图 + 邀请人信息 + QR 码 |
| 6 | 好友链权限 | 管理员看全树, 用户只看上下级 |
| 7 | 单股分析消耗 | 20 积分(首次跑) |
| 8 | 缓存命中消耗 | 10 积分(同股当日复用) |
| 9 | 单用户并发 | 1 只(超出排队) |
| 10 | 失败处理 | 自动全额退款 |
| 11 | 管理员手机号 | `18606099618` (config 写死) |
| 12 | 验证码方案 | 开发 mock(console 打印) → 灰度白名单 → 正式备案后阿里云 SMS |
| 13 | 健康检查频率 | 工作日 9:00-16:00 每小时定时 + 全天手动触发 |
| 14 | 报告呈现 | 互动 HTML 详情页(替代 PDF), 按需导出 PDF |
| 15 | UI 设计风格 | Linear 风格(主) + Sentry 数据密集元素 |
| 16 | 多语言 | 默认 zh-CN, 可切 en-US / zh-TW (UTF-8) |
| 17 | 数据源顺序 | Tushare → AKShare → TDX (已生效) |
| 18 | 数据库 | SQLite(起步) → PostgreSQL(>1k 用户时迁移) |
| 19 | 项目目录 | `web/` 子目录, 共享 `src/stockagent_analysis/` |
| 20 | 进度推送 | SSE (Server-Sent Events) |
| 21 | **评分类型分离** | LLM 全量(20pt) / 量化评分(1pt) 两类分开计算 |
| 22 | 量化评分等级 | 复用 `weak_buy/hold/weak_sell` (与 LLM 一致, 阈值同步) |
| 23 | 默认行为 | 已有 LLM 评分的股票, 默认仅做量化跟踪 |
| 24 | 量化日内缓存 | 同日同股复用结果, 但仍扣 1 积分 |
| 25 | 一键跟踪 | 用户可一键跟踪自己所有 LLM 评分过的股票 |
| 26 | 自动订阅 | 用户可订阅 → 每日 16:30 cron 自动量化 → 异常推送 |
| 27 | 异常推送渠道 | 后续接飞书/钉钉机器人(P11+) |

---

## 二、技术栈

```
后端     FastAPI + uvicorn + sse-starlette
任务     asyncio + Redis pub/sub (无 Celery, 轻量)
定时     APScheduler (健康检查 + 订阅自动跟踪)
DB       SQLAlchemy 2.0 async + Alembic 迁移
缓存/MQ  Redis 7 (Docker Desktop)
认证     JWT (HS256) + Cookie
前端     Jinja2 + HTMX + Alpine.js + Tailwind CSS
图表     ECharts (走势 + 好友链树)
海报     Pillow + qrcode
PDF导出  WeasyPrint (HTML→PDF)
日志     structlog (JSON 格式) + 分级文件
i18n     Babel + flask-babel/fastapi-babel
推送     飞书/钉钉 webhook (httpx) [P11]
部署     NSSM (Win 服务) + ngrok / Cloudflare Tunnel
测试     pytest + pytest-asyncio
```

---

## 三、目录结构

```
stockagent-analysis/
├── src/stockagent_analysis/      # 现有(不动)
├── web/                          # 新建 ★
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI 入口
│   │   ├── config.py             # 配置 (admin 手机号等)
│   │   ├── core/
│   │   │   ├── db.py             # SQLAlchemy 引擎+session
│   │   │   ├── redis.py          # Redis 连接
│   │   │   ├── security.py       # JWT 加解密
│   │   │   ├── deps.py           # 依赖注入
│   │   │   └── i18n.py           # 多语言
│   │   ├── models/               # ORM 模型
│   │   │   ├── user.py
│   │   │   ├── transaction.py
│   │   │   ├── job.py
│   │   │   ├── result.py
│   │   │   ├── invite.py
│   │   │   ├── healthcheck.py
│   │   │   └── log.py
│   │   ├── schemas/              # Pydantic
│   │   ├── routers/              # API 路由
│   │   │   ├── auth.py
│   │   │   ├── users.py
│   │   │   ├── analysis.py
│   │   │   ├── jobs.py
│   │   │   ├── results.py
│   │   │   ├── stocks.py
│   │   │   ├── share.py
│   │   │   ├── healthcheck.py
│   │   │   ├── admin.py
│   │   │   └── pages.py          # SSR 模板渲染
│   │   ├── services/             # 业务逻辑
│   │   │   ├── auth_service.py
│   │   │   ├── points_service.py
│   │   │   ├── invite_service.py
│   │   │   ├── analysis_runner.py # 调用 v3 + progress_cb
│   │   │   ├── poster_generator.py
│   │   │   ├── healthcheck_service.py
│   │   │   ├── pdf_export.py
│   │   │   └── sms_service.py
│   │   ├── tasks/                # 定时任务
│   │   │   └── healthcheck_cron.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       ├── logging.py
│   │       └── i18n.py
│   ├── templates/                # Jinja2
│   │   ├── base.html
│   │   ├── components/
│   │   ├── pages/
│   │   └── share/
│   ├── static/
│   │   ├── css/                  # 编译后 Tailwind
│   │   ├── js/                   # HTMX + Alpine + ECharts + 自定义
│   │   ├── img/
│   │   └── fonts/
│   ├── locales/
│   │   ├── zh_CN/LC_MESSAGES/messages.po
│   │   ├── en_US/LC_MESSAGES/messages.po
│   │   └── zh_TW/LC_MESSAGES/messages.po
│   ├── alembic/                  # DB 迁移
│   │   ├── env.py
│   │   └── versions/
│   ├── tests/
│   ├── data/                     # SQLite + 上传文件
│   │   └── app.db
│   ├── logs/                     # 日志文件
│   │   ├── app.log
│   │   ├── error.log
│   │   ├── analysis.log
│   │   ├── healthcheck.log
│   │   ├── api.log
│   │   └── llm.log
│   ├── alembic.ini
│   ├── requirements.txt
│   ├── .env.example
│   ├── babel.cfg
│   └── README.md
└── ...
```

---

## 四、数据库 Schema (核心 9 张表)

### 4.1 users
```sql
id INT PK
phone VARCHAR(11) UNIQUE NOT NULL
nickname VARCHAR(50)
avatar_url VARCHAR(255)
points INT DEFAULT 0
is_admin BOOL DEFAULT FALSE     -- 18606099618 注册自动 = TRUE
language VARCHAR(8) DEFAULT 'zh-CN'
invite_code VARCHAR(8) UNIQUE   -- A123456
invited_by_user_id INT FK NULL
invite_path VARCHAR(500)        -- "1/5/12/" 物化路径
invite_count INT DEFAULT 0
invite_earned_points INT DEFAULT 0
status ENUM('active','suspended') DEFAULT 'active'
created_at, last_login_at
```

### 4.2 sms_codes
```sql
id, phone, code, expires_at, used, ip_address, created_at
INDEX (phone, created_at)
```

### 4.3 point_transactions (强一致到 result_id)
```sql
id INT PK
user_id INT FK
delta INT                        -- 负=扣 / 正=加
reason ENUM(
  'register_bonus',              -- 注册赠送
  'invite_new_user_bonus',       -- 新人通过邀请码注册得 50
  'invite_referrer_bonus',       -- 介绍人得 100
  'analyze',                     -- 分析消耗 -20
  'cache_hit',                   -- 命中缓存 -10
  'refund',                      -- 失败退款
  'recharge',                    -- 管理员充值
  'admin_revoke'                 -- 管理员撤销
)
related_result_id INT FK NULL    -- 关联 analysis_results
related_invite_id INT FK NULL    -- 关联 invite_relations
related_user_id INT FK NULL      -- 充值场景: 操作者
note TEXT                        -- 备注 (充值原因等)
balance_before INT
balance_after INT
created_at TIMESTAMP

INDEX (user_id, created_at DESC)
```

### 4.4 analysis_jobs
```sql
id, user_id, symbols_count, total_points_charged
status ENUM('pending','running','partial_done','done','failed')
created_at, finished_at
```

### 4.5 analysis_results (核心)
```sql
id INT PK
job_id INT FK
symbol, name, run_dir                                  -- quant_only 不一定有 run_dir
analysis_type ENUM('full','quant_only') NOT NULL       -- ★ 评分类型
parent_full_result_id INT FK NULL                      -- ★ quant_only 关联到最近一次 full 评分
is_cache_hit BOOL                -- full: 是否命中当日缓存; quant_only: 是否命中当日 quant 缓存
source_result_id INT FK NULL     -- 命中缓存指向源记录
points_charged INT               -- 20 / 10 / 1
status ENUM('queued','running','done','failed','refunded')
current_phase VARCHAR(50)        -- quant_only 直接 'done', 不跑 SSE
progress_pct INT DEFAULT 0
final_score FLOAT                -- full: 走融合公式; quant_only: = quant_score
decision_level VARCHAR(20)       -- 两类共用 weak_buy/hold/weak_sell 等级
quant_score FLOAT                -- 两类都有
quant_components_json JSON       -- ★ 量化 4 维触发明细
trader_decision VARCHAR(10)      -- quant_only 为 NULL
expert_scores_json JSON          -- quant_only 为 NULL
score_components_json JSON       -- quant_only 为 NULL
error_message TEXT
duration_sec INT
created_at, finished_at
INDEX (symbol, created_at DESC)              -- 走势查询
INDEX (status, created_at)                    -- 队列管理
INDEX (analysis_type, symbol, created_at)     -- ★ 按类型查走势
```

### 4.6 progress_events (SSE 持久化, 断线重连用)
```sql
id, result_id FK, phase_id, percent, message
data_json JSON                   -- 阶段附带数据
created_at
INDEX (result_id, created_at)
```

### 4.7 invite_relations
```sql
id, inviter_user_id FK, invitee_user_id FK
invite_method ENUM('code','qr','poster','link')
inviter_reward_points INT
invitee_reward_points INT
poster_result_id INT FK NULL     -- 海报来源股票
created_at
```

### 4.8 health_checks
```sql
id, triggered_by_user_id FK NULL  -- NULL = 定时任务
trigger_type ENUM('manual','cron')
total_items, passed, failed
duration_ms
details_json JSON                 -- 每项 (api_name, status, latency, error)
market_snapshot_json JSON         -- 大盘指数快照
created_at
INDEX (created_at DESC)
```

### 4.9 app_logs (关键事件)
```sql
id, level, module, message
user_id FK NULL, request_id
context_json JSON, traceback TEXT
created_at
INDEX (level, created_at)
INDEX (user_id, created_at)
```

### 4.10 subscriptions (★ 自动跟踪订阅)
```sql
id INT PK
user_id INT FK
symbol VARCHAR(10)
name VARCHAR(50)
enabled BOOL DEFAULT TRUE
auto_quant_enabled BOOL DEFAULT TRUE          -- 每日自动量化
notify_on_change BOOL DEFAULT TRUE            -- 决策等级变化时通知
notify_threshold_score_delta INT DEFAULT 5    -- final_score 变化超此值触发通知
notify_channels JSON                          -- ['feishu','dingtalk','email']
last_quant_at TIMESTAMP NULL
last_quant_result_id INT FK NULL
created_at
UNIQUE (user_id, symbol)
INDEX (enabled, auto_quant_enabled)            -- cron 扫描用
```

### 4.11 push_notifications (★ 异常推送日志, P11+)
```sql
id INT PK
user_id INT FK
subscription_id INT FK NULL
type ENUM('decision_change','score_alert','quant_failed','admin_message')
title VARCHAR(200)
content TEXT
channel ENUM('feishu','dingtalk','email','site')
status ENUM('pending','sent','failed')
sent_at TIMESTAMP
related_result_id INT FK NULL
created_at
```

---

## 五、API 设计 (40+ 接口)

### 认证
```
POST /api/auth/send-code         {phone}
POST /api/auth/verify            {phone, code, invite_code?}  → JWT
POST /api/auth/logout
GET  /api/me                     → {user, points, team_summary}
PATCH /api/me                    {nickname, language, avatar_url}
```

### 分析
```
POST /api/analyze/preview        {symbols: [...]}  → 预判每只类型/积分(不扣款)
POST /api/analyze                {symbols: [...], force_full?: [...]}  → 提交分析
                                   返回 {job_id, breakdown:[{symbol, type, points}], total_points}
POST /api/analyze/quant          {symbols: [...]} → 强制全部量化(快捷方式)
POST /api/analyze/track-all      → 一键跟踪我所有 LLM 评分过的股票 ★
                                   返回 {symbols_count, total_points, job_id}
GET  /api/jobs                   → 我的任务分页
GET  /api/jobs/{id}              → 任务概览(含每只 status + type)
GET  /api/jobs/{id}/stream       SSE 实时进度推送 ★ (仅 full 类型)
GET  /api/results/{id}           → 单股分析详情 JSON
GET  /api/results/{id}/pdf       → 实时 HTML→PDF 导出
GET  /api/results/{id}/poster    → 生成分享海报 PNG
DELETE /api/jobs/{id}            → 取消(仅 pending)
```

### 订阅 (★ 自动跟踪)
```
GET    /api/me/subscriptions               → 我的订阅列表
POST   /api/me/subscriptions               {symbol, notify_channels, threshold}
PATCH  /api/me/subscriptions/{id}          {enabled, threshold, channels}
DELETE /api/me/subscriptions/{id}
GET    /api/me/subscriptions/{id}/history  → 该订阅的量化跟踪历史
POST   /api/me/notify-channels             {feishu_webhook, dingtalk_webhook}
GET    /api/me/notifications               → 历史推送记录
```

### 股票
```
GET  /api/stocks/{symbol}/history?limit=20&type=     → 时序所有分析(走势)
                                                       type=all/full/quant_only 筛选
GET  /api/stocks/leaderboard                          → 全平台高分榜
GET  /api/stocks/recent                               → 最近分析的股票
GET  /api/stocks/{symbol}/can-quant                   → 检查是否可走量化(需有 LLM 历史)
```

### 邀请 & 团队
```
GET  /api/me/team                → 上下级 (用户视角)
GET  /api/me/invite-info         → 我的邀请码 + 累计成绩
POST /api/share/poster           {result_id} → 海报 PNG
GET  /api/invite/{code}          → 落地页数据 (介绍人头像/昵称)
```

### 积分
```
GET  /api/me/transactions?type=&page=  → 流水分页 (含关联 result link)
```

### 健康检查
```
POST /api/healthcheck/run        → 手动触发 → 返回 check_id
GET  /api/healthcheck/{id}/stream  SSE 实时输出每项结果
GET  /api/healthcheck/history    → 历史检查 (24h 趋势图)
```

### 管理员
```
GET  /api/admin/users?search=&page=
POST /api/admin/users/{id}/recharge  {amount, note}
PATCH /api/admin/users/{id}          {status, is_admin, points_adjust}
GET  /api/admin/relations            → 完整邀请树 (echarts 数据)
GET  /api/admin/logs?level=&module=&page=
GET  /api/admin/stats                → 平台总览
POST /api/admin/refund/{result_id}   → 手动退款
```

### 页面 (SSR)
```
GET  /                       仪表盘
GET  /login
GET  /analyze                提交分析(智能预判类型)
GET  /jobs/{id}              进度页 (HTMX SSE; quant 直接显示结果)
GET  /jobs/{id}/result       结果汇总(混合 full/quant)
GET  /stock/{symbol}         详情页(互动 HTML)
GET  /stock/{symbol}/history 走势页(双线: full/quant)
GET  /me                     个人中心 (含我的团队 + 订阅入口)
GET  /me/jobs                我的历史
GET  /me/transactions        消费记录
GET  /me/subscriptions       ★ 我的订阅
GET  /me/notifications       ★ 推送记录
GET  /share                  我的邀请
GET  /invite/{code}          邀请落地页
GET  /system/health          健康检查
GET  /admin/users
GET  /admin/recharge
GET  /admin/relations
GET  /admin/logs
```

---

## 五点五、双模式评分核心逻辑 ★

### 路由分发

```python
# services/analysis_runner.py
async def determine_analysis_type(symbol: str, force_full: bool) -> tuple[str, int]:
    """返回 (analysis_type, points_to_charge)。"""
    has_llm = await has_any_full_score(symbol)   # 任何用户曾跑过 full
    
    if not has_llm:
        return ("full", settings.points_analyze_full_cost)            # 必须 full
    
    if force_full:
        if await full_cache_hit_today(symbol):
            return ("full", settings.points_analyze_full_cache_hit)   # 当日复用 10pt
        return ("full", settings.points_analyze_full_cost)             # 重新跑 20pt
    
    return ("quant_only", settings.points_analyze_quant_cost)         # 默认量化 1pt


async def submit_analysis(user, symbols: list[str], force_full: list[str] = None):
    """统一入口: 多只股票, 自动判断类型, 一次扣款, 异步执行。"""
    breakdown = []
    total_points = 0
    
    for symbol in symbols:
        a_type, pts = await determine_analysis_type(
            symbol, force_full=symbol in (force_full or []))
        breakdown.append({"symbol": symbol, "type": a_type, "points": pts})
        total_points += pts
    
    if user.points < total_points:
        raise InsufficientPointsError(need=total_points, have=user.points)
    
    job = await create_job(user, breakdown, total_points)
    await deduct_points(user, total_points, reason="analyze", related_job_id=job.id)
    
    # 异步执行: full 走 v3 流水线 + SSE; quant 同步几秒内完成
    for item in breakdown:
        if item["type"] == "full":
            asyncio.create_task(run_full_async(job.id, item["symbol"]))
        else:
            asyncio.create_task(run_quant_async(job.id, item["symbol"]))
    
    return job
```

### 量化评分独立模块

```python
# services/quant_runner.py
QUANT_LEVEL_THRESHOLDS = [
    (80, "strong_buy"), (72, "weak_buy"), (62, "hold"),
    (52, "watch_sell"), (42, "weak_sell"),
]   # < 42 → strong_sell

async def run_quant_only(symbol: str, user_id: int, job_id: int) -> AnalysisResult:
    """10s 内完成, 不调 LLM。"""
    # 1. 当日缓存检查 (复用结果但仍扣 1pt, 已在外层扣过)
    cached = await get_quant_cache(symbol, _trading_day_today())
    if cached:
        return await save_quant_result_referencing(cached, user_id, job_id)
    
    # 2. 拉新鲜 Tushare 数据
    ts_enrich = await asyncio.to_thread(
        enrich_with_tushare, symbol, run_dir=None, use_cache=False)
    
    # 3. 算 quant_score
    quant_info = compute_quant_score(ts_enrich)
    score = quant_info["quant_score"]
    
    # 4. 决策等级映射 (沿用 weak_buy/hold/weak_sell 与 LLM 一致)
    level = "strong_sell"
    for thr, name in QUANT_LEVEL_THRESHOLDS:
        if score >= thr:
            level = name
            break
    
    # 5. 关联最近一次 full 评分(用于走势对比)
    parent = await find_latest_full_score(symbol)
    
    return await save_quant_result(
        symbol=symbol, user_id=user_id, job_id=job_id,
        analysis_type="quant_only",
        parent_full_result_id=parent.id if parent else None,
        final_score=score,
        decision_level=level,
        quant_score=score,
        quant_components_json=quant_info,
        status="done",
    )
```

### 一键跟踪

```python
# routers/analysis.py
@router.post("/api/analyze/track-all")
async def track_all(user=Depends(current_user)):
    """对当前用户所有曾经做过 full 评分的股票, 批量跑量化。"""
    symbols = await db.execute("""
        SELECT DISTINCT symbol FROM analysis_results
        WHERE user_id = :uid AND analysis_type = 'full' AND status = 'done'
    """, {"uid": user.id})
    
    if not symbols:
        raise HTTPException(400, "你还没有 LLM 全量评分历史")
    
    return await submit_analysis(user, list(symbols), force_full=[])
```

### 自动订阅 cron (P10)

```python
# tasks/subscription_cron.py
@scheduler.scheduled_job(CronTrigger(
    hour=16, minute=30, day_of_week='mon-fri', timezone='Asia/Shanghai'))
async def auto_quant_subscriptions():
    """每个交易日 16:30 自动跑所有订阅 → 异常推送。"""
    subs = await db.query(Subscription).filter_by(
        enabled=True, auto_quant_enabled=True).all()
    
    for sub in subs:
        try:
            new_result = await run_quant_only(sub.symbol, user_id=sub.user_id, job_id=None)
            
            # 比较与上次结果
            if sub.notify_on_change and sub.last_quant_result_id:
                last = await get_result(sub.last_quant_result_id)
                if (last.decision_level != new_result.decision_level or 
                    abs(last.final_score - new_result.final_score) >= sub.notify_threshold_score_delta):
                    await push_notification(sub, new_result, last)
            
            sub.last_quant_at = datetime.utcnow()
            sub.last_quant_result_id = new_result.id
        except Exception as e:
            logger.exception(f"sub {sub.id} failed")
            await push_failure_notification(sub, str(e))
```

---

## 六、互动 HTML 详情页设计

替代 PDF, 关键互动:

```
┌────────────────────────────────────────────────────┐
│  600126 杭钢股份         77.7  weak_buy  [生成海报] │
│                                          [导出 PDF] │
├────────────────────────────────────────────────────┤
│  ◐ 评分拆解 (点击各项展开 reason)                   │
│   ▸ 专家共识  79.78 × 0.43  = 34.31                 │
│   ▸ Judge 仲裁 76.98 × 0.275 = 21.17                │
│   ▸ 风控映射  50.00 × 0.155 = 7.75                  │
│   ▸ 量化 4 维 68.00 × 0.14  = 9.52  ← 点击展开 4 因子│
│         └ ADX +10  Winner -5  主力 +5  股东 +8     │
│   ▸ 一致性奖励 +5.00                                │
│                                                    │
│  ◐ FOMC 点阵图 (互动 svg, hover 看详情)             │
│       K走势=85 ●━━━━━━━━━━━━━━━●                   │
│       波浪=78        ●━━━━━━━━━●                   │
│       短线=72                  ●●                   │
│       马丁=82             ●━━━━●                   │
│                                                    │
│  ◐ K 线图 (ECharts, 可缩放/添加均线)                │
│                                                    │
│  ◐ 多空辩论 (4 轮, 可折叠)                         │
│      ▾ 第 1 轮  Bull "..." Bear "..."              │
│                                                    │
│  ◐ 入场策略 (3 张卡片, hover 加深)                  │
│      [回踩 10.05 → T1 11.0 SL 10.05 RR 3.2]       │
│                                                    │
│  ◐ 风控纪律 (止损 / 止盈 / PM 总结)                 │
│                                                    │
│  ◐ Quant 4 维详情 (hover 每个因子看 reason)        │
└────────────────────────────────────────────────────┘
```

---

## 七、Windows 10 部署完整步骤

### 7.1 一次性环境准备

```powershell
# 1. Python 3.11+ ✓ (已有 3.14)

# 2. Docker Desktop  https://docker.com/products/docker-desktop
#    安装后启用 WSL 2 后端

# 3. Redis 容器
docker run -d --restart=always -p 6379:6379 --name redis redis:7-alpine

# 4. ngrok (选 a)
choco install ngrok          # 或直接下载 .exe
ngrok config add-authtoken <YOUR_TOKEN>   # 注册免费账号
# 或 (选 b) Cloudflare Tunnel - 免费固定域名
choco install cloudflared

# 5. 中文字体 (海报生成用)
# 安装 思源黑体 / 微软雅黑 (Win10 自带)

# 6. wkhtmltopdf 或 GTK runtime (WeasyPrint 依赖)
choco install wkhtmltopdf
# 或 WeasyPrint 用 GTK: 下载 https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
```

### 7.2 Python 依赖

```powershell
cd web
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:
```
fastapi==0.110+
uvicorn[standard]==0.27+
sqlalchemy[asyncio]==2.0+
aiosqlite
alembic
redis>=5.0
sse-starlette
pyjwt[crypto]
passlib[bcrypt]
python-multipart
jinja2
httpx
pillow>=10
qrcode[pil]
weasyprint
structlog
babel
APScheduler                  # 定时任务
pydantic-settings
```

### 7.3 启动

```powershell
# 数据库初始化
alembic upgrade head

# 启动 (开发)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 启动 (生产)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1

# ngrok 映射
ngrok http 8000   # 拿到 https://xxx.ngrok-free.app
```

### 7.4 服务化 (开机自启)

**方案 A: NSSM**
```powershell
# 下载 nssm.exe → C:\tools\nssm.exe
nssm install stockagent-web "D:\aicoding\stockagent-analysis\web\venv\Scripts\python.exe"
nssm set stockagent-web AppParameters "-m uvicorn app.main:app --host 0.0.0.0 --port 8000"
nssm set stockagent-web AppDirectory "D:\aicoding\stockagent-analysis\web"
nssm set stockagent-web AppStdout "D:\aicoding\stockagent-analysis\web\logs\nssm-stdout.log"
nssm set stockagent-web AppStderr "D:\aicoding\stockagent-analysis\web\logs\nssm-stderr.log"
nssm start stockagent-web
```

**方案 B: Task Scheduler + startup.bat (更简单)**
```batch
@echo off
cd /d D:\aicoding\stockagent-analysis\web
docker start redis 2>nul
start /min cmd /c "ngrok http 8000"
call venv\Scripts\activate.bat
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

任务计划 → 触发器 = 用户登录 → 启动 startup.bat

---

## 八、实施分阶段 (16-19 天)

### P0 · 项目骨架 (0.5 天)
- 创建 web/ 目录结构
- requirements.txt + .env.example
- pyproject 配置
- README 启动说明

### P1 · 数据库 (1.5 天)
- SQLAlchemy 模型 (9 张表)
- Alembic 初始迁移
- 测试: pytest + 内存 SQLite
- 种子数据: admin (18606099618)

### P2 · 认证 + i18n (2 天)
- JWT 中间件
- mock SMS service (验证码 console + DB)
- 8888 admin 测试码
- Babel + 三语 .po 文件
- 路由: send-code / verify / me

### P3 · 积分 + 邀请 (1.5 天)
- points_service: 扣款/退款/充值
- invite_service: 邀请码生成/绑定
- 物化路径维护
- transaction 强一致到 result_id

### P4 · 分析核心 (2.5 天) ★ 含双模式
- analysis_runner 路由分发: full vs quant_only
- run_full_analysis: 改造 orchestrator_v3 加 progress_cb (7min)
- run_quant_only: 独立模块, 仅调用 enrich_with_tushare + compute_quant_score (10s)
  - 决策等级映射沿用 weak_buy/hold/weak_sell 阈值
  - parent_full_result_id 关联最近一次 full 评分
- determine_analysis_type: 智能预判 (新股票/已评分/用户强制)
- POST /analyze/preview 预判 API (前端展示扣分)
- POST /analyze/track-all 一键跟踪 API
- Redis pub/sub
- 缓存命中检测 (按交易日, full 和 quant 各自缓存)

### P5 · SSE 进度推送 (1.5 天)
- sse-starlette endpoint
- progress_events 持久化
- 断线重连恢复
- 失败自动退款

### P6 · 健康检查模块 (1 天)
- 14 项检测函数 (各 API 联通性)
- 大盘数据快照
- APScheduler 定时 (9-16 点每小时)
- 历史趋势图

### P7 · 日志系统 (0.5 天)
- structlog 配置
- 6 个分级日志文件
- 滚动 (50MB × 7)
- /admin/logs 查看

### P8 · 前端互动 HTML (3 天)
- Linear 风格基底
- 16 个页面 (SSR)
- HTMX 异步局部刷新
- ECharts 走势/树图
- Pillow 海报生成
- WeasyPrint PDF 导出

### P9 · 部署 (1 天)
- Docker Compose (Redis)
- NSSM 服务
- ngrok 配置
- E2E 测试
- 备份脚本 (DB + run_dir)

### P10 · 订阅自动跟踪 (1.5 天) ★
- subscriptions / push_notifications 表迁移
- POST/GET/PATCH/DELETE /me/subscriptions API
- APScheduler cron 任务 (每个交易日 16:30 自动跑订阅列表)
- 决策等级变化检测 + final_score 阈值检测
- /me/subscriptions 管理页面
- 站内通知中心 (右上角铃铛 + 红点)

### P11 · 推送渠道 (1 天)
- 飞书机器人 webhook
- 钉钉机器人 webhook
- 邮件 (可选, 需 SMTP 配置)
- /me/notify-channels 配置页

### P12 · 后续可选
- 阿里云 SMS 接入(备案后)
- 性能优化(连接池/缓存)
- 监控仪表盘
- 移动端适配(主要页面响应式)
- 微信公众号通知 (需要公众号资质)

---

## 九、安全 & 风控

- 手机号字段加密存储 (AES, 显示打码 138****1234)
- JWT 过期 2h, refresh token 7 天
- 同 IP 24h 注册限制 5 个 (Redis rate limit)
- 同手机号当日发送验证码限制 5 次
- 邀请关系一旦建立不可改 (admin 可手动撤销奖励)
- SSE endpoint 带 user 校验
- 管理员路由独立 middleware 校验 is_admin
- 文件下载 (PDF/海报) 校验所有权或公开状态

---

## 十、关键 UX 决策

1. **失败必有反馈** - LLM 调用失败时 SSE 推 "失败重试中" → 自动退款时推 "已退还 N 积分"
2. **海报必有水印** - 防止他人盗用截图; 含日期防过期分歧
3. **走势页必有标注** - 决策变化点高亮 (从 hold → buy 处加旗帜)
4. **首页必有大盘** - 上证/深证/创业板实时, 不需登录可见 (吸引访客注册)
5. **个人中心必显示团队** - 上级/下级 (鼓励继续邀请)

---

## 十一、首期上线后运营建议

- 邀请奖励初期可加倍 (新人 +100 介绍人 +200) 拉新种子用户
- 健康检查异常告警 → admin 收 push (后续接微信机器人)
- 每日运营报表 → /admin/stats (新增/活跃/分析次数/积分流出入)
- 定期清理 30 天前的 progress_events / app_logs (保留 analysis_results)

---

## 十二、下一步

P0 + P1 + P2 立刻开干, 预计 4 天可见雏形 (后端骨架 + 登录可用)。

后续会议要决定:
- 灰度内测时间表
- 阿里云备案启动时间 (域名 + 短信网关)
- 是否要做微信小程序版 (远期)
