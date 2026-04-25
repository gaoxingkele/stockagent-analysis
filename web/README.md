# StockAgent Web 平台

> 多用户股票分析 SaaS · 基于 stockagent v3.1 评分系统

## 快速启动 (Windows 10)

### 1. 一次性环境

```powershell
# Python 3.11+
python --version

# Redis (Docker Desktop)
docker run -d --restart=always -p 6379:6379 --name redis redis:7-alpine

# (可选) wkhtmltopdf for PDF 导出
choco install wkhtmltopdf

# (可选) ngrok 公网映射
ngrok config add-authtoken <YOUR_TOKEN>
```

### 2. 项目依赖

```powershell
cd web
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 复制配置
copy .env.example .env
# 编辑 .env 填入 SECRET_KEY / TUSHARE_TOKEN / LLM keys
```

### 3. 数据库初始化

```powershell
alembic upgrade head
```

首次启动会自动:
- 创建 SQLite 文件 `data/app.db`
- 注册管理员账户 (手机号 `18606099618`)

### 4. 启动

```powershell
# 开发模式
uvicorn app.main:app --host 0.0.0.0 --port 9000 --reload

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 9000

# 公网映射
ngrok http 9000
```

访问 http://localhost:9000

### 5. 服务化 (开机自启)

详见 `docs/web-platform-implementation-plan.md` 第七章。

## 目录说明

```
web/
├── app/              # FastAPI 应用
│   ├── core/         # 核心 (DB / Redis / Security / i18n)
│   ├── models/       # ORM 模型
│   ├── schemas/      # Pydantic 数据模型
│   ├── routers/      # API 路由
│   ├── services/     # 业务逻辑
│   ├── tasks/        # 定时任务 (健康检查)
│   └── middleware/   # 中间件 (Auth / Logging / i18n)
├── templates/        # Jinja2 模板 (Linear 风格)
├── static/           # CSS / JS / 字体
├── locales/          # 多语言 (zh_CN / en_US / zh_TW)
├── alembic/          # DB 迁移
├── tests/            # pytest
├── data/             # SQLite + 上传文件
└── logs/             # 应用日志 (分级文件)
```

## 一键安装 (推荐)

```powershell
# 在 web/ 目录下
.\install.ps1
```

会自动: 创建 venv + 装依赖 + 复制 .env + 跑迁移 + 显示管理员账户.

## 启动 (开机自启)

```powershell
# 方式 A: 双击 startup.bat
# 方式 B: NSSM 注册成 Windows 服务
nssm install stockagent-web "D:\aicoding\stockagent-analysis\web\venv\Scripts\python.exe"
nssm set stockagent-web AppParameters "-m uvicorn app.main:app --host 0.0.0.0 --port 9000"
nssm set stockagent-web AppDirectory "D:\aicoding\stockagent-analysis\web"
nssm start stockagent-web
```

## 管理员账户

```
手机号: 18606099618
默认密码: Ab18606099618
登录路径: /login → 切到 "管理员密码登录"
```

或万能验证码 8888 通过手机号验证码登录.

## 关键设计

- **用户**: 手机号 + 验证码登录, JWT 认证
- **积分**: 注册 100 / 邀请码注册 +50 / 介绍人 +100 / 单股分析 -20 / 缓存命中 -10
- **邀请**: 1 字母+6 数字 (例 `A123456`), 无限层级, 管理员可看全树
- **分析**: 调用现有 v3.1 流水线, SSE 实时推送进度
- **多语言**: 默认 zh-CN, 可切 en-US / zh-TW
- **健康检查**: 工作日 9-16 点每小时定时 + 全天手动

## 文档

- 完整实施方案: [`../docs/web-platform-implementation-plan.md`](../docs/web-platform-implementation-plan.md)
- 现有 v3.1 评分系统: [`../docs/STATUS-2026-04-24.md`](../docs/STATUS-2026-04-24.md)
