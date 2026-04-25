# StockAgent Web 一次性环境安装脚本 (PowerShell)
# 用法: 在 web/ 目录下右键 "用 PowerShell 运行"

$ErrorActionPreference = "Stop"
Write-Host "=== StockAgent Web 环境安装 ===" -ForegroundColor Green

# 1. 检查 Python
$pyVer = python --version 2>&1
Write-Host "[1/5] Python: $pyVer"
if (-not $pyVer.ToString().StartsWith("Python 3.1")) {
  Write-Host "  需要 Python 3.11+, 当前 $pyVer" -ForegroundColor Yellow
}

# 2. 创建虚拟环境
Write-Host "[2/5] 创建虚拟环境 venv/..."
if (-not (Test-Path "venv")) {
  python -m venv venv
}

# 3. 安装依赖
Write-Host "[3/5] 安装 pip 依赖..."
.\venv\Scripts\pip.exe install --upgrade pip
.\venv\Scripts\pip.exe install -r requirements.txt

# 4. .env 配置
Write-Host "[4/5] 检查 .env..."
if (-not (Test-Path ".env")) {
  Copy-Item .env.example .env
  Write-Host "  已生成 .env, 请编辑填入 SECRET_KEY / TUSHARE_TOKEN / LLM keys" -ForegroundColor Yellow
}

# 5. 数据库初始化
Write-Host "[5/5] 数据库迁移..."
.\venv\Scripts\python.exe -m alembic upgrade head

Write-Host ""
Write-Host "=== 完成 ===" -ForegroundColor Green
Write-Host "管理员账户:" -ForegroundColor Cyan
Write-Host "  手机号: 18606099618"
Write-Host "  默认密码: Ab18606099618"
Write-Host ""
Write-Host "下一步:"
Write-Host "  1. 启动 Redis: docker run -d --restart=always -p 6379:6379 --name redis redis:7-alpine"
Write-Host "  2. 启动应用: .\venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 9000"
Write-Host "  3. ngrok 映射: ngrok http 9000"
Write-Host "  4. 访问: http://localhost:9000/login"
