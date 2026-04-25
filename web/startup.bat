@echo off
REM =====================================================
REM StockAgent Web 启动脚本 (Win10)
REM 用法: 双击运行 / 或 Task Scheduler 开机自启
REM =====================================================

cd /d "%~dp0"
echo [%TIME%] StockAgent Web starting...

REM 启动 Redis (Docker Desktop, 不存在则创建)
docker start redis 2>nul
if errorlevel 1 (
  echo [%TIME%] Creating new Redis container...
  docker run -d --restart=always -p 6379:6379 --name redis redis:7-alpine
)

REM 启动 ngrok 后台 (HTTP 8000)
where ngrok >nul 2>&1
if not errorlevel 1 (
  echo [%TIME%] Starting ngrok tunnel...
  start /min cmd /c "ngrok http 8000"
)

REM 数据库迁移
echo [%TIME%] Running alembic migrations...
call venv\Scripts\activate.bat 2>nul
if errorlevel 1 (
  echo [WARN] venv not found, using system python
)
alembic upgrade head

REM 启动 uvicorn
echo [%TIME%] Starting uvicorn on port 8000...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

pause
