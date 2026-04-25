#!/bin/bash
# stockagent-analysis 项目验证命令
# Ralph 每个迭代都会跑这个,失败时反馈给下次迭代

set -e

cd "$(dirname "$0")/../.."   # 回到项目根

# 1. web/ 模块 imports 健康
if [ -f web/app/config.py ]; then
  cd web
  python -c "from app.config import settings; print(f'[verify] config OK: {settings.app_name}')"
  python -c "from app.main import app; print(f'[verify] FastAPI OK: {len(app.routes)} routes')"
  cd ..
fi

# 2. pytest (有测试才跑)
if [ -d web/tests ] && find web/tests -name "test_*.py" -type f 2>/dev/null | grep -q .; then
  cd web
  python -m pytest tests/ -x --tb=short
  cd ..
else
  echo "[verify] no pytest tests yet (skipped)"
fi

# 3. 主项目核心模块 import 检查
python -c "
import sys
sys.path.insert(0, 'src')
from stockagent_analysis.tushare_enrich import compute_quant_score
from stockagent_analysis.agents_v3.orchestrator_v3 import _compose_final_score
print('[verify] core imports OK')
"

# 4. 关键 Python 文件语法检查
SYNTAX_FILES=$(find web/app src/stockagent_analysis -name "*.py" -type f 2>/dev/null | head -50)
if [ -n "$SYNTAX_FILES" ]; then
  python -m py_compile $SYNTAX_FILES
  echo "[verify] syntax OK ($(echo "$SYNTAX_FILES" | wc -l) files)"
fi

echo "[verify] ALL CHECKS PASSED"
