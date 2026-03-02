# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

# Windows 控制台 UTF-8 编码：避免中文和 Unicode 符号乱码
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 默认关闭代理，确保 Tushare 等数据源直连（参考 https://tushare.pro/document/2?doc_id=14）
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stockagent_analysis.main import main


if __name__ == "__main__":
    main()
