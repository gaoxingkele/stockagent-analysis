# -*- coding: utf-8 -*-
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any


def build_run_dir(root: Path, symbol: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "output" / "runs" / f"{ts}_{symbol}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "messages").mkdir(parents=True, exist_ok=True)
    (run_dir / "submissions").mkdir(parents=True, exist_ok=True)
    (run_dir / "data").mkdir(parents=True, exist_ok=True)
    return run_dir


def get_agent_logger(run_dir: Path, agent_id: str) -> logging.Logger:
    logger = logging.getLogger(f"stockagent.{run_dir.name}.{agent_id}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir / "logs" / f"{agent_id}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
