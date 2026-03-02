# -*- coding: utf-8 -*-
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_project_config(root: Path) -> dict[str, Any]:
    return load_json(root / "configs" / "project.json")


def load_agent_configs(root: Path) -> list[dict[str, Any]]:
    agent_dir = root / "configs" / "agents"
    return [load_json(p) for p in sorted(agent_dir.glob("*.json"))]


def split_agents(agent_cfgs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manager = None
    analysts: list[dict[str, Any]] = []
    for cfg in agent_cfgs:
        if not cfg.get("enabled", True):
            continue
        if cfg.get("dim_code") == "MASTER":
            manager = cfg
        else:
            analysts.append(cfg)
    if manager is None:
        raise ValueError("缺少管理智能体配置（dim_code=MASTER）")
    analysts.sort(key=lambda x: (not x.get("mandatory", False), -float(x.get("weight", 0.0)), x.get("agent_id", "")))
    return manager, analysts
