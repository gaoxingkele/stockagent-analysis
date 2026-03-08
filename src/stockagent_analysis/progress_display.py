# -*- coding: utf-8 -*-
"""集中进度渲染：流水线阶段指示、表格化进度、ANSI原地刷新。"""
from __future__ import annotations

import os
import sys
import threading
import time
import unicodedata
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .parallel_runner import ProviderProgress


# ────────────────────────────────────────────────
# 终端检测
# ────────────────────────────────────────────────

def _is_interactive_terminal() -> bool:
    """检测 stdout 是否为交互终端（vs 管道/重定向）。"""
    if os.environ.get("FORCE_INTERACTIVE", "").lower() in ("1", "true"):
        return True
    if os.environ.get("FORCE_NON_INTERACTIVE", "").lower() in ("1", "true"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


# ────────────────────────────────────────────────
# CJK 宽度辅助
# ────────────────────────────────────────────────

def _display_width(text: str) -> int:
    """计算终端显示宽度（CJK全角=2列，ASCII=1列）。"""
    w = 0
    for ch in text:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def _truncate_to_width(text: str, max_w: int) -> str:
    """截断文本使其显示宽度不超过 max_w。"""
    w = 0
    for i, ch in enumerate(text):
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > max_w:
            return text[:i]
        w += cw
    return text


def _pad_to_width(text: str, target_w: int) -> str:
    """用空格填充到目标显示宽度。"""
    current = _display_width(text)
    if current >= target_w:
        return text
    return text + " " * (target_w - current)


def _center_in_width(text: str, width: int) -> str:
    """在指定显示宽度内居中文本。"""
    tw = _display_width(text)
    if tw >= width:
        return _truncate_to_width(text, width)
    pad_total = width - tw
    left = pad_total // 2
    right = pad_total - left
    return " " * left + text + " " * right


# ────────────────────────────────────────────────
# Agent 中文名注册表
# ────────────────────────────────────────────────

class AgentNameRegistry:
    """从 analyst_cfg 构建 agent_id → 中文名 映射。"""

    def __init__(self, analyst_cfg: list[dict[str, Any]]):
        self._map: dict[str, str] = {}
        for c in analyst_cfg:
            agent_id = c.get("agent_id", "")
            # 优先用 name 字段，其次 role
            cn = c.get("name", "") or c.get("role", "") or agent_id
            self._map[agent_id] = cn

    def get(self, agent_id: str) -> str:
        return self._map.get(agent_id, agent_id)

    def as_dict(self) -> dict[str, str]:
        return dict(self._map)


# ────────────────────────────────────────────────
# 流水线阶段指示器
# ────────────────────────────────────────────────

_PIPELINE_STAGES = [
    "数据采集",
    "本地分析",
    "并行分析",
    "合并评分",
    "输出报告",
]

_STAGE_DONE = "[v]"
_STAGE_ACTIVE = "[>]"
_STAGE_PENDING = "[ ]"


class PipelineTracker:
    """流水线阶段追踪器：显示 ✓数据采集 → ▶本地分析 → ○并行分析 → …"""

    def __init__(self, stages: list[str] | None = None):
        self._stages = stages or list(_PIPELINE_STAGES)
        self._current_index = -1  # 尚未开始
        self._lock = threading.Lock()
        self._interactive = _is_interactive_terminal()

    def advance(self, stage_name: str | None = None) -> None:
        """推进到下一阶段（或按名称跳转）。"""
        with self._lock:
            if stage_name:
                for i, s in enumerate(self._stages):
                    if s == stage_name:
                        self._current_index = i
                        break
                else:
                    self._current_index += 1
            else:
                self._current_index += 1
            self._print()

    def _print(self) -> None:
        parts = []
        for i, s in enumerate(self._stages):
            if i < self._current_index:
                parts.append(f"{_STAGE_DONE}{s}")
            elif i == self._current_index:
                parts.append(f"{_STAGE_ACTIVE}{s}")
            else:
                parts.append(f"{_STAGE_PENDING}{s}")
        line = " → ".join(parts)
        print(f"\n=== {line} ===", flush=True)

    def finish(self) -> None:
        """所有阶段标记完成。"""
        with self._lock:
            self._current_index = len(self._stages)
            parts = [f"{_STAGE_DONE}{s}" for s in self._stages]
            line = " → ".join(parts)
            print(f"\n=== {line} ===", flush=True)


# ────────────────────────────────────────────────
# Provider 并行进度显示器（表格化）
# ────────────────────────────────────────────────

class ProviderProgressDisplay:
    """表格化进度渲染器：Provider列 × Agent行。

    交互模式：ANSI 转义码原地覆盖刷新。
    非交互模式：仅内容变化时追加输出。
    """

    _NAME_WIDTH = 12   # agent名称显示宽度（6个中文字）
    _COL_WIDTH = 8     # 每个provider列宽
    _NUM_WIDTH = 3     # 行号宽度

    def __init__(
        self,
        progresses: dict[str, "ProviderProgress"],
        agent_names: dict[str, str] | None = None,
        total_agents: int = 0,
        agent_order: list[str] | None = None,
        default_providers: list[str] | None = None,
    ):
        self._progresses = progresses
        self._agent_names = agent_names or {}
        self._total_agents = total_agents
        self._agent_order = agent_order or []
        self._providers = list(progresses.keys())
        self._default_providers = set(default_providers or [])
        self._interactive = _is_interactive_terminal()
        self._lock = threading.Lock()
        self._last_output = ""       # 非交互模式：变化检测
        self._last_line_count = 0    # 交互模式：上次输出的行数

    def _short_name(self, agent_id: str) -> str:
        """获取短显示名，去掉"分析师"后缀，截断到 _NAME_WIDTH 列宽。"""
        cn = self._agent_names.get(agent_id, "") or agent_id
        for suffix in ("分析师", "分析", "Agent", "agent"):
            if cn.endswith(suffix) and len(cn) > len(suffix):
                cn = cn[: -len(suffix)]
                break
        return _truncate_to_width(cn, self._NAME_WIDTH)

    def _cell_text(self, idx: int, agent_id: str, prog: "ProviderProgress") -> str:
        """计算单元格文本：agent×provider 交叉状态。

        ..  = 等待
        >研 = 研判中
        研  = 研判完
        >评 = 评分中
        数字 = 已出分
        --- = 超时/取消未完成
        """
        # 已有评分 → 直接显示
        if agent_id in prog.agent_scores:
            return str(int(prog.agent_scores[agent_id]))

        # 已结束且有错误 → 未完成的显示 ---
        if prog.finished and prog.error:
            return "---"

        # 已结束且无错误 → 应该都有分数（兜底）
        if prog.finished:
            return "--"

        # 尚未完成权重分配
        if not prog.weight_done:
            return ".."

        # 评分阶段（此时所有agent已完成研判）
        if prog.current_stage == "评分":
            if idx == prog.score_done and prog.current_agent == agent_id:
                return ">评"
            if idx > prog.score_done:
                return "研"  # 已研判，等待评分
            return "研"  # fallback

        # 研判阶段
        if prog.current_stage == "研判":
            if idx < prog.enrich_done:
                return "研"
            if idx == prog.enrich_done and prog.current_agent == agent_id:
                return ">研"
            return ".."

        return ".."

    def _provider_total(self, prog: "ProviderProgress") -> str:
        """加权总分：全部agent评分完成才显示数字。"""
        if not prog.agent_weights:
            if prog.finished and prog.error:
                return "---"
            return "--"
        if len(prog.agent_scores) < len(self._agent_order):
            if prog.finished and prog.error:
                return "---"
            return "--"
        # 计算加权总分
        total_w = sum(prog.agent_weights.values()) or 1.0
        weighted = sum(
            prog.agent_scores.get(aid, 50) * prog.agent_weights.get(aid, 0)
            for aid in self._agent_order
        )
        return f"{weighted / total_w:.1f}"

    def _elapsed_text(self, prog: "ProviderProgress") -> str:
        elapsed = time.time() - prog.start_time
        return f"{elapsed:.0f}s"

    def render(self) -> None:
        """渲染一帧进度信息。"""
        with self._lock:
            lines = self._build_lines()
            output = "\n".join(lines)

            if self._interactive:
                self._render_interactive(lines)
            else:
                self._render_noninteractive(output)

    def _col_sep(self, prev_p: str | None, cur_p: str) -> str:
        """列间分隔符。"""
        return " "

    def _build_lines(self) -> list[str]:
        providers = self._providers
        agents = self._agent_order
        NW = self._NUM_WIDTH
        AW = self._NAME_WIDTH
        CW = self._COL_WIDTH

        finished_count = sum(1 for p in self._progresses.values() if p.finished)
        total_p = len(providers)

        lines: list[str] = []

        # 标题行
        lines.append(f"[并行] {total_p}个默认大模型 | 完成 {finished_count}/{total_p} (候选按需激活)")

        # 表头行：行号 + 智能体 + 各provider名
        header = " " * NW + " " + _pad_to_width("智能体", AW)
        prev_p = None
        for p in providers:
            header += self._col_sep(prev_p, p) + _center_in_width(p, CW)
            prev_p = p
        lines.append(header)

        # 分隔线
        sep = " " * NW + " " + "-" * AW
        prev_p = None
        for p in providers:
            sep += self._col_sep(prev_p, p) + "-" * CW
            prev_p = p
        lines.append(sep)

        # Agent 行
        for i, agent_id in enumerate(agents):
            name = self._short_name(agent_id)
            row = f"{i + 1:>{NW}} " + _pad_to_width(name, AW)
            prev_p = None
            for p in providers:
                prog = self._progresses[p]
                cell = self._cell_text(i, agent_id, prog)
                row += self._col_sep(prev_p, p) + _center_in_width(cell, CW)
                prev_p = p
            lines.append(row)

        # 分隔线
        lines.append(sep)

        # 加权总分行
        total_row = " " * NW + " " + _pad_to_width("加权总分", AW)
        prev_p = None
        for p in providers:
            prog = self._progresses[p]
            t = self._provider_total(prog)
            total_row += self._col_sep(prev_p, p) + _center_in_width(t, CW)
            prev_p = p
        lines.append(total_row)

        # 耗时行
        time_row = " " * NW + " " + _pad_to_width("耗时", AW)
        prev_p = None
        for p in providers:
            prog = self._progresses[p]
            t = self._elapsed_text(prog)
            time_row += self._col_sep(prev_p, p) + _center_in_width(t, CW)
            prev_p = p
        lines.append(time_row)

        return lines

    def _render_interactive(self, lines: list[str]) -> None:
        """ANSI 转义码原地覆盖。"""
        # 先清除上次输出
        if self._last_line_count > 0:
            # 上移 n 行 + 每行清除
            sys.stdout.write(f"\033[{self._last_line_count}A")
            for _ in range(self._last_line_count):
                sys.stdout.write("\033[2K\n")
            sys.stdout.write(f"\033[{self._last_line_count}A")

        for line in lines:
            sys.stdout.write("\033[2K" + line + "\n")
        sys.stdout.flush()
        self._last_line_count = len(lines)

    def _render_noninteractive(self, output: str) -> None:
        """非交互模式：仅内容有变化时追加输出。"""
        if output != self._last_output:
            print(output, flush=True)
            self._last_output = output

    def render_final(self) -> None:
        """渲染最终状态（不再原地覆盖，直接追加）。"""
        with self._lock:
            lines = self._build_lines()
            if self._interactive and self._last_line_count > 0:
                # 清除最后的动态输出
                sys.stdout.write(f"\033[{self._last_line_count}A")
                for _ in range(self._last_line_count):
                    sys.stdout.write("\033[2K\n")
                sys.stdout.write(f"\033[{self._last_line_count}A")
                self._last_line_count = 0
            # 最终状态直接 print（不会再被覆盖）
            print("\n".join(lines), flush=True)


# ────────────────────────────────────────────────
# 刷新循环
# ────────────────────────────────────────────────

def run_display_loop(
    display: ProviderProgressDisplay,
    stop_event: threading.Event,
    interval: float = 1.0,
) -> None:
    """后台守护线程：按 interval 秒间隔刷新进度。"""
    while not stop_event.is_set():
        stop_event.wait(interval)
        if stop_event.is_set():
            break
        display.render()
