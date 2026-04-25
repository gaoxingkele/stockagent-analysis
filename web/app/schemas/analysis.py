"""分析相关 schemas。"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    symbols: list[str] = Field(..., min_length=1, max_length=50)
    force_full: list[str] = Field(default_factory=list)


class AnalyzePreviewItem(BaseModel):
    symbol: str
    type: str        # full / quant_only / cache_hit
    points: int
    last_full_score: float | None = None
    last_full_at: datetime | None = None


class AnalyzePreviewResponse(BaseModel):
    items: list[AnalyzePreviewItem]
    total_points: int
    user_points: int
    enough: bool


class JobBriefResponse(BaseModel):
    id: int
    user_id: int
    symbols_count: int
    total_points_charged: int
    status: str
    created_at: datetime
    finished_at: datetime | None
    breakdown: list[dict]   # [{symbol, type, status, score, level, ...}]


class ResultDetailResponse(BaseModel):
    id: int
    symbol: str
    name: str | None
    analysis_type: str
    is_cache_hit: bool
    parent_full_result_id: int | None
    status: str
    current_phase: str | None
    progress_pct: int
    points_charged: int
    final_score: float | None
    decision_level: str | None
    quant_score: float | None
    trader_decision: str | None
    expert_scores_json: dict | None
    score_components_json: dict | None
    quant_components_json: dict | None
    error_message: str | None
    duration_sec: int | None
    created_at: datetime
    finished_at: datetime | None
    run_dir: str | None
