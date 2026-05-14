"""V12 评分相关 schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class V12ScoreResponse(BaseModel):
    """单股 V12 实时评分."""
    ts_code: str
    trade_date: str
    industry: Optional[str] = None
    buy_score: float
    sell_score: float
    r10_pred: float
    r20_pred: float
    sell_10_v6_prob: float
    sell_20_v6_prob: float
    quadrant: str
    v7c_recommend: bool
    pyr_velocity_20_60: Optional[float] = None
    f1_neg1: Optional[float] = None
    f2_pos1: Optional[float] = None


class V12RecommendItem(BaseModel):
    rank: int
    ts_code: str
    industry: Optional[str] = None
    buy_score: float
    sell_score: float
    r20_pred: float
    sell_20_v6_prob: Optional[float] = None
    quadrant: str
    v12_source: str   # V7c-main / V11-rescued-contradiction
    # V7c 6 铁律新增 zombie 字段
    is_zombie: Optional[bool] = None
    zombie_days_pct: Optional[float] = None
    ma60_slope_short: Optional[float] = None


class V12RecommendResponse(BaseModel):
    date: str
    total: int
    main_count: int
    rescued_count: int
    items: list[V12RecommendItem]


class V12ContradictionItem(BaseModel):
    ts_code: str
    industry: Optional[str] = None
    buy_score: float
    sell_score: float
    r20_pred: float
    sell_20_v6_prob: Optional[float] = None
    bull_prob: Optional[float] = None
    base_prob: Optional[float] = None
    bear_prob: Optional[float] = None
    trend_strength: Optional[str] = None
    key_pattern: Optional[str] = None
    v11_status: Optional[str] = None


class V12DatesResponse(BaseModel):
    dates: list[str]
    latest: Optional[str] = None


class RunMarketRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{8}$")


class RunLlmFilterRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{8}$")
    symbols: Optional[list[str]] = None   # None = 跑全部矛盾段
    limit: Optional[int] = Field(None, ge=1, le=300)   # 仅取 r20_pred 前 N


class V12JobResponse(BaseModel):
    job_id: int
    result_id: int
    status: str
    points_charged: int
    message: str
