"""V12 服务层: 全市场推理 + LLM 视觉过滤 (异步 + SSE 进度).

设计:
- read_recommend(date)         读 v12_inference_final_*.csv (静态)
- score_stock_now(symbol, date) 单股实时评分 (同步 thread, 不入 job)
- submit_v12_market(db, user, date)  创建 v12_market job + 异步执行
- submit_v12_llm_filter(...)         创建 v12_llm_filter job + 异步执行

进度桥接:
  V12Scorer / V11VisionFilter 的 ProgressCb -> emit_progress (DB+Redis SSE)
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..core.db import get_session_factory
from ..models import (
    AnalysisJob, AnalysisResult, AnalysisType, JobStatus, ResultStatus, User,
    TransactionReason,
)
from .points_service import deduct_points, refund_points
from .progress_service import emit_progress, emit_done, emit_failed

logger = logging.getLogger(__name__)

# 把项目 src + 根 加进 sys.path (根: 复用 daily_dashboard.build_pools)
_PROJECT_SRC = settings.project_root / "src"
if str(_PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(_PROJECT_SRC))
if str(settings.project_root) not in sys.path:
    sys.path.insert(0, str(settings.project_root))

# 计费 (可放到 settings)
POINTS_V12_MARKET = 5
POINTS_V12_LLM_PER_STOCK = 1


# ──────── 静态读取 (csv) ────────

def _v12_dir() -> Path:
    return settings.project_root / "output" / "v12_inference"


def list_available_dates() -> list[str]:
    """扫 v12_inference/ 看跑过哪些日期."""
    d = _v12_dir()
    if not d.exists(): return []
    out = set()
    for p in d.glob("v12_inference_final_*.csv"):
        try:
            stem = p.stem.replace("v12_inference_final_", "")
            if len(stem) == 8 and stem.isdigit(): out.add(stem)
        except Exception: pass
    for p in d.glob("v7c_inference_*.csv"):
        try:
            stem = p.stem.replace("v7c_inference_", "")
            if len(stem) == 8 and stem.isdigit(): out.add(stem)
        except Exception: pass
    # v7c csv 也可能在 v7c_full_inference 目录
    v7c_d = settings.project_root / "output" / "v7c_full_inference"
    if v7c_d.exists():
        for p in v7c_d.glob("v7c_inference_*.csv"):
            stem = p.stem.replace("v7c_inference_", "")
            if len(stem) == 8 and stem.isdigit(): out.add(stem)
    return sorted(out)


def read_recommend(date: str) -> dict:
    """读 V12 final csv. 没有就 fallback v7c_inference_{date}.csv."""
    final_p = _v12_dir() / f"v12_inference_final_{date}.csv"
    v7c_p = settings.project_root / "output" / "v7c_full_inference" / f"v7c_inference_{date}.csv"
    used: Path
    if final_p.exists():
        used = final_p
    elif v7c_p.exists():
        used = v7c_p
    else:
        return {"date": date, "total": 0, "main_count": 0, "rescued_count": 0,
                 "items": [], "source_file": None}

    df = pd.read_csv(used, dtype={"ts_code": str})
    if "v12_source" not in df.columns:
        df["v12_source"] = "V7c-main"
    if "rank" not in df.columns:
        df = df.sort_values("r20_pred", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

    items = []
    for _, r in df.iterrows():
        items.append({
            "rank": int(r.get("rank", 0)),
            "ts_code": r["ts_code"],
            "industry": r.get("industry") if pd.notna(r.get("industry")) else None,
            "buy_score": float(r["buy_score"]),
            "sell_score": float(r["sell_score"]),
            "r20_pred": float(r["r20_pred"]),
            "sell_20_v6_prob": float(r["sell_20_v6_prob"]) if pd.notna(r.get("sell_20_v6_prob")) else None,
            "quadrant": str(r.get("quadrant", "")),
            "v12_source": str(r.get("v12_source", "V7c-main")),
            "is_zombie": bool(r["is_zombie"]) if pd.notna(r.get("is_zombie")) else None,
            "zombie_days_pct": float(r["zombie_days_pct"]) if pd.notna(r.get("zombie_days_pct")) else None,
            "ma60_slope_short": float(r["ma60_slope_short"]) if pd.notna(r.get("ma60_slope_short")) else None,
        })
    main = sum(1 for x in items if x["v12_source"] == "V7c-main")
    rescued = len(items) - main
    return {"date": date, "total": len(items),
             "main_count": main, "rescued_count": rescued,
             "items": items, "source_file": used.name}


def _dash_dir(date: str) -> Path:
    return settings.project_root / "output" / "daily_pick" / f"dashboard_{date}"


def list_pool_dates() -> list[str]:
    """扫 output/daily_pick/dashboard_* 看哪些日期有四池数据."""
    base = settings.project_root / "output" / "daily_pick"
    if not base.exists():
        return []
    out = set()
    for p in base.glob("dashboard_*"):
        stem = p.name.replace("dashboard_", "")
        if len(stem) == 8 and stem.isdigit() and (p / "poolA_system.csv").exists():
            out.add(stem)
    return sorted(out)


_POOL_FILES = {"A": "poolA_system.csv", "B": "pool_B.csv", "C": "pool_C.csv",
               "D": "poolD_ratio_ge5.csv", "E": "pool_E.csv"}
_POOL_META = {
    "A": {"name": "系统推荐", "dna": "抄底回调 (系统自营)", "desc": "V12.31 v7c主推 → ratio排序 → 行业cap → Top20"},
    "B": {"name": "自选", "dna": "用户关注", "desc": "自选清单打分排名"},
    "C": {"name": "基金重仓", "dna": "机构抱团 (动量追高)", "desc": "好基金优选池 (被≥2只高收益基金持有)"},
    "D": {"name": "追高动量", "dna": "高决断度 (与系统对立, 对照)", "desc": "全市场 ratio≥5"},
    "E": {"name": "外部信号", "dna": "外部程序 (每日固定调用)", "desc": "外部策略清单打分排名 (stock_benchmark)"},
}


def _pool_items(df) -> list[dict]:
    items = []
    for _, p in df.iterrows():
        ratio = p.get("ratio")
        items.append({
            "ts_code": str(p["ts_code"]),
            "name": p.get("name") if pd.notna(p.get("name")) else None,
            "industry": p.get("industry") if pd.notna(p.get("industry")) else None,
            "r20": float(p["buy_r20_score"]) if pd.notna(p.get("buy_r20_score")) else None,
            "pump_up": float(p["pump_score"]) if pd.notna(p.get("pump_score")) else None,
            "pump_down": float(p["pump_down_score"]) if pd.notna(p.get("pump_down_score")) else None,
            "ratio": float(ratio) if pd.notna(ratio) else None,
            "past5": float(p["past_r5"]) if pd.notna(p.get("past_r5")) else None,
            "v7c": bool(p["v7c_recommend"]) if pd.notna(p.get("v7c_recommend")) else False,
            "n_funds": int(p["n_funds"]) if pd.notna(p.get("n_funds")) else None,
        })
    return items


def _prev_scores_map(date: str) -> dict:
    """上一池日的 {ts_code: {r20, ratio}} (优先 scores_<prev>.parquet 全量, 否则四池CSV并集兜底)。"""
    base = settings.project_root / "output" / "daily_pick"
    prev_dates = [d for d in list_pool_dates() if d < date]
    if not prev_dates:
        return {}
    prev = prev_dates[-1]
    sp = base / f"scores_{prev}.parquet"
    out = {}
    if sp.exists():
        df = pd.read_parquet(sp)
        df["ts_code"] = df["ts_code"].astype(str)
        if "ratio" not in df.columns and "pump_score" in df.columns:
            df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
        for _, r in df.iterrows():
            out[r["ts_code"]] = {"r20": r.get("buy_r20_score"), "ratio": r.get("ratio")}
        return out
    # 兜底: 上一池日四池CSV并集
    pd_ = _dash_dir(prev)
    for fname in _POOL_FILES.values():
        fp = pd_ / fname
        if not fp.exists():
            continue
        df = pd.read_csv(fp, dtype={"ts_code": str})
        if "ratio" not in df.columns and "pump_score" in df.columns:
            df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
        for _, r in df.iterrows():
            out.setdefault(str(r["ts_code"]),
                           {"r20": r.get("buy_r20_score"), "ratio": r.get("ratio")})
    return out


def _dir(cur, prev, eps=1e-9):
    """涨跌方向: up/down/flat/None(无上日数据)。"""
    if cur is None or prev is None:
        return None
    diff = cur - prev
    if diff > eps:
        return "up"
    if diff < -eps:
        return "down"
    return "flat"


def read_four_pools(date: str) -> dict:
    """读四池看板 CSV (daily_dashboard build_pools 产出). 无则 total=0. 含与上一池日 r20/ratio 涨跌方向。"""
    d = _dash_dir(date)
    prev = _prev_scores_map(date)
    pools, total = {}, 0
    for key, fname in _POOL_FILES.items():
        fp = d / fname
        meta = dict(_POOL_META[key], key=key)
        if fp.exists():
            df = pd.read_csv(fp, dtype={"ts_code": str})
            if "ratio" not in df.columns and "pump_score" in df.columns:
                df["ratio"] = df["pump_score"] / (df["pump_down_score"] + 0.01)
            items = _pool_items(df)
            for it in items:
                pv = prev.get(it["ts_code"])
                it["r20_dir"] = _dir(it["r20"], pv["r20"] if pv else None)
                it["ratio_dir"] = _dir(it["ratio"], pv["ratio"] if pv else None)
            total += len(items)
            n_star = sum(1 for x in items if x["v7c"])
            pools[key] = {**meta, "count": len(items), "n_star": n_star, "items": items}
        else:
            pools[key] = {**meta, "count": 0, "n_star": 0, "items": []}
    # 大盘指数基准
    benchmark = []
    bm_fp = d / "benchmark_indices.csv"
    if bm_fp.exists():
        bm_df = pd.read_csv(bm_fp, dtype={"ts_code": str})
        for _, r in bm_df.iterrows():
            benchmark.append({
                "ts_code": r["ts_code"], "name": r["name"], "industry": r.get("industry", "指数"),
                "past_r5": float(r["past_r5"]) if pd.notna(r.get("past_r5")) else None,
            })
    return {"date": date, "total": total, "pools": pools, "benchmark": benchmark,
            "exists": total > 0, "prev_compared": bool(prev)}


# ── 池B 自选管理 (JSON 持久化, 复用 daily_dashboard 单一真相) ──
def get_watchlist_b() -> list[str]:
    import daily_dashboard as dd
    return dd.load_watchlist_b()


def add_watchlist_b(code: str) -> list[str]:
    import daily_dashboard as dd
    c = dd._norm_code(code)
    if not (c.endswith(".SZ") or c.endswith(".SH")):
        raise ValueError(f"非法代码 {code} (需 6位.SZ/.SH 或纯6位)")
    return dd.add_to_b(c)


def remove_watchlist_b(code: str) -> list[str]:
    import daily_dashboard as dd
    return dd.remove_from_b(code)


def score_pool_item(date: str, code: str) -> Optional[dict]:
    """从 scores_<date>.parquet 取单 code 的池-item (供池B加自选即时注入). 无则 None。"""
    import daily_dashboard as dd
    code = dd._norm_code(code)
    fp = settings.project_root / "output/daily_pick" / f"scores_{date}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    row = df[df["ts_code"] == code]
    if row.empty:
        return None
    r = row.iloc[0]
    industry = None
    try:
        basic = pd.read_parquet(settings.project_root / "output/tushare_cache/stock_basic.parquet")
        m = basic[basic["ts_code"] == code]
        if not m.empty:
            industry = m.iloc[0].get("industry")
    except Exception:
        pass
    f = lambda k: None if pd.isna(r.get(k)) else float(r.get(k))
    return {
        "ts_code": code, "name": (None if pd.isna(r.get("name")) else r.get("name")),
        "industry": industry,
        "r20": f("buy_r20_score") or 0.0, "pump_up": f("pump_score") or 0.0,
        "pump_down": f("pump_down_score") or 0.0, "ratio": f("ratio"),
        "past5": f("past_r5") or 0.0, "v7c": bool(r.get("v7c_recommend")), "n_funds": None,
    }


def read_contradiction(date: str) -> list[dict]:
    """读 V11 LLM filter 结果 (含 bull_prob 等)."""
    p = _v12_dir() / f"v11_filter_results_{date}.csv"
    if not p.exists():
        # fallback: 矛盾段 pending (未跑 LLM 时)
        p2 = _v12_dir() / f"v12_contradiction_pending_{date}.csv"
        if not p2.exists(): return []
        df = pd.read_csv(p2, dtype={"ts_code": str})
        return [{
            "ts_code": r["ts_code"], "industry": r.get("industry"),
            "buy_score": float(r["buy_score"]), "sell_score": float(r["sell_score"]),
            "r20_pred": float(r["r20_pred"]),
            "sell_20_v6_prob": float(r["sell_20_v6_prob"]) if pd.notna(r.get("sell_20_v6_prob")) else None,
            "v11_status": "pending",
        } for _, r in df.iterrows()]
    df = pd.read_csv(p, dtype={"ts_code": str})
    items = []
    for _, r in df.iterrows():
        items.append({
            "ts_code": r["ts_code"],
            "industry": r.get("industry") if pd.notna(r.get("industry")) else None,
            "buy_score": float(r["buy_score"]) if pd.notna(r.get("buy_score")) else None,
            "sell_score": float(r["sell_score"]) if pd.notna(r.get("sell_score")) else None,
            "r20_pred": float(r["r20_pred"]) if pd.notna(r.get("r20_pred")) else None,
            "sell_20_v6_prob": float(r["sell_20_v6_prob"]) if pd.notna(r.get("sell_20_v6_prob")) else None,
            "bull_prob": float(r["bull_prob"]) if pd.notna(r.get("bull_prob")) else None,
            "base_prob": float(r["base_prob"]) if pd.notna(r.get("base_prob")) else None,
            "bear_prob": float(r["bear_prob"]) if pd.notna(r.get("bear_prob")) else None,
            "trend_strength": str(r["trend_strength"]) if pd.notna(r.get("trend_strength")) else None,
            "key_pattern": str(r["key_pattern"]) if pd.notna(r.get("key_pattern")) else None,
            "v11_status": str(r.get("v11_status", "")),
        })
    return items


# ──────── 单股实时 (同步 thread) ────────

async def score_stock_now(symbol: str, date: str) -> dict:
    def _do():
        from stockagent_analysis.v12_scoring import V12Scorer
        scorer = V12Scorer.get(settings.project_root)
        return scorer.score_stock(symbol.upper(), date)
    return await asyncio.to_thread(_do)


# ──────── 异步 job ────────

async def submit_v12_market(
    db: AsyncSession, user: User, date: str,
) -> AnalysisResult:
    """创建 v12_market job (1 个 result), 扣分, 异步执行."""
    pts = POINTS_V12_MARKET
    if user.points < pts:
        from .points_service import InsufficientPointsError
        raise InsufficientPointsError(need=pts, have=user.points)

    job = AnalysisJob(
        user_id=user.id, symbols_count=1, total_points_charged=pts,
        status=JobStatus.running,
    )
    db.add(job); await db.flush()

    rec = AnalysisResult(
        job_id=job.id, user_id=user.id, symbol=f"V12_MARKET_{date}",
        analysis_type=AnalysisType.v12_market, points_charged=pts,
        status=ResultStatus.queued, progress_pct=0, current_phase="queued",
    )
    db.add(rec); await db.flush()

    await deduct_points(db, user, pts,
                         reason=TransactionReason.analyze_quant,
                         related_result_id=rec.id,
                         note=f"V12 全市场推理 {date}", auto_commit=False)
    await db.commit()

    factory = get_session_factory()
    asyncio.create_task(_do_v12_market(factory, rec.id, date))
    return rec


async def submit_v12_llm_filter(
    db: AsyncSession, user: User, date: str,
    symbols: Optional[list[str]] = None, limit: Optional[int] = None,
) -> AnalysisResult:
    """创建 v12_llm_filter job. symbols=None 时跑当日全部矛盾段 (限速保护用 limit)."""
    # 解析 symbols
    if symbols is None:
        contra = read_contradiction(date)
        symbols = [c["ts_code"] for c in contra if c.get("v11_status") in ("", "pending", None)]
        if not symbols:
            # 也许还没跑过 V12, 或者矛盾段 pending csv 不存在 - 报错
            raise ValueError(f"未找到 {date} 矛盾段清单, 请先跑 V12 全市场推理")
    if limit and len(symbols) > limit:
        symbols = symbols[:limit]

    n = len(symbols)
    if n == 0:
        raise ValueError("矛盾段股票列表为空")
    pts = n * POINTS_V12_LLM_PER_STOCK
    if user.points < pts:
        from .points_service import InsufficientPointsError
        raise InsufficientPointsError(need=pts, have=user.points)

    job = AnalysisJob(
        user_id=user.id, symbols_count=n, total_points_charged=pts,
        status=JobStatus.running,
    )
    db.add(job); await db.flush()

    rec = AnalysisResult(
        job_id=job.id, user_id=user.id, symbol=f"V12_LLM_{date}_{n}",
        analysis_type=AnalysisType.v12_llm_filter, points_charged=pts,
        status=ResultStatus.queued, progress_pct=0, current_phase="queued",
        extra_data_json={"date": date, "symbols": symbols, "n": n},
    )
    db.add(rec); await db.flush()

    await deduct_points(db, user, pts,
                         reason=TransactionReason.analyze_quant,
                         related_result_id=rec.id,
                         note=f"V12 LLM 视觉过滤 {date} ({n} 股)",
                         auto_commit=False)
    await db.commit()

    factory = get_session_factory()
    asyncio.create_task(_do_v12_llm_filter(factory, rec.id, date, symbols))
    return rec


# ──────── 异步执行 ────────

def _make_progress_bridge(loop: asyncio.AbstractEventLoop, factory, result_id: int):
    """同步线程内调用 -> 转回 asyncio loop emit_progress."""
    async def _emit(phase, pct, msg, data):
        async with factory() as db:
            try:
                await emit_progress(db, result_id, phase_id=phase[:50],
                                     percent=int(max(0, min(100, pct))),
                                     message=msg[:500], data=data)
            except Exception as e:
                logger.error("emit_progress fail: %s", e)

    def cb(phase: str, pct: int, msg: str, data):
        asyncio.run_coroutine_threadsafe(_emit(phase, pct, msg, data), loop)
    return cb


async def _do_v12_market(factory, result_id: int, date: str):
    async with factory() as db:
        rs = await db.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
        rec = rs.scalar_one_or_none()
        if rec is None: return
        job_id = rec.job_id
        rec.status = ResultStatus.running
        rec.current_phase = "starting"
        await db.commit()
        t0 = time.time()

        loop = asyncio.get_event_loop()
        cb = _make_progress_bridge(loop, factory, result_id)

        try:
            def _run():
                # 四池统一构建 (复用 daily_dashboard.build_pools, CLI/web 单一真相)
                import daily_dashboard as dd
                res = dd.build_pools(date, cb=cb, write_csv=True)
                p = res["pools"]
                return {
                    "n_pool_A": len(p["A"]["items"]), "n_A_universe": p["A"].get("n_pool", 0),
                    "n_pool_B": len(p["B"]["items"]), "n_pool_C": len(p["C"]["items"]),
                    "n_pool_D": len(p["D"]["items"]),
                    "n_main": len(p["A"]["items"]),
                }
            stats = await asyncio.to_thread(_run)

            async with factory() as db2:
                rs2 = await db2.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
                rec2 = rs2.scalar_one_or_none()
                if rec2 is not None:
                    rec2.status = ResultStatus.done
                    rec2.progress_pct = 100
                    rec2.current_phase = "done"
                    rec2.duration_sec = int(time.time() - t0)
                    rec2.finished_at = datetime.now(timezone.utc)
                    rec2.extra_data_json = {"date": date, **stats}
                    rec2.final_score = float(stats["n_main"])  # 主推数量当 final_score
                    rec2.decision_level = "v12_market"
                    await db2.commit()
                await emit_done(db2, result_id,
                                 final_score=float(stats["n_main"]),
                                 decision_level="v12_market",
                                 extra={"type": "v12_market", **stats})
                # job status
                jrs = await db2.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
                job = jrs.scalar_one_or_none()
                if job is not None:
                    job.status = JobStatus.done
                    job.finished_at = datetime.now(timezone.utc)
                    await db2.commit()
        except Exception as e:
            logger.exception("v12_market 失败: %s", e)
            await _mark_failed(factory, result_id, job_id, str(e))


async def submit_v12_update(
    db: AsyncSession, user: User, date: Optional[str] = None,
) -> AnalysisResult:
    """创建 v12_update job: 拉最新交易日数据 + 重算特征 + 重跑五池 (异步)."""
    pts = POINTS_V12_MARKET
    if user.points < pts:
        from .points_service import InsufficientPointsError
        raise InsufficientPointsError(need=pts, have=user.points)

    job = AnalysisJob(
        user_id=user.id, symbols_count=1, total_points_charged=pts,
        status=JobStatus.running,
    )
    db.add(job); await db.flush()

    rec = AnalysisResult(
        job_id=job.id, user_id=user.id, symbol=f"V12_UPDATE_{date or 'latest'}",
        analysis_type=AnalysisType.v12_market, points_charged=pts,
        status=ResultStatus.queued, progress_pct=0, current_phase="queued",
    )
    db.add(rec); await db.flush()

    await deduct_points(db, user, pts,
                         reason=TransactionReason.analyze_quant,
                         related_result_id=rec.id,
                         note=f"V12 更新数据+重跑五池 {date or 'latest'}", auto_commit=False)
    await db.commit()

    factory = get_session_factory()
    asyncio.create_task(_do_v12_update(factory, rec.id, date))
    return rec


async def _do_v12_update(factory, result_id: int, date: Optional[str]):
    """拉最新交易日数据 → 增量算特征 → build_pools 五池 (复用 daily_review.update_data)."""
    async with factory() as db:
        rs = await db.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
        rec = rs.scalar_one_or_none()
        if rec is None: return
        job_id = rec.job_id
        rec.status = ResultStatus.running
        rec.current_phase = "starting"
        await db.commit()
        t0 = time.time()

        loop = asyncio.get_event_loop()
        cb = _make_progress_bridge(loop, factory, result_id)

        try:
            def _run():
                import sys as _sys
                root = str(settings.project_root)
                if root not in _sys.path:
                    _sys.path.insert(0, root)
                import daily_review as dr
                import daily_dashboard as dd
                # update_data 占 0-55%, build_pools 的 0-100 映射到 55-100%
                def cb_build(phase, pct, msg, data):
                    cb(phase, 55 + int(pct * 0.45), msg, data)
                D = dr.update_data(date or None, cb=cb)
                # 更新 stock_benchmark 数据 + 生成 SEMAS / 池E 外部推荐 (不阻塞主流程)
                cb("semas", 56, "更新 SEMAS 外部推荐...", {})
                try:
                    dd.update_and_generate_semas()
                except Exception:
                    pass  # SEMAS 失败不影响主池
                # 池E: 日线数据已由上一步更新, 只需重跑 final_best_combo 生成
                cb("pool_e", 57, "生成池E 综合 Top30...", {})
                try:
                    dd.generate_pool_e_only()
                except Exception:
                    pass  # 池E 失败不影响主池
                res = dd.build_pools(D, cb=cb_build, write_csv=True)
                p = res["pools"]
                return {
                    "date": D,
                    "n_pool_A": len(p["A"]["items"]), "n_A_universe": p["A"].get("n_pool", 0),
                    "n_pool_B": len(p["B"]["items"]), "n_pool_C": len(p["C"]["items"]),
                    "n_pool_D": len(p["D"]["items"]), "n_pool_E": len(p["E"]["items"]),
                    "n_main": len(p["A"]["items"]),
                }
            stats = await asyncio.to_thread(_run)
            done_date = stats["date"]

            async with factory() as db2:
                rs2 = await db2.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
                rec2 = rs2.scalar_one_or_none()
                if rec2 is not None:
                    rec2.status = ResultStatus.done
                    rec2.progress_pct = 100
                    rec2.current_phase = "done"
                    rec2.duration_sec = int(time.time() - t0)
                    rec2.finished_at = datetime.now(timezone.utc)
                    rec2.extra_data_json = {**stats}
                    rec2.final_score = float(stats["n_main"])
                    rec2.decision_level = "v12_update"
                    await db2.commit()
                await emit_done(db2, result_id,
                                 final_score=float(stats["n_main"]),
                                 decision_level="v12_update",
                                 extra={"type": "v12_update", **stats})
                jrs = await db2.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
                job = jrs.scalar_one_or_none()
                if job is not None:
                    job.status = JobStatus.done
                    job.finished_at = datetime.now(timezone.utc)
                    await db2.commit()
        except Exception as e:
            logger.exception("v12_update 失败: %s", e)
            await _mark_failed(factory, result_id, job_id, str(e))


async def _do_v12_llm_filter(factory, result_id: int, date: str, symbols: list[str]):
    async with factory() as db:
        rs = await db.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
        rec = rs.scalar_one_or_none()
        if rec is None: return
        job_id = rec.job_id
        rec.status = ResultStatus.running
        rec.current_phase = "starting"
        await db.commit()
        t0 = time.time()

        loop = asyncio.get_event_loop()
        cb = _make_progress_bridge(loop, factory, result_id)

        try:
            cloubic = os.environ.get("CLOUBIC_API_KEY")
            if not cloubic:
                # 尝试从 .env / .env.cloubic 读
                from dotenv import load_dotenv
                load_dotenv(settings.project_root / ".env.cloubic")
                load_dotenv(settings.project_root / ".env")
                cloubic = os.environ.get("CLOUBIC_API_KEY")
            if not cloubic:
                raise RuntimeError("CLOUBIC_API_KEY 未配置")

            def _run():
                from stockagent_analysis.v11_vision import V11VisionFilter
                f = V11VisionFilter.get(settings.project_root, cloubic)
                results = f.filter_batch(symbols, date, cb=cb)
                # 保存 v11_filter_results_{date}.csv (与 v12_llm_filter_0508.py 兼容)
                out_dir = settings.project_root / "output" / "v12_inference"
                out_dir.mkdir(parents=True, exist_ok=True)
                df = pd.DataFrame(results)
                # 合并矛盾段元数据 (buy/sell/r20) 进结果
                pending_p = out_dir / f"v12_contradiction_pending_{date}.csv"
                if pending_p.exists():
                    pend = pd.read_csv(pending_p, dtype={"ts_code": str})
                    df = df.merge(pend, on="ts_code", how="left", suffixes=("", "_meta"))
                df.to_csv(out_dir / f"v11_filter_results_{date}.csv",
                          index=False, encoding="utf-8-sig")
                rescued = df[(df["status"] == "ok") & (df["bull_prob"] >= 0.5)] if "status" in df.columns else df.iloc[0:0]
                # 同时刷 final
                main_p = out_dir / f"v12_inference_{date}.csv"
                final_p = out_dir / f"v12_inference_final_{date}.csv"
                if main_p.exists():
                    main_df = pd.read_csv(main_p, dtype={"ts_code": str})
                    main_df["v12_source"] = "V7c-main"
                    if len(rescued) > 0:
                        # 标准化救出股的列名
                        r2 = rescued.rename(columns={"status": "v11_status"}).copy()
                        r2["v12_source"] = "V11-rescued-contradiction"
                        common_cols = [c for c in main_df.columns if c in r2.columns]
                        v12 = pd.concat([main_df[common_cols], r2[common_cols]], ignore_index=True)
                    else:
                        v12 = main_df
                    v12 = v12.sort_values("r20_pred", ascending=False).reset_index(drop=True)
                    v12["rank"] = v12.index + 1
                    v12.to_csv(final_p, index=False, encoding="utf-8-sig")
                ok = sum(1 for r in results if r.get("status") == "ok")
                fail_img = sum(1 for r in results if r.get("status") == "no_image")
                fail_parse = sum(1 for r in results if r.get("status") == "parse_error")
                fail_llm = sum(1 for r in results if r.get("status") == "llm_error")
                return {
                    "n_total": len(results), "n_ok": ok,
                    "n_no_image": fail_img, "n_parse_err": fail_parse,
                    "n_llm_err": fail_llm,
                    "n_rescued": int(len(rescued)),
                }
            stats = await asyncio.to_thread(_run)

            async with factory() as db2:
                rs2 = await db2.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
                rec2 = rs2.scalar_one_or_none()
                if rec2 is not None:
                    rec2.status = ResultStatus.done
                    rec2.progress_pct = 100
                    rec2.current_phase = "done"
                    rec2.duration_sec = int(time.time() - t0)
                    rec2.finished_at = datetime.now(timezone.utc)
                    rec2.extra_data_json = {**(rec2.extra_data_json or {}), "stats": stats}
                    rec2.final_score = float(stats["n_rescued"])
                    rec2.decision_level = "v12_llm_filter"
                    await db2.commit()
                await emit_done(db2, result_id,
                                 final_score=float(stats["n_rescued"]),
                                 decision_level="v12_llm_filter",
                                 extra={"type": "v12_llm_filter", **stats})
                jrs = await db2.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
                job = jrs.scalar_one_or_none()
                if job is not None:
                    job.status = JobStatus.done
                    job.finished_at = datetime.now(timezone.utc)
                    await db2.commit()
        except Exception as e:
            logger.exception("v12_llm_filter 失败: %s", e)
            await _mark_failed(factory, result_id, job_id, str(e))


async def _mark_failed(factory, result_id: int, job_id: int, err: str):
    async with factory() as db:
        rs = await db.execute(select(AnalysisResult).where(AnalysisResult.id == result_id))
        rec = rs.scalar_one_or_none()
        if rec is None: return
        rec.status = ResultStatus.failed
        rec.error_message = err[:500]
        rec.finished_at = datetime.now(timezone.utc)
        await db.commit()
        # 退款
        ur = await db.execute(select(User).where(User.id == rec.user_id))
        user = ur.scalar_one_or_none()
        if user:
            try:
                await refund_points(db, user, rec.points_charged,
                                     related_result_id=rec.id,
                                     note=f"V12 失败自动退: {err[:80]}")
                rec.status = ResultStatus.refunded
                await db.commit()
            except Exception:
                pass
        try:
            await emit_failed(db, result_id, error=err, refunded=rec.points_charged)
        except Exception: pass
        # job
        jrs = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
        job = jrs.scalar_one_or_none()
        if job:
            job.status = JobStatus.failed
            job.finished_at = datetime.now(timezone.utc)
            await db.commit()


# ── SEMAS 推荐面板 ──

def read_semas_stocks(date: str) -> dict:
    """读 SEMAS 推荐 CSV (build_pools 产出 semas_stocks.csv)."""
    d = _dash_dir(date)
    fp = d / "semas_stocks.csv"
    if not fp.exists():
        return {"date": date, "exists": False, "count": 0, "items": []}
    df = pd.read_csv(fp, dtype={"ts_code": str})
    items = []
    for _, r in df.iterrows():
        items.append({
            "ts_code": r["ts_code"],
            "name": r.get("name") if pd.notna(r.get("name")) else r.get("stock_name", ""),
            "industry": r.get("industry") if pd.notna(r.get("industry")) else None,
            "merged_rank": int(r.get("merged_rank", 0) or 0),
            "hit_count": int(r.get("hit_count", 0) or 0),
            "recommended_horizons": str(r.get("recommended_horizons", "")) if pd.notna(r.get("recommended_horizons")) else "",
            "avg_rank": round(float(r.get("avg_rank", 0) or 0), 1),
            "r20": float(r.get("buy_r20_score", 0) or 0),
            "pump_up": float(r.get("pump_score", 0) or 0),
            "pump_down": float(r.get("pump_down_score", 0) or 0),
            "ratio": float(r.get("ratio", 0) or 0),
            "past5": float(r.get("past_r5", 0) or 0),
            "v7c": bool(r.get("v7c_recommend", False)),
        })
    return {"date": date, "exists": True, "count": len(items), "items": items}


# ── 大盘指数指标 ──

_INDEX_CODES = [
    ("000001.SH", "上证指数", "指数"),
    ("000300.SH", "沪深300", "指数"),
    ("000905.SH", "中证500", "指数"),
    ("000852.SH", "中证1000", "指数"),
    ("399006.SZ", "创业板指", "指数"),
]

_ETF_CODES = [
    # 宽基
    ("510050.SH", "上证50ETF", "宽基"),
    ("510300.SH", "沪深300ETF", "宽基"),
    ("510500.SH", "中证500ETF", "宽基"),
    ("512100.SH", "中证1000ETF", "宽基"),
    ("588000.SH", "科创50ETF", "宽基"),
    ("562800.SH", "中证A500ETF", "宽基"),
    ("159915.SZ", "创业板ETF", "宽基"),
    # 行业
    ("512010.SH", "医药ETF", "行业"),
    ("512660.SH", "军工ETF", "行业"),
    ("512880.SH", "证券ETF", "行业"),
    ("515030.SH", "新能源ETF", "行业"),
    ("512480.SH", "半导体ETF", "行业"),
    ("159869.SZ", "游戏ETF", "行业"),
    # 跨境
    ("513100.SH", "纳指ETF", "跨境"),
    ("513050.SH", "中概互联ETF", "跨境"),
    ("159920.SZ", "恒生ETF", "跨境"),
]

_index_cache: dict[str, list[dict]] = {}
_index_models = None  # lazy load

_INDEX_FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "dev_ma5", "dev_ma10", "dev_ma20", "dev_ma60", "dev_ma120",
    "rsi_14", "vol_20d", "vol_ratio", "slope_20d",
    "macd_dif", "macd_hist", "bb_pos", "up_ratio_20d",
    "amplitude", "gap",
]


def _load_index_models():
    """懒加载指数评分模型 (3 个 LightGBM Booster)."""
    global _index_models
    if _index_models is not None:
        return _index_models
    model_dir = Path(__file__).resolve().parents[3] / "output" / "index_model"
    if not (model_dir / "r20_reg.txt").exists():
        return None
    try:
        import lightgbm as lgb
        _index_models = {
            "r20_reg": lgb.Booster(model_file=str(model_dir / "r20_reg.txt")),
            "pump_cls": lgb.Booster(model_file=str(model_dir / "pump_cls.txt")),
            "pump_down_cls": lgb.Booster(model_file=str(model_dir / "pump_down_cls.txt")),
        }
        return _index_models
    except Exception as e:
        logger.warning("[index] model load failed: %s", e)
        return None


def _score_index_features(c, o, h, l, v, pre_close):
    """从 OHLCV 序列计算最新一行的 20 个特征 (与训练脚本一致)."""
    import numpy as np
    n = len(c)
    feats = {}

    # 收益率
    for w in [1, 5, 10, 20, 60]:
        feats[f"ret_{w}d"] = c[-1] / c[-1 - w] - 1 if n > w else 0

    # MA偏离
    for w in [5, 10, 20, 60, 120]:
        ma = np.mean(c[-w:]) if n >= w else np.mean(c)
        feats[f"dev_ma{w}"] = c[-1] / (ma + 1e-10) - 1

    # RSI-14
    if n >= 15:
        delta = np.diff(c[-15:])
        gain = np.mean(np.maximum(delta, 0))
        loss = np.mean(np.maximum(-delta, 0))
        feats["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-10))
    else:
        feats["rsi_14"] = 50

    # 波动率
    if n >= 21:
        feats["vol_20d"] = np.std(np.diff(c[-21:]) / c[-21:-1])
    else:
        feats["vol_20d"] = 0

    # 量比
    if n >= 20:
        vol_ma20 = np.mean(v[-20:])
        feats["vol_ratio"] = v[-1] / (vol_ma20 + 1e-10)
    else:
        feats["vol_ratio"] = 1.0

    # 动量斜率
    if n >= 20:
        x = np.arange(20)
        y = c[-20:]
        feats["slope_20d"] = np.polyfit(x, y, 1)[0] / np.mean(y) * 100
    else:
        feats["slope_20d"] = 0

    # MACD
    if n >= 26:
        ema12 = pd.Series(c).ewm(span=12).mean().iloc[-1]
        ema26 = pd.Series(c).ewm(span=26).mean().iloc[-1]
        dif = ema12 - ema26
        dea = pd.Series(c).ewm(span=12).mean().ewm(span=9).mean().iloc[-1] - \
              pd.Series(c).ewm(span=26).mean().ewm(span=9).mean().iloc[-1]
        feats["macd_dif"] = dif / (c[-1] + 1e-10)
        feats["macd_hist"] = (dif - dea) / (c[-1] + 1e-10)
    else:
        feats["macd_dif"] = 0
        feats["macd_hist"] = 0

    # 布林带位置
    if n >= 20:
        bb_mid = np.mean(c[-20:])
        bb_std = np.std(c[-20:])
        feats["bb_pos"] = (c[-1] - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)
    else:
        feats["bb_pos"] = 0.5

    # 20日涨跌比
    if n >= 21:
        changes = np.diff(c[-21:])
        up = (changes > 0).sum()
        dn = (changes < 0).sum()
        feats["up_ratio_20d"] = up / (dn + 1)
    else:
        feats["up_ratio_20d"] = 1.0

    # 振幅
    feats["amplitude"] = (h[-1] - l[-1]) / (c[-1] + 1e-10)

    # 缺口
    feats["gap"] = (o[-1] - pre_close[-1]) / (pre_close[-1] + 1e-10)

    return np.array([feats.get(col, 0) for col in _INDEX_FEATURE_COLS]).reshape(1, -1)


def _build_row(ts_code, name, asset_type, df, models) -> dict | None:
    """从日线 DataFrame 构建一行指标 + 模型评分."""
    import numpy as np
    if df is None or len(df) < 21:
        return None
    df = df.sort_values("trade_date").reset_index(drop=True)
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["vol"].values.astype(float)
    pc = df["pre_close"].values.astype(float)

    close = c[-1]
    ret5 = (c[-1] / c[-6] - 1) * 100 if len(c) >= 6 else 0
    ret20 = (c[-1] / c[-21] - 1) * 100 if len(c) >= 21 else 0
    ret60 = (c[-1] / c[-61] - 1) * 100 if len(c) >= 61 else 0

    delta = np.diff(c[-15:])
    gain = np.mean(np.maximum(delta, 0))
    loss = np.mean(np.maximum(-delta, 0))
    rsi = 100 - 100 / (1 + gain / (loss + 1e-10))

    changes = np.diff(c[-21:])
    up_days = int((changes > 0).sum())
    down_days = int((changes < 0).sum())
    up_ratio = up_days / (down_days + 1)

    vol_ratio = v[-1] / np.mean(v[-20:]) if np.mean(v[-20:]) > 0 else 1.0

    ma20 = np.mean(c[-20:])
    ma60 = np.mean(c[-60:]) if len(c) >= 60 else np.mean(c)
    dev_ma20 = (close / ma20 - 1) * 100
    dev_ma60 = (close / ma60 - 1) * 100

    row = {
        "ts_code": ts_code, "name": name, "asset_type": asset_type,
        "trade_date": str(df["trade_date"].iloc[-1]),
        "close": round(close, 2),
        "ret_5d": round(ret5, 1), "ret_20d": round(ret20, 1), "ret_60d": round(ret60, 1),
        "rsi_14": round(rsi, 1),
        "up_days": up_days, "down_days": down_days, "up_ratio": round(up_ratio, 2),
        "vol_ratio": round(vol_ratio, 2),
        "dev_ma20": round(dev_ma20, 1), "dev_ma60": round(dev_ma60, 1),
    }

    if models and len(c) >= 120:
        X = _score_index_features(c, o, h, l, v, pc)
        r20_raw = models["r20_reg"].predict(X)[0]
        pump_prob = models["pump_cls"].predict(X)[0]
        pump_down_prob = models["pump_down_cls"].predict(X)[0]
        r20_score = max(0, min(100, (r20_raw + 0.15) / 0.30 * 100))
        ratio = pump_prob / (pump_down_prob + 0.01)
        row["r20_score"] = round(r20_score, 0)
        row["pump_score"] = round(pump_prob * 100, 1)
        row["pump_down_score"] = round(pump_down_prob * 100, 1)
        row["model_ratio"] = round(ratio, 2)
        row["r20_pred"] = round(r20_raw * 100, 2)

    return row


def compute_index_metrics(date: str = "") -> list[dict]:
    """计算 5 大指数 + 16 ETF 的量化指标 + 模型评分. 同 date 缓存."""
    if date and date in _index_cache:
        return _index_cache[date]
    try:
        import tushare as ts
        pro = ts.pro_api()
    except Exception as e:
        logger.warning("[index] tushare init failed: %s", e)
        return []

    end_date = date or datetime.now().strftime("%Y%m%d")
    start_date = str(int(end_date[:4]) - 1) + end_date[4:]
    models = _load_index_models()

    results = []
    # 指数 (index_daily)
    for ts_code, name, atype in _INDEX_CODES:
        try:
            df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            row = _build_row(ts_code, name, atype, df, models)
            if row:
                results.append(row)
        except Exception as e:
            logger.warning("[index] %s failed: %s", ts_code, e)
    # ETF (fund_daily)
    for ts_code, name, category in _ETF_CODES:
        try:
            df = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            row = _build_row(ts_code, name, f"ETF-{category}", df, models)
            if row:
                results.append(row)
        except Exception as e:
            logger.warning("[index] ETF %s failed: %s", ts_code, e)

    if date:
        _index_cache[date] = results
    return results
