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


_POOL_FILES = {"A": "poolA_system.csv", "B": "pool_B.csv", "C": "pool_C.csv", "D": "poolD_ratio_ge5.csv"}
_POOL_META = {
    "A": {"name": "系统推荐", "dna": "抄底回调 (系统自营)", "desc": "V12.31 v7c主推 → ratio排序 → 行业cap → Top20"},
    "B": {"name": "自选", "dna": "用户关注", "desc": "自选清单打分排名"},
    "C": {"name": "基金重仓", "dna": "机构抱团 (动量追高)", "desc": "好基金优选池 (被≥2只高收益基金持有)"},
    "D": {"name": "追高动量", "dna": "高决断度 (与系统对立, 对照)", "desc": "全市场 ratio≥5"},
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
    return {"date": date, "total": total, "pools": pools, "exists": total > 0, "prev_compared": bool(prev)}


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
