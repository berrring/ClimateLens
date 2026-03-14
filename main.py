from pathlib import Path
from typing import Optional

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ai_module
import cache
import sentinel_config

logger = logging.getLogger("climatelens")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    ai_module.ensure_sample_data()
    cache.init_db()
    logger.info("sentinel_config=%s", sentinel_config.settings_summary())
    yield


app = FastAPI(title="ClimateLens", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AnalyzeRequest(BaseModel):
    query: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    mode: str = "simple"
    year_start: int = ai_module.DEFAULT_YEAR_START
    year_end: int = ai_module.DEFAULT_YEAR_END


class ExplainRequest(BaseModel):
    query: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    mode: str = "simple"
    question: Optional[str] = None
    timeline_year: Optional[int] = None
    year_start: int = ai_module.DEFAULT_YEAR_START
    year_end: int = ai_module.DEFAULT_YEAR_END
    require_cached: bool = False
    analysis_id: Optional[str] = None
    analysis_summary: Optional[str] = None
    analysis_metrics: Optional[dict] = None


@app.get("/")
def index():
    return FileResponse(INDEX_FILE)


@app.post("/analyze_location")
def analyze_location(req: AnalyzeRequest):
    normalized_query = ai_module.normalize_query(req.query)
    location, error = ai_module.resolve_location_info(normalized_query, req.lat, req.lon)
    logger.info("analyze_location query=%s normalized=%s lat=%s lon=%s", req.query, normalized_query, req.lat, req.lon)
    if error:
        return _error_response("location_not_found", error, status_code=404)

    logger.info(
        "resolved location=%s (%s, %s) source=%s years=%s-%s",
        location["name"],
        location["lat"],
        location["lon"],
        location.get("source"),
        req.year_start,
        req.year_end,
    )
    logger.info("sentinel_runtime=%s", sentinel_config.settings_summary())
    cache_key = _cache_key(location, req.year_start, req.year_end)
    analysis = cache.get_cached_analysis(cache_key)
    cached = True
    if analysis is not None and not _has_source_diagnostics(analysis):
        logger.info("cached analysis is missing source diagnostics; recomputing key=%s", cache_key)
        analysis = None
    if analysis is not None and not analysis.get("cacheable", True):
        logger.info("cached analysis is not reusable; recomputing key=%s source_mode=%s", cache_key, analysis.get("source_mode"))
        analysis = None
    if analysis is None:
        analysis = ai_module.analyze_location(location, req.year_start, req.year_end)
        cache.set_cached_analysis(cache_key, analysis)
        cached = False
    cache.add_history(cache_key, location, req.year_start, req.year_end)
    payload = ai_module.analysis_payload(analysis, mode=req.mode)
    payload["cached"] = cached
    payload["success"] = True
    if payload.get("messages"):
        logger.info("fallback notice=%s", payload["messages"])
    if payload.get("source_mode") and payload["source_mode"] != "sentinel":
        logger.info("analysis source=%s source_mode=%s", payload.get("source"), payload.get("source_mode"))
    return payload


@app.post("/get_charts_data")
def get_charts_data(req: AnalyzeRequest):
    normalized_query = ai_module.normalize_query(req.query)
    location, error = ai_module.resolve_location_info(normalized_query, req.lat, req.lon)
    if error:
        return _error_response("location_not_found", error, status_code=404)
    cache_key = _cache_key(location, req.year_start, req.year_end)
    analysis = cache.get_cached_analysis(cache_key)
    if analysis is not None and not _has_source_diagnostics(analysis):
        analysis = None
    if analysis is None:
        analysis = ai_module.analyze_location(location, req.year_start, req.year_end)
        cache.set_cached_analysis(cache_key, analysis)
    payload = ai_module.charts_payload(analysis)
    payload["success"] = True
    return payload


@app.post("/ai_explanation")
def ai_explanation(req: ExplainRequest):
    normalized_query = ai_module.normalize_query(req.query)
    logger.info(
        "assistant question=%s analysis_id=%s year=%s mode=%s",
        req.question,
        req.analysis_id,
        req.timeline_year,
        req.mode,
    )
    location, error = ai_module.resolve_location_info(normalized_query, req.lat, req.lon)
    if error:
        return _error_response("location_not_found", error, status_code=404)
    logger.info(
        "assistant location=%s (%s, %s) years=%s-%s",
        location["name"],
        location["lat"],
        location["lon"],
        req.year_start,
        req.year_end,
    )
    cache_key = _cache_key(location, req.year_start, req.year_end)
    analysis = cache.get_cached_analysis(cache_key)
    if analysis is not None and not _has_source_diagnostics(analysis):
        analysis = None
    if analysis is None:
        if req.require_cached:
            return {"success": False, "summary": "Please analyze a location first.", "messages": []}
        analysis = ai_module.analyze_location(location, req.year_start, req.year_end)
        cache.set_cached_analysis(cache_key, analysis)
    source_mode = analysis.get("source_mode", analysis.get("source", "demo"))
    if source_mode != "sentinel":
        logger.info("assistant using fallback source=%s", source_mode)
    response = ai_module.answer_question(
        analysis,
        question=req.question,
        mode=req.mode,
        year_focus=req.timeline_year,
    )
    response["success"] = True
    return response


@app.get("/history")
def history(limit: int = 20):
    return {"items": cache.get_history(limit=limit)}


def _cache_key(location, year_start, year_end):
    data_mode = "sentinel" if sentinel_config.is_configured() else "demo"
    return f"{location['lat']:.4f},{location['lon']:.4f}:{year_start}-{year_end}:{data_mode}"


def _error_response(code, message, status_code=400):
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": {"code": code, "message": message}, "messages": []},
    )


def _has_source_diagnostics(analysis):
    return "source_mode" in analysis and "source_details" in analysis
