# ClimateLens
**See how Earth is changing through AI-powered satellite analysis.**

## Overview
ClimateLens is an AI-powered satellite monitoring dashboard for exploring environmental change across time. It detects patterns such as **vegetation change, urban growth, water change, and ice/glacier variation** using **Sentinel-2 imagery when available** and a **location-aware fallback mode** when live imagery is unavailable.

The platform combines:
- interactive map overlays
- time-series trend charts
- timeline exploration
- Simple / Expert modes
- AI-generated explanations and recommendations

## Features
- Multi-year environmental change analysis
- NDVI, NDWI, and NDBI computation
- Land cover classification and change detection
- Interactive Leaflet map with overlays
- Plotly trend charts and forecasts
- Timeline explorer
- Simple Mode and Expert Mode
- AI assistant for context-aware explanations
- Sentinel-2 live mode with graceful fallback support
- Caching for analysis and imagery

## Quick Start

### 1. Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Sentinel Hub (optional, for live data)
```powershell
$env:SENTINELHUB_CLIENT_ID="your-client-id"
$env:SENTINELHUB_CLIENT_SECRET="your-client-secret"
$env:SENTINELHUB_ENABLED="true"
$env:SENTINELHUB_MAX_CLOUD="0.2"
```

If you want to run without live Sentinel-2 data, omit the credentials or disable Sentinel mode:

```powershell
$env:SENTINELHUB_ENABLED="false"
```

### 4. Run the app
```powershell
python -m uvicorn main:app --reload
```

Open the UI at:

`http://127.0.0.1:8000`

## Architecture

- FastAPI backend for analysis endpoints and static asset serving
- Sentinel Hub integration in `sentinel_client.py` for live Sentinel-2 imagery
- Satellite processing in `satellite_processing.py` for NDVI / NDWI / NDBI and land cover classification
- Analysis orchestration in `ai_module.py` for change detection, time series, timeline logic, fallback behavior, and forecasting

### Caching
- SQLite analysis cache: `data/cache.db`
- Sentinel imagery cache: `data/sentinel_cache/`

### Frontend
- React (CDN)
- Leaflet map layers
- Plotly charts

## Satellite Analysis Workflow

1. Resolve a location from city name or coordinates.
2. Fetch Sentinel-2 bands (B02, B03, B04, B08, B11) for multiple years when available.
3. Compute:
   - NDVI for vegetation
   - NDWI for water
   - NDBI for built-up / urban areas
4. Classify land cover into vegetation, water, urban, and ice.
5. Compare years (2016, 2018, 2020, 2022, 2024) for change detection.
6. Generate heatmaps, previews, and time-series charts.
7. Forecast the next 5 years using a linear trend model.
8. Produce AI-generated explanations based on current analysis.

## Source Modes

ClimateLens can operate in different source modes depending on data availability:

- `sentinel` - live Sentinel-2 imagery was used
- `mixed` - some years used Sentinel-2 and some years used fallback data
- `demo` - location-aware fallback imagery was used
- `config_missing` - Sentinel config is missing
- `auth_failed` - Sentinel authentication failed
- `no_scenes` - no suitable Sentinel scenes were found
- `download_failed` - imagery request/download failed
- `processing_failed` - imagery processing failed

## Sentinel Hub Setup

Set environment variables locally and never commit secrets.

Required:
- `SENTINELHUB_CLIENT_ID`
- `SENTINELHUB_CLIENT_SECRET`

Optional:
- `SENTINELHUB_MAX_CLOUD` (default: `0.2`)
- `SENTINELHUB_CACHE_DIR` (default: `data/sentinel_cache`)
- `SENTINELHUB_ENABLED` (`true` / `false`)

## Notes

- Sample scenes are generated on first run under `data/`
- Analysis caching is stored in `data/cache.db`
- Sentinel imagery cache is stored in `data/sentinel_cache/`
- The API logs whether Sentinel credentials are present, whether authentication succeeds, whether scenes are found, and the exact fallback reason when live data is not used
- The map uses OpenStreetMap and optional Esri satellite tiles when online

## Known Limitations

- Some locations or years may fall back to demo mode because no suitable Sentinel scenes are available
- Cloud filtering and scene selection are simplified
- The AI assistant is context-aware and constrained, but not a general-purpose LLM analyst
- Demo fallback mode is synthetic, although it is location-aware

## Tests
```powershell
python -m pytest
```
