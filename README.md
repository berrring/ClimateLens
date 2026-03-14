# ClimateLens

See how Earth is changing through AI-powered satellite analysis.

## Overview
ClimateLens is an AI-powered satellite monitoring dashboard that detects environmental change (vegetation, urban growth, water, ice) using Sentinel-2 imagery or a location-aware demo fallback. The UI pairs Leaflet heatmaps with Plotly trend charts, a timeline explorer, Simple / Expert modes, and natural language insights.

## Quick Start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:SENTINELHUB_CLIENT_ID="your-client-id"
$env:SENTINELHUB_CLIENT_SECRET="your-client-secret"
$env:SENTINELHUB_ENABLED="true"
$env:SENTINELHUB_MAX_CLOUD="0.2"

.\.venv\Scripts\uvicorn.exe main:app --reload
```

Open `http://127.0.0.1:8000`.

If you want to run without live Sentinel-2 data, omit the credentials or set:

```powershell
$env:SENTINELHUB_ENABLED="false"
```

## Architecture
- **FastAPI** backend serving analysis endpoints and static UI assets.
- **Sentinel Hub integration** in `sentinel_client.py` for live Sentinel-2 imagery (L2A).
- **Satellite processing** in `satellite_processing.py` for NDVI / NDWI / NDBI + land cover classification.
- **Analysis orchestration** in `ai_module.py` with change detection, time series, and forecasting.
- **Caching**: SQLite analysis cache (`data/cache.db`) and imagery cache (`data/sentinel_cache/`).
- **Frontend**: React (CDN), Leaflet map layers, Plotly charts.

## Satellite Analysis Workflow
1. Fetch Sentinel-2 bands (B02/B03/B04/B08/B11) by coordinate + date range.
2. Compute NDVI, NDWI, NDBI.
3. Classify land cover (vegetation, water, urban, ice) via index thresholds.
4. Compare years (2016, 2018, 2020, 2022, 2024) for change detection.
5. Generate heatmaps and time-series charts.
6. Forecast the next 5 years using a linear trend model.

## Sentinel Hub Setup
Set environment variables (do not commit secrets):

```powershell
$env:SENTINELHUB_CLIENT_ID="your-client-id"
$env:SENTINELHUB_CLIENT_SECRET="your-client-secret"
```

Optional:
- `SENTINELHUB_MAX_CLOUD` (default `0.2`)
- `SENTINELHUB_CACHE_DIR` (default `data/sentinel_cache`)
- `SENTINELHUB_ENABLED` (`true` / `false`)

If credentials are missing or the API is unavailable, the system automatically falls back to a demo dataset.
Responses and logs now expose the exact source mode:

- `sentinel`: live Sentinel-2 imagery was used
- `demo`: demo fallback imagery was used
- `config_missing`, `auth_failed`, `no_scenes`, `download_failed`, `processing_failed`: explicit fallback reasons
- `mixed`: some years used Sentinel-2 and some used fallback data

## Run Locally
1. Create environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the API from the project virtual environment:

```powershell
.\.venv\Scripts\uvicorn.exe main:app --reload
```

3. Open the UI:

- http://127.0.0.1:8000

## Notes
- Sample scenes are generated on first run under `data/`.
- Analysis caching is stored in `data/cache.db` (SQLite).
- Sentinel imagery cache is stored in `data/sentinel_cache/`.
- The API logs whether Sentinel credentials are present, whether authentication succeeds, whether scenes are found, and the exact fallback reason when live data is not used.
- The map uses OpenStreetMap and optional Esri satellite tiles when online.

## Tests
```powershell
python -m pytest
```
