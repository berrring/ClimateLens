import hashlib
import logging
from itertools import islice
from pathlib import Path

import numpy as np

try:
    from sentinelhub import (
        BBox,
        CRS,
        DataCollection,
        DownloadFailedException,
        MimeType,
        MosaickingOrder,
        SentinelHubCatalog,
        SentinelHubRequest,
        SentinelHubSession,
    )
    SENTINEL_IMPORT_ERROR = None
except Exception as err:  # pragma: no cover - optional dependency
    BBox = None
    CRS = None
    DataCollection = None
    DownloadFailedException = Exception
    MimeType = None
    MosaickingOrder = None
    SentinelHubCatalog = None
    SentinelHubRequest = None
    SentinelHubSession = None
    SENTINEL_IMPORT_ERROR = str(err)

from sentinel_config import get_sh_config, load_settings, settings_summary


EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02", "B03", "B04", "B08", "B11"], units: "REFLECTANCE" }],
    output: { bands: 5, sampleType: "FLOAT32" }
  };
}

function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11];
}
"""

logger = logging.getLogger("climatelens.sentinel")


def _bbox_from_location(lat, lon, half_size=0.1):
    return [lon - half_size, lat - half_size, lon + half_size, lat + half_size]


def _cache_key(lat, lon, year, size, max_cloud, time_interval):
    payload = f"{lat:.4f},{lon:.4f}:{year}:{size[0]}x{size[1]}:{max_cloud}:{time_interval}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir, key):
    return Path(cache_dir) / f"sentinel_{key}.npz"


def _runtime_details(settings=None):
    settings = settings or load_settings()
    details = settings_summary()
    details["max_cloud"] = settings.max_cloud
    return details


def _scene_dates(scenes):
    dates = []
    for scene in scenes or []:
        props = scene.get("properties", {}) if isinstance(scene, dict) else {}
        date_value = props.get("datetime") or props.get("start_datetime") or scene.get("id")
        if date_value:
            dates.append(date_value)
    return dates


def _classify_exception(err):
    message = str(err).lower()
    if any(term in message for term in ["401", "403", "unauthorized", "invalid_client", "oauth", "token"]):
        return "auth_failed"
    return "download_failed"


def _failure(status, message, *, location, year, time_interval, bbox=None, scenes=None, runtime=None, err=None, cached=False):
    diagnostics = {
        "status": status,
        "message": message,
        "location": location.get("name"),
        "year": year,
        "time_interval": list(time_interval) if time_interval else None,
        "bbox": list(bbox) if bbox else None,
        "scenes_found": len(scenes or []),
        "scene_dates": _scene_dates(scenes),
        "runtime": runtime or _runtime_details(),
        "cached": cached,
    }
    if err is not None:
        diagnostics["error"] = str(err)
    logger.warning("Sentinel fallback status=%s diagnostics=%s", status, diagnostics)
    return {
        "ok": False,
        "status": status,
        "source": "demo",
        "message": message,
        "cached": cached,
        "time_interval": list(time_interval) if time_interval else None,
        "bbox": list(bbox) if bbox else None,
        "scenes_found": len(scenes or []),
        "scene_dates": _scene_dates(scenes),
        "runtime": diagnostics["runtime"],
        "error": diagnostics.get("error"),
    }


def _success(payload, *, location, year, time_interval, bbox=None, scenes=None, runtime=None, cached=False):
    logger.info(
        "Sentinel fetch succeeded location=%s year=%s cached=%s scenes_found=%s interval=%s",
        location.get("name"),
        year,
        cached,
        len(scenes or []),
        time_interval,
    )
    payload.update(
        {
            "ok": True,
            "status": "sentinel",
            "source": "sentinel",
            "message": "Using Sentinel-2 imagery.",
            "cached": cached,
            "time_interval": list(time_interval) if time_interval else None,
            "bbox": list(bbox) if bbox else None,
            "scenes_found": len(scenes or []),
            "scene_dates": _scene_dates(scenes),
            "runtime": runtime or _runtime_details(),
        }
    )
    return payload


def _authenticate(config, *, location, year, runtime):
    try:
        session = SentinelHubSession(config=config)
        token = session.token
        token_expires_at = token.get("expires_at") if isinstance(token, dict) else None
        logger.info(
            "Sentinel authentication succeeded location=%s year=%s token_expires_at=%s",
            location.get("name"),
            year,
            token_expires_at,
        )
        return {"ok": True, "token_expires_at": token_expires_at}
    except Exception as err:
        message = "Sentinel Hub authentication failed."
        logger.warning(
            "Sentinel authentication failed location=%s year=%s error=%s runtime=%s",
            location.get("name"),
            year,
            err,
            runtime,
        )
        return {"ok": False, "status": "auth_failed", "message": message, "error": err}


def _find_scenes(config, bbox, time_interval, max_cloud, *, location, year):
    if SentinelHubCatalog is None:
        return None, None

    try:
        catalog = SentinelHubCatalog(config=config)
        search = catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=bbox,
            time=time_interval,
            filter=f"eo:cloud_cover <= {max_cloud * 100:.1f}",
            fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []},
            limit=5,
        )
        scenes = list(islice(search, 5))
        logger.info(
            "Sentinel catalog query location=%s year=%s scenes_found=%s interval=%s",
            location.get("name"),
            year,
            len(scenes),
            time_interval,
        )
        return scenes, None
    except Exception as err:
        logger.warning(
            "Sentinel catalog query failed location=%s year=%s interval=%s error=%s",
            location.get("name"),
            year,
            time_interval,
            err,
        )
        return None, err


def fetch_sentinel_bands(
    location,
    year,
    size=(256, 256),
    max_cloud=None,
    half_size=0.1,
    time_interval=None,
):
    settings = load_settings()
    runtime = _runtime_details(settings)

    if max_cloud is None:
        max_cloud = settings.max_cloud
    max_cloud = float(max(0.0, min(1.0, max_cloud)))

    lat = float(location["lat"])
    lon = float(location["lon"])
    bbox_values = _bbox_from_location(lat, lon, half_size=half_size)
    if time_interval is None:
        time_interval = (f"{year}-05-01", f"{year}-09-30")

    logger.info(
        "Sentinel fetch requested location=%s year=%s bbox=%s interval=%s runtime=%s",
        location.get("name"),
        year,
        bbox_values,
        time_interval,
        runtime,
    )

    if SentinelHubRequest is None or BBox is None or CRS is None:
        message = "Sentinel Hub library is unavailable in the current Python environment."
        return _failure(
            "library_missing",
            message,
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            runtime=runtime,
            err=SENTINEL_IMPORT_ERROR,
        )

    if not settings.enabled:
        message = "Sentinel Hub is disabled via SENTINELHUB_ENABLED."
        return _failure(
            "disabled",
            message,
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            runtime=runtime,
        )

    if not settings.client_id or not settings.client_secret:
        message = "Sentinel Hub credentials are missing at runtime."
        return _failure(
            "config_missing",
            message,
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            runtime=runtime,
        )

    config = get_sh_config()
    if config is None:
        message = "Sentinel Hub configuration could not be created."
        return _failure(
            "config_missing",
            message,
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            runtime=runtime,
        )

    cache_dir = settings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(lat, lon, year, size, max_cloud, time_interval)
    path = _cache_path(cache_dir, key)
    if path.exists():
        data = np.load(path)
        return _success(
            {
                "blue": data["blue"],
                "green": data["green"],
                "red": data["red"],
                "nir": data["nir"],
                "swir": data["swir"],
            },
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            runtime=runtime,
            cached=True,
        )

    auth = _authenticate(config, location=location, year=year, runtime=runtime)
    if not auth["ok"]:
        return _failure(
            auth["status"],
            auth["message"],
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            runtime=runtime,
            err=auth["error"],
        )

    bbox = BBox(bbox=bbox_values, crs=CRS.WGS84)
    scenes, catalog_error = _find_scenes(
        config,
        bbox,
        time_interval,
        max_cloud,
        location=location,
        year=year,
    )
    if scenes == []:
        return _failure(
            "no_scenes",
            "No Sentinel-2 scenes matched the requested location and date range.",
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            scenes=scenes,
            runtime=runtime,
        )

    try:
        request = SentinelHubRequest(
            evalscript=EVALSCRIPT,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    mosaicking_order=MosaickingOrder.LEAST_CC,
                    maxcc=max_cloud,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=config,
        )
    except Exception as err:
        return _failure(
            "download_failed",
            "Sentinel request generation failed.",
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            scenes=scenes,
            runtime=runtime,
            err=err,
        )

    try:
        data = request.get_data()
    except DownloadFailedException as err:
        return _failure(
            _classify_exception(err),
            "Sentinel imagery download failed.",
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            scenes=scenes,
            runtime=runtime,
            err=err,
        )
    except Exception as err:
        return _failure(
            _classify_exception(err),
            "Sentinel imagery download failed.",
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            scenes=scenes,
            runtime=runtime,
            err=err,
        )

    if not data:
        if catalog_error is None:
            return _failure(
                "no_scenes",
                "Sentinel returned no imagery for the requested location and date range.",
                location=location,
                year=year,
                time_interval=time_interval,
                bbox=bbox_values,
                scenes=scenes,
                runtime=runtime,
            )
        return _failure(
            "download_failed",
            "Sentinel imagery request returned no data after catalog lookup failed.",
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            scenes=scenes,
            runtime=runtime,
            err=catalog_error,
        )

    try:
        bands = np.asarray(data[0], dtype=np.float32)
        if bands.ndim != 3 or bands.shape[2] < 5:
            raise ValueError(f"Unexpected band shape: {bands.shape}")
        bands = np.nan_to_num(bands, nan=0.0, posinf=0.0, neginf=0.0)
        bands = np.clip(bands, 0.0, 1.0)
    except Exception as err:
        return _failure(
            "processing_failed",
            "Sentinel imagery processing failed after download.",
            location=location,
            year=year,
            time_interval=time_interval,
            bbox=bbox_values,
            scenes=scenes,
            runtime=runtime,
            err=err,
        )

    payload = {
        "blue": bands[..., 0],
        "green": bands[..., 1],
        "red": bands[..., 2],
        "nir": bands[..., 3],
        "swir": bands[..., 4],
    }

    np.savez_compressed(path, **payload)
    return _success(
        payload,
        location=location,
        year=year,
        time_interval=time_interval,
        bbox=bbox_values,
        scenes=scenes,
        runtime=runtime,
    )
