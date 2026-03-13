import hashlib
from pathlib import Path

import numpy as np

try:
    from sentinelhub import (
        BBox,
        CRS,
        DataCollection,
        MimeType,
        MosaickingOrder,
        SentinelHubRequest,
    )
except Exception:  # pragma: no cover - optional dependency
    BBox = None
    CRS = None
    DataCollection = None
    MimeType = None
    MosaickingOrder = None
    SentinelHubRequest = None

from sentinel_config import get_sh_config, load_settings


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


def _bbox_from_location(lat, lon, half_size=0.1):
    return [lon - half_size, lat - half_size, lon + half_size, lat + half_size]


def _cache_key(lat, lon, year, size, max_cloud, time_interval):
    payload = f"{lat:.4f},{lon:.4f}:{year}:{size[0]}x{size[1]}:{max_cloud}:{time_interval}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(cache_dir, key):
    return Path(cache_dir) / f"sentinel_{key}.npz"


def fetch_sentinel_bands(
    location,
    year,
    size=(256, 256),
    max_cloud=None,
    half_size=0.1,
    time_interval=None,
):
    if SentinelHubRequest is None:
        return None
    config = get_sh_config()
    if config is None:
        return None

    settings = load_settings()
    if max_cloud is None:
        max_cloud = settings.max_cloud

    lat = float(location["lat"])
    lon = float(location["lon"])
    if time_interval is None:
        time_interval = (f"{year}-05-01", f"{year}-09-30")

    cache_dir = settings.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(lat, lon, year, size, max_cloud, time_interval)
    path = _cache_path(cache_dir, key)
    if path.exists():
        data = np.load(path)
        return {
            "blue": data["blue"],
            "green": data["green"],
            "red": data["red"],
            "nir": data["nir"],
            "swir": data["swir"],
            "source": "sentinel",
            "cached": True,
            "time_interval": time_interval,
        }

    max_cloud = float(max(0.0, min(1.0, max_cloud)))
    bbox = BBox(bbox=_bbox_from_location(lat, lon, half_size=half_size), crs=CRS.WGS84)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order=MosaickingOrder.LEAST_CC,
                data_filter={"maxcc": max_cloud},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )

    try:
        data = request.get_data()
    except Exception:
        return None

    if not data:
        return None

    bands = np.asarray(data[0], dtype=np.float32)
    if bands.ndim != 3 or bands.shape[2] < 5:
        return None
    bands = np.nan_to_num(bands, nan=0.0, posinf=0.0, neginf=0.0)
    bands = np.clip(bands, 0.0, 1.0)

    payload = {
        "blue": bands[..., 0],
        "green": bands[..., 1],
        "red": bands[..., 2],
        "nir": bands[..., 3],
        "swir": bands[..., 4],
        "source": "sentinel",
        "cached": False,
        "time_interval": time_interval,
    }

    np.savez_compressed(path, **{k: payload[k] for k in ["blue", "green", "red", "nir", "swir"]})
    return payload
