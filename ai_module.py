import base64
import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import satellite_processing as sp
import sentinel_client
from geocode import geocode_city

try:
    import torch
except Exception:  # pragma: no cover - optional
    torch = None

try:
    import rasterio
except Exception:  # pragma: no cover - optional
    rasterio = None

try:
    import geopandas as gpd
    from shapely.geometry import box
except Exception:  # pragma: no cover - optional
    gpd = None
    box = None

DATA_DIR = Path(__file__).parent / "data"
PREVIEW_DIR = DATA_DIR / "previews"

AVAILABLE_YEARS = [2016, 2018, 2020, 2022, 2024]
DEFAULT_YEAR_START = 2016
DEFAULT_YEAR_END = 2024

DEFAULT_LOCATION = {"name": "San Francisco, CA", "lat": 37.7749, "lon": -122.4194}

CITY_DB = {
    "san francisco": ("San Francisco, CA", 37.7749, -122.4194),
    "new york": ("New York, NY", 40.7128, -74.0060),
    "london": ("London, UK", 51.5074, -0.1278),
    "tokyo": ("Tokyo, JP", 35.6895, 139.6917),
    "bishkek": ("Bishkek, KG", 42.8746, 74.5698),
    "almaty": ("Almaty, KZ", 43.2389, 76.8897),
    "nairobi": ("Nairobi, KE", -1.2921, 36.8219),
    "sydney": ("Sydney, AU", -33.8688, 151.2093),
    "reykjavik": ("Reykjavik, IS", 64.1466, -21.9426),
    "mumbai": ("Mumbai, IN", 19.0760, 72.8777),
}


def normalize_query(query):
    if query is None:
        return None
    normalized = " ".join(str(query).strip().split())
    return normalized or None


def _validate_coordinates(lat, lon):
    if lat is None or lon is None:
        return False
    return -90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0


@dataclass
class Scene:
    year: int
    red: np.ndarray
    green: np.ndarray
    blue: np.ndarray
    nir: np.ndarray
    swir: np.ndarray
    source: str = "synthetic"


class LinearTrendModel:
    def predict(self, years, values, forecast_years):
        years = np.asarray(years, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32)
        forecast_years = np.asarray(forecast_years, dtype=np.float32)

        if torch is not None:
            x = torch.tensor(years, dtype=torch.float32).unsqueeze(1)
            y = torch.tensor(values, dtype=torch.float32).unsqueeze(1)
            ones = torch.ones_like(x)
            X = torch.cat([x, ones], dim=1)
            solution = torch.linalg.lstsq(X, y).solution
            slope = solution[0, 0].item()
            intercept = solution[1, 0].item()
        else:
            slope, intercept = np.polyfit(years, values, 1)

        preds = slope * forecast_years + intercept
        return preds.tolist()


TREND_MODEL = LinearTrendModel()


def _resolve_years(year_start, year_end):
    if year_start > year_end:
        year_start, year_end = year_end, year_start
    base_years = [y for y in AVAILABLE_YEARS if year_start <= y <= year_end]
    extras = []
    if year_start not in base_years:
        extras.append(year_start)
    if year_end not in base_years:
        extras.append(year_end)
    years = sorted(set(base_years + extras))
    if not years:
        years = sorted({year_start, year_end})
    return years


def resolve_location_info(query=None, lat=None, lon=None):
    if lat is not None and lon is not None:
        if not _validate_coordinates(lat, lon):
            return (None, "Coordinates out of range. Use lat (-90..90) and lon (-180..180).")
        return (
            {"name": f"{float(lat):.4f}, {float(lon):.4f}", "lat": float(lat), "lon": float(lon), "source": "coords"},
            None,
        )

    normalized = normalize_query(query)
    if normalized:
        m = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*", normalized)
        if m:
            lat_val = float(m.group(1))
            lon_val = float(m.group(2))
            if not _validate_coordinates(lat_val, lon_val):
                return (None, "Coordinates out of range. Use lat (-90..90) and lon (-180..180).")
            return (
                {
                    "name": f"{lat_val:.4f}, {lon_val:.4f}",
                    "lat": lat_val,
                    "lon": lon_val,
                    "source": "coords",
                },
                None,
            )
        key = normalized.lower()
        if key in CITY_DB:
            name, lat_val, lon_val = CITY_DB[key]
            return ({"name": name, "lat": float(lat_val), "lon": float(lon_val), "source": "city_db"}, None)

        geocoded = geocode_city(normalized)
        if geocoded:
            return (
                {"name": geocoded["name"], "lat": geocoded["lat"], "lon": geocoded["lon"], "source": "geocode"},
                None,
            )

        return (None, "Location not found. Try another city or use coordinates.")

    fallback = DEFAULT_LOCATION.copy()
    fallback["source"] = "default"
    return (fallback, None)


def resolve_location(query=None, lat=None, lon=None):
    location, _ = resolve_location_info(query=query, lat=lat, lon=lon)
    return location


def location_to_bbox(lat, lon, half_size=0.5):
    bbox = {
        "north": lat + half_size,
        "south": lat - half_size,
        "east": lon + half_size,
        "west": lon - half_size,
    }

    if gpd is not None and box is not None:
        _ = gpd.GeoDataFrame({"geometry": [box(bbox["west"], bbox["south"], bbox["east"], bbox["north"])]})

    return bbox


def ensure_sample_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for year, seed in [(2018, 42), (2023, 84)]:
        npz_path = DATA_DIR / f"scene_{year}.npz"
        preview_path = PREVIEW_DIR / f"preview_{year}.png"
        tiff_path = DATA_DIR / f"scene_{year}.tif"

        has_swir = False
        if npz_path.exists():
            try:
                with np.load(npz_path) as data:
                    has_swir = "swir" in data
            except Exception:
                has_swir = False

        if npz_path.exists() and preview_path.exists() and has_swir:
            continue

        scene = _generate_scene(seed=seed, year=year)
        np.savez_compressed(
            npz_path,
            red=scene.red,
            green=scene.green,
            blue=scene.blue,
            nir=scene.nir,
            swir=scene.swir,
        )

        rgb = np.stack([scene.red, scene.green, scene.blue], axis=-1)
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        img.save(preview_path)

        if rasterio is not None:
            try:
                if not tiff_path.exists():
                    height, width = scene.red.shape
                    with rasterio.open(
                        tiff_path,
                        "w",
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=5,
                        dtype="float32",
                    ) as dst:
                        dst.write(scene.red.astype(np.float32), 1)
                        dst.write(scene.green.astype(np.float32), 2)
                        dst.write(scene.blue.astype(np.float32), 3)
                        dst.write(scene.nir.astype(np.float32), 4)
                        dst.write(scene.swir.astype(np.float32), 5)
            except Exception:
                pass


def _generate_scene(seed, year, size=256):
    rng = np.random.default_rng(seed)
    h = w = size

    veg = np.zeros((h, w), dtype=bool)
    water = np.zeros((h, w), dtype=bool)
    urban = np.zeros((h, w), dtype=bool)
    ice = np.zeros((h, w), dtype=bool)

    if year <= 2018:
        _add_circle(veg, (120, 120), 80)
        _add_circle(veg, (60, 180), 45)
        _add_circle(water, (190, 70), 30)
        _add_rect(urban, (40, 40), (100, 90))
        _add_rect(ice, (180, 180), (230, 230))
    else:
        _add_circle(veg, (125, 125), 65)
        _add_circle(veg, (70, 185), 35)
        _add_circle(water, (200, 80), 25)
        _add_rect(urban, (30, 30), (120, 110))
        _add_rect(ice, (190, 190), (225, 225))

    base = rng.normal(0.15, 0.02, size=(h, w))
    red = base.copy()
    green = base.copy()
    blue = base.copy()
    nir = base.copy()
    swir = base.copy()

    _apply_class(red, green, blue, nir, swir, veg, (0.12, 0.45, 0.10, 0.68, 0.20), rng)
    _apply_class(red, green, blue, nir, swir, water, (0.04, 0.10, 0.22, 0.02, 0.05), rng)
    _apply_class(red, green, blue, nir, swir, urban, (0.62, 0.22, 0.45, 0.30, 0.55), rng)
    _apply_class(red, green, blue, nir, swir, ice, (0.85, 0.90, 0.92, 0.84, 0.70), rng)

    red = np.clip(red, 0, 1).astype(np.float32)
    green = np.clip(green, 0, 1).astype(np.float32)
    blue = np.clip(blue, 0, 1).astype(np.float32)
    nir = np.clip(nir, 0, 1).astype(np.float32)
    swir = np.clip(swir, 0, 1).astype(np.float32)

    return Scene(year=year, red=red, green=green, blue=blue, nir=nir, swir=swir)


def _add_circle(mask, center, radius):
    cy, cx = center
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    dist = (xx - cx) ** 2 + (yy - cy) ** 2
    mask[dist <= radius**2] = True


def _add_rect(mask, top_left, bottom_right):
    y0, x0 = top_left
    y1, x1 = bottom_right
    mask[y0:y1, x0:x1] = True


def _apply_class(red, green, blue, nir, swir, mask, values, rng):
    r, g, b, n, s = values
    noise = rng.normal(0, 0.015, size=mask.sum())
    red[mask] = r + noise
    green[mask] = g + noise
    blue[mask] = b + noise
    nir[mask] = n + noise
    swir[mask] = s + noise


def _load_scene(year):
    npz_path = DATA_DIR / f"scene_{year}.npz"
    if not npz_path.exists():
        ensure_sample_data()
    if not npz_path.exists():
        seed = int(year) * 7 % 10000
        scene = _generate_scene(seed=seed, year=year)
        np.savez_compressed(
            npz_path,
            red=scene.red,
            green=scene.green,
            blue=scene.blue,
            nir=scene.nir,
            swir=scene.swir,
        )
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        preview_path = PREVIEW_DIR / f"preview_{year}.png"
        _preview_from_scene(scene).save(preview_path)
        return scene

    data = np.load(npz_path)
    red = data["red"]
    green = data["green"]
    blue = data["blue"]
    nir = data["nir"]
    swir = data["swir"] if "swir" in data else np.clip(red * 0.8 + 0.05, 0, 1)
    return Scene(year=year, red=red, green=green, blue=blue, nir=nir, swir=swir, source="synthetic")


def _load_scene_for_year(location, year, size=256):
    payload = sentinel_client.fetch_sentinel_bands(location, year, size=(size, size))
    if payload:
        return Scene(
            year=year,
            red=payload["red"],
            green=payload["green"],
            blue=payload["blue"],
            nir=payload["nir"],
            swir=payload["swir"],
            source="sentinel",
        )
    return _load_scene(year)


def _encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _preview_from_scene(scene):
    rgb = np.stack([scene.red, scene.green, scene.blue], axis=-1)
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    return img


def _load_preview_image(year):
    preview_path = PREVIEW_DIR / f"preview_{year}.png"
    if not preview_path.exists():
        ensure_sample_data()
    return Image.open(preview_path).convert("RGB")


def _load_preview(year):
    image = _load_preview_image(year)
    return _encode_image(image)


def _blend_preview(year_start, year_end, year):
    start_img = _load_preview_image(year_start)
    end_img = _load_preview_image(year_end)
    span = max(year_end - year_start, 1)
    alpha = (year - year_start) / span

    if alpha <= 1:
        blended = Image.blend(start_img, end_img, float(alpha))
    else:
        blended = end_img.copy()
        tint = Image.new("RGB", blended.size, (238, 200, 170))
        blended = Image.blend(blended, tint, 0.2)

    return _encode_image(blended)


def _blend_preview_images(start_img, end_img, year_start, year_end, year):
    span = max(year_end - year_start, 1)
    alpha = (year - year_start) / span
    if alpha <= 1:
        blended = Image.blend(start_img, end_img, float(alpha))
    else:
        blended = end_img.copy()
        tint = Image.new("RGB", blended.size, (238, 200, 170))
        blended = Image.blend(blended, tint, 0.2)
    return _encode_image(blended)


def _compute_indices(scene):
    indices = sp.calculate_indices(scene.red, scene.green, scene.nir, scene.swir)
    brightness = sp.calculate_brightness(scene.red, scene.green, scene.blue)
    return indices["ndvi"], indices["ndwi"], indices["ndbi"], brightness


def _classify(scene):
    return sp.classify_landcover(scene.red, scene.green, scene.blue, scene.nir, scene.swir)


def _mask_stats(mask):
    return float(mask.sum()) / float(mask.size) * 100.0


def _scene_stats(scene, classes):
    return sp.stats_from_classes(classes)


def _generate_timeseries(stat_start, stat_end, years):
    years = list(years)
    values = np.linspace(stat_start, stat_end, len(years))
    noise = np.random.default_rng(123).normal(0, 0.3, size=len(years))
    return (values + noise).tolist()


def _render_heatmap(change, color_neg, color_pos, scale=1.0):
    change_scaled = np.clip(change.astype(np.float32) * float(scale), -1.0, 1.0)
    norm = (change_scaled + 1.0) / 2.0
    alpha = np.clip(np.abs(change_scaled), 0, 1)

    img = np.zeros((*change.shape, 4), dtype=np.uint8)
    for i, channel in enumerate([0, 1, 2]):
        start = color_neg[i]
        end = color_pos[i]
        img[..., channel] = (start + (end - start) * norm).astype(np.uint8)
    img[..., 3] = (alpha * 200).astype(np.uint8)

    image = Image.fromarray(img, mode="RGBA")
    return _encode_image(image)


def _render_index_heatmap(index, color_low=(170, 120, 90), color_high=(60, 160, 90)):
    values = np.clip(index.astype(np.float32), -1.0, 1.0)
    norm = (values + 1.0) / 2.0
    alpha = np.clip(np.abs(values), 0, 1)

    img = np.zeros((*index.shape, 4), dtype=np.uint8)
    for i, channel in enumerate([0, 1, 2]):
        start = color_low[i]
        end = color_high[i]
        img[..., channel] = (start + (end - start) * norm).astype(np.uint8)
    img[..., 3] = (alpha * 180).astype(np.uint8)

    image = Image.fromarray(img, mode="RGBA")
    return _encode_image(image)


def _change_maps(classes_2018, classes_2023):
    return sp.change_maps(classes_2018, classes_2023)


def _overlay_images_from_change(change_maps, scale=1.0):
    overlays = {}
    palettes = {
        "vegetation": ((210, 61, 61), (67, 168, 76)),
        "water": ((180, 100, 40), (45, 110, 210)),
        "urban": ((44, 133, 204), (228, 134, 47)),
        "ice": ((190, 190, 190), (220, 240, 255)),
    }

    for key, change in change_maps.items():
        neg, pos = palettes[key]
        overlays[key] = _render_heatmap(change, neg, pos, scale=scale)
    return overlays


def analyze_location(location, year_start=DEFAULT_YEAR_START, year_end=DEFAULT_YEAR_END):
    ensure_sample_data()
    years = _resolve_years(int(year_start), int(year_end))
    year_start = years[0]
    year_end = years[-1]

    year_scenes = {}
    year_stats = {}
    year_classes = {}
    sources = {}

    for year in years:
        scene = _load_scene_for_year(location, year)
        year_scenes[year] = scene
        sources[str(year)] = scene.source
        classes = _classify(scene)
        stats = _scene_stats(scene, classes)
        year_classes[year] = classes
        year_stats[year] = stats

    stats_start = year_stats[year_start]
    stats_end = year_stats[year_end]
    classes_start = year_classes[year_start]
    classes_end = year_classes[year_end]

    changes = {
        key: stats_end[key] - stats_start[key]
        for key in ["vegetation", "water", "urban", "ice"]
    }

    change_maps = _change_maps(classes_start, classes_end)
    overlays = _overlay_images_from_change(change_maps, scale=1.0)

    series = {
        "vegetation": [year_stats[y]["vegetation"] for y in years],
        "water": [year_stats[y]["water"] for y in years],
        "urban": [year_stats[y]["urban"] for y in years],
        "ice": [year_stats[y]["ice"] for y in years],
    }

    temperature = np.linspace(0.2, 1.4, len(years)) + np.random.default_rng(222).normal(0, 0.1, len(years))
    series["temperature"] = temperature.tolist()

    forecast_years = list(range(year_end + 1, year_end + 6))
    forecast = {
        key: TREND_MODEL.predict(years, series[key], forecast_years)
        for key in ["vegetation", "water", "urban", "ice", "temperature"]
    }

    preview_images = {year: _preview_from_scene(scene) for year, scene in year_scenes.items()}
    start_preview_img = preview_images[year_start]
    end_preview_img = preview_images[year_end]

    timeline_years = list(years)
    forecast_anchor = year_end + 5
    if forecast_anchor not in timeline_years:
        timeline_years.append(forecast_anchor)

    timeline_overlays = {}
    timeline_previews = {}
    timeline_ndvi = {}
    span = max(year_end - year_start, 1)
    for year in timeline_years:
        if year <= year_end:
            factor = (year - year_start) / span
        else:
            factor = 1 + (year - year_end) / span
        timeline_overlays[str(year)] = _overlay_images_from_change(change_maps, scale=factor)

        if year in preview_images:
            timeline_previews[str(year)] = _encode_image(preview_images[year])
        else:
            timeline_previews[str(year)] = _blend_preview_images(
                start_preview_img,
                end_preview_img,
                year_start,
                year_end,
                year,
            )

        if year in year_classes:
            ndvi_map = year_classes[year]["ndvi"]
        else:
            ndvi_map = classes_start["ndvi"] + factor * (classes_end["ndvi"] - classes_start["ndvi"])
        timeline_ndvi[str(year)] = _render_index_heatmap(ndvi_map)

    indices_by_year = {
        str(year): {
            "ndvi_mean": year_stats[year]["ndvi_mean"],
            "ndwi_mean": year_stats[year]["ndwi_mean"],
            "ndbi_mean": year_stats[year]["ndbi_mean"],
        }
        for year in years
    }

    notices = []
    if any(value != "sentinel" for value in sources.values()):
        notices.append({"code": "fallback_data", "message": "Using demo fallback data for this location."})

    source_mode = _summarize_source_mode(sources)

    analysis = {
        "analysis_id": uuid.uuid4().hex,
        "location": location,
        "bbox": location_to_bbox(location["lat"], location["lon"]),
        "years": years,
        "sources": sources,
        "source_mode": source_mode,
        "notices": notices,
        "stats": {"start": stats_start, "end": stats_end, "by_year": {str(y): year_stats[y] for y in years}},
        "changes": changes,
        "indices": {
            "ndvi_mean_start": stats_start["ndvi_mean"],
            "ndvi_mean_end": stats_end["ndvi_mean"],
            "ndwi_mean_start": stats_start["ndwi_mean"],
            "ndwi_mean_end": stats_end["ndwi_mean"],
            "ndbi_mean_start": stats_start["ndbi_mean"],
            "ndbi_mean_end": stats_end["ndbi_mean"],
            "by_year": indices_by_year,
        },
        "overlays": overlays,
        "previews": {
            "start": _encode_image(start_preview_img),
            "end": _encode_image(end_preview_img),
        },
        "timeline": {
            "years": timeline_years,
            "overlays": timeline_overlays,
            "previews": timeline_previews,
            "ndvi": timeline_ndvi,
        },
        "series": series,
        "forecast_years": forecast_years,
        "forecast": forecast,
    }

    return analysis


def _summarize_source_mode(sources):
    if not sources:
        return "demo"
    values = {value for value in sources.values()}
    if values == {"sentinel"}:
        return "sentinel"
    if values == {"synthetic"}:
        return "demo"
    if "sentinel" in values and len(values) > 1:
        return "mixed"
    return "demo"


def generate_recommendations(changes):
    recs = []
    if changes["vegetation"] < -2:
        recs.append("Increase urban green coverage and protect remaining vegetation zones.")
    if changes["water"] < -1:
        recs.append("Monitor water bodies and investigate upstream withdrawals or drought stress.")
    if changes["urban"] > 2:
        recs.append("Rising urbanization detected; review zoning growth to reduce heat island effects and runoff.")
    if changes["ice"] < -0.5:
        recs.append("Track glacier melt rates and plan for seasonal water variability.")
    if not recs:
        recs.append("Maintain current conservation measures and continue periodic monitoring.")
    return recs


def generate_explanation(analysis, mode="simple", question=None, year_focus=None):
    changes = analysis["changes"]
    stats_start = analysis["stats"]["start"]
    stats_end = analysis["stats"]["end"]
    idx = analysis["indices"]
    year_start = analysis["years"][0]
    year_end = analysis["years"][-1]

    recommendations = generate_recommendations(changes)

    key_name, delta = max(changes.items(), key=lambda x: abs(x[1]))
    delta_desc = f"{delta:+.1f} percentage points"

    focus_note = ""
    if year_focus is not None:
        span = max(year_end - year_start, 1)
        if year_focus <= year_end:
            factor = (year_focus - year_start) / span
        else:
            factor = 1 + (year_focus - year_end) / span
        factor = float(factor)

        focus_changes = {k: v * factor for k, v in changes.items()}
        focus_ndvi = idx["ndvi_mean_start"] + factor * (idx["ndvi_mean_end"] - idx["ndvi_mean_start"])
        focus_ndwi = idx["ndwi_mean_start"] + factor * (idx["ndwi_mean_end"] - idx["ndwi_mean_start"])
        focus_ndbi = idx["ndbi_mean_start"] + factor * (idx["ndbi_mean_end"] - idx["ndbi_mean_start"])
        focus_note = (
            f"Timeline focus {year_focus}: vegetation {focus_changes['vegetation']:+.1f} pp, "
            f"water {focus_changes['water']:+.1f} pp, urban {focus_changes['urban']:+.1f} pp, "
            f"ice {focus_changes['ice']:+.1f} pp. NDVI {focus_ndvi:.2f}, NDWI {focus_ndwi:.2f}, NDBI {focus_ndbi:.2f}. "
        )

    if mode == "expert":
        detail = (
            f"Vegetation {stats_start['vegetation']:.1f}% → {stats_end['vegetation']:.1f}% "
            f"({changes['vegetation']:+.1f} pp), water {stats_start['water']:.1f}% → {stats_end['water']:.1f}% "
            f"({changes['water']:+.1f} pp), urban {stats_start['urban']:.1f}% → {stats_end['urban']:.1f}% "
            f"({changes['urban']:+.1f} pp), ice {stats_start['ice']:.1f}% → {stats_end['ice']:.1f}% "
            f"({changes['ice']:+.1f} pp). "
            f"NDVI {idx['ndvi_mean_start']:.2f} → {idx['ndvi_mean_end']:.2f}, "
            f"NDWI {idx['ndwi_mean_start']:.2f} → {idx['ndwi_mean_end']:.2f}, "
            f"NDBI {idx['ndbi_mean_start']:.2f} → {idx['ndbi_mean_end']:.2f}. "
            f"Largest shift: {key_name} at {delta_desc}. "
        )
    else:
        detail = (
            f"Vegetation changed {changes['vegetation']:+.1f} pp and urban changed {changes['urban']:+.1f} pp "
            f"since {year_start}. Largest shift: {key_name} at {delta_desc}. "
        )

    impact_notes = []
    if changes["vegetation"] < -2:
        impact_notes.append("Reduced vegetation can raise surface temperatures and increase runoff.")
    if changes["urban"] > 2:
        impact_notes.append("Urban expansion often increases impermeable surfaces and heat-island effects.")
    if changes["water"] < -1:
        impact_notes.append("Water loss may stress ecosystems and local water supply reliability.")
    if changes["ice"] < -0.5:
        impact_notes.append("Ice loss signals melt trends that can alter seasonal water availability.")
    if not impact_notes:
        impact_notes.append("Overall land cover remains relatively stable with localized shifts.")

    cause_notes = []
    if changes["urban"] > 2 and changes["vegetation"] < -1:
        cause_notes.append("Likely land conversion from vegetation to built-up areas.")
    if changes["water"] < -1:
        cause_notes.append("Potential drought conditions or upstream withdrawals.")
    if changes["ice"] < -0.5:
        cause_notes.append("Warming trends or shorter snow seasons.")
    if not cause_notes:
        cause_notes.append("Changes may reflect seasonal variability or localized land management.")

    forecast_notes = []
    forecast_years = analysis.get("forecast_years", [])
    forecast = analysis.get("forecast", {})
    if forecast_years and forecast:
        horizon = forecast_years[-1]
        veg_pred = forecast.get("vegetation", [None])[-1]
        urban_pred = forecast.get("urban", [None])[-1]
        if veg_pred is not None and urban_pred is not None:
            forecast_notes.append(
                f"By {horizon}, vegetation is projected near {veg_pred:.1f}% and urban near {urban_pred:.1f}% if trends continue."
            )
    if not forecast_notes:
        forecast_notes.append("Forecasts indicate gradual change if current trends persist.")

    if question:
        question = question.strip()
    if question:
        response = (
            f"You asked: '{question}'. {detail}{focus_note}"
            f"Key recommendation: {recommendations[0]}"
        )
    else:
        response = (
            f"Changes detected between {year_start} and {year_end}. {detail}{focus_note}"
            f"Recommended action: {recommendations[0]}"
        )

    return {
        "summary": response,
        "recommendations": recommendations,
        "insights": {
            "impact": " ".join(impact_notes),
            "causes": " ".join(cause_notes),
            "forecast": " ".join(forecast_notes),
        },
    }


def answer_question(analysis, question=None, mode="simple", year_focus=None):
    if not question or not str(question).strip():
        return {"summary": "Please analyze a location first.", "recommendations": [], "insights": {}}

    q = question.strip().lower()
    explanation = generate_explanation(analysis, mode=mode, question=question, year_focus=year_focus)

    changes = analysis["changes"]
    stats_end = analysis["stats"]["end"]
    year_start = analysis["years"][0]
    year_end = analysis["years"][-1]
    location_name = analysis["location"]["name"]

    by_year = analysis.get("stats", {}).get("by_year", {})
    idx_by_year = analysis.get("indices", {}).get("by_year", {})
    forecast_years = analysis.get("forecast_years", [])
    forecast = analysis.get("forecast", {})

    focus_year = year_focus if year_focus is not None else year_end
    focus_year_int = int(focus_year)
    focus_key = str(focus_year_int)
    focus_is_forecast = focus_year_int > year_end and focus_year_int in forecast_years

    def _forecast_value(metric):
        if focus_year_int not in forecast_years:
            return None
        idx = forecast_years.index(focus_year_int)
        values = forecast.get(metric, [])
        if idx < len(values):
            return values[idx]
        return None

    def _metric_value(metric):
        if focus_key in by_year:
            return by_year[focus_key].get(metric), False
        forecast_value = _forecast_value(metric)
        if forecast_value is not None:
            return forecast_value, True
        return stats_end.get(metric), False

    focus_label = f"{focus_year_int} (forecast)" if focus_is_forecast else str(focus_year_int)
    focus_stats = by_year.get(focus_key, stats_end)
    focus_indices = idx_by_year.get(focus_key, None)

    def _two_sentences(first, second):
        return f"{first.strip()} {second.strip()}"

    key_name, delta = max(changes.items(), key=lambda x: abs(x[1]))
    delta_desc = f"{delta:+.1f} pp"
    key_label = key_name.replace("_", " ")

    topic_flags = {
        "biggest": any(term in q for term in ["biggest", "largest", "most change"]),
        "vegetation": "vegetation" in q,
        "water": "water" in q,
        "urban": "urban" in q or "urbanization" in q,
        "ndvi": "ndvi" in q,
        "ndwi": "ndwi" in q,
        "ndbi": "ndbi" in q,
        "cause": any(term in q for term in ["why", "cause", "driver"]),
        "recommend": any(term in q for term in ["monitor", "should", "recommend"]),
    }

    if not any(topic_flags.values()):
        summary = _two_sentences(
            "I can answer about vegetation, water, urban growth, NDVI/NDWI/NDBI, biggest changes, and recommendations.",
            f"Try asking about vegetation change in {location_name} or the biggest change from {year_start} to {year_end}.",
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    if topic_flags["biggest"]:
        summary = _two_sentences(
            f"In {location_name}, the biggest change from {year_start} to {year_end} is {key_label} ({delta_desc}).",
            f"Vegetation is {changes['vegetation']:+.1f} pp, urban is {changes['urban']:+.1f} pp, and water is {changes['water']:+.1f} pp.",
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    if topic_flags["vegetation"]:
        veg_value, _ = _metric_value("vegetation")
        ndvi_value = focus_indices["ndvi_mean"] if focus_indices else None
        second = f"Selected year {focus_label}: vegetation {veg_value:.1f}%." if veg_value is not None else f"Selected year {focus_label}: vegetation data is unavailable."
        if ndvi_value is not None:
            second = f"{second} NDVI {ndvi_value:.2f}."
        summary = _two_sentences(
            f"In {location_name}, vegetation changed {changes['vegetation']:+.1f} pp between {year_start} and {year_end}.",
            second,
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    if topic_flags["urban"]:
        urban_value, _ = _metric_value("urban")
        ndbi_value = focus_indices["ndbi_mean"] if focus_indices else None
        trend = "increasing" if changes["urban"] > 0 else "decreasing"
        second = f"Selected year {focus_label}: urban {urban_value:.1f}%." if urban_value is not None else f"Selected year {focus_label}: urban data is unavailable."
        if ndbi_value is not None:
            second = f"{second} NDBI {ndbi_value:.2f}."
        summary = _two_sentences(
            f"In {location_name}, urban cover is {trend} ({changes['urban']:+.1f} pp) from {year_start} to {year_end}.",
            second,
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    if topic_flags["water"]:
        water_value, _ = _metric_value("water")
        ndwi_value = focus_indices["ndwi_mean"] if focus_indices else None
        second = f"Selected year {focus_label}: water {water_value:.1f}%." if water_value is not None else f"Selected year {focus_label}: water data is unavailable."
        if ndwi_value is not None:
            second = f"{second} NDWI {ndwi_value:.2f}."
        summary = _two_sentences(
            f"In {location_name}, water cover changed {changes['water']:+.1f} pp between {year_start} and {year_end}.",
            second,
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    if topic_flags["ndvi"]:
        ndvi_value = focus_indices["ndvi_mean"] if focus_indices else None
        second = f"In {location_name}, NDVI is {ndvi_value:.2f} for {focus_label}." if ndvi_value is not None else f"NDVI is unavailable for {focus_label}."
        summary = _two_sentences(
            "NDVI measures vegetation health using near infrared and red reflectance.",
            f"{second} Vegetation changed {changes['vegetation']:+.1f} pp since {year_start}.",
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    if topic_flags["ndwi"]:
        ndwi_value = focus_indices["ndwi_mean"] if focus_indices else None
        second = f"In {location_name}, NDWI is {ndwi_value:.2f} for {focus_label}." if ndwi_value is not None else f"NDWI is unavailable for {focus_label}."
        summary = _two_sentences(
            "NDWI highlights surface water using green and near infrared reflectance.",
            f"{second} Water changed {changes['water']:+.1f} pp since {year_start}.",
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    if topic_flags["ndbi"]:
        ndbi_value = focus_indices["ndbi_mean"] if focus_indices else None
        second = f"In {location_name}, NDBI is {ndbi_value:.2f} for {focus_label}." if ndbi_value is not None else f"NDBI is unavailable for {focus_label}."
        summary = _two_sentences(
            "NDBI indicates built-up areas using SWIR and near infrared reflectance.",
            f"{second} Urban change is {changes['urban']:+.1f} pp since {year_start}.",
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    if topic_flags["cause"]:
        cause = explanation["insights"]["causes"].split(".")[0].strip()
        summary = _two_sentences(
            f"Possible drivers in {location_name}: {cause}.",
            f"Key changes from {year_start} to {year_end} include vegetation {changes['vegetation']:+.1f} pp and urban {changes['urban']:+.1f} pp.",
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    if topic_flags["recommend"]:
        summary = _two_sentences(
            f"Recommendation: {explanation['recommendations'][0]}.",
            f"This targets {key_label} change in {location_name} for {year_start} to {year_end}.",
        )
        return {"summary": summary, "recommendations": explanation["recommendations"], "insights": explanation["insights"]}

    return explanation


def charts_payload(analysis):
    df = pd.DataFrame({"year": analysis["years"], **analysis["series"]})
    series = {col: df[col].tolist() for col in df.columns if col != "year"}
    return {
        "years": df["year"].tolist(),
        "series": series,
        "forecast_years": analysis["forecast_years"],
        "forecast": analysis["forecast"],
    }


def analysis_payload(analysis, mode):
    explanation = generate_explanation(analysis, mode=mode)
    location = analysis["location"]
    coordinates = {"lat": float(location["lat"]), "lon": float(location["lon"])}
    metrics = {
        "changes": analysis["changes"],
        "indices": analysis["indices"],
        "stats": analysis["stats"],
    }
    return {
        "success": True,
        "analysis_id": analysis["analysis_id"],
        "location": location,
        "coordinates": coordinates,
        "bbox": analysis["bbox"],
        "years": analysis["years"],
        "sources": analysis.get("sources", {}),
        "source": analysis.get("source_mode", "demo"),
        "notices": analysis.get("notices", []),
        "messages": analysis.get("notices", []),
        "stats": analysis["stats"],
        "changes": analysis["changes"],
        "indices": analysis["indices"],
        "metrics": metrics,
        "overlays": analysis["overlays"],
        "previews": analysis["previews"],
        "timeline": analysis["timeline"],
        "summary": explanation["summary"],
        "recommendations": explanation["recommendations"],
        "insights": explanation.get("insights", {}),
    }
