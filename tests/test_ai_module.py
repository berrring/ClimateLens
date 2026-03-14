import base64
from pathlib import Path

import ai_module


def _use_tmp_dirs(tmp_path):
    ai_module.DATA_DIR = tmp_path / "data"
    ai_module.PREVIEW_DIR = ai_module.DATA_DIR / "previews"


def test_analyze_location(tmp_path):
    _use_tmp_dirs(tmp_path)
    location = ai_module.resolve_location("San Francisco")
    analysis = ai_module.analyze_location(location, 2018, 2023)

    assert "changes" in analysis
    assert "overlays" in analysis
    assert "timeline" in analysis

    timeline_years = analysis["timeline"]["years"]
    assert 2018 in timeline_years
    assert 2023 in timeline_years
    assert 2028 in timeline_years
    assert 2024 in timeline_years
    assert 2027 in timeline_years

    overlay_2023 = analysis["timeline"]["overlays"].get("2023")
    assert overlay_2023
    assert "vegetation" in overlay_2023

    preview_2018 = analysis["timeline"]["previews"].get("2018")
    assert preview_2018.startswith("data:image/png;base64,")

    # Ensure changes are not all zeros
    total_change = sum(abs(value) for value in analysis["changes"].values())
    assert total_change > 0


def test_generate_explanation_modes(tmp_path):
    _use_tmp_dirs(tmp_path)
    location = ai_module.resolve_location("San Francisco")
    analysis = ai_module.analyze_location(location, 2018, 2023)

    simple = ai_module.generate_explanation(analysis, mode="simple")
    expert = ai_module.generate_explanation(analysis, mode="expert")

    assert "summary" in simple
    assert "summary" in expert
    assert "NDVI" in expert["summary"]
    assert "NDVI" not in simple["summary"]
    assert "San Francisco" in simple["summary"]


def test_preview_encoding(tmp_path):
    _use_tmp_dirs(tmp_path)
    ai_module.ensure_sample_data()
    preview = ai_module._load_preview(2018)
    assert preview.startswith("data:image/png;base64,")
    base64.b64decode(preview.split(",", 1)[1])


def test_demo_analysis_is_location_aware(tmp_path, monkeypatch):
    _use_tmp_dirs(tmp_path)
    monkeypatch.delenv("SENTINELHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("SENTINELHUB_CLIENT_SECRET", raising=False)

    san_francisco = ai_module.resolve_location("San Francisco")
    tokyo = ai_module.resolve_location("Tokyo")

    sf_analysis = ai_module.analyze_location(san_francisco, 2016, 2024)
    tokyo_analysis = ai_module.analyze_location(tokyo, 2016, 2024)

    assert sf_analysis["source"] == "demo"
    assert sf_analysis["source_mode"] == "config_missing"
    assert sf_analysis["changes"] != tokyo_analysis["changes"]
    assert sf_analysis["series"]["vegetation"] != tokyo_analysis["series"]["vegetation"]
