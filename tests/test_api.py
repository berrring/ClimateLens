from fastapi.testclient import TestClient

import ai_module
import cache
import main


def _configure_tmp(tmp_path):
    ai_module.DATA_DIR = tmp_path / "data"
    ai_module.PREVIEW_DIR = ai_module.DATA_DIR / "previews"
    cache.DATA_DIR = tmp_path
    cache.DB_PATH = tmp_path / "cache.db"


def test_endpoints(tmp_path):
    _configure_tmp(tmp_path)
    cache.init_db()

    with TestClient(main.app) as client:
        resp = client.post("/analyze_location", json={"query": "San Francisco", "mode": "simple"})
        assert resp.status_code == 200
        data = resp.json()
        assert "changes" in data
        assert "timeline" in data
        assert "cached" in data
        assert "source_mode" in data
        assert "source_details" in data

        chart_resp = client.post("/get_charts_data", json={"query": "San Francisco"})
        assert chart_resp.status_code == 200
        chart_data = chart_resp.json()
        assert "years" in chart_data
        assert "series" in chart_data

        explain_resp = client.post("/ai_explanation", json={"query": "San Francisco", "question": "Why?"})
        assert explain_resp.status_code == 200
        assert "summary" in explain_resp.json()

        history_resp = client.get("/history")
        assert history_resp.status_code == 200
        assert "items" in history_resp.json()


def test_endpoint_reports_explicit_auth_failure(tmp_path, monkeypatch):
    _configure_tmp(tmp_path)
    cache.init_db()

    def fake_fetch(location, year, size=(256, 256), max_cloud=None, half_size=0.1, time_interval=None):
        return {
            "ok": False,
            "status": "auth_failed",
            "source": "demo",
            "message": "Sentinel Hub authentication failed.",
            "cached": False,
            "time_interval": list(time_interval or (f"{year}-05-01", f"{year}-09-30")),
            "bbox": [location["lon"] - 0.1, location["lat"] - 0.1, location["lon"] + 0.1, location["lat"] + 0.1],
            "scenes_found": 0,
            "scene_dates": [],
            "runtime": {
                "enabled": True,
                "has_client_id": True,
                "has_client_secret": True,
                "configured": True,
                "sentinelhub_available": True,
            },
            "error": "invalid_client",
        }

    monkeypatch.setattr(ai_module.sentinel_client, "fetch_sentinel_bands", fake_fetch)

    with TestClient(main.app) as client:
        resp = client.post("/analyze_location", json={"query": "San Francisco", "mode": "simple"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "demo"
        assert data["source_mode"] == "auth_failed"
        assert data["messages"][0]["code"] == "auth_failed"
        assert data["source_details"]["by_year"]["2016"]["mode"] == "auth_failed"
