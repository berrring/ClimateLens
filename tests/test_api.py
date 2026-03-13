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
