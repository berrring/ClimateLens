import cache


def test_cache_roundtrip(tmp_path):
    cache.DATA_DIR = tmp_path
    cache.DB_PATH = tmp_path / "cache.db"
    cache.init_db()

    key = "test:2018-2023"
    payload = {"hello": "world", "value": 42}
    cache.set_cached_analysis(key, payload)
    loaded = cache.get_cached_analysis(key)

    assert loaded == payload


def test_history(tmp_path):
    cache.DATA_DIR = tmp_path
    cache.DB_PATH = tmp_path / "cache.db"
    cache.init_db()

    location = {"name": "Test", "lat": 1.0, "lon": 2.0}
    cache.add_history("k1", location, 2018, 2023)
    items = cache.get_history(limit=5)

    assert len(items) == 1
    assert items[0]["location"] == "Test"
