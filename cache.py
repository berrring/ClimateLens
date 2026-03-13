import json
import sqlite3
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "cache.db"


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_cache (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                location_name TEXT,
                lat REAL,
                lon REAL,
                year_start INTEGER,
                year_end INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )


def _now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def get_cached_analysis(key):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT payload FROM analysis_cache WHERE key = ?", (key,)).fetchone()
    if not row:
        return None
    return json.loads(row[0])


def set_cached_analysis(key, payload):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO analysis_cache (key, payload, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET payload = excluded.payload, created_at = excluded.created_at
            """,
            (key, json.dumps(payload), _now_iso()),
        )


def add_history(key, location, year_start, year_end):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO analysis_history (key, location_name, lat, lon, year_start, year_end, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                location.get("name"),
                float(location.get("lat")),
                float(location.get("lon")),
                int(year_start),
                int(year_end),
                _now_iso(),
            ),
        )


def get_history(limit=20):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT key, location_name, lat, lon, year_start, year_end, created_at
            FROM analysis_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "key": row[0],
            "location": row[1],
            "lat": row[2],
            "lon": row[3],
            "year_start": row[4],
            "year_end": row[5],
            "created_at": row[6],
        }
        for row in rows
    ]
