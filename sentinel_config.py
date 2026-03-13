import os
from dataclasses import dataclass
from pathlib import Path

try:
    from sentinelhub import SHConfig
except Exception:  # pragma: no cover - optional dependency
    SHConfig = None


@dataclass
class SentinelSettings:
    client_id: str | None
    client_secret: str | None
    max_cloud: float
    cache_dir: Path
    enabled: bool


def _env_bool(value, default=True):
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def load_settings():
    client_id = os.getenv("SENTINELHUB_CLIENT_ID") or os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET") or os.getenv("SH_CLIENT_SECRET")
    max_cloud = float(os.getenv("SENTINELHUB_MAX_CLOUD", "0.2"))
    cache_dir = Path(os.getenv("SENTINELHUB_CACHE_DIR", Path(__file__).parent / "data" / "sentinel_cache"))
    enabled = _env_bool(os.getenv("SENTINELHUB_ENABLED"), default=True)
    return SentinelSettings(
        client_id=client_id,
        client_secret=client_secret,
        max_cloud=max_cloud,
        cache_dir=cache_dir,
        enabled=enabled,
    )


def get_sh_config():
    settings = load_settings()
    if SHConfig is None:
        return None
    if not settings.enabled:
        return None
    if not settings.client_id or not settings.client_secret:
        return None
    config = SHConfig()
    config.sh_client_id = settings.client_id
    config.sh_client_secret = settings.client_secret
    return config


def is_configured():
    return get_sh_config() is not None
