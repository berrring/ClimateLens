import os
import sys
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
    base_url: str | None
    token_url: str | None
    enabled: bool


def _env_bool(value, default=True):
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def load_settings():
    client_id = os.getenv("SENTINELHUB_CLIENT_ID") or os.getenv("SH_CLIENT_ID")
    client_secret = os.getenv("SENTINELHUB_CLIENT_SECRET") or os.getenv("SH_CLIENT_SECRET")
    try:
        max_cloud = float(os.getenv("SENTINELHUB_MAX_CLOUD", "0.2"))
    except ValueError:
        max_cloud = 0.2
    cache_dir = Path(os.getenv("SENTINELHUB_CACHE_DIR", Path(__file__).parent / "data" / "sentinel_cache"))
    base_url = os.getenv("SENTINELHUB_BASE_URL") or os.getenv("SH_BASE_URL")
    token_url = os.getenv("SENTINELHUB_TOKEN_URL") or os.getenv("SH_TOKEN_URL")
    enabled = _env_bool(os.getenv("SENTINELHUB_ENABLED"), default=True)
    return SentinelSettings(
        client_id=client_id,
        client_secret=client_secret,
        max_cloud=max_cloud,
        cache_dir=cache_dir,
        base_url=base_url,
        token_url=token_url,
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
    if settings.base_url:
        config.sh_base_url = settings.base_url
    if settings.token_url:
        config.sh_token_url = settings.token_url
    return config


def is_configured():
    return get_sh_config() is not None


def settings_summary():
    settings = load_settings()
    configured = bool(SHConfig is not None and settings.enabled and settings.client_id and settings.client_secret)
    return {
        "enabled": settings.enabled,
        "has_client_id": bool(settings.client_id),
        "has_client_secret": bool(settings.client_secret),
        "base_url": settings.base_url,
        "token_url": settings.token_url,
        "cache_dir": str(settings.cache_dir),
        "configured": configured,
        "sentinelhub_available": SHConfig is not None,
        "python_executable": sys.executable,
    }
