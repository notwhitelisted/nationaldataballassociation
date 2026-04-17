"""Application configuration using pydantic-settings.

Loads from .env file and environment variables. All config is centralized here
so there's a single source of truth for settings across the application.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── NBA API ──────────────────────────────────────────────────────────
    nba_api_delay: float = 0.75  # seconds between API calls to avoid rate limiting
    nba_api_proxy: str = ""  # optional proxy for nba_api requests

    # ── Data Storage ─────────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    models_dir: Path = Path("./models")

    # ── Seasons ──────────────────────────────────────────────────────────
    # Default seasons to collect (year = start year, e.g. 2024 = 2024-25 season)
    default_seasons: list[int] = [2019, 2020, 2021, 2022, 2023, 2024, 2025]

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── External APIs (optional) ─────────────────────────────────────────
    odds_api_key: str = ""

    # ── Database (future) ────────────────────────────────────────────────
    database_url: str = ""

    def ensure_directories(self) -> None:
        """Create data and model directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "features").mkdir(exist_ok=True)

    def season_string(self, season: int) -> str:
        """Convert season year to NBA format string, e.g. 2024 -> '2024-25'."""
        next_year = str(season + 1)[-2:]
        return f"{season}-{next_year}"

    def nba_api_season_string(self, season: int) -> str:
        """Convert season year to nba_api format, e.g. 2024 -> '2024-25'."""
        return self.season_string(season)


# Singleton instance — import this throughout the app
settings = Settings()
