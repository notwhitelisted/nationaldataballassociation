"""Data storage utilities for saving and loading collected data.

Uses Parquet format for efficient storage and fast loading. All data goes to
the configured DATA_DIR with a consistent directory structure:

    data/
    ├── raw/
    │   ├── games/
    │   │   ├── games_2019.parquet
    │   │   ├── games_2020.parquet
    │   │   └── ...
    │   └── player_logs/
    │       ├── player_logs_2019.parquet
    │       ├── player_logs_2020.parquet
    │       └── ...
    ├── processed/
    │   └── ...
    └── features/
        └── ...
"""

from pathlib import Path

import pandas as pd

from app.config import settings
from app.models import Game, PlayerGameLog, Team, Player
from app.utils import logger


class DataStore:
    """Handles persistence of collected NBA data."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or settings.data_dir
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create directory structure."""
        for subdir in ["raw/games", "raw/player_logs", "raw/teams", "processed", "features"]:
            (self.data_dir / subdir).mkdir(parents=True, exist_ok=True)

    #save methods 

    def save_teams(self, teams: list[Team]) -> Path:
        """Save team data."""
        df = pd.DataFrame([t.model_dump() for t in teams])
        path = self.data_dir / "raw" / "teams" / "teams.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved {} teams to {}", len(df), path)
        return path

    def save_season_games(self, games: list[Game], season: int) -> Path:
        """Save game data for a single season."""
        df = pd.DataFrame([g.model_dump() for g in games])
        path = self.data_dir / "raw" / "games" / f"games_{season}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved {} games for season {} to {}", len(df), season, path)
        return path

    def save_season_player_logs(
        self, logs: list[PlayerGameLog], season: int
    ) -> Path:
        """Save player game logs for a single season."""
        df = pd.DataFrame([log.model_dump() for log in logs])
        path = self.data_dir / "raw" / "player_logs" / f"player_logs_{season}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved {} player logs for season {} to {}", len(df), season, path)
        return path

    #load Methods 

    def load_teams(self) -> pd.DataFrame:
        """Load team data."""
        path = self.data_dir / "raw" / "teams" / "teams.parquet"
        if not path.exists():
            logger.warning("No team data found at {}", path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_season_games(self, season: int) -> pd.DataFrame:
        """Load game data for a single season."""
        path = self.data_dir / "raw" / "games" / f"games_{season}.parquet"
        if not path.exists():
            logger.warning("No game data found for season {} at {}", season, path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_all_games(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Load and combine game data across multiple seasons."""
        seasons = seasons or settings.default_seasons
        dfs = []
        for season in seasons:
            df = self.load_season_games(season)
            if not df.empty:
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        logger.info("Loaded {} total games across {} seasons", len(combined), len(dfs))
        return combined

    def load_season_player_logs(self, season: int) -> pd.DataFrame:
        """Load player game logs for a single season."""
        path = self.data_dir / "raw" / "player_logs" / f"player_logs_{season}.parquet"
        if not path.exists():
            logger.warning("No player log data for season {} at {}", season, path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_all_player_logs(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Load and combine player logs across multiple seasons."""
        seasons = seasons or settings.default_seasons
        dfs = []
        for season in seasons:
            df = self.load_season_player_logs(season)
            if not df.empty:
                dfs.append(df)
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        logger.info("Loaded {} total player logs across {} seasons", len(combined), len(dfs))
        return combined

    #processed/feature Data 

    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        """Save a processed DataFrame."""
        path = self.data_dir / "processed" / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved processed data '{}' ({} rows) to {}", name, len(df), path)
        return path

    def load_processed(self, name: str) -> pd.DataFrame:
        """Load a processed DataFrame."""
        path = self.data_dir / "processed" / f"{name}.parquet"
        if not path.exists():
            logger.warning("No processed data found: {}", path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def save_features(self, df: pd.DataFrame, name: str) -> Path:
        """Save a feature matrix."""
        path = self.data_dir / "features" / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved feature matrix '{}' ({} rows, {} cols) to {}", name, len(df), len(df.columns), path)
        return path

    def load_features(self, name: str) -> pd.DataFrame:
        """Load a feature matrix."""
        path = self.data_dir / "features" / f"{name}.parquet"
        if not path.exists():
            logger.warning("No feature data found: {}", path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    #utilities

    def list_available_seasons(self) -> list[int]:
        """List seasons that have been collected."""
        games_dir = self.data_dir / "raw" / "games"
        if not games_dir.exists():
            return []
        seasons = []
        for f in games_dir.glob("games_*.parquet"):
            try:
                season = int(f.stem.split("_")[1])
                seasons.append(season)
            except (ValueError, IndexError):
                continue
        return sorted(seasons)
