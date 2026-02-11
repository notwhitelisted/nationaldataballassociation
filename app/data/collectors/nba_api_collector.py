"""NBA API data collector.

Wraps the nba_api package to collect games, player stats, and team data.
Handles rate limiting, error recovery, and data normalization into our
Pydantic models.

Usage:
    from app.data.collectors.nba_api_collector import NBAApiCollector

    collector = NBAApiCollector()
    teams = collector.get_all_teams()
    games = collector.get_season_games(2024)
    logs = collector.get_player_game_logs(player_id=2544, season=2024)
"""

import time
from datetime import datetime

import pandas as pd
from nba_api.stats.endpoints import (
    LeagueGameFinder,
    PlayerGameLog as NBAPlayerGameLog,
)
from nba_api.stats.static import players as static_players, teams as static_teams

from app.config import settings
from app.models import Game, Player, PlayerGameLog, Team
from app.utils import logger


class NBAApiCollector:
    """Collects NBA data from the nba_api package.

    Attributes:
        delay: Seconds to wait between API calls to respect rate limits.
        proxy: Optional proxy URL for nba_api requests.
    """

    def __init__(
        self,
        delay: float | None = None,
        proxy: str | None = None,
    ):
        self.delay = delay or settings.nba_api_delay
        self.proxy = proxy or settings.nba_api_proxy or None
        self._request_kwargs = {}
        if self.proxy:
            self._request_kwargs["proxies"] = {"https": self.proxy}

    def _sleep(self) -> None:
        """Rate-limit pause between API calls."""
        time.sleep(self.delay)

    # ── Static Data (local, no API call) ────────────────────────────────

    def get_all_teams(self) -> list[Team]:
        """Return all NBA teams from nba_api static data (no API call)."""
        raw_teams = static_teams.get_teams()
        teams = []
        for t in raw_teams:
            teams.append(
                Team(
                    team_id=t["id"],
                    abbreviation=t["abbreviation"],
                    full_name=t["full_name"],
                    nickname=t["nickname"],
                    city=t["city"],
                    state=t["state"],
                    year_founded=t["year_founded"],
                )
            )
        logger.info("Loaded {} teams from static data", len(teams))
        return teams

    def get_all_players(self) -> list[Player]:
        """Return all NBA players from nba_api static data (no API call)."""
        raw_players = static_players.get_players()
        players = []
        for p in raw_players:
            players.append(
                Player(
                    player_id=p["id"],
                    full_name=p["full_name"],
                    first_name=p["first_name"],
                    last_name=p["last_name"],
                    is_active=p["is_active"],
                )
            )
        logger.info(
            "Loaded {} players ({} active) from static data",
            len(players),
            sum(1 for p in players if p.is_active),
        )
        return players

    # ── Season Games ────────────────────────────────────────────────────

    def get_season_games(self, season: int) -> list[Game]:
        """Fetch all regular-season and playoff games for a given season.

        Args:
            season: Start year of the season (e.g. 2024 for 2024-25).

        Returns:
            List of Game objects, one per game (home + away combined).
        """
        season_str = settings.nba_api_season_string(season)
        logger.info("Fetching games for season {}", season_str)

        # nba_api returns one row per team per game, so we get duplicates.
        # We query once and then pair up home/away rows by game_id.
        self._sleep()
        try:
            finder = LeagueGameFinder(
                season_nullable=season_str,
                league_id_nullable="00",  # NBA
                season_type_nullable="Regular Season",
                **self._request_kwargs,
            )
            df = finder.get_data_frames()[0]
        except Exception as e:
            logger.error("Failed to fetch games for season {}: {}", season_str, e)
            raise

        if df.empty:
            logger.warning("No games found for season {}", season_str)
            return []

        games = self._process_game_finder_df(df, season)

        # Also fetch playoff games
        self._sleep()
        try:
            finder_playoffs = LeagueGameFinder(
                season_nullable=season_str,
                league_id_nullable="00",
                season_type_nullable="Playoffs",
                **self._request_kwargs,
            )
            df_playoffs = finder_playoffs.get_data_frames()[0]
            if not df_playoffs.empty:
                playoff_games = self._process_game_finder_df(df_playoffs, season)
                games.extend(playoff_games)
                logger.info("Added {} playoff games", len(playoff_games))
        except Exception as e:
            logger.warning("Failed to fetch playoff games for {}: {}", season_str, e)

        logger.info("Total games collected for season {}: {}", season_str, len(games))
        return games

    def _process_game_finder_df(self, df: pd.DataFrame, season: int) -> list[Game]:
        """Convert LeagueGameFinder DataFrame into Game objects.

        The DataFrame has one row per team per game. We need to pair home/away
        rows by GAME_ID to create a single Game record.

        The MATCHUP column tells us home vs away:
            - "LAL vs. GSW" means LAL is home
            - "LAL @ GSW" means LAL is away (GSW is home)
        """
        games = []
        seen_game_ids: set[str] = set()

        for _, row in df.iterrows():
            game_id = str(row["GAME_ID"])
            if game_id in seen_game_ids:
                continue

            # Find both rows for this game
            game_rows = df[df["GAME_ID"] == row["GAME_ID"]]
            if len(game_rows) < 2:
                # Incomplete game data, skip
                continue

            # Determine which row is home, which is away
            home_row = None
            away_row = None
            for _, grow in game_rows.iterrows():
                matchup = str(grow["MATCHUP"])
                if " vs. " in matchup:
                    home_row = grow
                elif " @ " in matchup:
                    away_row = grow

            if home_row is None or away_row is None:
                logger.debug("Could not determine home/away for game {}", game_id)
                continue

            try:
                game_date = pd.to_datetime(home_row["GAME_DATE"]).date()
                home_score = int(home_row["PTS"]) if pd.notna(home_row["PTS"]) else 0
                away_score = int(away_row["PTS"]) if pd.notna(away_row["PTS"]) else 0

                game = Game(
                    game_id=game_id,
                    season=season,
                    game_date=game_date,
                    home_team_id=int(home_row["TEAM_ID"]),
                    away_team_id=int(away_row["TEAM_ID"]),
                    home_team_abbr=str(home_row["TEAM_ABBREVIATION"]),
                    away_team_abbr=str(away_row["TEAM_ABBREVIATION"]),
                    home_score=home_score,
                    away_score=away_score,
                    home_win=home_score > away_score,
                )
                games.append(game)
                seen_game_ids.add(game_id)
            except Exception as e:
                logger.debug("Error processing game {}: {}", game_id, e)
                continue

        return games

    # ── Player Game Logs ────────────────────────────────────────────────

    def get_player_game_logs(
        self, player_id: int, season: int
    ) -> list[PlayerGameLog]:
        """Fetch all game logs for a player in a given season.

        Args:
            player_id: NBA player ID.
            season: Start year of the season (e.g. 2024 for 2024-25).

        Returns:
            List of PlayerGameLog objects, one per game played.
        """
        season_str = settings.nba_api_season_string(season)
        self._sleep()

        try:
            game_log = NBAPlayerGameLog(
                player_id=player_id,
                season=season_str,
                season_type_all_star="Regular Season",
                **self._request_kwargs,
            )
            df = game_log.get_data_frames()[0]
        except Exception as e:
            logger.error(
                "Failed to fetch game logs for player {} season {}: {}",
                player_id,
                season_str,
                e,
            )
            raise

        if df.empty:
            logger.debug("No game logs for player {} in {}", player_id, season_str)
            return []

        logs = self._process_player_game_log_df(df, season)
        logger.debug(
            "Fetched {} game logs for player {} in {}",
            len(logs),
            player_id,
            season_str,
        )
        return logs

    def _process_player_game_log_df(
        self, df: pd.DataFrame, season: int
    ) -> list[PlayerGameLog]:
        """Convert PlayerGameLog DataFrame into PlayerGameLog model objects."""
        logs = []

        for _, row in df.iterrows():
            try:
                # Parse minutes from "MM:SS" format or float
                minutes = self._parse_minutes(row.get("MIN", 0))

                game_date = pd.to_datetime(row["GAME_DATE"]).date()

                log = PlayerGameLog(
                    player_id=int(row["Player_ID"]),
                    player_name=str(row.get("PLAYER_NAME", "")),
                    team_id=int(row.get("TEAM_ID", 0)),
                    team_abbr=str(row.get("TEAM_ABBREVIATION", "")),
                    game_id=str(row["Game_ID"]),
                    game_date=game_date,
                    season=season,
                    minutes=minutes,
                    points=int(row.get("PTS", 0) or 0),
                    fgm=int(row.get("FGM", 0) or 0),
                    fga=int(row.get("FGA", 0) or 0),
                    fg_pct=float(row.get("FG_PCT", 0) or 0),
                    fg3m=int(row.get("FG3M", 0) or 0),
                    fg3a=int(row.get("FG3A", 0) or 0),
                    fg3_pct=float(row.get("FG3_PCT", 0) or 0),
                    ftm=int(row.get("FTM", 0) or 0),
                    fta=int(row.get("FTA", 0) or 0),
                    ft_pct=float(row.get("FT_PCT", 0) or 0),
                    oreb=int(row.get("OREB", 0) or 0),
                    dreb=int(row.get("DREB", 0) or 0),
                    reb=int(row.get("REB", 0) or 0),
                    ast=int(row.get("AST", 0) or 0),
                    stl=int(row.get("STL", 0) or 0),
                    blk=int(row.get("BLK", 0) or 0),
                    tov=int(row.get("TOV", 0) or 0),
                    pf=int(row.get("PF", 0) or 0),
                    plus_minus=float(row.get("PLUS_MINUS", 0) or 0),
                )
                # Compute DFS fantasy points
                log.compute_fantasy_points()
                logs.append(log)

            except Exception as e:
                logger.debug("Error processing player game log row: {}", e)
                continue

        return logs

    @staticmethod
    def _parse_minutes(raw_min) -> float:
        """Parse minutes from various formats nba_api returns.

        Could be "34:21" (MM:SS), 34.35 (float), 34 (int), or None.
        """
        if raw_min is None or (isinstance(raw_min, float) and pd.isna(raw_min)):
            return 0.0
        if isinstance(raw_min, (int, float)):
            return float(raw_min)
        if isinstance(raw_min, str) and ":" in raw_min:
            parts = raw_min.split(":")
            try:
                return float(parts[0]) + float(parts[1]) / 60.0
            except (ValueError, IndexError):
                return 0.0
        try:
            return float(raw_min)
        except (ValueError, TypeError):
            return 0.0

    # ── Bulk Collection Helpers ─────────────────────────────────────────

    def get_active_player_ids(self) -> list[int]:
        """Return IDs of all currently active players."""
        players = self.get_all_players()
        return [p.player_id for p in players if p.is_active]

    def get_season_player_logs(
        self,
        season: int,
        player_ids: list[int] | None = None,
        top_n: int | None = None,
    ) -> list[PlayerGameLog]:
        """Fetch game logs for multiple players in a season.

        Args:
            season: Start year of the season.
            player_ids: Specific player IDs to fetch. If None, fetches active players.
            top_n: If set, only fetch logs for this many players (useful for testing).

        Returns:
            Combined list of all PlayerGameLog objects.
        """
        if player_ids is None:
            player_ids = self.get_active_player_ids()

        if top_n is not None:
            player_ids = player_ids[:top_n]

        logger.info(
            "Fetching game logs for {} players in season {}",
            len(player_ids),
            settings.nba_api_season_string(season),
        )

        all_logs: list[PlayerGameLog] = []
        failed: list[int] = []

        for i, pid in enumerate(player_ids):
            try:
                logs = self.get_player_game_logs(pid, season)
                all_logs.extend(logs)
                if (i + 1) % 50 == 0:
                    logger.info(
                        "Progress: {}/{} players fetched ({} total logs)",
                        i + 1,
                        len(player_ids),
                        len(all_logs),
                    )
            except Exception as e:
                logger.warning("Failed to fetch logs for player {}: {}", pid, e)
                failed.append(pid)

        if failed:
            logger.warning("{} players failed: {}", len(failed), failed[:10])

        logger.info(
            "Collected {} total game logs for season {}",
            len(all_logs),
            settings.nba_api_season_string(season),
        )
        return all_logs
