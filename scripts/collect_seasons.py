"""Collect NBA season data from nba_api and save to local storage.

Usage:
    # Collect specific seasons
    python -m scripts.collect_seasons --seasons 2022 2023 2024

    # Collect all default seasons (2019-2024)
    python -m scripts.collect_seasons

    # Collect games only (skip player logs — much faster)
    python -m scripts.collect_seasons --games-only

    # Collect with a limit on players (for testing)
    python -m scripts.collect_seasons --seasons 2024 --top-n-players 20
"""

import argparse
import time

from app.config import settings
from app.data.collectors.nba_api_collector import NBAApiCollector
from app.data.storage import DataStore
from app.utils import logger


def main():
    parser = argparse.ArgumentParser(description="Collect NBA season data")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=settings.default_seasons,
        help="Season start years to collect (e.g. 2024 for 2024-25)",
    )
    parser.add_argument(
        "--games-only",
        action="store_true",
        help="Only collect game results, skip player logs",
    )
    parser.add_argument(
        "--player-logs-only",
        action="store_true",
        help="Only collect player logs, skip games",
    )
    parser.add_argument(
        "--top-n-players",
        type=int,
        default=None,
        help="Limit player log collection to top N players (for testing)",
    )
    args = parser.parse_args()

    collector = NBAApiCollector()
    store = DataStore()

    logger.info("Starting data collection for seasons: {}", args.seasons)
    start_time = time.time()

    # Collect teams (one-time, static)
    teams = collector.get_all_teams()
    store.save_teams(teams)

    for season in args.seasons:
        season_str = settings.nba_api_season_string(season)
        logger.info("=" * 60)
        logger.info("Collecting season {}", season_str)
        logger.info("=" * 60)

        # ── Games ───────────────────────────────────────────────────
        if not args.player_logs_only:
            try:
                games = collector.get_season_games(season)
                store.save_season_games(games, season)
                logger.info(
                    "Season {}: {} games collected and saved",
                    season_str,
                    len(games),
                )
            except Exception as e:
                logger.error("Failed to collect games for {}: {}", season_str, e)

        # ── Player Game Logs ────────────────────────────────────────
        if not args.games_only:
            try:
                logs = collector.get_season_player_logs(
                    season=season,
                    top_n=args.top_n_players,
                )
                store.save_season_player_logs(logs, season)
                logger.info(
                    "Season {}: {} player game logs collected and saved",
                    season_str,
                    len(logs),
                )
            except Exception as e:
                logger.error(
                    "Failed to collect player logs for {}: {}", season_str, e
                )

    elapsed = time.time() - start_time
    logger.info("Data collection complete in {:.1f} minutes", elapsed / 60)


if __name__ == "__main__":
    main()
