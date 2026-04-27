"""Player impact analysis and availability features.

Collects player game logs from the NBA API, identifies each team's key players,
calculates team performance with/without each player, and generates player
availability features for the prediction model.

Usage:
    # Collect player data and analyze impact
    python -m scripts.player_impact --season 2025

    # Quick check — just show top players and their impact
    python -m scripts.player_impact --season 2025 --summary-only

    # Collect for multiple seasons
    python -m scripts.player_impact --seasons 2024 2025
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import PlayerGameLog as NBAPlayerGameLog
from nba_api.stats.static import players as static_players

from app.data.storage import DataStore
from app.utils import logger


class PlayerImpactAnalyzer:
    """Analyzes how individual player availability affects team performance.

    For each team:
    1. Identifies top players by minutes played
    2. Matches player game logs to team game results
    3. Calculates team record WITH vs WITHOUT each player
    4. Computes a "player impact score" measuring how much each player matters

    This data can then be used as features for the prediction model —
    when a key player is listed as OUT, the model can adjust its probability.
    """

    def __init__(self, delay: float = 0.75):
        self.delay = delay
        self.store = DataStore()

    def collect_season_player_logs(
        self,
        season: int,
        top_n_per_team: int = 8,
    ) -> pd.DataFrame:
        """Collect game logs for top players on every team.

        Args:
            season: Season start year (e.g. 2025 for 2025-26)
            top_n_per_team: Number of top players per team to collect

        Returns:
            DataFrame with all player game logs
        """
        season_str = f"{season}-{str(season + 1)[-2:]}"
        logger.info("Collecting player game logs for season {}", season_str)

        # Load team games to get team IDs and game counts
        games = self.store.load_season_games(season)
        if games.empty:
            logger.error("No game data found for season {}", season)
            return pd.DataFrame()

        # Get all unique team abbreviations and IDs
        teams = {}
        for _, game in games.iterrows():
            teams[game["home_team_abbr"]] = game["home_team_id"]
            teams[game["away_team_abbr"]] = game["away_team_id"]

        logger.info("Found {} teams in season {}", len(teams), season_str)

        # Get all active players
        all_players = static_players.get_players()
        active_players = [p for p in all_players if p["is_active"]]
        logger.info("Total active players: {}", len(active_players))

        # Collect game logs for all active players
        all_logs = []
        failed = []
        collected = 0

        for i, player in enumerate(active_players):
            pid = player["id"]
            name = player["full_name"]

            time.sleep(self.delay)

            try:
                plog = NBAPlayerGameLog(player_id=pid, season=season_str)
                df = plog.get_data_frames()[0]

                if df.empty:
                    continue

                df["PLAYER_NAME"] = name
                df["PLAYER_ID"] = pid
                all_logs.append(df)
                collected += 1

                if collected % 50 == 0:
                    logger.info("Progress: {}/{} players collected ({} with data)",
                               i + 1, len(active_players), collected)

            except Exception as e:
                failed.append(name)
                continue

        if not all_logs:
            logger.warning("No player logs collected")
            return pd.DataFrame()

        combined = pd.concat(all_logs, ignore_index=True)
        logger.info("Collected {} game log entries for {} players",
                     len(combined), collected)

        if failed:
            logger.info("{} players had no data or errors", len(failed))

        # Save raw player logs
        self._save_player_logs(combined, season)

        return combined

    def collect_top_players_only(
        self,
        season: int,
        top_n_per_team: int = 8,
    ) -> pd.DataFrame:
        """Faster collection — only get top players per team using roster data.

        Instead of collecting all 530 players, this identifies the top players
        by checking roster pages and collecting only those who play significant minutes.

        Args:
            season: Season start year
            top_n_per_team: Top N players per team to collect

        Returns:
            DataFrame with player game logs for top players only
        """
        season_str = f"{season}-{str(season + 1)[-2:]}"
        logger.info("Collecting top {} players per team for season {}",
                     top_n_per_team, season_str)

        # Load games to get team info
        games = self.store.load_season_games(season)
        if games.empty:
            logger.error("No game data for season {}", season)
            return pd.DataFrame()

        # Get unique teams
        teams = {}
        for _, game in games.iterrows():
            teams[game["home_team_abbr"]] = game["home_team_id"]
            teams[game["away_team_abbr"]] = game["away_team_id"]

        # For each team, find their most active players
        all_players = static_players.get_players()
        active_players = {p["id"]: p for p in all_players if p["is_active"]}

        all_logs = []
        players_collected = set()

        for team_abbr, team_id in teams.items():
            logger.info("Collecting players for {} (team_id: {})", team_abbr, team_id)

            # Get all player logs for this team using LeagueDashPlayerStats
            # or just iterate through known active players
            team_logs = []
            team_player_count = 0

            for pid, player in active_players.items():
                if pid in players_collected:
                    continue

                time.sleep(self.delay)

                try:
                    plog = NBAPlayerGameLog(player_id=pid, season=season_str)
                    df = plog.get_data_frames()[0]

                    if df.empty:
                        continue

                    # Check if this player plays for this team
                    if "TEAM_ABBREVIATION" in df.columns:
                        player_team = df.iloc[0]["TEAM_ABBREVIATION"]
                        # Normalize abbreviations
                        if player_team == "PHX":
                            player_team = "PHX"
                        elif player_team == "CHA":
                            player_team = "CHA"

                        if player_team != team_abbr:
                            continue

                    df["PLAYER_NAME"] = player["full_name"]
                    df["PLAYER_ID"] = pid
                    team_logs.append(df)
                    players_collected.add(pid)
                    team_player_count += 1

                    if team_player_count >= top_n_per_team:
                        break

                except Exception:
                    continue

            if team_logs:
                combined_team = pd.concat(team_logs, ignore_index=True)
                all_logs.append(combined_team)
                logger.info("  {} — {} players collected", team_abbr, team_player_count)

        if not all_logs:
            return pd.DataFrame()

        combined = pd.concat(all_logs, ignore_index=True)
        logger.info("Total: {} game log entries for {} players",
                     len(combined), len(players_collected))

        self._save_player_logs(combined, season)
        return combined

    def analyze_player_impact(self, season: int) -> pd.DataFrame:
        """Analyze how each player's presence/absence affects their team.

        Loads player game logs and team game results, then for each key player:
        - Identifies which team games they played in vs missed
        - Calculates team win rate, average margin, and average total WITH them
        - Calculates the same WITHOUT them
        - Computes impact scores

        Args:
            season: Season start year

        Returns:
            DataFrame with player impact metrics
        """
        # Load player logs
        player_logs = self._load_player_logs(season)
        if player_logs.empty:
            logger.error("No player logs for season {}. Run collection first.", season)
            return pd.DataFrame()

        # Load team games
        team_games = self.store.load_season_games(season)

        # Build game_id to result mapping
        game_results = {}
        for _, game in team_games.iterrows():
            gid = str(game["game_id"])
            game_results[gid] = {
                "home_team": game["home_team_abbr"],
                "away_team": game["away_team_abbr"],
                "home_score": game["home_score"],
                "away_score": game["away_score"],
                "home_win": game["home_win"],
                "margin": game["home_score"] - game["away_score"],
                "total": game["home_score"] + game["away_score"],
            }

        # Get total games per team
        team_total_games = {}
        team_wins = {}
        team_margins = {}
        team_totals = {}

        for _, game in team_games.iterrows():
            for team_col, is_home in [("home_team_abbr", True), ("away_team_abbr", False)]:
                team = game[team_col]
                if team not in team_total_games:
                    team_total_games[team] = 0
                    team_wins[team] = 0
                    team_margins[team] = []
                    team_totals[team] = []

                team_total_games[team] += 1
                won = game["home_win"] if is_home else not game["home_win"]
                if won:
                    team_wins[team] += 1
                margin = game["home_score"] - game["away_score"]
                team_margins[team].append(margin if is_home else -margin)
                team_totals[team].append(game["home_score"] + game["away_score"])

        # Analyze each player
        impact_records = []

        # Group player logs by player
        if "PLAYER_ID" not in player_logs.columns:
            logger.error("Player logs missing PLAYER_ID column")
            return pd.DataFrame()
        # Extract team abbreviation from MATCHUP column if TEAM_ABBREVIATION doesn't exist
        if "TEAM_ABBREVIATION" not in player_logs.columns and "MATCHUP" in player_logs.columns:
            player_logs["TEAM_ABBREVIATION"] = player_logs["MATCHUP"].str.split(" ").str[0]
            logger.info("Extracted team abbreviations from MATCHUP column")

        for pid, player_df in player_logs.groupby("PLAYER_ID"):
            name = player_df.iloc[0].get("PLAYER_NAME", str(pid))
            team = player_df.iloc[0].get("TEAM_ABBREVIATION", "")
            games_played = len(player_df)
            avg_min = player_df["MIN"].astype(float).mean()
            avg_pts = player_df["PTS"].astype(float).mean()

            # Skip players with very few minutes (bench warmers)
            if avg_min < 15 or games_played < 20:
                continue

            total_team_games = team_total_games.get(team, 0)
            if total_team_games == 0:
                continue

            games_missed = total_team_games - games_played

            # Find game IDs this player played in
            player_game_ids = set(player_df["Game_ID"].astype(str).values)

            # Calculate team record WITH this player
            wins_with = 0
            games_with = 0
            margins_with = []
            totals_with = []

            wins_without = 0
            games_without = 0
            margins_without = []
            totals_without = []

            for _, game in team_games.iterrows():
                gid = str(game["game_id"])
                is_home = game["home_team_abbr"] == team

                if not (game["home_team_abbr"] == team or game["away_team_abbr"] == team):
                    continue

                won = game["home_win"] if is_home else not game["home_win"]
                margin = (game["home_score"] - game["away_score"]) if is_home else (game["away_score"] - game["home_score"])
                total = game["home_score"] + game["away_score"]

                # Check if player played in this game
                # Game IDs from NBA API might have different format
                # Try matching with the last part of the ID
                player_in_game = gid in player_game_ids

                if player_in_game:
                    games_with += 1
                    if won:
                        wins_with += 1
                    margins_with.append(margin)
                    totals_with.append(total)
                else:
                    games_without += 1
                    if won:
                        wins_without += 1
                    margins_without.append(margin)
                    totals_without.append(total)

            # Calculate impact metrics
            win_rate_with = wins_with / games_with if games_with > 0 else 0
            win_rate_without = wins_without / games_without if games_without > 0 else 0
            avg_margin_with = np.mean(margins_with) if margins_with else 0
            avg_margin_without = np.mean(margins_without) if margins_without else 0
            avg_total_with = np.mean(totals_with) if totals_with else 0
            avg_total_without = np.mean(totals_without) if totals_without else 0

            impact_records.append({
                "player_id": pid,
                "player_name": name,
                "team_abbr": team,
                "games_played": games_played,
                "games_missed": games_missed,
                "avg_minutes": round(avg_min, 1),
                "avg_points": round(avg_pts, 1),
                "win_rate_with": round(win_rate_with, 3),
                "win_rate_without": round(win_rate_without, 3),
                "win_rate_impact": round(win_rate_with - win_rate_without, 3),
                "avg_margin_with": round(avg_margin_with, 1),
                "avg_margin_without": round(avg_margin_without, 1),
                "margin_impact": round(avg_margin_with - avg_margin_without, 1),
                "avg_total_with": round(avg_total_with, 1),
                "avg_total_without": round(avg_total_without, 1),
                "total_impact": round(avg_total_with - avg_total_without, 1),
                "season": season,
            })

        impact_df = pd.DataFrame(impact_records)

        if not impact_df.empty:
            impact_df = impact_df.sort_values("margin_impact", ascending=False).reset_index(drop=True)
            self._save_impact_data(impact_df, season)
            logger.info("Analyzed impact for {} players across {} teams",
                         len(impact_df), impact_df["team_abbr"].nunique())

        return impact_df

    def print_impact_summary(self, impact_df: pd.DataFrame) -> None:
        """Print a formatted summary of player impact analysis."""
        if impact_df.empty:
            print("No impact data available.")
            return

        print(f"\n{'='*85}")
        print(f"  PLAYER IMPACT ANALYSIS — {len(impact_df)} players analyzed")
        print(f"{'='*85}")

        # Top 10 most impactful players
        print(f"\n  TOP 15 MOST IMPACTFUL PLAYERS (by margin impact):")
        print(f"  {'Player':<22} {'Team':<5} {'GP':<4} {'PPG':<6} {'MPG':<6} "
              f"{'WR With':<9} {'WR Without':<11} {'Impact':<8} {'Margin Δ':<9}")
        print(f"  {'-'*80}")

        for _, row in impact_df.head(15).iterrows():
            wr_with = f"{row['win_rate_with']:.1%}"
            wr_without = f"{row['win_rate_without']:.1%}"
            impact = f"{row['win_rate_impact']:+.1%}"
            margin = f"{row['margin_impact']:+.1f}"
            print(f"  {row['player_name']:<22} {row['team_abbr']:<5} "
                  f"{row['games_played']:<4} {row['avg_points']:<6.1f} {row['avg_minutes']:<6.1f} "
                  f"{wr_with:<9} {wr_without:<11} {impact:<8} {margin:<9}")

        # Bottom 5 (players whose absence helps the team)
        print(f"\n  LEAST IMPACTFUL (team performs similarly or better without):")
        print(f"  {'Player':<22} {'Team':<5} {'GP':<4} "
              f"{'WR With':<9} {'WR Without':<11} {'Impact':<8}")
        print(f"  {'-'*65}")

        for _, row in impact_df.tail(5).iterrows():
            wr_with = f"{row['win_rate_with']:.1%}"
            wr_without = f"{row['win_rate_without']:.1%}"
            impact = f"{row['win_rate_impact']:+.1%}"
            print(f"  {row['player_name']:<22} {row['team_abbr']:<5} "
                  f"{row['games_played']:<4} {wr_with:<9} {wr_without:<11} {impact:<8}")

        # Team summary
        print(f"\n  TEAM DEPENDENCY (most reliant on key players):")
        team_max_impact = impact_df.groupby("team_abbr")["margin_impact"].max().sort_values(ascending=False)
        for team, impact in team_max_impact.head(10).items():
            top_player = impact_df[impact_df["team_abbr"] == team].iloc[0]["player_name"]
            print(f"    {team}: {top_player} ({impact:+.1f} margin impact)")

        print()

    def _save_player_logs(self, df: pd.DataFrame, season: int) -> None:
        path = Path(f"data/raw/player_logs/player_logs_{season}.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Saved {} player logs for season {} to {}", len(df), season, path)

    def _load_player_logs(self, season: int) -> pd.DataFrame:
        path = Path(f"data/raw/player_logs/player_logs_{season}.parquet")
        if not path.exists():
            logger.warning("No player logs found at {}", path)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def _save_impact_data(self, df: pd.DataFrame, season: int) -> None:
        path = Path(f"data/processed/player_impact_{season}.parquet")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info("Saved player impact data to {}", path)

    def load_impact_data(self, season: int) -> pd.DataFrame:
        path = Path(f"data/processed/player_impact_{season}.parquet")
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)


def main():
    parser = argparse.ArgumentParser(description="Player impact analysis")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2025],
        help="Seasons to analyze (default: 2025)",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect player game logs from NBA API (takes ~10 minutes per season)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze player impact from collected data",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Just print the impact summary (requires prior analysis)",
    )
    args = parser.parse_args()

    analyzer = PlayerImpactAnalyzer()

    for season in args.seasons:
        if args.summary_only:
            impact = analyzer.load_impact_data(season)
            if impact.empty:
                print(f"No impact data for season {season}. Run with --collect --analyze first.")
            else:
                analyzer.print_impact_summary(impact)
            continue

        if args.collect:
            logger.info("=" * 60)
            logger.info("COLLECTING PLAYER GAME LOGS — Season {}", season)
            logger.info("This will take ~10 minutes (530 players × 0.75s delay)")
            logger.info("=" * 60)
            logs = analyzer.collect_season_player_logs(season)
            logger.info("Collection complete: {} log entries", len(logs))

        if args.analyze:
            logger.info("=" * 60)
            logger.info("ANALYZING PLAYER IMPACT — Season {}", season)
            logger.info("=" * 60)
            impact = analyzer.analyze_player_impact(season)
            analyzer.print_impact_summary(impact)

        if not args.collect and not args.analyze and not args.summary_only:
            # Default: do both
            logger.info("=" * 60)
            logger.info("COLLECTING PLAYER GAME LOGS — Season {}", season)
            logger.info("This will take ~10 minutes (530 players × 0.75s delay)")
            logger.info("=" * 60)
            logs = analyzer.collect_season_player_logs(season)

            logger.info("=" * 60)
            logger.info("ANALYZING PLAYER IMPACT — Season {}", season)
            logger.info("=" * 60)
            impact = analyzer.analyze_player_impact(season)
            analyzer.print_impact_summary(impact)


if __name__ == "__main__":
    main()