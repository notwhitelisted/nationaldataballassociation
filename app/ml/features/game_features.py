"""Feature engineering for NBA game outcome prediction.

Builds a feature matrix from raw game and player log data. Each row represents
a single game, with features computed from historical data available BEFORE
that game was played (no data leakage).

Feature Categories:
    1. Team rolling stats (last N games): win%, points scored/allowed, etc.
    2. Head-to-head history
    3. Rest days and schedule factors
    4. Home/away splits
    5. Player availability and impact (future: player-specific models)

Usage:
    from app.ml.features.game_features import GameFeatureBuilder

    builder = GameFeatureBuilder(games_df, player_logs_df)
    feature_matrix = builder.build()
"""

import pandas as pd
import numpy as np

from app.utils import logger


class GameFeatureBuilder:
    """Builds game-level features for outcome prediction.

    All features are computed using only data available BEFORE the game
    in question — this is critical to avoid data leakage.
    """

    #rolling windows to compute stats over
    WINDOWS = [3, 5, 10, 15]

    def __init__(self, games_df: pd.DataFrame, player_logs_df: pd.DataFrame | None = None):
        """
        Args:
            games_df: DataFrame of Game records with columns matching the Game model.
            player_logs_df: Optional DataFrame of PlayerGameLog records.
        """
        self.games = games_df.copy().sort_values("game_date").reset_index(drop=True)
        self.player_logs = player_logs_df

        #pre-compute team game histories for fast lookups
        self._build_team_histories()

    def _build_team_histories(self) -> None:
        """Create per-team game history DataFrames for rolling computations."""
        records = []
        for _, game in self.games.iterrows():
            #home team perspective
            records.append({
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "team_id": game["home_team_id"],
                "opponent_id": game["away_team_id"],
                "is_home": True,
                "points_for": game["home_score"],
                "points_against": game["away_score"],
                "win": game["home_win"],
                "margin": game["home_score"] - game["away_score"],
            })
            #away team perspective
            records.append({
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "team_id": game["away_team_id"],
                "opponent_id": game["home_team_id"],
                "is_home": False,
                "points_for": game["away_score"],
                "points_against": game["home_score"],
                "win": not game["home_win"],
                "margin": game["away_score"] - game["home_score"],
            })

        self.team_games = pd.DataFrame(records)
        self.team_games = self.team_games.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    def build(self) -> pd.DataFrame:
        """Build the complete feature matrix.

        Returns:
            DataFrame with one row per game, containing all features plus the
            target variable (home_win).
        """
        logger.info("Building feature matrix for {} games...", len(self.games))

        features = []
        for idx, game in self.games.iterrows():
            row = self._build_game_features(game)
            if row is not None:
                features.append(row)

        df = pd.DataFrame(features)
        logger.info(
            "Feature matrix built: {} games, {} features",
            len(df),
            len(df.columns) - 3,  #subtract game_id, game_date, target
        )
        return df

    def _build_game_features(self, game: pd.Series) -> dict | None:
        """Build features for a single game.

        Returns None if insufficient history exists for reliable features.
        """
        game_date = game["game_date"]
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        #get historical games for each team BEFORE this game
        home_hist = self.team_games[
            (self.team_games["team_id"] == home_id)
            & (self.team_games["game_date"] < game_date)
        ]
        away_hist = self.team_games[
            (self.team_games["team_id"] == away_id)
            & (self.team_games["game_date"] < game_date)
        ]

        #require minimum history
        min_games = self.WINDOWS[0]  # at least 3 games
        if len(home_hist) < min_games or len(away_hist) < min_games:
            return None

        feat: dict = {
            "game_id": game["game_id"],
            "game_date": game["game_date"],
            "season": game["season"],
            "home_win": game["home_win"],  # TARGET
        }

        #rolling team stats 
        for window in self.WINDOWS:
            h = home_hist.tail(window)
            a = away_hist.tail(window)

            if len(h) < window or len(a) < window:
                #not enough games for this window, use what we have
                if len(h) < min_games or len(a) < min_games:
                    continue

            prefix_h = f"home_L{window}"
            prefix_a = f"away_L{window}"

            #win percentage
            feat[f"{prefix_h}_win_pct"] = h["win"].mean()
            feat[f"{prefix_a}_win_pct"] = a["win"].mean()

            #scoring
            feat[f"{prefix_h}_pts_for_avg"] = h["points_for"].mean()
            feat[f"{prefix_h}_pts_against_avg"] = h["points_against"].mean()
            feat[f"{prefix_a}_pts_for_avg"] = a["points_for"].mean()
            feat[f"{prefix_a}_pts_against_avg"] = a["points_against"].mean()

            #point differential
            feat[f"{prefix_h}_margin_avg"] = h["margin"].mean()
            feat[f"{prefix_a}_margin_avg"] = a["margin"].mean()

            #scoring variance (consistency)
            feat[f"{prefix_h}_pts_for_std"] = h["points_for"].std()
            feat[f"{prefix_a}_pts_for_std"] = a["points_for"].std()

            #home/away specific performance
            h_home = h[h["is_home"]]
            a_away = a[~a["is_home"]]
            if len(h_home) > 0:
                feat[f"{prefix_h}_home_win_pct"] = h_home["win"].mean()
            if len(a_away) > 0:
                feat[f"{prefix_a}_away_win_pct"] = a_away["win"].mean()

        #differential features (home advantage) 
        for window in self.WINDOWS:
            h_key = f"home_L{window}_win_pct"
            a_key = f"away_L{window}_win_pct"
            if h_key in feat and a_key in feat:
                feat[f"diff_L{window}_win_pct"] = feat[h_key] - feat[a_key]
                feat[f"diff_L{window}_margin"] = (
                    feat.get(f"home_L{window}_margin_avg", 0)
                    - feat.get(f"away_L{window}_margin_avg", 0)
                )
                feat[f"diff_L{window}_pts_for"] = (
                    feat.get(f"home_L{window}_pts_for_avg", 0)
                    - feat.get(f"away_L{window}_pts_for_avg", 0)
                )

        #rest days 
        if len(home_hist) > 0:
            last_home_game = pd.to_datetime(home_hist.iloc[-1]["game_date"])
            feat["home_rest_days"] = (pd.to_datetime(game_date) - last_home_game).days
        if len(away_hist) > 0:
            last_away_game = pd.to_datetime(away_hist.iloc[-1]["game_date"])
            feat["away_rest_days"] = (pd.to_datetime(game_date) - last_away_game).days

        if "home_rest_days" in feat and "away_rest_days" in feat:
            feat["rest_advantage"] = feat["home_rest_days"] - feat["away_rest_days"]

        #head-to-head history 
        h2h = home_hist[home_hist["opponent_id"] == away_id]
        if len(h2h) > 0:
            feat["h2h_home_win_pct"] = h2h["win"].mean()
            feat["h2h_games_played"] = len(h2h)
            feat["h2h_avg_margin"] = h2h["margin"].mean()
        else:
            feat["h2h_home_win_pct"] = 0.5  #no history, assume neutral
            feat["h2h_games_played"] = 0
            feat["h2h_avg_margin"] = 0.0

        #season-long stats 
        feat["home_season_win_pct"] = home_hist["win"].mean()
        feat["away_season_win_pct"] = away_hist["win"].mean()
        feat["home_season_games"] = len(home_hist)
        feat["away_season_games"] = len(away_hist)

        return feat

    def get_feature_names(self) -> list[str]:
        """Return list of feature column names (excludes target and metadata)."""
        exclude = {"game_id", "game_date", "season", "home_win"}
        df = self.build()
        return [c for c in df.columns if c not in exclude]
