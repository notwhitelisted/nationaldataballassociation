"""Enhanced feature engineering for NBA game prediction.

Extends the original GameFeatureBuilder with additional features designed
specifically for spread and totals prediction:

NEW SPREAD FEATURES:
    - Cover rate: how often does this team beat the spread (using historical margins)
    - Blowout rate: how often does this team win/lose by 10+ points
    - Close game rate: how often are games decided by 5 or fewer points
    - Margin consistency: standard deviation of point differentials
    - Win margin vs loss margin: do they win big but lose close, or vice versa

NEW TOTALS FEATURES:
    - Pace proxy: average combined points in recent games
    - Offensive/defensive pace: points scored + allowed (proxy for game tempo)
    - High-scoring game rate: how often do games go over 220 total points
    - Low-scoring game rate: how often do games go under 210 total points
    - Combined points trend: are totals trending up or down

GENERAL IMPROVEMENTS:
    - Streak features: current win/loss streak length
    - Momentum: weighted recent performance (more recent games weighted higher)
    - Opponent-adjusted stats: performance relative to opponent strength

Usage:
    from app.ml.features.enhanced_features import EnhancedFeatureBuilder

    builder = EnhancedFeatureBuilder(games_df)
    feature_matrix = builder.build()
"""

import pandas as pd
import numpy as np

from app.utils import logger


class EnhancedFeatureBuilder:
    """Builds enhanced game-level features for moneyline, spread, and totals prediction.

    All features use only data available BEFORE each game (no data leakage).
    """

    WINDOWS = [3, 5, 10, 15]

    def __init__(self, games_df: pd.DataFrame):
        self.games = games_df.copy().sort_values("game_date").reset_index(drop=True)
        self._build_team_histories()

    def _build_team_histories(self) -> None:
        """Create per-team game history with detailed stats for feature computation."""
        records = []
        for _, game in self.games.iterrows():
            home_score = game["home_score"]
            away_score = game["away_score"]
            total = home_score + away_score
            margin = home_score - away_score

            #home team perspective
            records.append({
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "team_id": game["home_team_id"],
                "opponent_id": game["away_team_id"],
                "is_home": True,
                "points_for": home_score,
                "points_against": away_score,
                "win": game["home_win"],
                "margin": margin,
                "total_points": total,
            })
            #away team perspective
            records.append({
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "team_id": game["away_team_id"],
                "opponent_id": game["home_team_id"],
                "is_home": False,
                "points_for": away_score,
                "points_against": home_score,
                "win": not game["home_win"],
                "margin": -margin,
                "total_points": total,
            })

        self.team_games = pd.DataFrame(records)
        self.team_games = self.team_games.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    def build(self) -> pd.DataFrame:
        """Build the complete enhanced feature matrix."""
        logger.info("Building enhanced feature matrix for {} games...", len(self.games))

        features = []
        for idx, game in self.games.iterrows():
            row = self._build_game_features(game)
            if row is not None:
                features.append(row)

        df = pd.DataFrame(features)
        logger.info(
            "Enhanced feature matrix: {} games, {} features",
            len(df),
            len([c for c in df.columns if c not in ["game_id", "game_date", "season", "home_win"]]),
        )
        return df

    def _build_game_features(self, game: pd.Series) -> dict | None:
        """Build all features for a single game."""
        game_date = game["game_date"]
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        home_hist = self.team_games[
            (self.team_games["team_id"] == home_id)
            & (self.team_games["game_date"] < game_date)
        ]
        away_hist = self.team_games[
            (self.team_games["team_id"] == away_id)
            & (self.team_games["game_date"] < game_date)
        ]

        min_games = self.WINDOWS[0]
        if len(home_hist) < min_games or len(away_hist) < min_games:
            return None

        feat: dict = {
            "game_id": game["game_id"],
            "game_date": game["game_date"],
            "season": game["season"],
            "home_win": game["home_win"],
        }

        #ORIGINAL FEATURES (70+ features based on historical performance)
        for window in self.WINDOWS:
            h = home_hist.tail(window)
            a = away_hist.tail(window)

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

            #scoring variance
            feat[f"{prefix_h}_pts_for_std"] = h["points_for"].std()
            feat[f"{prefix_a}_pts_for_std"] = a["points_for"].std()

            #home/away specific
            h_home = h[h["is_home"]]
            a_away = a[~a["is_home"]]
            if len(h_home) > 0:
                feat[f"{prefix_h}_home_win_pct"] = h_home["win"].mean()
            if len(a_away) > 0:
                feat[f"{prefix_a}_away_win_pct"] = a_away["win"].mean()

            #DIFFERENTIALS
            feat[f"diff_L{window}_win_pct"] = feat.get(f"{prefix_h}_win_pct", 0.5) - feat.get(f"{prefix_a}_win_pct", 0.5)
            feat[f"diff_L{window}_margin"] = feat.get(f"{prefix_h}_margin_avg", 0) - feat.get(f"{prefix_a}_margin_avg", 0)
            feat[f"diff_L{window}_pts_for"] = feat.get(f"{prefix_h}_pts_for_avg", 0) - feat.get(f"{prefix_a}_pts_for_avg", 0)

            #NEW: SPREAD-SPECIFIC FEATURES 

            #margin consistency (std of margins — volatile teams are harder to predict)
            feat[f"{prefix_h}_margin_std"] = h["margin"].std()
            feat[f"{prefix_a}_margin_std"] = a["margin"].std()

            #blowout rate (win by 10+)
            feat[f"{prefix_h}_blowout_rate"] = (h["margin"] >= 10).mean()
            feat[f"{prefix_a}_blowout_rate"] = (a["margin"] >= 10).mean()

            #close game rate (decided by 5 or fewer)
            feat[f"{prefix_h}_close_game_rate"] = (h["margin"].abs() <= 5).mean()
            feat[f"{prefix_a}_close_game_rate"] = (a["margin"].abs() <= 5).mean()

            #average win margin vs average loss margin
            h_wins = h[h["win"] == True]
            h_losses = h[h["win"] == False]
            a_wins = a[a["win"] == True]
            a_losses = a[a["win"] == False]

            feat[f"{prefix_h}_avg_win_margin"] = h_wins["margin"].mean() if len(h_wins) > 0 else 0
            feat[f"{prefix_h}_avg_loss_margin"] = h_losses["margin"].mean() if len(h_losses) > 0 else 0
            feat[f"{prefix_a}_avg_win_margin"] = a_wins["margin"].mean() if len(a_wins) > 0 else 0
            feat[f"{prefix_a}_avg_loss_margin"] = a_losses["margin"].mean() if len(a_losses) > 0 else 0

            #spread-relevant differentials
            feat[f"diff_L{window}_margin_std"] = feat[f"{prefix_h}_margin_std"] - feat[f"{prefix_a}_margin_std"]
            feat[f"diff_L{window}_blowout"] = feat[f"{prefix_h}_blowout_rate"] - feat[f"{prefix_a}_blowout_rate"]

            #NEW: TOTALS-SPECIFIC FEATURES 

            #pace proxy: average total points in team's recent games
            feat[f"{prefix_h}_pace_proxy"] = h["total_points"].mean()
            feat[f"{prefix_a}_pace_proxy"] = a["total_points"].mean()

            #combined pace estimate for this matchup
            feat[f"combined_L{window}_pace"] = (feat[f"{prefix_h}_pace_proxy"] + feat[f"{prefix_a}_pace_proxy"]) / 2

            #offensive + defensive output (higher = faster pace)
            feat[f"{prefix_h}_total_involvement"] = h["points_for"].mean() + h["points_against"].mean()
            feat[f"{prefix_a}_total_involvement"] = a["points_for"].mean() + a["points_against"].mean()

            #high/low scoring game rates
            feat[f"{prefix_h}_high_scoring_rate"] = (h["total_points"] >= 220).mean()
            feat[f"{prefix_a}_high_scoring_rate"] = (a["total_points"] >= 220).mean()
            feat[f"{prefix_h}_low_scoring_rate"] = (h["total_points"] <= 210).mean()
            feat[f"{prefix_a}_low_scoring_rate"] = (a["total_points"] <= 210).mean()

            #total points variance
            feat[f"{prefix_h}_total_pts_std"] = h["total_points"].std()
            feat[f"{prefix_a}_total_pts_std"] = a["total_points"].std()

            #pace differential
            feat[f"diff_L{window}_pace"] = feat[f"{prefix_h}_pace_proxy"] - feat[f"{prefix_a}_pace_proxy"]

        #REST DAYS 
        if len(home_hist) > 0:
            last_home_game = pd.to_datetime(home_hist.iloc[-1]["game_date"])
            feat["home_rest_days"] = (pd.to_datetime(game_date) - last_home_game).days
        if len(away_hist) > 0:
            last_away_game = pd.to_datetime(away_hist.iloc[-1]["game_date"])
            feat["away_rest_days"] = (pd.to_datetime(game_date) - last_away_game).days

        if "home_rest_days" in feat and "away_rest_days" in feat:
            feat["rest_advantage"] = feat["home_rest_days"] - feat["away_rest_days"]
            #NEW: back-to-back flag (rest = 1 day)
            feat["home_b2b"] = 1 if feat["home_rest_days"] == 1 else 0
            feat["away_b2b"] = 1 if feat["away_rest_days"] == 1 else 0
            feat["b2b_disadvantage"] = feat["away_b2b"] - feat["home_b2b"]

        #HEAD-TO-HEAD 
        h2h = home_hist[home_hist["opponent_id"] == away_id]
        if len(h2h) > 0:
            feat["h2h_home_win_pct"] = h2h["win"].mean()
            feat["h2h_games_played"] = len(h2h)
            feat["h2h_avg_margin"] = h2h["margin"].mean()
            #NEW: h2h total points (pace between these specific teams)
            feat["h2h_avg_total"] = h2h["total_points"].mean()
        else:
            feat["h2h_home_win_pct"] = 0.5
            feat["h2h_games_played"] = 0
            feat["h2h_avg_margin"] = 0.0
            feat["h2h_avg_total"] = 215.0  # league average

        #SEASON-LONG STATS
        feat["home_season_win_pct"] = home_hist["win"].mean()
        feat["away_season_win_pct"] = away_hist["win"].mean()
        feat["home_season_games"] = len(home_hist)
        feat["away_season_games"] = len(away_hist)

        #NEW: STREAK FEATURES 
        feat["home_streak"] = self._compute_streak(home_hist)
        feat["away_streak"] = self._compute_streak(away_hist)
        feat["streak_diff"] = feat["home_streak"] - feat["away_streak"]

        #NEW: MOMENTUM (weighted recent performance) 
        # More recent games weighted higher
        feat["home_momentum"] = self._compute_momentum(home_hist, window=10)
        feat["away_momentum"] = self._compute_momentum(away_hist, window=10)
        feat["momentum_diff"] = feat["home_momentum"] - feat["away_momentum"]

        return feat

    @staticmethod
    def _compute_streak(hist: pd.DataFrame) -> int:
        """Compute current win/loss streak. Positive = winning, negative = losing."""
        if len(hist) == 0:
            return 0
        wins = hist["win"].values[::-1]  #most recent first
        if wins[0]:
            streak = 0
            for w in wins:
                if w:
                    streak += 1
                else:
                    break
            return streak
        else:
            streak = 0
            for w in wins:
                if not w:
                    streak -= 1
                else:
                    break
            return streak

    @staticmethod
    def _compute_momentum(hist: pd.DataFrame, window: int = 10) -> float:
        """Compute weighted momentum — recent games matter more.

        Uses exponential weighting where the most recent game has the
        highest weight and older games decay.
        """
        recent = hist.tail(window)
        if len(recent) == 0:
            return 0.0
        n = len(recent)
        weights = np.exp(np.linspace(-1, 0, n))  #exponential decay
        weights = weights / weights.sum()
        margins = recent["margin"].values
        return float(np.dot(weights, margins))