"""Spread and totals prediction models.

Extends the game prediction pipeline to support point spread and over/under
betting by training regression models that predict:
    1. Point differential (home_score - away_score) for spread bets
    2. Total points (home_score + away_score) for over/under bets

Uses the same 70 features from GameFeatureBuilder. The key difference from
GamePredictor is that these are regression problems (predicting a number)
rather than classification (predicting win/loss).

Usage:
    from app.ml.models.spread_totals_predictor import SpreadTotalsPredictor

    predictor = SpreadTotalsPredictor()
    predictor.load_data("game_features_all_seasons", "data/raw/nba_2008-2025.csv")
    results = predictor.train_and_evaluate()
    predictor.print_results(results)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from app.config import settings
from app.data.storage import DataStore
from app.data.processors.odds_processor import OddsProcessor
from app.utils import logger


@dataclass
class SpreadModelResult:
    """Results from training and evaluating a spread/totals model."""

    model_name: str
    target_type: str  # "spread" or "total"

    # Regression metrics
    mae: float  # mean absolute error (average points off)
    rmse: float  # root mean squared error
    r2: float  # R-squared (how much variance explained)

    # Betting performance
    bets_placed: int = 0
    bets_won: int = 0
    win_rate: float = 0.0
    profit: float = 0.0
    roi: float = 0.0
    total_wagered: float = 0.0

    # Raw predictions for analysis
    y_true: np.ndarray = field(default=None, repr=False)
    y_pred: np.ndarray = field(default=None, repr=False)
    lines: np.ndarray = field(default=None, repr=False)  # sportsbook lines


class SpreadTotalsPredictor:
    """Trains and evaluates spread and totals prediction models.

    For spread betting:
        - Predicts home_score - away_score (point differential)
        - Compares prediction to sportsbook spread
        - If model says home wins by 8 and spread is -5.5, bet home to cover

    For totals betting:
        - Predicts home_score + away_score (combined points)
        - Compares prediction to sportsbook over/under line
        - If model says 220 total points and line is 212.5, bet the over
    """

    MODELS = {
        "ridge": lambda: Ridge(alpha=1.0),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": lambda: XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            random_state=42,
            verbosity=0,
        ),
    }

    def __init__(self):
        self.store = DataStore()
        self.scaler = StandardScaler()
        self.features_df: pd.DataFrame | None = None
        self.feature_cols: list[str] = []

    def load_data(
        self,
        features_name: str = "game_features_all_seasons",
        odds_csv_path: str = "data/raw/nba_2008-2025.csv",
    ) -> None:
        """Load features and merge with real odds data.

        This combines your feature matrix with the Kaggle odds dataset
        so each game has both ML features and the real sportsbook lines.
        """
        # Load features
        self.features_df = self.store.load_features(features_name)
        if self.features_df.empty:
            raise FileNotFoundError(f"No feature data found: {features_name}")

        # Identify feature columns
        exclude = {"game_id", "game_date", "season", "home_win"}
        self.feature_cols = [c for c in self.features_df.columns if c not in exclude]

        # Load and merge odds
        games = self.store.load_all_games()
        processor = OddsProcessor(odds_csv_path)
        games_with_odds = processor.merge_with_games(games)

        # Add score differential and total to features
        # We need these as regression targets
        score_cols = games_with_odds[["game_id", "home_score", "away_score",
                                       "home_spread", "total", "implied_home_win_prob"]].copy()
        score_cols["point_diff"] = score_cols["home_score"] - score_cols["away_score"]
        score_cols["total_points"] = score_cols["home_score"] + score_cols["away_score"]

        # Merge into features
        self.features_df = self.features_df.merge(
            score_cols[["game_id", "point_diff", "total_points", "home_spread", "total",
                         "home_score", "away_score"]],
            on="game_id",
            how="inner",
        )

        # Sort by date and drop NaN
        self.features_df = self.features_df.sort_values("game_date").reset_index(drop=True)
        before = len(self.features_df)
        self.features_df = self.features_df.dropna(subset=self.feature_cols + ["point_diff", "total_points", "home_spread", "total"])
        after = len(self.features_df)

        logger.info(
            "Loaded {} games with features + odds (dropped {} incomplete)",
            after, before - after,
        )

    def train_and_evaluate(
        self,
        test_seasons: list[int] | None = None,
        model_names: list[str] | None = None,
    ) -> dict[str, SpreadModelResult]:
        """Train spread and totals models, evaluate with real odds.

        Returns results for each model on both spread and totals prediction.
        """
        if self.features_df is None:
            raise RuntimeError("Call load_data() first")

        if model_names is None:
            model_names = list(self.MODELS.keys())

        if test_seasons is None:
            all_seasons = sorted(self.features_df["season"].unique())
            test_seasons = [all_seasons[-1]]

        # Split temporally
        train_df = self.features_df[~self.features_df["season"].isin(test_seasons)]
        test_df = self.features_df[self.features_df["season"].isin(test_seasons)]

        logger.info("Train: {} games, Test: {} games", len(train_df), len(test_df))

        # Prepare features
        X_train = self.scaler.fit_transform(train_df[self.feature_cols].values)
        X_test = self.scaler.transform(test_df[self.feature_cols].values)

        results = {}

        # Train for both target types
        for target_type in ["spread", "total"]:
            if target_type == "spread":
                y_train = train_df["point_diff"].values
                y_test = test_df["point_diff"].values
                lines = test_df["home_spread"].values  # sportsbook spread
            else:
                y_train = train_df["total_points"].values
                y_test = test_df["total_points"].values
                lines = test_df["total"].values  # sportsbook over/under

            for name in model_names:
                key = f"{name}_{target_type}"
                logger.info("Training {} for {} prediction", name, target_type)

                result = self._train_single_model(
                    name=name,
                    target_type=target_type,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    lines=lines,
                )
                results[key] = result

        return results

    def _train_single_model(
        self,
        name: str,
        target_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        lines: np.ndarray,
    ) -> SpreadModelResult:
        """Train a single regression model and evaluate against real lines."""

        # Train
        model = self.MODELS[name]()
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info("{} {} — MAE: {:.2f}, RMSE: {:.2f}, R²: {:.4f}",
                     name, target_type, mae, rmse, r2)

        # Backtest against real sportsbook lines
        bet_results = self._backtest_against_lines(
            y_true=y_test,
            y_pred=y_pred,
            lines=lines,
            target_type=target_type,
        )

        return SpreadModelResult(
            model_name=name,
            target_type=target_type,
            mae=mae,
            rmse=rmse,
            r2=r2,
            y_true=y_test,
            y_pred=y_pred,
            lines=lines,
            **bet_results,
        )

    def _backtest_against_lines(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lines: np.ndarray,
        target_type: str,
        bet_size: float = 10.0,
        min_edge: float = 1.5,  # minimum points of disagreement to bet
    ) -> dict:
        """Backtest predictions against real sportsbook lines.

        For spreads:
            - If model predicts home wins by 8 and spread is -5.5,
              the model thinks home covers by 2.5 points → bet home
            - If model predicts home wins by 3 and spread is -5.5,
              the model thinks home doesn't cover → bet away (or skip)

        For totals:
            - If model predicts 220 total and line is 212.5,
              model is 7.5 points over → bet the over
            - If model predicts 208 total and line is 212.5,
              model is 4.5 points under → bet the under

        Standard odds for spread and totals bets: -110 (decimal 1.909)
        This means you risk $110 to win $100.

        Args:
            min_edge: Minimum points of disagreement between model and line
                      to place a bet. Higher = more selective = fewer bets.
        """
        standard_decimal_odds = 1.909  # -110 odds (standard for spread/total bets)
        bets = 0
        wins = 0
        profit = 0.0

        for i in range(len(y_true)):
            if np.isnan(lines[i]):
                continue

            if target_type == "spread":
                # Model's predicted margin vs sportsbook spread
                # home_spread is negative when home is favored
                model_margin = y_pred[i]
                book_spread = lines[i]

                # Edge = how much our prediction disagrees with the line
                # If model says +8 and spread is -5.5, edge = 8 - (-5.5) = 13.5
                # Positive edge = model thinks home covers
                edge = model_margin - book_spread

                if abs(edge) < min_edge:
                    continue

                bets += 1
                actual_margin = y_true[i]

                if edge > 0:
                    # Bet home to cover: home needs to win by more than spread
                    covered = actual_margin > abs(book_spread) if book_spread < 0 else actual_margin > -book_spread
                    # Simplified: did home beat the spread?
                    actual_vs_spread = actual_margin + book_spread  # positive = home covered
                    if actual_vs_spread > 0:
                        wins += 1
                        profit += bet_size * (standard_decimal_odds - 1)
                    elif actual_vs_spread == 0:
                        pass  # push, no win no loss
                    else:
                        profit -= bet_size
                else:
                    # Bet away to cover: home needs to lose or win by less than spread
                    actual_vs_spread = actual_margin + book_spread
                    if actual_vs_spread < 0:
                        wins += 1
                        profit += bet_size * (standard_decimal_odds - 1)
                    elif actual_vs_spread == 0:
                        pass  # push
                    else:
                        profit -= bet_size

            else:  # totals
                model_total = y_pred[i]
                book_total = lines[i]
                edge = model_total - book_total

                if abs(edge) < min_edge:
                    continue

                bets += 1
                actual_total = y_true[i]

                if edge > 0:
                    # Bet the over
                    if actual_total > book_total:
                        wins += 1
                        profit += bet_size * (standard_decimal_odds - 1)
                    elif actual_total == book_total:
                        pass  # push
                    else:
                        profit -= bet_size
                else:
                    # Bet the under
                    if actual_total < book_total:
                        wins += 1
                        profit += bet_size * (standard_decimal_odds - 1)
                    elif actual_total == book_total:
                        pass  # push
                    else:
                        profit -= bet_size

        total_wagered = bets * bet_size
        roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
        win_rate = (wins / bets * 100) if bets > 0 else 0

        return {
            "bets_placed": bets,
            "bets_won": wins,
            "win_rate": round(win_rate, 1),
            "profit": round(profit, 2),
            "roi": round(roi, 2),
            "total_wagered": round(total_wagered, 2),
        }

    def print_results(self, results: dict[str, SpreadModelResult]) -> None:
        """Print formatted results for all models."""

        # Group by target type
        for target_type in ["spread", "total"]:
            type_results = {k: v for k, v in results.items() if v.target_type == target_type}
            if not type_results:
                continue

            label = "SPREAD" if target_type == "spread" else "TOTALS (OVER/UNDER)"
            print(f"\n{'='*70}")
            print(f"{label} PREDICTION RESULTS")
            print(f"{'='*70}")

            print(f"\n{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} "
                  f"{'Bets':>6} {'Wins':>6} {'Win%':>7} {'Profit':>10} {'ROI':>8}")
            print("-" * 90)

            for key, r in type_results.items():
                print(
                    f"{r.model_name:<20} "
                    f"{r.mae:>8.2f} "
                    f"{r.rmse:>8.2f} "
                    f"{r.r2:>8.4f} "
                    f"{r.bets_placed:>6} "
                    f"{r.bets_won:>6} "
                    f"{r.win_rate:>6.1f}% "
                    f"${r.profit:>9.2f} "
                    f"{r.roi:>+7.2f}%"
                )

            print("-" * 90)
            print(f"  Break-even win rate at -110 odds: 52.4%")
            print(f"  Bet size: $10 flat | Min edge: 1.5 points")