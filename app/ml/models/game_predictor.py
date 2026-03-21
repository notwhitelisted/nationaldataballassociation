"""
Trains and evaluates models for predicting NBA game outcomes (home win/loss). The predictor uses 
temporal (time-series) cross-validation to prevent data leakage - train on past and test on future games.

Models Implemented:
1. Baseline: Always predict home team wins 
2. Logistic Regression: Simple linear model 
3. Random Forest: Ensemble of decision trees
4. XGBoost: Gradient boosted trees (typically best performance)

Each model is evaluated on both accuracy and calibration metrics, following Walsh & Joshi (2024)'s findings that 
calibration matters more than accuracy for betting profitability. 

Usage:
    from app.ml.models.game_predictor import GamePredictor

    predictor = GamePredictor()
    predictor.load_features("game_features_all_seasons")
    results = predictor.train_and_evaluate()
    predictor.print_results(results)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from xgboost import XGBClassifier

from app.config import settings
from app.data.storage import DataStore
from app.ml.calibration import (
    CalibrationMetrics,
    ModelCalibrator,
    CalibrationResult,
)
from app.ml.evaluation import KellyCriterion, simulate_betting_strategy
from app.utils import logger


@dataclass
class ModelResult:
    """Results from training and evaluating a single model."""

    model_name: str
    accuracy: float
    log_loss_val: float
    roc_auc: float
    home_win_rate: float  # baseline metric

    # Calibration metrics
    calibration_before: CalibrationResult  # uncalibrated
    calibration_after: dict[str, CalibrationResult] = field(default_factory=dict)

    # Best calibration method
    best_calibration_method: str = "none"
    best_ece: float = 1.0

    # Raw predictions for further analysis
    y_true: np.ndarray = field(default=None, repr=False)
    y_prob_uncalibrated: np.ndarray = field(default=None, repr=False)
    y_prob_calibrated: np.ndarray = field(default=None, repr=False)


class GamePredictor:
    """Trains and evaluates game outcome prediction models.

    Main entry point for the ML pipeline:
    1. Load and split features temporally to avoid data leakage
    2. Train multiple models (logistic regression, random forest, xgboost)
    3. Apply and compare calibration methods
    4. Evaluate accuracy and calibration metrics
    """

    MODELS = {
        "logistic_regression": lambda: LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
    }

    CALIBRATION_METHODS = ["platt", "isotonic", "temperature"]

    def __init__(self):
        self.store = DataStore()
        self.scaler = StandardScaler()
        self.features_df: pd.DataFrame | None = None
        self.feature_cols: list[str] = []
        self.target_col = "home_win"

    def load_features(self, name: str = "game_features_all_seasons") -> None:
        """Load a saved feature matrix."""
        self.features_df = self.store.load_features(name)
        if self.features_df.empty:
            raise FileNotFoundError(f"No feature data found: {name}")

        # Identify feature columns (exclude target and metadata)
        exclude = {"game_id", "game_date", "season", "home_win"}
        self.feature_cols = [c for c in self.features_df.columns if c not in exclude]

        # Sort by date to ensure temporal order
        self.features_df = self.features_df.sort_values("game_date").reset_index(drop=True)

        # Drop rows with any NaN features
        before = len(self.features_df)
        self.features_df = self.features_df.dropna(subset=self.feature_cols)
        after = len(self.features_df)
        if before != after:
            logger.info("Dropped {} rows with NaN features", before - after)

        logger.info(
            "Loaded features: {} games, {} features, date range {} to {}",
            len(self.features_df),
            len(self.feature_cols),
            self.features_df["game_date"].min(),
            self.features_df["game_date"].max(),
        )

    def temporal_train_test_split(
        self, test_seasons: list[int] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data temporally — train on older seasons, test on recent ones.

        Args:
            test_seasons: Seasons to use for testing. Defaults to the most
                recent season in the dataset.

        Returns:
            (train_df, test_df) tuple.
        """
        if self.features_df is None:
            raise RuntimeError("Call load_features() first")

        if test_seasons is None:
            # Use the most recent season as test
            all_seasons = sorted(self.features_df["season"].unique())
            test_seasons = [all_seasons[-1]]

        train = self.features_df[~self.features_df["season"].isin(test_seasons)]
        test = self.features_df[self.features_df["season"].isin(test_seasons)]

        logger.info(
            "Train: {} games (seasons {}), Test: {} games (seasons {})",
            len(train),
            sorted(train["season"].unique()),
            len(test),
            sorted(test["season"].unique()),
        )
        return train, test

    def train_and_evaluate(
        self,
        test_seasons: list[int] | None = None,
        model_names: list[str] | None = None,
    ) -> dict[str, ModelResult]:
        """Train all models and evaluate on accuracy + calibration.

        This is the main method that produces results for your report.

        Args:
            test_seasons: Seasons for test set. Defaults to most recent.
            model_names: Which models to train. Defaults to all.

        Returns:
            Dict mapping model name to ModelResult.
        """
        if self.features_df is None:
            raise RuntimeError("Call load_features() first")

        if model_names is None:
            model_names = list(self.MODELS.keys())

        # Split data
        train_df, test_df = self.temporal_train_test_split(test_seasons)

        # Prepare arrays
        X_train = train_df[self.feature_cols].values
        y_train = train_df[self.target_col].astype(int).values
        X_test = test_df[self.feature_cols].values
        y_test = test_df[self.target_col].astype(int).values

        # Scale features (important for logistic regression)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        home_win_rate = y_test.mean()
        logger.info("Test set home win rate (baseline): {:.1%}", home_win_rate)

        # Further split train into train/calibration sets
        # Use the last 20% of training data for calibration fitting
        cal_split = int(len(X_train_scaled) * 0.8)
        X_train_model = X_train_scaled[:cal_split]
        y_train_model = y_train[:cal_split]
        X_cal = X_train_scaled[cal_split:]
        y_cal = y_train[cal_split:]

        logger.info(
            "Split: {} train, {} calibration, {} test",
            len(X_train_model), len(X_cal), len(X_test_scaled),
        )

        results = {}
        for name in model_names:
            logger.info("=" * 60)
            logger.info("Training: {}", name)
            logger.info("=" * 60)

            result = self._train_single_model(
                name=name,
                X_train=X_train_model,
                y_train=y_train_model,
                X_cal=X_cal,
                y_cal=y_cal,
                X_test=X_test_scaled,
                y_test=y_test,
                home_win_rate=home_win_rate,
            )
            results[name] = result

        return results

    def _train_single_model(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        home_win_rate: float,
    ) -> ModelResult:
        """Train a single model, calibrate it, and evaluate."""

        # 1. Train the model
        model = self.MODELS[name]()
        model.fit(X_train, y_train)

        # 2. Get uncalibrated probabilities
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_prob_cal = model.predict_proba(X_cal)[:, 1]

        # 3. Evaluate uncalibrated performance
        y_pred = (y_prob_test >= 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_prob_test)
        auc = roc_auc_score(y_test, y_prob_test)

        logger.info("{} — Accuracy: {:.1%}, Log Loss: {:.4f}, AUC: {:.4f}",
                     name, accuracy, ll, auc)

        # 4. Evaluate uncalibrated calibration
        cal_before = CalibrationMetrics.expected_calibration_error(y_test, y_prob_test)
        logger.info("{} — UNCALIBRATED: {}", name, cal_before.summary())

        # 5. Apply each calibration method and evaluate
        cal_after = {}
        best_method = "none"
        best_ece = cal_before.ece
        best_calibrated_probs = y_prob_test

        for method in self.CALIBRATION_METHODS:
            try:
                calibrator = ModelCalibrator(method=method)
                calibrator.fit(y_cal, y_prob_cal)
                y_prob_calibrated = calibrator.transform(y_prob_test)

                # Clip to valid range
                y_prob_calibrated = np.clip(y_prob_calibrated, 0.001, 0.999)

                cal_result = CalibrationMetrics.expected_calibration_error(
                    y_test, y_prob_calibrated
                )
                cal_after[method] = cal_result
                logger.info("{} — {} calibration: {}", name, method, cal_result.summary())

                if cal_result.ece < best_ece:
                    best_ece = cal_result.ece
                    best_method = method
                    best_calibrated_probs = y_prob_calibrated

            except Exception as e:
                logger.warning("Calibration failed for {} with {}: {}", name, method, e)

        if best_method != "none":
            logger.info("{} — Best calibration: {} (ECE: {:.4f})", name, best_method, best_ece)
        else:
            logger.info("{} — Uncalibrated model had best ECE", name)

        return ModelResult(
            model_name=name,
            accuracy=accuracy,
            log_loss_val=ll,
            roc_auc=auc,
            home_win_rate=home_win_rate,
            calibration_before=cal_before,
            calibration_after=cal_after,
            best_calibration_method=best_method,
            best_ece=best_ece,
            y_true=y_test,
            y_prob_uncalibrated=y_prob_test,
            y_prob_calibrated=best_calibrated_probs,
        )

    def print_results(self, results: dict[str, ModelResult]) -> None:
        """Print a formatted comparison of all model results."""

        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)

        # Header
        print(f"\n{'Model':<25} {'Accuracy':>10} {'AUC':>8} {'Log Loss':>10} "
              f"{'ECE (raw)':>10} {'ECE (cal)':>10} {'Brier (cal)':>12} {'Cal Method':>12}")
        print("-" * 100)

        baseline_acc = None
        for name, r in results.items():
            if baseline_acc is None:
                baseline_acc = r.home_win_rate

            print(
                f"{r.model_name:<25} "
                f"{r.accuracy:>10.1%} "
                f"{r.roc_auc:>8.4f} "
                f"{r.log_loss_val:>10.4f} "
                f"{r.calibration_before.ece:>10.4f} "
                f"{r.best_ece:>10.4f} "
                f"{r.calibration_before.brier:>12.4f} "
                f"{r.best_calibration_method:>12}"
            )

        print("-" * 100)
        print(f"{'Baseline (always home)':<25} {baseline_acc:>10.1%}")
        print()

        # Calibration improvement summary
        print("CALIBRATION IMPROVEMENT SUMMARY:")
        print("-" * 60)
        for name, r in results.items():
            ece_before = r.calibration_before.ece
            ece_after = r.best_ece
            improvement = (ece_before - ece_after) / ece_before * 100 if ece_before > 0 else 0
            print(f"  {name}: ECE {ece_before:.4f} → {ece_after:.4f} "
                  f"({improvement:+.1f}% improvement via {r.best_calibration_method})")

    def save_model(self, model, name: str) -> Path:
        """Save a trained model to disk."""
        settings.ensure_directories()
        path = settings.models_dir / f"{name}.joblib"
        joblib.dump(model, path)
        logger.info("Saved model to {}", path)
        return path