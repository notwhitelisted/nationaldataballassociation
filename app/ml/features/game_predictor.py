"""
Trains and evaluates models for predicting NBA game outcomes (home win/loss). The predictor looks to use 
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

from ctypes.util import test

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
from xgboost import XGBClassifier, train

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
    #results from training and evaluating a single model
    model_name: str
    accuracy: float
    log_loss_val: float
    roc_auc: float
    home_win_rate: float #baseline metric

    #calibration metrics 
    calibration_before: CalibrationResult #uncalibrated
    calibration_after: dict[str, CalibrationResult]  = field(default_factory=dict) #calibrated with different methods

    #best calibration method
    best_calibration_method: str = None
    best_ece: float = None

    #raw predictions and probabilities for further analysis
    y_true: np.ndarray = field(default=None, repr=False)
    y_prob_uncalibrated: np.ndarray = field(default=None, repr=False)
    y_prob_calibrated: np.ndarray = field(default=None, repr=False)

    class GamePredictor:
        #trains and evaluates game outcome prediction models
        """
        Main point for ML pipeline to train and evaluate game outcome prediction models. This class will:
        1. Load and split features temporally to avoid data leakage
        2. Train multiple models (baseline, logistic regression, random forest, xgboost)
        3. Apply and compare calibration methods
        4. Evaluate accuracry and calibration metrics 
        """

        MODELS = {
            "logistic_regression": lambda: LogisticRegression(
                C=1.0, max_iter=1000, random_state=42
            ),
            "random_forest": lambda: RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=20, random_state=42, n_jobs=-1,
            ),
            "xgboost": lambda: XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, random_state=42, eval_metric="logloss", verbosity=0,
            ),
        }

        CALIBRATION_METHODS = ["platt", "isotonic", "temperature"]
        
        def __init__(self):
            self.data_store = DataStore()
            self.scaler = StandardScaler() #for logistic regression
            self.features_df: pd.DataFrame | None = None
            self.features_cols: list[str] = []
            self.target.col = "home_win"

        def load_features(self, name: str = "game_features_all_seasons") -> None:
            #load saved feature matrix
            self.features_df = self.data_store.load(name)
            if self.features_df.empty:
                raise FileNotFoundError(f"No feature data found: {name}")
            
            #identify feature columns (exclude target and metadata)
            exclude = {"game_id", "game_date", "season", "home_win"}
            self.features_cols = [c for c in self.features_df.columns if c not in exclude]

            #sort by date to ensure temporal order for time-series cross-validation
            self.features_df = self.features_df.sort_values("game_date").reset_index(drop=True)

            #drop rows with any NaN features
            self.features_df = self.features_df.dropna(subset=self.features_cols)

            #drop rows with any Nan features
            before = len(self.features_df)
            self.features_df = self.features_df.dropna(subset=self.features_cols)
            after = len(self.features_df)
            if before != after:
                logger.info("Dropped {} rows with NaN features", before - after)
            
            logger.info("Loaded features: {} games, {} features, data range {} to {}",
                len(self.features_df), 
                len(self.features_cols), 
                self.features_df["game_date"].min(), 
                self.features_df["game_date"].max(),
            )
        
        def temporal_train_test_split(self, test_seasons: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
            """
            split data temporally - train on older seasons, test on recent ones

            args:
                test_seasons: list of seasons to use as test set (e.g. [2023, 2024]). defaults to most recent season in dataset

            returns:
                (train_df, test_df) tuples of dataframes for training and testing
            """
            if self.features_df is None:
                raise RuntimeError("Call load_features() first")
            
            if test_seasons is None:
            #use the most recent season as test
                all_seasons = sorted(self.features_df["season"].unique())
                test_seasons = [all_seasons[-1]]
            
            train = self.features_df[~self.features_df["season"].isin(test_seasons)]
            test = self.features_df[self.features_df["season"].isin(test_seasons)]

            logger.info("Train: {} games (seasons {}), Test: {} games (seasons {})", len(train), sorted(train["season"].unique()), len(test), sorted(test["season"].unique()),
            )
            return train, test