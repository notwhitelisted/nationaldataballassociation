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
    #results from training and evaluating a single model
    model_name: str
    accuracy: float
    log_loss_val: float
    roc_auc: float
    home_win_rate: float #baseline metric

    #calibration metrics 
    calibration_before: CalibrationResult #uncalibrated
    calibration_after: dict[str, CalibrationResult]  = field(default_factory=dict) #calibrated with different methods