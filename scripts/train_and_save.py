"""Train all models and save to disk for reuse.

Run this ONCE to train and save everything:
    python -m scripts.train_and_save

After this, your Streamlit app and bet tracker just load the saved models.
Models are saved to the models/ directory.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from app.config import settings
from app.data.storage import DataStore
from app.data.processors.odds_processor import OddsProcessor
from app.ml.calibration import ModelCalibrator, CalibrationMetrics
from app.ml.models.game_predictor import GamePredictor
from app.utils import logger


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # ── Load Data ────────────────────────────────────────────────────────
    logger.info("Loading features and odds data...")
    store = DataStore()

    # Load features
    features_df = store.load_features("game_features_all_seasons")
    exclude = {"game_id", "game_date", "season", "home_win"}
    feature_cols = [c for c in features_df.columns if c not in exclude]
    features_df = features_df.sort_values("game_date").reset_index(drop=True)
    features_df = features_df.dropna(subset=feature_cols)

    # Load and merge odds
    games = store.load_all_games()
    processor = OddsProcessor("data/raw/nba_2008-2025.csv")
    games_with_odds = processor.merge_with_games(games)

    # Merge scores and odds into features for spread/totals
    score_cols = games_with_odds[["game_id", "home_score", "away_score",
                                   "home_spread", "total", "implied_home_win_prob"]].copy()
    score_cols["point_diff"] = score_cols["home_score"] - score_cols["away_score"]
    score_cols["total_points"] = score_cols["home_score"] + score_cols["away_score"]

    features_with_odds = features_df.merge(
        score_cols[["game_id", "point_diff", "total_points", "home_spread", "total"]],
        on="game_id",
        how="inner",
    ).dropna(subset=feature_cols + ["point_diff", "total_points", "home_spread", "total"])

    logger.info("Features: {} games, Features with odds: {} games",
                len(features_df), len(features_with_odds))

    # ── Split Data ───────────────────────────────────────────────────────
    # Train on all seasons except the most recent for backtesting
    all_seasons = sorted(features_df["season"].unique())
    test_season = all_seasons[-1]

    train_df = features_df[features_df["season"] != test_season]
    train_odds_df = features_with_odds[features_with_odds["season"] != test_season]

    logger.info("Training on seasons {}, test season: {}",
                [s for s in all_seasons if s != test_season], test_season)

    # ── Prepare Training Data ────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(train_df[feature_cols].values)
    y_train_ml = train_df["home_win"].astype(int).values

    # Split for model training vs calibration (80/20)
    split = int(len(X_train_all) * 0.8)
    X_train = X_train_all[:split]
    y_train = y_train_ml[:split]
    X_cal = X_train_all[split:]
    y_cal = y_train_ml[split:]

    logger.info("Train: {}, Calibration: {}", len(X_train), len(X_cal))

    # ── Train Moneyline Models ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING MONEYLINE MODELS")
    logger.info("=" * 60)

    ml_models = {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }

    for name, model in ml_models.items():
        logger.info("Training moneyline model: {}", name)
        model.fit(X_train, y_train)

        # Evaluate
        y_prob = model.predict_proba(X_cal)[:, 1]
        cal_before = CalibrationMetrics.expected_calibration_error(y_cal, y_prob)
        logger.info("  {} uncalibrated ECE: {:.4f}", name, cal_before.ece)

        # Save model
        joblib.dump(model, models_dir / f"moneyline_{name}.joblib")
        logger.info("  Saved: moneyline_{}.joblib", name)

    # ── Train Calibrator (Platt scaling on best model — Random Forest) ──
    logger.info("Training Platt scaling calibrator...")
    rf_model = ml_models["random_forest"]
    cal_probs = rf_model.predict_proba(X_cal)[:, 1]

    calibrator = ModelCalibrator(method="platt")
    calibrator.fit(y_cal, cal_probs)

    joblib.dump(calibrator, models_dir / "calibrator_platt.joblib")
    logger.info("Saved: calibrator_platt.joblib")

    # ── Train Spread Model ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING SPREAD MODEL")
    logger.info("=" * 60)

    # Use the odds-merged data for spread training
    scaler_spread = StandardScaler()
    X_train_spread = scaler_spread.fit_transform(train_odds_df[feature_cols].values)
    y_train_spread = train_odds_df["point_diff"].values

    spread_model = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    spread_model.fit(X_train_spread, y_train_spread)

    joblib.dump(spread_model, models_dir / "spread_random_forest.joblib")
    logger.info("Saved: spread_random_forest.joblib")

    # ── Train Totals Model ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING TOTALS MODEL")
    logger.info("=" * 60)

    y_train_total = train_odds_df["total_points"].values

    totals_model = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    totals_model.fit(X_train_spread, y_train_total)

    joblib.dump(totals_model, models_dir / "totals_random_forest.joblib")
    logger.info("Saved: totals_random_forest.joblib")

    # ── Save Scaler and Feature Columns ──────────────────────────────────
    joblib.dump(scaler, models_dir / "scaler_moneyline.joblib")
    joblib.dump(scaler_spread, models_dir / "scaler_spread.joblib")
    joblib.dump(feature_cols, models_dir / "feature_cols.joblib")
    logger.info("Saved: scalers and feature columns")

    # ── Save Strategy Parameters ─────────────────────────────────────────
    strategy_params = {
        # Moneyline strategy
        "moneyline_model": "random_forest",
        "moneyline_calibration": "platt",
        "moneyline_min_edge": 0.03,
        "moneyline_bet_size": 10.0,

        # Spread strategy (combined model)
        "spread_min_edge": 4.0,
        "spread_ml_confidence": 0.68,
        "spread_bet_size": 10.0,
        "spread_standard_odds": 1.909,  # -110

        # Totals: NOT RECOMMENDED (documented finding)
        "totals_profitable": False,

        # Training info
        "train_seasons": [s for s in all_seasons if s != test_season],
        "test_season": int(test_season),
        "n_features": len(feature_cols),
        "n_training_games": len(X_train),
    }
    joblib.dump(strategy_params, models_dir / "strategy_params.joblib")
    logger.info("Saved: strategy_params.joblib")

    # ── Create Bet Tracker CSV ───────────────────────────────────────────
    tracker_path = Path("data/bet_tracker.csv")
    if not tracker_path.exists():
        tracker_df = pd.DataFrame(columns=[
            "date", "home_team", "away_team", "bet_type",
            "bet_side", "model_probability", "market_odds",
            "edge", "bet_amount", "result", "profit_loss",
            "cumulative_profit", "notes",
        ])
        tracker_df.to_csv(tracker_path, index=False)
        logger.info("Created bet tracker: {}", tracker_path)
    else:
        logger.info("Bet tracker already exists: {}", tracker_path)

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ALL MODELS SAVED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("Saved files:")
    for f in sorted(models_dir.glob("*.joblib")):
        logger.info("  {}", f.name)

    logger.info("\nTo use in Streamlit or scripts:")
    logger.info("  model = joblib.load('models/moneyline_random_forest.joblib')")
    logger.info("  scaler = joblib.load('models/scaler_moneyline.joblib')")
    logger.info("  calibrator = joblib.load('models/calibrator_platt.joblib')")


if __name__ == "__main__":
    main()
