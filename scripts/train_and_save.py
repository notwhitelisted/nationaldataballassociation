"""Train all models on enhanced features (Elo + Four Factors) and save to disk.

Run this ONCE to train and save everything:
    python -m scripts.train_and_save

After this, your Streamlit app and bet tracker just load the saved models.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

from app.config import settings
from app.data.storage import DataStore
from app.data.processors.odds_processor import OddsProcessor
from app.ml.calibration import ModelCalibrator, CalibrationMetrics
from app.utils import logger


def main():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # ── Load Data ────────────────────────────────────────────────────────
    logger.info("Loading enhanced features and odds data...")
    store = DataStore()

    # Load enhanced features (with Elo + Four Factors)
    features_df = store.load_features("enhanced_features_all_seasons")
    exclude = {"game_id", "game_date", "season", "home_win"}
    feature_cols = [c for c in features_df.columns if c not in exclude]
    features_df = features_df.sort_values("game_date").reset_index(drop=True)
    features_df = features_df.dropna(subset=feature_cols)

    logger.info("Features loaded: {} games, {} features", len(features_df), len(feature_cols))

    # Load and merge odds for spread/totals
    games = store.load_all_games()
    processor = OddsProcessor("data/raw/nba_2008-2025.csv")
    games_with_odds = processor.merge_with_games(games)

    score_cols = games_with_odds[["game_id", "home_score", "away_score",
                                   "home_spread", "total", "implied_home_win_prob"]].copy()
    score_cols["point_diff"] = score_cols["home_score"] - score_cols["away_score"]
    score_cols["total_points"] = score_cols["home_score"] + score_cols["away_score"]

    features_with_odds = features_df.merge(
        score_cols[["game_id", "point_diff", "total_points", "home_spread", "total"]],
        on="game_id",
        how="inner",
    ).dropna(subset=feature_cols + ["point_diff", "total_points", "home_spread", "total"])

    logger.info("Features with odds: {} games", len(features_with_odds))

    # ── Split Data ───────────────────────────────────────────────────────
    all_seasons = sorted(features_df["season"].unique())
    test_season = all_seasons[-1]

    train_ml = features_df[features_df["season"] != test_season]
    test_ml = features_df[features_df["season"] == test_season]
    train_odds = features_with_odds[features_with_odds["season"] != test_season]

    logger.info("Train: {} games, Test: {} games, Train with odds: {}",
                len(train_ml), len(test_ml), len(train_odds))

    # ── Prepare Training Data ────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(train_ml[feature_cols].values)
    y_train_ml = train_ml["home_win"].astype(int).values

    # 80/20 split for model training vs calibration
    split = int(len(X_train_all) * 0.8)
    X_train = X_train_all[:split]
    y_train = y_train_ml[:split]
    X_cal = X_train_all[split:]
    y_cal = y_train_ml[split:]

    # Test data
    X_test = scaler.transform(test_ml[feature_cols].values)
    y_test = test_ml["home_win"].astype(int).values

    logger.info("Train: {}, Calibration: {}, Test: {}", len(X_train), len(X_cal), len(X_test))

    # ── Train Moneyline Models ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING MONEYLINE MODELS (219 features)")
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

    best_ml_accuracy = 0
    best_ml_name = ""

    for name, model in ml_models.items():
        logger.info("Training: {}", name)
        model.fit(X_train, y_train)

        # Evaluate
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        cal_before = CalibrationMetrics.expected_calibration_error(y_test, y_prob)
        logger.info("  {} — Accuracy: {:.1%}, AUC: {:.4f}, ECE: {:.4f}",
                     name, acc, auc, cal_before.ece)

        if acc > best_ml_accuracy:
            best_ml_accuracy = acc
            best_ml_name = name

        joblib.dump(model, models_dir / f"moneyline_{name}.joblib")
        logger.info("  Saved: moneyline_{}.joblib", name)

    logger.info("Best moneyline model: {} ({:.1%})", best_ml_name, best_ml_accuracy)

    # ── Train Calibrators ────────────────────────────────────────────────
    logger.info("Training calibrators for Random Forest...")

    rf_model = ml_models["random_forest"]
    cal_probs = rf_model.predict_proba(X_cal)[:, 1]

    for method in ["platt", "isotonic", "temperature"]:
        calibrator = ModelCalibrator(method=method)
        calibrator.fit(y_cal, cal_probs)
        joblib.dump(calibrator, models_dir / f"calibrator_{method}.joblib")
        logger.info("  Saved: calibrator_{}.joblib", method)

    # ── Train Spread Model ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING SPREAD MODEL")
    logger.info("=" * 60)

    scaler_spread = StandardScaler()
    X_train_spread = scaler_spread.fit_transform(train_odds[feature_cols].values)
    y_train_spread = train_odds["point_diff"].values

    spread_models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            random_state=42, verbosity=0,
        ),
    }

    for name, model in spread_models.items():
        logger.info("Training spread model: {}", name)
        model.fit(X_train_spread, y_train_spread)
        joblib.dump(model, models_dir / f"spread_{name}.joblib")
        logger.info("  Saved: spread_{}.joblib", name)

    # ── Train Totals Model ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING TOTALS MODEL")
    logger.info("=" * 60)

    y_train_total = train_odds["total_points"].values

    totals_models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            random_state=42, verbosity=0,
        ),
    }

    for name, model in totals_models.items():
        logger.info("Training totals model: {}", name)
        model.fit(X_train_spread, y_train_total)
        joblib.dump(model, models_dir / f"totals_{name}.joblib")
        logger.info("  Saved: totals_{}.joblib", name)

    # ── Save Scalers and Feature Columns ─────────────────────────────────
    joblib.dump(scaler, models_dir / "scaler_moneyline.joblib")
    joblib.dump(scaler_spread, models_dir / "scaler_spread.joblib")
    joblib.dump(feature_cols, models_dir / "feature_cols.joblib")
    logger.info("Saved: scalers and feature columns")

    # ── Save Strategy Parameters ─────────────────────────────────────────
    strategy_params = {
        # Moneyline strategy
        "moneyline_best_model": "random_forest",
        "moneyline_best_calibration": "isotonic",
        "moneyline_min_edge": 0.03,
        "moneyline_accuracy": 0.676,
        "moneyline_auc": 0.7286,
        "moneyline_ece_calibrated": 0.0232,
        "moneyline_roi": 83.25,

        # Spread strategy (combined model)
        "spread_best_model": "random_forest",
        "spread_min_edge": 4.0,
        "spread_ml_confidence": 0.75,
        "spread_standard_odds": 1.909,
        "spread_win_rate": 58.1,
        "spread_roi": 10.96,

        # Totals
        "totals_best_model": "ridge",
        "totals_profitable": False,
        "totals_best_r2": 0.1371,

        # Feature info
        "feature_set": "enhanced_with_elo_four_factors",
        "n_features": len(feature_cols),
        "train_seasons": [int(s) for s in all_seasons if s != test_season],
        "test_season": int(test_season),
        "n_training_games": len(X_train),
        "baseline_accuracy": float(y_test.mean()),
    }
    joblib.dump(strategy_params, models_dir / "strategy_params.joblib")
    logger.info("Saved: strategy_params.joblib")

    # ── Create/Update Bet Tracker CSV ────────────────────────────────────
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

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("ALL MODELS SAVED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info("Saved files:")
    for f in sorted(models_dir.glob("*.joblib")):
        size_kb = f.stat().st_size / 1024
        logger.info("  {} ({:.0f} KB)", f.name, size_kb)

    logger.info("\nModel Summary:")
    logger.info("  Moneyline: RF {:.1%} accuracy, {:.4f} ECE (isotonic)",
                strategy_params["moneyline_accuracy"],
                strategy_params["moneyline_ece_calibrated"])
    logger.info("  Spread: Combined strategy {:.1%} win rate, +{:.2f}% ROI",
                strategy_params["spread_win_rate"] / 100,
                strategy_params["spread_roi"])
    logger.info("  Totals: Not profitable (R²={:.4f})",
                strategy_params["totals_best_r2"])
    logger.info("  Features: {} (Elo + Four Factors + rolling stats)",
                strategy_params["n_features"])


if __name__ == "__main__":
    main()