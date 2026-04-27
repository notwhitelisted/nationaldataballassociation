"""Fetch today's NBA odds and generate predictions for all games.

Usage:
    python -m scripts.predict_today

Pulls live odds from The Odds API, builds features for each game
using recent team performance data, runs predictions through saved
models, and displays bet recommendations.
"""

import pandas as pd
import numpy as np

from app.data.storage import DataStore
from app.ml.features.enhanced_features import EnhancedFeatureBuilder
from app.data.collectors.odds_api_collector import OddsAPICollector, LivePredictor
from app.utils import logger
from pathlib import Path


# Your API key — you can also move this to .env later
ODDS_API_KEY = "f08a75c6c952195ea0fb6badab1e631b"


def main():
    # ── Load game data and build feature histories ───────────────────
    logger.info("Loading game data and building feature histories...")
    store = DataStore()
    import pandas as pd
    all_seasons = []
    for season in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        df = store.load_season_games(season)
        if not df.empty:
            all_seasons.append(df)
    all_games = pd.concat(all_seasons, ignore_index=True)
    logger.info("Loaded {} total games across {} seasons", len(all_games), len(all_seasons))

    # Build features with Elo (skip four factors scraping for speed)
    builder = EnhancedFeatureBuilder(all_games, scrape_four_factors=False)
    # Load injury data and player impact
    from app.data.collectors.injury_collector import InjuryCollector
    injury_collector = InjuryCollector()
    
    impact_path = Path("data/processed/player_impact_2025.parquet")
    if impact_path.exists():
        impact_df = pd.read_parquet(impact_path)
        logger.info("Loaded player impact data: {} players", len(impact_df))
    else:
        impact_df = pd.DataFrame()
        logger.warning("No player impact data found — run: python -m scripts.player_impact --seasons 2025 --analyze")

    # ── Load saved models ────────────────────────────────────────────
    predictor = LivePredictor()

    # ── Fetch live odds ──────────────────────────────────────────────
    collector = OddsAPICollector(api_key=ODDS_API_KEY)
    games = collector.get_current_odds()

    if not games:
        print("\nNo upcoming NBA games found.")
        return

    # ── Build features and predict each game ─────────────────────────
    predictions = []
    for game in games:
        # Find team IDs from our historical data
        home_matches = all_games[all_games["home_team_abbr"] == game.home_abbr]
        away_matches = all_games[all_games["away_team_abbr"] == game.away_abbr]

        if len(home_matches) == 0 or len(away_matches) == 0:
            logger.warning("Skipping {} @ {} — team not found in data",
                          game.away_abbr, game.home_abbr)
            continue

        home_id = home_matches.iloc[-1]["home_team_id"]
        away_id = away_matches.iloc[-1]["away_team_id"]

        # Use the latest date in our data so all history is available
        latest_date = all_games["game_date"].max()

        # Build a game row for the feature builder
        fake_game = pd.Series({
            "game_id": game.game_id,
            "game_date": latest_date,
            "season": 2025,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_abbr": game.home_abbr,
            "away_team_abbr": game.away_abbr,
            "home_score": 0,
            "away_score": 0,
            "home_win": False,
        })

        # Compute features using team histories
        features = builder._build_game_features(fake_game)

        if features is None:
            logger.warning("Skipping {} @ {} — not enough team history",
                          game.away_abbr, game.home_abbr)
            continue

        # Generate prediction
        # Generate prediction
        prediction = predictor.predict_game(game, features)

        # Apply injury adjustments
        adj = injury_collector.get_game_adjustment(game.home_abbr, game.away_abbr, impact_df)
        prediction["injury_adjustment"] = adj
        
        # Adjust calibrated probability
        original_prob = prediction["home_win_prob_calibrated"]
        adjusted_prob = max(0.05, min(0.95, original_prob + adj["prob_adjustment"]))
        prediction["home_win_prob_adjusted"] = round(adjusted_prob, 4)
        
        # Adjust predicted margin
        original_margin = prediction["predicted_margin"]
        prediction["predicted_margin_adjusted"] = round(original_margin + adj["spread_adjustment"], 1)
        # Recalculate spread recommendation with adjusted margin
        if prediction["book_spread"] is not None:
            adjusted_spread_edge = prediction["predicted_margin_adjusted"] - prediction["book_spread"]
            prediction["spread_edge"] = round(adjusted_spread_edge, 1)
            min_spread_edge = 3.0
            
            if adjusted_spread_edge >= min_spread_edge:
                prediction["spread_recommendation"] = f"Bet {game.home_abbr} {prediction['book_spread']:+.1f}"
            elif adjusted_spread_edge <= -min_spread_edge:
                away_spread = -prediction["book_spread"]
                prediction["spread_recommendation"] = f"Bet {game.away_abbr} {away_spread:+.1f}"
            else:
                prediction["spread_recommendation"] = "No bet"
        
        predictions.append(prediction)

    # ── Display results ──────────────────────────────────────────────
    if predictions:
        predictor.print_predictions(predictions)

        # Summary of recommended bets
        ml_bets = [p for p in predictions if "Bet" in p["ml_recommendation"]]
        spread_bets = [p for p in predictions if "Bet" in p["spread_recommendation"]]

        print(f"\n{'='*75}")
        print(f"  BET RECOMMENDATIONS SUMMARY")
        print(f"{'='*75}")

        if ml_bets:
            print(f"\n  Moneyline bets ({len(ml_bets)}):")
            for p in ml_bets:
                print(f"    {p['ml_recommendation']} — Edge: {p['ml_edge']*100:+.1f}% "
                      f"({p['away_abbr']} @ {p['home_abbr']})")
        else:
            print("\n  No moneyline bets recommended (no sufficient edge found)")

        if spread_bets:
            print(f"\n  Spread bets ({len(spread_bets)}):")
            for p in spread_bets:
                print(f"    {p['spread_recommendation']} — Edge: {p['spread_edge']:+.1f} pts "
                      f"({p['away_abbr']} @ {p['home_abbr']})")
        else:
            print("\n  No spread bets recommended (combined strategy filter not met)")

        print(f"\n  Totals: Not recommended (market too efficient)")
        print()
    else:
        print("\nCould not generate predictions — check team data availability.")


if __name__ == "__main__":
    main()