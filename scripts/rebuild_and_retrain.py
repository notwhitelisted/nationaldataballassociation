"""Rebuild features with 2025-26 season and retrain all models."""

import pandas as pd
from app.data.storage import DataStore
from app.ml.features.enhanced_features import EnhancedFeatureBuilder
from app.utils import logger


def main():
    store = DataStore()

    all_seasons = []
    for season in [2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        df = store.load_season_games(season)
        if not df.empty:
            all_seasons.append(df)
    games = pd.concat(all_seasons, ignore_index=True)
    logger.info("Total games: {}", len(games))

    builder = EnhancedFeatureBuilder(games, scrape_four_factors=False)
    features = builder.build()
    logger.info("Feature matrix: {}", features.shape)

    store.save_features(features, "enhanced_features_all_seasons")
    logger.info("Features saved!")


if __name__ == "__main__":
    main()