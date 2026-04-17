# %% [markdown]
# # NBA Data Exploration
#
# This notebook explores the collected NBA data and validates the data pipeline.
# Run `python -m scripts.collect_seasons --seasons 2024 --games-only` first.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from app.data.storage import DataStore
from app.config import settings

sns.set_theme(style="whitegrid")
store = DataStore()

# %% [markdown]
# ## 1. Load and Inspect Game Data

# %%
games = store.load_all_games()
print(f"Total games: {len(games)}")
print(f"Seasons: {games['season'].unique()}")
print(f"Date range: {games['game_date'].min()} to {games['game_date'].max()}")
games.head()

# %%
# Home win percentage (should be ~58-60% historically)
home_win_pct = games["home_win"].mean()
print(f"Home win %: {home_win_pct:.1%}")

# By season
games.groupby("season")["home_win"].mean().plot(kind="bar", title="Home Win % by Season")
plt.ylabel("Home Win %")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Score Distributions

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
games["home_score"].hist(bins=30, ax=axes[0], alpha=0.7, label="Home")
games["away_score"].hist(bins=30, ax=axes[0], alpha=0.7, label="Away")
axes[0].set_title("Score Distribution")
axes[0].legend()

margin = games["home_score"] - games["away_score"]
margin.hist(bins=40, ax=axes[1], alpha=0.7)
axes[1].set_title("Home Margin Distribution")
axes[1].axvline(0, color="red", linestyle="--")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Feature Engineering Preview

# %%
from app.ml.features.game_features import GameFeatureBuilder

builder = GameFeatureBuilder(games)
features = builder.build()
print(f"Feature matrix shape: {features.shape}")
print(f"\nFeature columns:")
for col in sorted(features.columns):
    if col not in ["game_id", "game_date", "season", "home_win"]:
        print(f"  {col}")

# %%
# Correlation with target
feature_cols = [c for c in features.columns if c not in ["game_id", "game_date", "season", "home_win"]]
correlations = features[feature_cols].corrwith(features["home_win"]).sort_values(ascending=False)
print("Top features correlated with home_win:")
print(correlations.head(15))
print("\nBottom features:")
print(correlations.tail(5))
