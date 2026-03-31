"""Odds data integration for realistic backtesting.

Loads historical NBA betting odds from the Kaggle dataset and merges them
with our existing game data. Converts point spreads to implied win
probabilities using the established logistic relationship.

Usage:
    from app.data.processors.odds_processor import OddsProcessor

    processor = OddsProcessor("path/to/nba_2008-2025.csv")
    games_with_odds = processor.merge_with_games(games_df)
"""

import numpy as np
import pandas as pd

from app.utils import logger


#Team abbreviation mapping 
#Kaggle dataset uses lowercase, sometimes different abbreviations.
#Map from Kaggle format → our BBRef format.

KAGGLE_TO_BBREF = {
    "atl": "ATL", "bos": "BOS", "bkn": "BRK", "cha": "CHO",
    "chi": "CHI", "cle": "CLE", "dal": "DAL", "den": "DEN",
    "det": "DET", "gs": "GSW", "hou": "HOU", "ind": "IND",
    "lac": "LAC", "lal": "LAL", "mem": "MEM", "mia": "MIA",
    "mil": "MIL", "min": "MIN", "no": "NOP", "ny": "NYK",
    "okc": "OKC", "orl": "ORL", "phi": "PHI", "phx": "PHO",
    "por": "POR", "sa": "SAS", "sac": "SAC", "tor": "TOR",
    "utah": "UTA", "wsh": "WAS",
}


class OddsProcessor:
    """Processes and merges betting odds data with game results."""

    def __init__(self, csv_path: str):
        """Load the Kaggle odds CSV.

        Args:
            csv_path: Path to nba_2008-2025.csv
        """
        self.raw_df = pd.read_csv(csv_path)
        self.raw_df["date"] = pd.to_datetime(self.raw_df["date"]).dt.date
        logger.info("Loaded {} games from odds dataset", len(self.raw_df))

    def get_odds_for_seasons(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """Get processed odds data for specified seasons.

        The Kaggle dataset uses 'season' as the ending year (e.g. 2025 = 2024-25 season).
        Our project uses the starting year (e.g. 2024 = 2024-25 season).
        So Kaggle season 2025 = our season 2024.

        Args:
            seasons: Our season format (start year). e.g. [2019, 2020, 2021, 2022, 2023, 2024]

        Returns:
            DataFrame with normalized team abbreviations and implied probabilities.
        """
        df = self.raw_df.copy()

        #convert Kaggle season (end year) to our season (start year)
        df["our_season"] = df["season"] - 1

        if seasons is not None:
            df = df[df["our_season"].isin(seasons)]

        #normalize team abbreviations
        df["home_abbr"] = df["home"].map(KAGGLE_TO_BBREF)
        df["away_abbr"] = df["away"].map(KAGGLE_TO_BBREF)

        #check for unmapped teams
        unmapped_home = df[df["home_abbr"].isnull()]["home"].unique()
        unmapped_away = df[df["away_abbr"].isnull()]["away"].unique()
        if len(unmapped_home) > 0 or len(unmapped_away) > 0:
            logger.warning("Unmapped teams: home={}, away={}", unmapped_home, unmapped_away)

        #convert spread to home team perspective
        #in the Kaggle data: spread is always positive, whos_favored tells direction
        #we want: negative spread = home favored, positive = home underdog
        df["home_spread"] = df.apply(self._compute_home_spread, axis=1)

        #convert spread to implied win probability
        df["implied_home_win_prob"] = df["home_spread"].apply(self.spread_to_probability)

        #convert moneylines to decimal odds (where available)
        df["home_decimal_odds"] = df["moneyline_home"].apply(self.american_to_decimal)
        df["away_decimal_odds"] = df["moneyline_away"].apply(self.american_to_decimal)

        #where moneylines are missing, derive decimal odds from spread-implied probability
        mask = df["home_decimal_odds"].isnull()
        df.loc[mask, "home_decimal_odds"] = 1.0 / df.loc[mask, "implied_home_win_prob"].clip(0.01, 0.99)
        #add ~4% vig to make odds realistic (books don't offer fair odds)
        df.loc[mask, "home_decimal_odds"] = df.loc[mask, "home_decimal_odds"] * 0.96

        df.loc[mask, "away_decimal_odds"] = 1.0 / (1 - df.loc[mask, "implied_home_win_prob"]).clip(0.01, 0.99)
        df.loc[mask, "away_decimal_odds"] = df.loc[mask, "away_decimal_odds"] * 0.96

        logger.info(
            "Processed odds for {} games across seasons {}",
            len(df),
            sorted(df["our_season"].unique()),
        )

        return df[[ 
            "date", "our_season", "home_abbr", "away_abbr",
            "spread", "total", "home_spread",
            "moneyline_home", "moneyline_away",
            "home_decimal_odds", "away_decimal_odds",
            "implied_home_win_prob", "whos_favored",
            "score_home", "score_away",
        ]].rename(columns={"our_season": "season"})

    def merge_with_games(self, games_df: pd.DataFrame, seasons: list[int] | None = None) -> pd.DataFrame:
        """Merge odds data into the existing games DataFrame.

        Matches on date + home team + away team.

        Args:
            games_df: Our games DataFrame from DataStore.
            seasons: Which seasons to include.

        Returns:
            games_df with added odds columns.
        """
        if seasons is None:
            seasons = sorted(games_df["season"].unique())

        odds = self.get_odds_for_seasons(seasons)

        #ensure date types match
        games = games_df.copy()
        games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
        odds["date"] = pd.to_datetime(odds["date"]).dt.date

        #merge on date + home + away
        merged = games.merge(
            odds,
            left_on=["game_date", "home_team_abbr", "away_team_abbr"],
            right_on=["date", "home_abbr", "away_abbr"],
            how="left",
            suffixes=("", "_odds"),
        )

        matched = merged["implied_home_win_prob"].notna().sum()
        total = len(merged)
        logger.info(
            "Merged odds: {}/{} games matched ({:.1%})",
            matched, total, matched / total if total > 0 else 0,
        )

        #clean up duplicate columns
        drop_cols = [c for c in merged.columns if c.endswith("_odds")]
        drop_cols += ["date", "home_abbr", "away_abbr"]
        merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

        return merged

    @staticmethod
    def _compute_home_spread(row) -> float:
        """Convert the Kaggle spread format to home team spread.

        Kaggle format: spread is always positive, whos_favored indicates direction.
        Our format: negative = home favored, positive = home underdog.
        """
        spread = row["spread"]
        if pd.isna(spread):
            return 0.0
        if row["whos_favored"] == "home":
            return -abs(spread)  #home favored = negative spread
        else:
            return abs(spread)  #away favored = positive spread for home

    @staticmethod
    def spread_to_probability(spread: float) -> float:
        """Convert a point spread to implied win probability.

        Uses the logistic function calibrated to historical NBA data.
        A spread of 0 = 50% win probability.
        Each point of spread ≈ 3-4% change in win probability.

        The coefficient 0.148 is derived from historical NBA data where
        a 1-point spread change corresponds to roughly 3.6% probability change.
        This is consistent with the academic literature (Stern 1991, Paul & Weinbach 2014).

        Args:
            spread: Home team spread (negative = favored).

        Returns:
            Implied probability of home team winning.
        """
        #logistic function: P(home win) = 1 / (1 + exp(k * spread))
        #k ≈ 0.148 calibrated to NBA historical data
        #negative spread (home favored) → probability > 0.5
        k = 0.148
        prob = 1.0 / (1.0 + np.exp(k * spread))
        return np.clip(prob, 0.01, 0.99)

    @staticmethod
    def american_to_decimal(american_odds) -> float | None:
        """Convert American odds to decimal odds.

        +150 → 2.50 (bet $100 to win $150)
        -200 → 1.50 (bet $200 to win $100)
        """
        if pd.isna(american_odds):
            return None
        if american_odds > 0:
            return american_odds / 100.0 + 1.0
        elif american_odds < 0:
            return 100.0 / abs(american_odds) + 1.0
        else:
            return 2.0  #even money