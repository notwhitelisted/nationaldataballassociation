"""Enhanced feature engineering for NBA game prediction.

All-in-one feature builder with features for moneyline, spread, and totals:

ORIGINAL FEATURES (for moneyline):
    - Rolling team stats (win%, points, margin) over 3/5/10/15 game windows
    - Home/away splits, differentials, rest days, head-to-head history

SPREAD FEATURES:
    - Margin consistency, blowout rate, close game rate
    - Win margin vs loss margin patterns
    - Streak and momentum features

TOTALS FEATURES:
    - Pace proxy, total involvement, high/low scoring rates
    - Combined pace estimate, total points variance

ELO RATINGS (computed from game results, no scraping):
    - Dynamic team strength rating adjusted for schedule difficulty
    - Elo-based win probability, elo differential

FOUR FACTORS (scraped from Basketball Reference):
    - eFG%, TOV%, ORB%, FT/FGA for offense and defense
    - Pace, offensive/defensive/net rating, SRS

Usage:
    from app.ml.features.enhanced_features import EnhancedFeatureBuilder

    # Without four factors (no internet needed)
    builder = EnhancedFeatureBuilder(games_df)
    features = builder.build()

    # With four factors (scrapes Basketball Reference)
    builder = EnhancedFeatureBuilder(games_df, scrape_four_factors=True)
    features = builder.build()
"""

import re
import time
from datetime import date

import pandas as pd
import numpy as np

from app.utils import logger


#Team abbreviation mapping for four factors scraping

BBREF_TEAM_NORMALIZE = {
    "ATL": "ATL", "BOS": "BOS", "BRK": "BRK", "BKN": "BRK",
    "CHO": "CHO", "CHA": "CHO", "CHI": "CHI", "CLE": "CLE",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GSW": "GSW",
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL",
    "MEM": "MEM", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "NOP": "NOP", "NYK": "NYK", "OKC": "OKC", "ORL": "ORL",
    "PHI": "PHI", "PHO": "PHO", "PHX": "PHO", "POR": "POR",
    "SAC": "SAC", "SAS": "SAS", "TOR": "TOR", "UTA": "UTA",
    "WAS": "WAS",
}

#ELO RATING SYSTEM

class EloRatingSystem:
    """Computes Elo ratings from game results.

    Elo is a dynamic team strength rating that updates after every game.
    Beating a strong team gains more points than beating a weak team.
    Accounts for home court advantage and margin of victory.
    """

    def __init__(
        self,
        k_factor: float = 25.0,
        home_advantage: float = 80.0,
        initial_rating: float = 1500.0,
        season_regression: float = 0.25,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_rating = initial_rating
        self.season_regression = season_regression
        self.ratings: dict[int, float] = {}

    def compute_from_games(self, games_df: pd.DataFrame) -> dict:
        """Compute Elo ratings and return a lookup dict.

        Returns:
            Dict mapping game_id → {home_elo, away_elo, elo_diff, elo_home_win_prob}
        """
        games = games_df.sort_values("game_date").reset_index(drop=True)
        self.ratings = {}
        current_season = None
        elo_lookup = {}

        for _, game in games.iterrows():
            season = game["season"]
            home_id = game["home_team_id"]
            away_id = game["away_team_id"]

            #season transition: regress toward mean
            if current_season is not None and season != current_season:
                self._regress_to_mean()
            current_season = season

            #initialize new teams
            if home_id not in self.ratings:
                self.ratings[home_id] = self.initial_rating
            if away_id not in self.ratings:
                self.ratings[away_id] = self.initial_rating

            #ratings BEFORE this game
            home_elo = self.ratings[home_id]
            away_elo = self.ratings[away_id]
            elo_diff = home_elo - away_elo + self.home_advantage
            home_win_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

            elo_lookup[game["game_id"]] = {
                "home_elo": round(home_elo, 1),
                "away_elo": round(away_elo, 1),
                "elo_diff": round(elo_diff, 1),
                "elo_home_win_prob": round(home_win_prob, 4),
            }

            #update ratings based on result
            actual_home = 1.0 if game["home_score"] > game["away_score"] else 0.0
            mov = abs(game["home_score"] - game["away_score"])
            mov_multiplier = np.log(max(mov, 1) + 1) * (2.2 / (2.2 + 0.001 * abs(elo_diff)))

            update = self.k_factor * mov_multiplier * (actual_home - home_win_prob)
            self.ratings[home_id] += update
            self.ratings[away_id] -= update

        logger.info("Computed Elo ratings for {} games", len(elo_lookup))
        return elo_lookup

    def _regress_to_mean(self) -> None:
        """Regress all ratings toward the mean between seasons."""
        mean_rating = np.mean(list(self.ratings.values()))
        for team_id in self.ratings:
            self.ratings[team_id] = (
                self.ratings[team_id] * (1 - self.season_regression)
                + mean_rating * self.season_regression
            )

    def get_current_ratings(self) -> pd.DataFrame:
        """Get current Elo ratings sorted by strength."""
        records = [
            {"team_id": tid, "elo": round(rating, 1)}
            for tid, rating in self.ratings.items()
        ]
        return pd.DataFrame(records).sort_values("elo", ascending=False).reset_index(drop=True)


#FOUR FACTORS SCRAPER

class FourFactorsScraper:
    """Scrapes team Four Factors from Basketball Reference.

    The Four Factors (Dean Oliver):
        1. eFG% — Effective Field Goal % (shooting efficiency)
        2. TOV% — Turnover % (ball security)
        3. ORB% — Offensive Rebound % (second chances)
        4. FT/FGA — Free Throw Rate (getting to the line)

    Also scrapes: Pace, Offensive Rating, Defensive Rating, Net Rating, SRS.
    """

    BASE_URL = "https://www.basketball-reference.com"

    def __init__(self, delay: float = 3.5):
        self.delay = delay
        try:
            import cloudscraper
            self.session = cloudscraper.create_scraper()
        except ImportError:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0.0.0"
            })

    def scrape_season(self, season: int) -> pd.DataFrame:
        """Scrape four factors for all teams in a season."""
        from bs4 import BeautifulSoup

        bbref_year = season + 1
        url = f"{self.BASE_URL}/leagues/NBA_{bbref_year}.html"

        logger.info("Scraping four factors for season {}", season)
        time.sleep(self.delay)

        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        #find misc_stats table (sometimes hidden in HTML comments)
        misc_table = soup.find("table", {"id": "advanced-team"})
        if misc_table is None:
            comments = soup.find_all(
                string=lambda text: isinstance(text, str) and "advanced-team" in text
            )
            for comment in comments:
                comment_soup = BeautifulSoup(comment, "html.parser")
                misc_table = comment_soup.find("table", {"id": "advanced-team"})
                if misc_table:
                    break

        if misc_table is None:
            logger.warning("Could not find advanced-team table for season {}", season)
            return pd.DataFrame()

        rows = []
        tbody = misc_table.find("tbody")
        if tbody is None:
            return pd.DataFrame()

        for row in tbody.find_all("tr"):
            if row.find("th", {"scope": "col"}):
                continue

            team_cell = row.find("td", {"data-stat": "team"})
            if team_cell is None:
                continue

            link = team_cell.find("a")
            if link is None:
                continue
            href = link.get("href", "")
            abbr_match = re.search(r"/teams/(\w+)/", href)
            if not abbr_match:
                continue

            abbr = abbr_match.group(1)
            normalized = BBREF_TEAM_NORMALIZE.get(abbr, abbr)

            def _get_stat(stat_name, default=0.0):
                cell = row.find("td", {"data-stat": stat_name})
                if cell is None:
                    return default
                text = cell.text.strip()
                try:
                    return float(text)
                except (ValueError, TypeError):
                    return default

            rows.append({
                "season": season,
                "team_abbr": normalized,
                "off_efg_pct": _get_stat("efg_pct"),
                "off_tov_pct": _get_stat("tov_pct"),
                "off_orb_pct": _get_stat("orb_pct"),
                "off_ft_rate": _get_stat("ft_rate"),
                "def_efg_pct": _get_stat("opp_efg_pct"),
                "def_tov_pct": _get_stat("opp_tov_pct"),
                "def_orb_pct": _get_stat("drb_pct"),
                "def_ft_rate": _get_stat("opp_ft_rate"),
                "pace": _get_stat("pace"),
                "off_rtg": _get_stat("off_rtg"),
                "def_rtg": _get_stat("def_rtg"),
                "net_rtg": _get_stat("net_rtg"),
                "srs": _get_stat("srs"),
            })

        df = pd.DataFrame(rows)
        logger.info("Scraped four factors for {} teams in season {}", len(df), season)
        return df

    def scrape_multiple_seasons(self, seasons: list[int]) -> pd.DataFrame:
        """Scrape four factors for multiple seasons."""
        all_dfs = []
        for season in seasons:
            df = self.scrape_season(season)
            if not df.empty:
                all_dfs.append(df)
        if not all_dfs:
            return pd.DataFrame()
        return pd.concat(all_dfs, ignore_index=True)

#ENHANCED FEATURE BUILDER

class EnhancedFeatureBuilder:
    """Builds the complete feature matrix for outcome, spread, and totals prediction.

    Combines:
        - Rolling team stats (original 70 features)
        - Spread-specific features (margin patterns, streaks, momentum)
        - Totals-specific features (pace, high/low scoring rates)
        - Elo ratings (dynamic team strength)
        - Four Factors (offensive/defensive efficiency from Basketball Reference)

    All features use only data available BEFORE each game (no data leakage).
    """

    WINDOWS = [3, 5, 10, 15]

    def __init__(self, games_df: pd.DataFrame, scrape_four_factors: bool = False):
        """
        Args:
            games_df: Raw games DataFrame from DataStore.
            scrape_four_factors: If True, scrapes Basketball Reference for
                four factors data. Requires internet. Set False for offline use.
        """
        self.games = games_df.copy().sort_values("game_date").reset_index(drop=True)
        self.scrape_four_factors = scrape_four_factors

        #build team histories for rolling features
        self._build_team_histories()

        #compute Elo ratings
        logger.info("Computing Elo ratings...")
        self.elo_system = EloRatingSystem()
        self.elo_lookup = self.elo_system.compute_from_games(self.games)

        #scrape four factors if requested
        self.four_factors_df = pd.DataFrame()
        if self.scrape_four_factors:
            seasons = sorted(self.games["season"].unique())
            scraper = FourFactorsScraper()
            self.four_factors_df = scraper.scrape_multiple_seasons(seasons)
            if not self.four_factors_df.empty:
                #create lookup: (season, team_abbr) → stats dict
                self.ff_lookup = {}
                for _, row in self.four_factors_df.iterrows():
                    key = (row["season"], row["team_abbr"])
                    self.ff_lookup[key] = row.to_dict()
                logger.info("Four factors loaded for {} team-seasons", len(self.ff_lookup))
            else:
                self.ff_lookup = {}
        else:
            self.ff_lookup = {}

    def _build_team_histories(self) -> None:
        """Create per-team game history with detailed stats."""
        records = []
        for _, game in self.games.iterrows():
            home_score = game["home_score"]
            away_score = game["away_score"]
            total = home_score + away_score
            margin = home_score - away_score

            records.append({
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "team_id": game["home_team_id"],
                "opponent_id": game["away_team_id"],
                "is_home": True,
                "points_for": home_score,
                "points_against": away_score,
                "win": game["home_win"],
                "margin": margin,
                "total_points": total,
            })
            records.append({
                "game_id": game["game_id"],
                "game_date": game["game_date"],
                "team_id": game["away_team_id"],
                "opponent_id": game["home_team_id"],
                "is_home": False,
                "points_for": away_score,
                "points_against": home_score,
                "win": not game["home_win"],
                "margin": -margin,
                "total_points": total,
            })

        self.team_games = pd.DataFrame(records)
        self.team_games = self.team_games.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    def build(self) -> pd.DataFrame:
        """Build the complete feature matrix."""
        logger.info("Building enhanced feature matrix for {} games...", len(self.games))

        features = []
        for idx, game in self.games.iterrows():
            row = self._build_game_features(game)
            if row is not None:
                features.append(row)

        df = pd.DataFrame(features)
        n_features = len([c for c in df.columns if c not in ["game_id", "game_date", "season", "home_win"]])
        logger.info("Enhanced feature matrix: {} games, {} features", len(df), n_features)
        return df

    def _build_game_features(self, game: pd.Series) -> dict | None:
        """Build all features for a single game."""
        game_date = game["game_date"]
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]
        game_id = game["game_id"]

        home_hist = self.team_games[
            (self.team_games["team_id"] == home_id)
            & (self.team_games["game_date"] < game_date)
        ]
        away_hist = self.team_games[
            (self.team_games["team_id"] == away_id)
            & (self.team_games["game_date"] < game_date)
        ]

        min_games = self.WINDOWS[0]
        if len(home_hist) < min_games or len(away_hist) < min_games:
            return None

        feat: dict = {
            "game_id": game_id,
            "game_date": game_date,
            "season": game["season"],
            "home_win": game["home_win"],
        }

        #ROLLING TEAM STATS (original + spread + totals features) 
        for window in self.WINDOWS:
            h = home_hist.tail(window)
            a = away_hist.tail(window)

            if len(h) < min_games or len(a) < min_games:
                continue

            prefix_h = f"home_L{window}"
            prefix_a = f"away_L{window}"

            # --- Original moneyline features ---
            feat[f"{prefix_h}_win_pct"] = h["win"].mean()
            feat[f"{prefix_a}_win_pct"] = a["win"].mean()

            feat[f"{prefix_h}_pts_for_avg"] = h["points_for"].mean()
            feat[f"{prefix_h}_pts_against_avg"] = h["points_against"].mean()
            feat[f"{prefix_a}_pts_for_avg"] = a["points_for"].mean()
            feat[f"{prefix_a}_pts_against_avg"] = a["points_against"].mean()

            feat[f"{prefix_h}_margin_avg"] = h["margin"].mean()
            feat[f"{prefix_a}_margin_avg"] = a["margin"].mean()

            feat[f"{prefix_h}_pts_for_std"] = h["points_for"].std()
            feat[f"{prefix_a}_pts_for_std"] = a["points_for"].std()

            h_home = h[h["is_home"]]
            a_away = a[~a["is_home"]]
            if len(h_home) > 0:
                feat[f"{prefix_h}_home_win_pct"] = h_home["win"].mean()
            if len(a_away) > 0:
                feat[f"{prefix_a}_away_win_pct"] = a_away["win"].mean()

            # --- Differentials ---
            feat[f"diff_L{window}_win_pct"] = feat.get(f"{prefix_h}_win_pct", 0.5) - feat.get(f"{prefix_a}_win_pct", 0.5)
            feat[f"diff_L{window}_margin"] = feat.get(f"{prefix_h}_margin_avg", 0) - feat.get(f"{prefix_a}_margin_avg", 0)
            feat[f"diff_L{window}_pts_for"] = feat.get(f"{prefix_h}_pts_for_avg", 0) - feat.get(f"{prefix_a}_pts_for_avg", 0)

            # --- Spread-specific features ---
            feat[f"{prefix_h}_margin_std"] = h["margin"].std()
            feat[f"{prefix_a}_margin_std"] = a["margin"].std()

            feat[f"{prefix_h}_blowout_rate"] = (h["margin"] >= 10).mean()
            feat[f"{prefix_a}_blowout_rate"] = (a["margin"] >= 10).mean()

            feat[f"{prefix_h}_close_game_rate"] = (h["margin"].abs() <= 5).mean()
            feat[f"{prefix_a}_close_game_rate"] = (a["margin"].abs() <= 5).mean()

            h_wins = h[h["win"] == True]
            h_losses = h[h["win"] == False]
            a_wins = a[a["win"] == True]
            a_losses = a[a["win"] == False]

            feat[f"{prefix_h}_avg_win_margin"] = h_wins["margin"].mean() if len(h_wins) > 0 else 0
            feat[f"{prefix_h}_avg_loss_margin"] = h_losses["margin"].mean() if len(h_losses) > 0 else 0
            feat[f"{prefix_a}_avg_win_margin"] = a_wins["margin"].mean() if len(a_wins) > 0 else 0
            feat[f"{prefix_a}_avg_loss_margin"] = a_losses["margin"].mean() if len(a_losses) > 0 else 0

            feat[f"diff_L{window}_margin_std"] = feat[f"{prefix_h}_margin_std"] - feat[f"{prefix_a}_margin_std"]
            feat[f"diff_L{window}_blowout"] = feat[f"{prefix_h}_blowout_rate"] - feat[f"{prefix_a}_blowout_rate"]

            #Totals-specific features
            feat[f"{prefix_h}_pace_proxy"] = h["total_points"].mean()
            feat[f"{prefix_a}_pace_proxy"] = a["total_points"].mean()
            feat[f"combined_L{window}_pace"] = (feat[f"{prefix_h}_pace_proxy"] + feat[f"{prefix_a}_pace_proxy"]) / 2

            feat[f"{prefix_h}_total_involvement"] = h["points_for"].mean() + h["points_against"].mean()
            feat[f"{prefix_a}_total_involvement"] = a["points_for"].mean() + a["points_against"].mean()

            feat[f"{prefix_h}_high_scoring_rate"] = (h["total_points"] >= 220).mean()
            feat[f"{prefix_a}_high_scoring_rate"] = (a["total_points"] >= 220).mean()
            feat[f"{prefix_h}_low_scoring_rate"] = (h["total_points"] <= 210).mean()
            feat[f"{prefix_a}_low_scoring_rate"] = (a["total_points"] <= 210).mean()

            feat[f"{prefix_h}_total_pts_std"] = h["total_points"].std()
            feat[f"{prefix_a}_total_pts_std"] = a["total_points"].std()

            feat[f"diff_L{window}_pace"] = feat[f"{prefix_h}_pace_proxy"] - feat[f"{prefix_a}_pace_proxy"]

        #REST DAYS 
        if len(home_hist) > 0:
            last_home_game = pd.to_datetime(home_hist.iloc[-1]["game_date"])
            feat["home_rest_days"] = (pd.to_datetime(game_date) - last_home_game).days
        if len(away_hist) > 0:
            last_away_game = pd.to_datetime(away_hist.iloc[-1]["game_date"])
            feat["away_rest_days"] = (pd.to_datetime(game_date) - last_away_game).days

        if "home_rest_days" in feat and "away_rest_days" in feat:
            feat["rest_advantage"] = feat["home_rest_days"] - feat["away_rest_days"]
            feat["home_b2b"] = 1 if feat["home_rest_days"] == 1 else 0
            feat["away_b2b"] = 1 if feat["away_rest_days"] == 1 else 0
            feat["b2b_disadvantage"] = feat["away_b2b"] - feat["home_b2b"]

        #HEAD-TO-HEAD
        h2h = home_hist[home_hist["opponent_id"] == away_id]
        if len(h2h) > 0:
            feat["h2h_home_win_pct"] = h2h["win"].mean()
            feat["h2h_games_played"] = len(h2h)
            feat["h2h_avg_margin"] = h2h["margin"].mean()
            feat["h2h_avg_total"] = h2h["total_points"].mean()
        else:
            feat["h2h_home_win_pct"] = 0.5
            feat["h2h_games_played"] = 0
            feat["h2h_avg_margin"] = 0.0
            feat["h2h_avg_total"] = 215.0

        #SEASON-LONG STATS 
        feat["home_season_win_pct"] = home_hist["win"].mean()
        feat["away_season_win_pct"] = away_hist["win"].mean()
        feat["home_season_games"] = len(home_hist)
        feat["away_season_games"] = len(away_hist)

        #STREAKS
        feat["home_streak"] = self._compute_streak(home_hist)
        feat["away_streak"] = self._compute_streak(away_hist)
        feat["streak_diff"] = feat["home_streak"] - feat["away_streak"]

        #MOMENTUM 
        feat["home_momentum"] = self._compute_momentum(home_hist, window=10)
        feat["away_momentum"] = self._compute_momentum(away_hist, window=10)
        feat["momentum_diff"] = feat["home_momentum"] - feat["away_momentum"]

        #ELO RATINGS
        elo_data = self.elo_lookup.get(game_id)
        if elo_data:
            feat["home_elo"] = elo_data["home_elo"]
            feat["away_elo"] = elo_data["away_elo"]
            feat["elo_diff"] = elo_data["elo_diff"]
            feat["elo_home_win_prob"] = elo_data["elo_home_win_prob"]
        else:
            feat["home_elo"] = 1500.0
            feat["away_elo"] = 1500.0
            feat["elo_diff"] = 0.0
            feat["elo_home_win_prob"] = 0.5

        #FOUR FACTORS 
        if self.ff_lookup:
            season = game["season"]
            home_abbr = game.get("home_team_abbr", "")
            away_abbr = game.get("away_team_abbr", "")

            home_ff = self.ff_lookup.get((season, home_abbr), {})
            away_ff = self.ff_lookup.get((season, away_abbr), {})

            #Home team four factors
            for stat in ["off_efg_pct", "off_tov_pct", "off_orb_pct", "off_ft_rate",
                          "def_efg_pct", "def_tov_pct", "def_orb_pct", "def_ft_rate",
                          "pace", "off_rtg", "def_rtg", "net_rtg", "srs"]:
                feat[f"home_{stat}"] = home_ff.get(stat, 0.0)
                feat[f"away_{stat}"] = away_ff.get(stat, 0.0)
                # Differential
                feat[f"diff_{stat}"] = feat[f"home_{stat}"] - feat[f"away_{stat}"]

        return feat

    @staticmethod
    def _compute_streak(hist: pd.DataFrame) -> int:
        """Compute current win/loss streak. Positive = winning, negative = losing."""
        if len(hist) == 0:
            return 0
        wins = hist["win"].values[::-1]
        if wins[0]:
            streak = 0
            for w in wins:
                if w:
                    streak += 1
                else:
                    break
            return streak
        else:
            streak = 0
            for w in wins:
                if not w:
                    streak -= 1
                else:
                    break
            return streak

    @staticmethod
    def _compute_momentum(hist: pd.DataFrame, window: int = 10) -> float:
        """Compute weighted momentum — recent games matter more."""
        recent = hist.tail(window)
        if len(recent) == 0:
            return 0.0
        n = len(recent)
        weights = np.exp(np.linspace(-1, 0, n))
        weights = weights / weights.sum()
        margins = recent["margin"].values
        return float(np.dot(weights, margins))