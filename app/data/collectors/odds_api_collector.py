"""Real-time NBA odds collector and prediction generator.

Connects to The Odds API to fetch current NBA game odds from major
sportsbooks (DraftKings, FanDuel, BetMGM, etc.), then runs predictions
through the saved models for moneyline, spread, and totals markets.

Free tier: 500 requests/month. Each call to get odds costs 1 request
per region per market. We fetch h2h (moneyline), spreads, and totals
for the US region = 3 requests per call.

Usage:
    from app.data.collectors.odds_api_collector import OddsAPICollector

    collector = OddsAPICollector(api_key="your_key_here")
    games = collector.get_upcoming_games()
    odds = collector.get_current_odds()
    collector.print_todays_odds()
"""

import json
from datetime import datetime, date
from dataclasses import dataclass, field

import requests
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from app.utils import logger


# Team name mapping: The Odds API uses full names, our models use abbreviations
TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

ABBR_TO_TEAM_NAME = {v: k for k, v in TEAM_NAME_TO_ABBR.items()}

# Team abbreviation to team_id mapping (matches our feature engineering)
ABBR_TO_TEAM_ID = {
    "ATL": 1, "BOS": 2, "BRK": 3, "CHA": 4, "CHI": 5,
    "CLE": 6, "DAL": 7, "DEN": 8, "DET": 9, "GSW": 10,
    "HOU": 11, "IND": 12, "LAC": 13, "LAL": 14, "MEM": 15,
    "MIA": 16, "MIL": 17, "MIN": 18, "NOP": 19, "NYK": 20,
    "OKC": 21, "ORL": 22, "PHI": 23, "PHX": 24, "POR": 25,
    "SAC": 26, "SAS": 27, "TOR": 28, "UTA": 29, "WAS": 30,
}


@dataclass
class GameOdds:
    """Odds for a single upcoming NBA game."""

    game_id: str
    commence_time: datetime
    home_team: str  # full name
    away_team: str  # full name
    home_abbr: str
    away_abbr: str

    # Moneyline odds (best available from bookmakers)
    home_ml: int | None = None  # American odds, e.g. -150
    away_ml: int | None = None  # American odds, e.g. +130
    home_ml_decimal: float | None = None
    away_ml_decimal: float | None = None
    ml_bookmaker: str = ""

    # Spread
    spread: float | None = None  # home team spread, e.g. -5.5
    spread_odds_home: int | None = None  # typically -110
    spread_odds_away: int | None = None
    spread_bookmaker: str = ""

    # Totals (over/under)
    total: float | None = None  # e.g. 218.5
    over_odds: int | None = None
    under_odds: int | None = None
    totals_bookmaker: str = ""


class OddsAPICollector:
    """Fetches real-time NBA odds from The Odds API."""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "basketball_nba"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.remaining_requests: int | None = None

    def get_current_odds(self) -> list[GameOdds]:
        """Fetch current NBA odds for all upcoming/live games.

        Fetches moneyline (h2h), spreads, and totals from US bookmakers.
        Costs 3 requests from your monthly quota (1 per market).

        Returns:
            List of GameOdds objects with odds from the best available bookmaker.
        """
        url = f"{self.BASE_URL}/sports/{self.SPORT}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
        }

        logger.info("Fetching NBA odds from The Odds API...")
        response = requests.get(url, params=params, timeout=30)

        # Track remaining requests
        self.remaining_requests = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        logger.info("API requests — used: {}, remaining: {}", used, self.remaining_requests)

        if response.status_code != 200:
            logger.error("API error {}: {}", response.status_code, response.text)
            response.raise_for_status()

        data = response.json()
        logger.info("Found {} upcoming/live games", len(data))

        games = []
        for event in data:
            game = self._parse_event(event)
            if game:
                games.append(game)

        return games

    def get_upcoming_games(self) -> list[dict]:
        """Fetch just the list of upcoming games (no odds, no API cost).

        This uses the events endpoint which is free (0 requests).
        """
        url = f"{self.BASE_URL}/sports/{self.SPORT}/events"
        params = {"apiKey": self.api_key}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _parse_event(self, event: dict) -> GameOdds | None:
        """Parse a single event from the API response into a GameOdds object."""
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        home_abbr = TEAM_NAME_TO_ABBR.get(home_team, "")
        away_abbr = TEAM_NAME_TO_ABBR.get(away_team, "")

        if not home_abbr or not away_abbr:
            logger.warning("Unknown team: {} or {}", home_team, away_team)
            return None

        commence = datetime.fromisoformat(
            event["commence_time"].replace("Z", "+00:00")
        )

        game = GameOdds(
            game_id=event["id"],
            commence_time=commence,
            home_team=home_team,
            away_team=away_team,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
        )

        # Parse bookmaker odds — use the first available US bookmaker for each market
        bookmakers = event.get("bookmakers", [])
        for bm in bookmakers:
            bm_name = bm.get("title", "")
            for market in bm.get("markets", []):
                market_key = market.get("key", "")
                outcomes = market.get("outcomes", [])

                if market_key == "h2h" and game.home_ml is None:
                    self._parse_moneyline(game, outcomes, bm_name)

                elif market_key == "spreads" and game.spread is None:
                    self._parse_spread(game, outcomes, home_team, bm_name)

                elif market_key == "totals" and game.total is None:
                    self._parse_totals(game, outcomes, bm_name)

        return game

    def _parse_moneyline(self, game: GameOdds, outcomes: list, bookmaker: str) -> None:
        """Extract moneyline odds from outcomes."""
        for outcome in outcomes:
            name = outcome.get("name", "")
            price = outcome.get("price", 0)
            if name == game.home_team:
                game.home_ml = price
                game.home_ml_decimal = self._american_to_decimal(price)
            elif name == game.away_team:
                game.away_ml = price
                game.away_ml_decimal = self._american_to_decimal(price)
        game.ml_bookmaker = bookmaker

    def _parse_spread(self, game: GameOdds, outcomes: list, home_team: str, bookmaker: str) -> None:
        """Extract spread from outcomes."""
        for outcome in outcomes:
            name = outcome.get("name", "")
            point = outcome.get("point", 0)
            price = outcome.get("price", 0)
            if name == home_team:
                game.spread = point
                game.spread_odds_home = price
            else:
                game.spread_odds_away = price
        game.spread_bookmaker = bookmaker

    def _parse_totals(self, game: GameOdds, outcomes: list, bookmaker: str) -> None:
        """Extract totals (over/under) from outcomes."""
        for outcome in outcomes:
            name = outcome.get("name", "")
            point = outcome.get("point", 0)
            price = outcome.get("price", 0)
            if name == "Over":
                game.total = point
                game.over_odds = price
            elif name == "Under":
                game.under_odds = price
        game.totals_bookmaker = bookmaker

    @staticmethod
    def _american_to_decimal(american: int) -> float:
        """Convert American odds to decimal odds."""
        if american > 0:
            return american / 100.0 + 1.0
        elif american < 0:
            return 100.0 / abs(american) + 1.0
        return 2.0

    def print_todays_odds(self) -> None:
        """Fetch and display today's odds in a readable format."""
        games = self.get_current_odds()
        if not games:
            print("No upcoming NBA games found.")
            return

        print(f"\n{'='*75}")
        print(f"  NBA ODDS — {len(games)} upcoming games")
        print(f"  Remaining API requests: {self.remaining_requests}")
        print(f"{'='*75}\n")

        for g in games:
            from datetime import timezone, timedelta
            pst = timezone(timedelta(hours=-7))  # PDT (Pacific Daylight Time)
            local_time = g.commence_time.astimezone(pst)
            time_str = local_time.strftime("%b %d, %I:%M %p PT")
            print(f"  {g.away_team} @ {g.home_team}")
            print(f"  {time_str}")
            print(f"  {'─'*50}")

            if g.home_ml is not None:
                print(f"  Moneyline:  {g.home_team}: {g.home_ml:+d}  |  "
                      f"{g.away_team}: {g.away_ml:+d}  ({g.ml_bookmaker})")

            if g.spread is not None:
                print(f"  Spread:     {g.home_team}: {g.spread:+.1f} ({g.spread_odds_home:+d})  |  "
                      f"{g.away_team}: {-g.spread:+.1f} ({g.spread_odds_away:+d})  ({g.spread_bookmaker})")

            if g.total is not None:
                print(f"  Total:      O/U {g.total}  |  Over: {g.over_odds:+d}  Under: {g.under_odds:+d}  ({g.totals_bookmaker})")

            print()


class LivePredictor:
    """Generates predictions for upcoming games using saved models and live odds.

    Loads pre-trained models from the models/ directory and combines them
    with real-time odds to produce bet recommendations.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self._load_models()

    def _load_models(self) -> None:
        """Load all saved models, scalers, and parameters."""
        logger.info("Loading saved models...")

        # Moneyline
        self.ml_model = joblib.load(self.models_dir / "moneyline_random_forest.joblib")
        self.ml_scaler = joblib.load(self.models_dir / "scaler_moneyline.joblib")
        self.calibrator = joblib.load(self.models_dir / "calibrator_isotonic.joblib")

        # Spread
        self.spread_model = joblib.load(self.models_dir / "spread_random_forest.joblib")
        self.spread_scaler = joblib.load(self.models_dir / "scaler_spread.joblib")

        # Totals
        self.totals_model = joblib.load(self.models_dir / "totals_random_forest.joblib")

        # Feature columns and strategy params
        self.feature_cols = joblib.load(self.models_dir / "feature_cols.joblib")
        self.params = joblib.load(self.models_dir / "strategy_params.joblib")

        logger.info("Loaded {} feature columns, {} models",
                     len(self.feature_cols), "all")

    def predict_game(self, game_odds: GameOdds, features: dict) -> dict:
        """Generate predictions for a single game across all bet types.

        Args:
            game_odds: Live odds for the game.
            features: Dict of feature values for this game (from feature builder).

        Returns:
            Dict with predictions, edges, and recommendations for each bet type.
        """
        # Build feature vector
        feature_values = [features.get(col, 0.0) for col in self.feature_cols]
        X = np.array(feature_values).reshape(1, -1)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # ── Moneyline Prediction ─────────────────────────────────────
        X_ml = self.ml_scaler.transform(X)
        raw_prob = self.ml_model.predict_proba(X_ml)[0, 1]
        cal_prob = float(np.clip(self.calibrator.transform(np.array([raw_prob])), 0.001, 0.999)[0])

        # Calculate edge vs market
        ml_edge = None
        ml_recommendation = "No bet"
        if game_odds.home_ml_decimal and game_odds.away_ml_decimal:
            home_market_implied = 1.0 / game_odds.home_ml_decimal
            away_market_implied = 1.0 / game_odds.away_ml_decimal

            home_edge = cal_prob - home_market_implied
            away_edge = (1 - cal_prob) - away_market_implied

            min_edge = self.params.get("moneyline_min_edge", 0.03)

            # Filter out heavy underdogs — don't bet on teams with +250 or worse odds
            home_is_heavy_dog = game_odds.home_ml is not None and game_odds.home_ml >= 250
            away_is_heavy_dog = game_odds.away_ml is not None and game_odds.away_ml >= 250

            if home_edge >= min_edge and home_edge > away_edge and not home_is_heavy_dog:
                ml_edge = home_edge
                ml_recommendation = f"Bet {game_odds.home_abbr} ML"
            elif away_edge >= min_edge and away_edge > home_edge and not away_is_heavy_dog:
                ml_edge = away_edge
                ml_recommendation = f"Bet {game_odds.away_abbr} ML"
            else:
                ml_edge = home_edge

        # ── Spread Prediction ────────────────────────────────────────
        X_sp = self.spread_scaler.transform(X)
        predicted_margin = float(self.spread_model.predict(X_sp)[0])

        spread_edge = None
        spread_recommendation = "No bet"
        if game_odds.spread is not None:
            spread_edge = predicted_margin - game_odds.spread
            min_spread_edge = self.params.get("spread_min_edge", 3.0)
            ml_conf_threshold = self.params.get("spread_ml_confidence", 0.55)

            if spread_edge >= min_spread_edge:
                spread_recommendation = f"Bet {game_odds.home_abbr} {game_odds.spread:+.1f}"
            elif spread_edge <= -min_spread_edge:
                away_spread = -game_odds.spread
                spread_recommendation = f"Bet {game_odds.away_abbr} {away_spread:+.1f}"

        # ── Totals Prediction ────────────────────────────────────────
        predicted_total = float(self.totals_model.predict(X_sp)[0])

        totals_edge = None
        totals_recommendation = "No bet"
        if game_odds.total is not None:
            totals_edge = predicted_total - game_odds.total
            # Only recommend totals with large edge (8+ points)
            if totals_edge >= 8.0:
                totals_recommendation = f"Bet OVER {game_odds.total}"
            elif totals_edge <= -8.0:
                totals_recommendation = f"Bet UNDER {game_odds.total}"

        return {
            "game_id": game_odds.game_id,
            "home_team": game_odds.home_team,
            "away_team": game_odds.away_team,
            "home_abbr": game_odds.home_abbr,
            "away_abbr": game_odds.away_abbr,
            "commence_time": game_odds.commence_time,

            # Moneyline
            "home_win_prob_raw": round(raw_prob, 4),
            "home_win_prob_calibrated": round(cal_prob, 4),
            "home_ml": game_odds.home_ml,
            "away_ml": game_odds.away_ml,
            "ml_edge": round(ml_edge, 4) if ml_edge else None,
            "ml_recommendation": ml_recommendation,

            # Spread
            "predicted_margin": round(predicted_margin, 1),
            "book_spread": game_odds.spread,
            "spread_odds_home": game_odds.spread_odds_home,
            "spread_odds_away": game_odds.spread_odds_away,
            "spread_edge": round(spread_edge, 1) if spread_edge is not None else None,
            "spread_edge_abs": round(abs(spread_edge), 1) if spread_edge is not None else None,
            "spread_recommendation": spread_recommendation,
            "spread_bookmaker": game_odds.spread_bookmaker,

            # Totals
            "predicted_total": round(predicted_total, 1),
            "book_total": game_odds.total,
            "over_odds": game_odds.over_odds,
            "under_odds": game_odds.under_odds,
            "totals_edge": round(totals_edge, 1) if totals_edge else None,
            "totals_recommendation": totals_recommendation,
            "totals_bookmaker": game_odds.totals_bookmaker,
        }

    def print_predictions(self, predictions: list[dict]) -> None:
        """Print formatted predictions for all games."""
        if not predictions:
            print("No predictions available.")
            return

        print(f"\n{'='*75}")
        print(f"  NBA PREDICTIONS — {len(predictions)} games")
        print(f"{'='*75}")

        for p in predictions:
            from datetime import timezone, timedelta
            pst = timezone(timedelta(hours=-7))
            local_time = p["commence_time"].astimezone(pst)
            time_str = local_time.strftime("%b %d, %I:%M %p PT")
            print(f"\n  {p['away_team']} @ {p['home_team']}")
            print(f"  {time_str}")
            print(f"  {'─'*60}")

            # Moneyline
            prob_pct = p['home_win_prob_calibrated'] * 100
            print(f"  MONEYLINE:")
            print(f"    Model: {p['home_abbr']} {prob_pct:.1f}% win probability (calibrated)")
            if p['home_ml']:
                print(f"    Market: {p['home_abbr']} {p['home_ml']:+d}  |  {p['away_abbr']} {p['away_ml']:+d}")
            if p['ml_edge']:
                edge_pct = p['ml_edge'] * 100
                print(f"    Edge: {edge_pct:+.1f}%")
            # Show injury adjustments if available
            adj = p.get("injury_adjustment")
            if adj and (adj["home_out"] or adj["away_out"]):
                if adj["home_out"]:
                    print(f"    ⚠️  {p['home_abbr']} missing: {', '.join(adj['home_out'][:3])}")
                if adj["away_out"]:
                    print(f"    ⚠️  {p['away_abbr']} missing: {', '.join(adj['away_out'][:3])}")
                if abs(adj["prob_adjustment"]) >= 0.01:
                    orig = p.get("home_win_prob_calibrated", 0) * 100
                    adjusted = p.get("home_win_prob_adjusted", p.get("home_win_prob_calibrated", 0)) * 100
                    print(f"    📊 Adjusted: {p['home_abbr']} {orig:.1f}% → {adjusted:.1f}%")
            marker = " <<<" if "Bet" in p['ml_recommendation'] else ""
            print(f"    >>> {p['ml_recommendation']}{marker}")

            # Spread
            print(f"  SPREAD:")
            adj_margin = p.get('predicted_margin_adjusted', p['predicted_margin'])
            print(f"    Model: {p['home_abbr']} by {adj_margin:+.1f} points")
            if p['book_spread'] is not None:
                print(f"    Line: {p['home_abbr']} {p['book_spread']:+.1f}")
            if p['spread_edge'] is not None:
                print(f"    Edge: {p['spread_edge']:+.1f} points")
            marker = " <<<" if "Bet" in p['spread_recommendation'] else ""
            print(f"    >>> {p['spread_recommendation']}{marker}")

            # Totals
            print(f"  TOTALS:")
            print(f"    Model: {p['predicted_total']:.1f} total points")
            if p['book_total'] is not None:
                print(f"    Line: O/U {p['book_total']}")
            if p['totals_edge'] is not None:
                direction = "OVER" if p['totals_edge'] > 0 else "UNDER"
                print(f"    Edge: {abs(p['totals_edge']):.1f} points {direction}")
            print(f"    >>> {p['totals_recommendation']}")