"""Pydantic data models representing NBA entities.

These models serve as the contract between data collection and the rest of the
pipeline. Raw API responses get validated and normalized into these shapes, so
downstream code never has to worry about missing fields or bad types.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


# ── Team ─────────────────────────────────────────────────────────────────────


class Team(BaseModel):
    """An NBA team."""

    team_id: int
    abbreviation: str
    full_name: str
    nickname: str
    city: str
    state: str
    year_founded: int


# ── Player ───────────────────────────────────────────────────────────────────


class Player(BaseModel):
    """An NBA player (static info)."""

    player_id: int
    full_name: str
    first_name: str
    last_name: str
    is_active: bool


# ── Game ─────────────────────────────────────────────────────────────────────


class Game(BaseModel):
    """A single NBA game with final results.

    Each row from LeagueGameFinder gives one team's perspective. We combine
    home + away into a single Game record during processing.
    """

    game_id: str  # e.g. "0022400123"
    season: int  # start year, e.g. 2024
    game_date: date
    home_team_id: int
    away_team_id: int
    home_team_abbr: str
    away_team_abbr: str
    home_score: int
    away_score: int
    home_win: bool

    # Optional betting data (populated later from odds sources)
    spread: Optional[float] = None  # home team spread (negative = favored)
    over_under: Optional[float] = None
    home_ml: Optional[int] = None  # moneyline odds
    away_ml: Optional[int] = None


# ── Player Game Log ──────────────────────────────────────────────────────────


class PlayerGameLog(BaseModel):
    """A single player's stats for a single game.

    This is the core unit of data for player-level feature engineering.
    All stat fields default to 0 so we handle DNPs and partial data gracefully.
    """

    player_id: int
    player_name: str
    team_id: int
    team_abbr: str
    game_id: str
    game_date: date
    season: int

    # Availability
    minutes: float = 0.0

    # Scoring
    points: int = 0
    fgm: int = 0  # field goals made
    fga: int = 0  # field goals attempted
    fg_pct: float = 0.0
    fg3m: int = 0  # three pointers made
    fg3a: int = 0  # three pointers attempted
    fg3_pct: float = 0.0
    ftm: int = 0  # free throws made
    fta: int = 0  # free throws attempted
    ft_pct: float = 0.0

    # Rebounds
    oreb: int = 0  # offensive rebounds
    dreb: int = 0  # defensive rebounds
    reb: int = 0  # total rebounds

    # Playmaking & defense
    ast: int = 0  # assists
    stl: int = 0  # steals
    blk: int = 0  # blocks
    tov: int = 0  # turnovers
    pf: int = 0  # personal fouls

    # Plus/minus
    plus_minus: float = 0.0

    # Fantasy (computed)
    fantasy_points: Optional[float] = None

    def compute_fantasy_points(self, scoring: str = "draftkings") -> float:
        """Compute fantasy points using standard DFS scoring.

        DraftKings NBA scoring:
            Point = 1, 3PM = 0.5, Reb = 1.25, Ast = 1.5,
            Stl = 2, Blk = 2, TO = -0.5,
            Double-double bonus = 1.5, Triple-double bonus = 3
        """
        if scoring != "draftkings":
            raise ValueError(f"Unsupported scoring system: {scoring}")

        pts = (
            self.points * 1.0
            + self.fg3m * 0.5
            + self.reb * 1.25
            + self.ast * 1.5
            + self.stl * 2.0
            + self.blk * 2.0
            + self.tov * -0.5
        )

        # Double-double / triple-double bonuses
        stat_categories = [self.points, self.reb, self.ast, self.stl, self.blk]
        doubles = sum(1 for s in stat_categories if s >= 10)
        if doubles >= 3:
            pts += 3.0  # triple-double
        elif doubles >= 2:
            pts += 1.5  # double-double

        self.fantasy_points = round(pts, 2)
        return self.fantasy_points


# ── Prediction (output of ML pipeline) ───────────────────────────────────────


class GamePrediction(BaseModel):
    """A prediction for a single game, emphasizing calibrated probabilities."""

    game_id: str
    model_name: str
    model_version: str

    # Core prediction
    home_win_probability: float = Field(ge=0.0, le=1.0)
    predicted_home_win: bool

    # Calibration metrics for this prediction
    calibration_method: str = "none"  # platt, isotonic, temperature, beta

    # Spread / total predictions (regression)
    predicted_spread: Optional[float] = None
    predicted_total: Optional[float] = None

    # Confidence
    confidence_lower: Optional[float] = None  # lower bound of probability CI
    confidence_upper: Optional[float] = None  # upper bound of probability CI

    # Kelly criterion output
    kelly_bet_fraction: Optional[float] = None
    kelly_edge: Optional[float] = None  # expected edge over the line


class PlayerPropPrediction(BaseModel):
    """A prediction for a player stat prop (e.g. LeBron over 25.5 points)."""

    player_id: int
    player_name: str
    game_id: str
    stat_type: str  # "points", "rebounds", "assists", etc.
    line: float  # the prop line (e.g. 25.5)
    over_probability: float = Field(ge=0.0, le=1.0)
    predicted_value: float  # point estimate
    calibration_method: str = "none"
