"""Injury report scraper and player availability features.

Scrapes current NBA injury data from Basketball Reference and combines
it with player impact scores to adjust model predictions.

Usage:
    from app.data.collectors.injury_collector import InjuryCollector

    collector = InjuryCollector()
    injuries = collector.get_current_injuries()
    collector.print_injuries()

    # Get availability adjustment for a specific game
    adjustment = collector.get_game_adjustment("CLE", "TOR", impact_df)
"""

import time
import re

import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup

from app.utils import logger


# Map BBRef team names to abbreviations
BBREF_TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}


class InjuryCollector:
    """Scrapes current NBA injury reports from Basketball Reference."""

    INJURIES_URL = "https://www.basketball-reference.com/friv/injuries.fcgi"

    def __init__(self, delay: float = 3.5):
        self.delay = delay
        self.session = cloudscraper.create_scraper()
        self._injuries_cache = None

    def get_current_injuries(self) -> pd.DataFrame:
        """Scrape current injury report from Basketball Reference.

        Returns:
            DataFrame with columns: player_name, team_abbr, status, description
        """
        if self._injuries_cache is not None:
            return self._injuries_cache

        logger.info("Fetching injury report from Basketball Reference...")
        time.sleep(self.delay)

        response = self.session.get(self.INJURIES_URL, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        table = soup.find("table")
        if table is None:
            logger.warning("No injury table found")
            return pd.DataFrame()

        tbody = table.find("tbody")
        if tbody is None:
            return pd.DataFrame()

        records = []

        for row in tbody.find_all("tr"):
            # Parse using data-stat attributes
            player_cell = row.find(["th", "td"], {"data-stat": "player"})
            team_cell = row.find("td", {"data-stat": "team_name"})
            note_cell = row.find("td", {"data-stat": "note"})

            if not player_cell or not team_cell or not note_cell:
                continue

            # Player name
            player_link = player_cell.find("a")
            player_name = player_link.text.strip() if player_link else player_cell.text.strip()

            # Team abbreviation — extract from link href
            team_link = team_cell.find("a")
            team_abbr = ""
            if team_link:
                import re
                match = re.search(r"/teams/(\w+)/", team_link.get("href", ""))
                if match:
                    team_abbr = match.group(1)

            if not team_abbr:
                # Fallback to name mapping
                team_name = team_cell.text.strip()
                team_abbr = BBREF_TEAM_NAME_TO_ABBR.get(team_name, "")

            if not team_abbr:
                continue

            # Status — parse from note text
            note_text = note_cell.text.strip()
            status = "unknown"
            note_lower = note_text.lower()
            if note_lower.startswith("out for season"):
                status = "out_for_season"
            elif note_lower.startswith("out"):
                status = "out"
            elif "doubtful" in note_lower:
                status = "doubtful"
            elif "questionable" in note_lower:
                status = "questionable"
            elif "probable" in note_lower:
                status = "probable"
            elif "day-to-day" in note_lower:
                status = "day_to_day"

            records.append({
                "player_name": player_name,
                "team_abbr": team_abbr,
                "status": status,
                "description": note_text[:200],
            })

        df = pd.DataFrame(records)
        if not df.empty:
            logger.info("Found {} injured/unavailable players across {} teams",
                         len(df), df["team_abbr"].nunique())
        self._injuries_cache = df
        return df

    def get_team_injuries(self, team_abbr: str) -> pd.DataFrame:
        """Get injuries for a specific team."""
        injuries = self.get_current_injuries()
        if injuries.empty:
            return pd.DataFrame()
        return injuries[injuries["team_abbr"] == team_abbr]

    def get_players_out(self, team_abbr: str) -> list[str]:
        """Get list of player names currently OUT for a team."""
        team_injuries = self.get_team_injuries(team_abbr)
        if team_injuries.empty:
            return []
        out_statuses = ["out", "out_for_season", "doubtful", "unknown"]
        out_players = team_injuries[team_injuries["status"].isin(out_statuses)]
        return out_players["player_name"].tolist()

    def get_game_adjustment(
        self,
        home_abbr: str,
        away_abbr: str,
        impact_df: pd.DataFrame,
    ) -> dict:
        """Calculate prediction adjustments based on player availability.

        Looks up which players are OUT for each team, finds their impact
        scores, and returns adjustments to win probability and spread.

        Args:
            home_abbr: Home team abbreviation
            away_abbr: Away team abbreviation
            impact_df: Player impact DataFrame from PlayerImpactAnalyzer

        Returns:
            Dict with home/away adjustments for probability and margin
        """
        home_out = self.get_players_out(home_abbr)
        away_out = self.get_players_out(away_abbr)

        home_impact = 0.0
        away_impact = 0.0
        home_missing = []
        away_missing = []

        if not impact_df.empty:
            for player in home_out:
                match = impact_df[
                    (impact_df["player_name"].str.contains(player.split(" ")[-1], case=False, na=False))
                    & (impact_df["team_abbr"] == home_abbr)
                ]
                if not match.empty:
                    margin_impact = match.iloc[0]["margin_impact"]
                    win_rate_impact = match.iloc[0]["win_rate_impact"]
                    home_impact += margin_impact
                    home_missing.append({
                        "name": player,
                        "margin_impact": margin_impact,
                        "win_rate_impact": win_rate_impact,
                    })

            for player in away_out:
                match = impact_df[
                    (impact_df["player_name"].str.contains(player.split(" ")[-1], case=False, na=False))
                    & (impact_df["team_abbr"] == away_abbr)
                ]
                if not match.empty:
                    margin_impact = match.iloc[0]["margin_impact"]
                    win_rate_impact = match.iloc[0]["win_rate_impact"]
                    away_impact += margin_impact
                    away_missing.append({
                        "name": player,
                        "margin_impact": margin_impact,
                        "win_rate_impact": win_rate_impact,
                    })

        # Net adjustment: if home is missing more impact, probability goes down
        # Scale margin impact to probability adjustment (~3 points = ~10% probability)
        net_margin_impact = away_impact - home_impact  # positive = home benefits
        prob_adjustment = net_margin_impact * 0.033  # ~3.3% per point of margin

        return {
            "home_out": home_out,
            "away_out": away_out,
            "home_missing_details": home_missing,
            "away_missing_details": away_missing,
            "home_total_margin_impact": round(home_impact, 1),
            "away_total_margin_impact": round(away_impact, 1),
            "net_margin_impact": round(net_margin_impact, 1),
            "prob_adjustment": round(prob_adjustment, 4),
            "spread_adjustment": round(net_margin_impact, 1),
        }

    def print_injuries(self) -> None:
        """Print formatted injury report."""
        injuries = self.get_current_injuries()
        if injuries.empty:
            print("No injury data available.")
            return

        print(f"\n{'='*70}")
        print(f"  NBA INJURY REPORT — {len(injuries)} players")
        print(f"{'='*70}")

        for team in sorted(injuries["team_abbr"].unique()):
            team_injuries = injuries[injuries["team_abbr"] == team]
            print(f"\n  {team}:")
            for _, row in team_injuries.iterrows():
                status_emoji = {
                    "out": "🔴", "out_for_season": "⛔",
                    "doubtful": "🟠", "questionable": "🟡",
                    "probable": "🟢", "day_to_day": "🟡",
                }.get(row["status"], "⚪")
                print(f"    {status_emoji} {row['player_name']} — {row['status'].replace('_', ' ').title()}")

        print()