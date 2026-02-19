"""
Baketball Reference data collector

Scrapes game results and player stats from basketball-reference.com.
This is a backup data source when nba_api and stats.nba.com is blocking requests.

Usage:
    from app.data.collectors.bbref_collector import BBRefCollector

    collector = BBRefCollector()
    games = collector.get_season_games(2024)
    logs = collector.get_player_game_logs_by_name("LeBron James", 2024)
"""

import re
import time
from datetime import date, datetime
from io import StringIO
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.config import settings
from app.models import Game, Player, PlayerGameLog, Team
from app.utils import logger

#static team data
BBREF_TEAMS = {
    {"team_id": 1, "abbreviation": "ATL", "full_name": "Atlanta Hawks", "nickname": "Hawks", "city": "Atlanta", "state": "Georgia", "year_founded": 1949},
    {"team_id": 2, "abbreviation": "BOS", "full_name": "Boston Celtics", "nickname": "Celtics", "city": "Boston", "state": "Massachusetts", "year_founded": 1946},
    {"team_id": 3, "abbreviation": "BRK", "full_name": "Brooklyn Nets", "nickname": "Nets", "city": "Brooklyn", "state": "New York", "year_founded": 1976},
    {"team_id": 4, "abbreviation": "CHO", "full_name": "Charlotte Hornets", "nickname": "Hornets", "city": "Charlotte", "state": "North Carolina", "year_founded": 1988},
    {"team_id": 5, "abbreviation": "CHI", "full_name": "Chicago Bulls", "nickname": "Bulls", "city": "Chicago", "state": "Illinois", "year_founded": 1966},
    {"team_id": 6, "abbreviation": "CLE", "full_name": "Cleveland Cavaliers", "nickname": "Cavaliers", "city": "Cleveland", "state": "Ohio", "year_founded": 1970},
    {"team_id": 7, "abbreviation": "DAL", "full_name": "Dallas Mavericks", "nickname": "Mavericks", "city": "Dallas", "state": "Texas", "year_founded": 1980},
    {"team_id": 8, "abbreviation": "DEN", "full_name": "Denver Nuggets", "nickname": "Nuggets", "city": "Denver", "state": "Colorado", "year_founded": 1976},
    {"team_id": 9, "abbreviation": "DET", "full_name": "Detroit Pistons", "nickname": "Pistons", "city": "Detroit", "state": "Michigan", "year_founded": 1948},
    {"team_id": 10, "abbreviation": "GSW", "full_name": "Golden State Warriors", "nickname": "Warriors", "city": "San Francisco", "state": "California", "year_founded": 1946},
    {"team_id": 11, "abbreviation": "HOU", "full_name": "Houston Rockets", "nickname": "Rockets", "city": "Houston", "state": "Texas", "year_founded": 1967},
    {"team_id": 12, "abbreviation": "IND", "full_name": "Indiana Pacers", "nickname": "Pacers", "city": "Indianapolis", "state": "Indiana", "year_founded": 1976},
    {"team_id": 13, "abbreviation": "LAC", "full_name": "Los Angeles Clippers", "nickname": "Clippers", "city": "Los Angeles", "state": "California", "year_founded": 1970},
    {"team_id": 14, "abbreviation": "LAL", "full_name": "Los Angeles Lakers", "nickname": "Lakers", "city": "Los Angeles", "state": "California", "year_founded": 1947},
    {"team_id": 15, "abbreviation": "MEM", "full_name": "Memphis Grizzlies", "nickname": "Grizzlies", "city": "Memphis", "state": "Tennessee", "year_founded": 1995},
    {"team_id": 16, "abbreviation": "MIA", "full_name": "Miami Heat", "nickname": "Heat", "city": "Miami", "state": "Florida", "year_founded": 1988},
    {"team_id": 17, "abbreviation": "MIL", "full_name": "Milwaukee Bucks", "nickname": "Bucks", "city": "Milwaukee", "state": "Wisconsin", "year_founded": 1968},
    {"team_id": 18, "abbreviation": "MIN", "full_name": "Minnesota Timberwolves", "nickname": "Timberwolves", "city": "Minneapolis", "state": "Minnesota", "year_founded": 1989},
    {"team_id": 19, "abbreviation": "NOP", "full_name": "New Orleans Pelicans", "nickname": "Pelicans", "city": "New Orleans", "state": "Louisiana", "year_founded": 2002},
    {"team_id": 20, "abbreviation": "NYK", "full_name": "New York Knicks", "nickname": "Knicks", "city": "New York", "state": "New York", "year_founded": 1946},
    {"team_id": 21, "abbreviation": "OKC", "full_name": "Oklahoma City Thunder", "nickname": "Thunder", "city": "Oklahoma City", "state": "Oklahoma", "year_founded": 1967},
    {"team_id": 22, "abbreviation": "ORL", "full_name": "Orlando Magic", "nickname": "Magic", "city": "Orlando", "state": "Florida", "year_founded": 1989},
    {"team_id": 23, "abbreviation": "PHI", "full_name": "Philadelphia 76ers", "nickname": "76ers", "city": "Philadelphia", "state": "Pennsylvania", "year_founded": 1949},
    {"team_id": 24, "abbreviation": "PHO", "full_name": "Phoenix Suns", "nickname": "Suns", "city": "Phoenix", "state": "Arizona", "year_founded": 1968},
    {"team_id": 25, "abbreviation": "POR", "full_name": "Portland Trail Blazers", "nickname": "Trail Blazers", "city": "Portland", "state": "Oregon", "year_founded": 1970},
    {"team_id": 26, "abbreviation": "SAC", "full_name": "Sacramento Kings", "nickname": "Kings", "city": "Sacramento", "state": "California", "year_founded": 1948},
    {"team_id": 27, "abbreviation": "SAS", "full_name": "San Antonio Spurs", "nickname": "Spurs", "city": "San Antonio", "state": "Texas", "year_founded": 1976},
    {"team_id": 28, "abbreviation": "TOR", "full_name": "Toronto Raptors", "nickname": "Raptors", "city": "Toronto", "state": "Ontario", "year_founded": 1995},
    {"team_id": 29, "abbreviation": "UTA", "full_name": "Utah Jazz", "nickname": "Jazz", "city": "Salt Lake City", "state": "Utah", "year_founded": 1974},
    {"team_id": 30, "abbreviation": "WAS", "full_name": "Washington Wizards", "nickname": "Wizards", "city": "Washington", "state": "District of Columbia", "year_founded": 1961},
}

#map bbref team abbreviations to team_id 
_ABBR_TO_ID = {t["abbreviation"]: t["team_id"] for t in BBREF_TEAMS}

#note all abbreviations between sources
_ABBR_NORMALIZE = {
    "BRK": "BRK", "BKN": "BRK", "NJN": "BRK", 
    "CHO": "CHO", "CHA": "CHO", "CHH": "CHO",
    "NOP": "NOP", "NOH": "NOP", "NOK": "NOP",
    "PHO": "PHO", "PHX": "PHO",
    "SA": "SAS", "SAS": "SAS", "SAN": "SAS",
    "NY": "NYK", "NYK": "NYK",
    "UTAH": "UTA", "UTA": "UTA",
    "WSH": "WAS", "WAS": "WAS",
}

def _normalize_abbr(abbr: str) -> str:
    #normalize team abbreviation to standard format
    return _ABBR_NORMALIZE.get(abbr.upper(), abbr.upper())

def _abbr_to_team_id(abbr: str) -> int:
    #convert abbreviation to team_id, with normalization
    normalized = _normalize_abbr(abbr)
    return _ABBR_TO_ID.get(normalized, 0)