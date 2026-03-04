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

import requests
import cloudscraper

import pandas as pd
import requests
from bs4 import BeautifulSoup

from app.config import settings
from app.models import Game, Player, PlayerGameLog, Team
from app.utils import logger

#static team data
BBREF_TEAMS = [
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
]

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

class BBRefCollector:
    """
    collects NBA data from basketball-reference.com
    attributes:
        delay: seconds to wait between requests to avoid rate-limiting
    """
    BASE_URL = "https://www.basketball-reference.com"

    def __init__(self, delay: float = 3.5):
        self.delay = delay
        self.session = cloudscraper.create_scraper()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
        })
        self._max_retries = 3

    def _sleep(self) -> None:
        #rate limit pause between requests
        time.sleep(self.delay)
    
    def _fetch_page(self, url: str) -> BeautifulSoup:
        #fetch webpage and return pased BeautifulSoup object
        for attempt in range(1, self._max_retries + 1):
            logger.debug("Fetching (attempt {}): {}", attempt, url)
            response = self.session.get(url, timeout=30)

            if response.status_code == 200:
                return BeautifulSoup(response.content, "html.parser")

            if response.status_code in (403, 429):
                if attempt < self._max_retries:
                    wait = self.delay * attempt * 2  # exponential backoff
                    logger.warning(
                        "Got {} from {}. Retrying in {:.0f}s (attempt {}/{})",
                        response.status_code, url, wait, attempt, self._max_retries,
                    )
                    time.sleep(wait)
                    continue

            # For non-retryable errors or final attempt, raise
            response.raise_for_status()

        # Should not reach here, but just in case
        response.raise_for_status()
        return BeautifulSoup(response.content, "html.parser")
    
    #teams (static, no HTTP)
    def get_all_teams(self) -> list[Team]:
        #return all 30 NBA teams from static data
        teams = [Team(**t) for t in BBREF_TEAMS]
        logger.info("Loaded {} teams from static data", len(teams))
        return teams
    
    #season games
    def get_season_games(self, season: int) -> list[Game]:
        """
        fetch all games for a season from basketball-reference.com
        
        scrapes monthly schedule pages for given season
        basketball reference organizes schedules by month:
            /leagues/NBA_2025_games-<month>.html 
        
        Args: season - start year of season (e.g. 2024 for 2024-25)

        Returns: list of Game objects 
        """
        bbref_year = season + 1 #bbref uses end year for season pages
        months = ["october", "november", "december", "january", "february", "march", "april", "may", "june"]

        all_games: list[Game] = []

        for month in months:
            url = f"{self.BASE_URL}/leagues/NBA_{bbref_year}_games-{month}.html"
            self._sleep()

            try:
                soup = self._fetch_page(url)
                games = self._parse_schedule_page(soup, season)
                all_games.extend(games)
                logger.info(
                    "Season {} - {}: {} games found",
                    settings.season_string(season),
                    month.capitalize(),
                    len(games),
                )
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status == 404:
                    #month doesn't exist yet (future months in current season)
                    logger.deug("No schedule page for {} {}", season, month)
                    continue
                elif status in (403, 429):
                    logger.warning("Blocked ({}) fetching {} {}. Waiting and retrying...", status, season, month)
                    time.sleep(10)
                else:
                    logger.error("HTTP error fetching {} {}: {}", season, month, e)
                    raise
            except Exception as e:
                logger.error("Error fetching {} {}: {}", season, month, e)
                continue
        
        logger.info(
            "Total games collected for season {}: {}",
            settings.season_string(season),
            len(all_games),
        )
        return all_games 
    
    def _parse_schedule_page(self, soup: BeautifulSoup, season: int) -> list[Game]:
        #parse monthly schedule page into Game objects

        #the schedule table has columns: Date, Start (ET), Visitor/Neutral, PTS, Home/Neutral, PTS, etc...)
        table = soup.find("table", id="schedule")
        if table is None:
            return []
        
        games = []
        tbody = table.find("tbody")
        if tbody is None:
            return []
        
        for row in tbody.find_all("tr"):
            #skip header rows within tbody
            if row.find("th", {"scope": "col"}):
                continue

            cells = row.find_all(["th", "td"])
            if len(cells) < 6:
                continue

            try:
                #parse date
                date_cell = row.find("th", {"data-stat": "date_game"})
                if date_cell is None:
                    continue
                date_link = date_cell.find("a")
                if date_link is None:
                    continue
                date_text = date_link.text.strip()

                #bbref date format: "Fri, Oct 22, 2024" 
                game_date = self._parse_bbref_date(date_text)
                if game_date is None:
                    continue

                #parse teams
                away_cell = row.find("td", {"data-stat": "visitor_team_name"})
                home_cell = row.find("td", {"data-stat": "home_team_name"})
                if away_cell is None or home_cell is None:
                    continue

                away_link = away_cell.find("a")
                home_link = home_cell.find("a")
                if away_link is None or home_link is None:
                    continue

                #extract abbreviation from the link href
                #e.g. /teams/LAL/2025.html -> LAL
                away_abbr = self._extract_team_abbr(away_link.get("href", ""))
                home_abbr = self._extract_team_abbr(home_link.get("href", ""))

                if not away_abbr or not home_abbr:
                    continue

                #parse scores
                away_score_cell = row.find("td", {"data-stat": "visitor_pts"})
                home_score_cell = row.find("td", {"data-stat": "home_pts"})

                if away_score_cell is None or home_score_cell is None:
                    continue

                away_score_text = away_score_cell.text.strip()
                home_score_text = home_score_cell.text.strip()

                #skip games that haven't been played yet (no score)
                if not away_score_text or not home_score_text:
                    continue

                away_score = int(away_score_text)
                home_score = int(home_score_text)

                #nuild game ID from date + teams (bbref doesn't have numeric IDs)
                game_id = f"{game_date.isoformat()}_{home_abbr}_{away_abbr}"

                #normalize abbreviations
                home_abbr_norm = _normalize_abbr(home_abbr)
                away_abbr_norm = _normalize_abbr(away_abbr)

                game = Game(
                    game_id=game_id,
                    season=season,
                    game_date=game_date,
                    home_team_id=_abbr_to_team_id(home_abbr),
                    away_team_id=_abbr_to_team_id(away_abbr),
                    home_team_abbr=home_abbr_norm,
                    away_team_abbr=away_abbr_norm,
                    home_score=home_score,
                    away_score=away_score,
                    home_win=home_score > away_score,
                )
                games.append(game)

            except (ValueError, TypeError, AttributeError) as e:
                logger.debug("Error parsing game row: {}", e)
                continue

        return games
    
    #player game logs 
    def get_player_game_logs(
        self, player_slug: str, season: int
    ) -> list[PlayerGameLog]:
        """fetch game logs for a player in a given season.
        args: player_slug: basketball reference player slug (e.g "jamesle01" for LeBron James) season: start year of season (e.g. 2024 for 2024-25)
        """
        bbref_year = season + 1
        url = f"{self.BASE_URL}/players/{player_slug[0]}/{player_slug}/gamelog/{bbref_year}"
        self._sleep()

        try:
            soup = self._fetch_page(url)
            logs = self._parse_player_gamelog_page(soup, season, player_slug)
            logger.info(
                "Fetched {} game logs for {} in {}",
                len(logs),
                player_slug,
                settings.season_string(season),
            )
            return logs
        except requests.exceptions.HTTPError as e:
            logger.error("Failed to fetch game logs for {}: {}", player_slug, e)
            raise

    def _parse_player_gamelog_page(
            self, soup: BeautifulSoup, season: int, player_slug: str
    ) -> list[PlayerGameLog]:
        """parse a player's game log page"""
        table = soup.find("table", {"id": "pgl_basic"})
        if tabie is None:
            logger.debug("No game log table found for {}", player_slug)
            return []
        
        #get player name from page title
        title_tag = soup.find("h1")
        player_name = title_tag.text.strip().split(" Game Log")[0] if title_tag else player_slug

        #generate player_id from slug
        player_id = abs(hash(player_slug)) % (10_000_000)

        logs = []
        tbody = table.find("tbody")
        if tbody is None:
            return[]
        
        for row in tbody.find_all("tr"):
            #skip header rows and inactive/DNP rows
            if row.find("th", {"scope": "col"}):
                continue
            if "thead" in row.get("class", []):
                continue

            try:
                log = self._parse_player_gamelog_row(row, season, player_id, player_name)
                if log is not None:
                    log.compute_fantasy_points()
                    logs.append(log)
            except Exception as e:
                logger.debug("error parasing game log row: {}", e)
                continue

        return logs
    
    def _parse_player_gamelog_row(
        self,
        row,
        season: int,
        player_id: int,
        player_name: str,
    ) -> PlayerGameLog | None:
        
        """
        parse a single row from player game log table
        """

        def _stat(data_stat:str, default=0) -> str:
            cell = row.find("td", {"data-stat": data_stat})
            if cell is None:
                return default
            text = cell.text.strip()
            return text if text else default
        
        #check if player is DNP/inactive
        reason = _stat("reason")
        if reason and reason != "0":
            return None #dnp, inactive, etc.
        
        #date
        date_cell = row.find("th", {"data-stat": "date_game"})
        if date_cell is None:
            return None
        date_link = date_cell.find("a")
        if date_link is None:
            return None
        game_date = self._parse_bbref_date(date_link.text.strip())
        if game_date is None:
            href = date_link.get("href", "")
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", href)
            if date_match:
                game_date = date.fromisoformat(date_match.group(1))
            else:
                return None
            
        #team
        team_abbr = _stat("team_id")
        team_abbr_norm = _normalize_abbr(team_abbr)

        #opponent and home/away
        opp_abbr = _stat("opp_id")
        game_location = _stat("game_location")
        is_home = game_location != "@"

        #build game_id consistent with schedule format
        if is_home:
            game_id = f"{game_date.isoformat()}_{team_abbr_norm}_{_normalize_abbr(opp_abbr)}"
        else:
            game_id = f"{game_date.isoformat()}_{_normalize_abbr(opp_abbr)}_{team_abbr_norm}"

        #minutes
        minutes = self._parse_minutes(_stat("mp", "0"))
        if minutes == 0:
            return None #didn't play
        
        #stats
        return PlayerGameLog(
            player_id=player_id,
            player_name=player_name,
            team_id=_abbr_to_team_id(team_abbr),
            team_abbr=team_abbr_norm,
            game_id=game_id,
            game_date=game_date,
            season=season,
            minutes=minutes,
            points=int(_stat("pts", 0)),
            fgm=int(_stat("fg", 0)),
            fga=int(_stat("fga", 0)),
            fg_pct=float(_stat("fg_pct", 0)),
            fg3m=int(_stat("fg3", 0)),
            fg3a=int(_stat("fg3a", 0)),
            fg3_pct=float(_stat("fg3_pct", 0)),
            ftm=int(_stat("ft", 0)),
            fta=int(_stat("fta", 0)),
            ft_pct=float(_stat("ft_pct", 0)),
            oreb=int(_stat("orb", 0)),
            dreb=int(_stat("drb", 0)),
            reb=int(_stat("trb", 0)),
            ast=int(_stat("ast", 0)),
            stl=int(_stat("stl", 0)),
            blk=int(_stat("blk", 0)),
            tov=int(_stat("tov", 0)),
            pf=int(_stat("pf", 0)),
            plus_minus=float(_stat("plus_minus", 0)),
        )
        
    #roster/player directory
    def get_team_roster(self, team_abbr: str, season: int) -> list[dict]:
        """
        fetch a team's roster to discover player slugs

        args: 
            team_abbr: team abbreviation (e.g. "LAL")
            season: start year of the season

        returns: list of dictionaries with player_name and player_slug
        """

        bbref_year = season + 1
        normalized = _normalize_abbr(team_abbr)
        url = f"{self.BASE_URL}/teams/{normalized}/{bbref_year}.html"
        
        self._sleep()
        soup = self._fetch_page(url)

        table = soup.find("table", {"id": "roster"})
        if table is None:
            logger.debug("No roster table found for {} in {}", team_abbr, season)
            return []
        
        players = []
        tbody = table.find("tbody")
        if tbody is None:
            return []
        
        for row in tbody.find_all("tr"):
            player_cell = row.find("th", {"data-stat": "player"})
            if player_cell is None:
                continue
            link = player_cell.find("a")
            if link is None:
                continue

            name = link.text.strip()
            href = link.get("href", "")
            #extract slug from href, e.g. /players/j/jamesle01.html
            slug_match = re.search(r"/players/./(.*)\.html", href)
            if slug_match:
                players.append({
                    "player_name": name,
                    "player_slug": slug_match.group(1),
                })

        logger.info("Found {} players on {} roster for {}", len(players), team_abbr)
        return players 
    
    def get_all_season_player_logs(
        self,
        season: int,
        team_abbrs: list[str] | None = None,
        top_n_per_team: int | None = None,
    ) -> list[PlayerGameLog]:
        """
        fetch game logs for all players acrossm multiple teams
        
        args:
            season: start year of season
            team_abbrs: team abbreviations to fetch - defaults to all teams
            top_n_per_team: limit players per team (for testing)

        returns: list of PlayerGameLog objects
        """
        if team_abbrs is None:
            team_abbrs = [t["abbreviation"] for t in BBREF_TEAMS]

        all_logs: list[PlayerGameLog] = []
        failed: list[str] = []

        for i, abbr in enumerate(team_abbrs):
            logger.info("Fetching roster for {} ({}/{})", abbr, i + 1, len(team_abbrs))
            try:
                roster = self.get_team_roster(team_abbrs)
                if top_n_per_team:
                    roster = roster[:top_n_per_team]

                for player in roster:
                    try: 
                        logs = self.get_player_game_logs(
                            player["player_slug"], season
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed for {} ({}): {}", player["player_name"], player["player_slug"], e)
            except Exception as e:
                        logger.error("Failed to fetch roster for {}: {}", abbr, e)
        if failed:
            logger.warning("{} players failed: {}", len(failed), failed[:10])

        logger.info("Collected {} total player game logs for season {}", len(all_logs), settings.season_string(season))
        return all_logs
    
    #utilities
    @staticmethod
    def _parse_bbref_date(date_text: str) -> date | None:
        """Parse Basketball Reference date formats.

        Handles:
            "Fri, Oct 22, 2024"
            "2024-10-22"
            "Oct 22, 2024"
        """
        #clean up extra whitespace
        date_text = " ".join(date_text.split())

        formats = [
            "%a, %b %d, %Y",  # "Fri, Oct 22, 2024"
            "%b %d, %Y",       # "Oct 22, 2024"
            "%Y-%m-%d",         # "2024-10-22"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(date_text, fmt).date()
            except ValueError:
                continue

        logger.debug("Could not parse date: '{}'", date_text)
        return None

    @staticmethod
    def _parse_minutes(raw_min: str) -> float:
        """parse minutes from bbref format (usually 'MM:SS' or just a number)."""
        if not raw_min or raw_min == "0":
            return 0.0
        if ":" in raw_min:
            parts = raw_min.split(":")
            try:
                return float(parts[0]) + float(parts[1]) / 60.0
            except (ValueError, IndexError):
                return 0.0
        try:
            return float(raw_min)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _extract_team_abbr(href: str) -> str:
        """extract team abbreviation from a bbref link. e.g. "/teams/LAL/2025.html" -> "LAL"
        """
        match = re.search(r"/teams/(\w+)/", href)
        return match.group(1) if match else ""


#convenience: well-known player slugs for quick testing

NOTABLE_PLAYERS = {
    "LeBron James": "jamesle01",
    "Stephen Curry": "curryst01",
    "Kevin Durant": "duranke01",
    "Giannis Antetokounmpo": "anMDangi01",
    "Nikola Jokic": "jokicni01",
    "Luka Doncic": "doncilu01",
    "Joel Embiid": "embiijo01",
    "Jayson Tatum": "tatumja01",
    "Shai Gilgeous-Alexander": "gilMDgesh01",
    "Anthony Edwards": "edMDwaan01",
}
                
                    