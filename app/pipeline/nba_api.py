import time
import logging
from nba_api.stats.endpoints import (LeagueGameFinder, PlayerGameLog as NBAPlayerGameLog,)
from nba_api.stats.library.http import HEADERS
from nba_api.stats.static import players as static_players, teams as static_teams
from app.config import settings

logger = logging.getLogger(__name__)

class nbaAPI:
    def __init__(self):
        self.delay = settings.NBA_API_DELAY
        self.proxy = settings.NBA_API_PROXY

    def sleep(self):
        time.sleep(self.delay)

    #static data
    def get_all_teams(self) -> list[dict]:
        """returns list of all NBA teams from static data. this is bundled locally with package"""
        #todo: implement caching?
        raise NotImplementedError("get_all_teams is not yet implemented")

    def get_all_players(self) -> list[dict]:
        """returns list of all NBA players from static data. this is bundled locally with package"""
        #todo: implement caching?
        raise NotImplementedError("get_all_players is not yet implemented")
    
    #game results
    def get_season_games(self, season: int) -> list[dict]:
        """returns list of all games in a given season, e.g 2024 for the 2024-25 season"""
        #returns a list of dict with keys matching the game model

        #todo: implement using LeagueGameFinder
        raise NotImplementedError("get_season_games is not yet implemented")
    
    #player game logs
    def get_player_game_logs(self, player_id: int, season: int) -> list[dict]:
        """returns list of all game logs for a given player in a given season, e.g 2024 for the 2024-25 season"""
        #returns a list of dict with keys matching the PlayerGameLog model

        #todo: implement using NBAPlayerGameLog
        raise NotImplementedError("get_player_game_logs is not yet implemented")