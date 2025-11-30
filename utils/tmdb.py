"""
TMDB API utilities for fetching movie posters and configuration.
"""
import os
import requests
from typing import Tuple, Optional


class TMDBClient:
    """Client for interacting with TMDB API."""
    
    def __init__(self):
        """Initialize TMDB client with API token from environment."""
        self.token = os.getenv("TMDB_TOKEN")
        if not self.token:
            raise ValueError("TMDB_TOKEN environment variable not found")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "accept": "application/json"
        }
        self.base_url = None
        self.poster_sizes = None
        self._configure()
    
    def _configure(self):
        """Fetch TMDB API configuration."""
        url = "https://api.themoviedb.org/3/configuration"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "images" not in data:
                raise ValueError(f"Invalid TMDB API response: {data}")
            
            self.base_url = data["images"]["secure_base_url"]
            self.poster_sizes = data["images"]["poster_sizes"]
        except Exception as e:
            raise RuntimeError(f"Failed to get TMDB configuration: {str(e)}")
    
    def get_movie_poster(self, tmdb_id: int, size: str = "w500") -> Optional[str]:
        """
        Fetch movie poster URL from TMDB API.
        
        Args:
            tmdb_id: TMDB movie ID
            size: Poster size (default: w500)
        
        Returns:
            Poster URL or None if not found
        """
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/images?language=en"
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data.get("posters"):
                poster_path = data["posters"][0]["file_path"]
                return f"{self.base_url}{size}{poster_path}"
        except Exception:
            pass
        
        return None
