"""
Movie recommendation engine using semantic embeddings and hybrid scoring.
"""
import numpy as np
import pandas as pd
from typing import List, Set
from utils.embeddings import EmbeddingManager


class MovieRecommender:
    """Hybrid movie recommendation system combining semantic and metadata signals."""
    
    def __init__(self, embedding_manager: EmbeddingManager, movies_df: pd.DataFrame):
        """
        Initialize recommender.
        
        Args:
            embedding_manager: EmbeddingManager instance
            movies_df: DataFrame containing movie data
        """
        self.emb_manager = embedding_manager
        self.movies_df = movies_df
        self.id_to_index = embedding_manager.id_to_index
    
    def recommend(self, movie_id1: int, movie_id2: int, top_n: int = 6) -> pd.DataFrame:
        """
        Generate movie recommendations based on two seed movies.
        
        Uses a hybrid approach:
        1. Semantic similarity via embeddings (finds movies similar to each seed)
        2. Intersection scoring (prefers movies similar to both seeds)
        3. Genre-aware reranking (bonus for matching genres)
        
        Args:
            movie_id1: First seed movie ID
            movie_id2: Second seed movie ID
            top_n: Number of recommendations to return
        
        Returns:
            DataFrame with recommended movies
        """
        # Validate movie IDs
        if movie_id1 not in self.id_to_index or movie_id2 not in self.id_to_index:
            raise ValueError("One or both movie IDs not found")
        
        idx1 = self.id_to_index[movie_id1]
        idx2 = self.id_to_index[movie_id2]
        
        # Get embeddings
        emb1 = self.emb_manager.embeddings[idx1]
        emb2 = self.emb_manager.embeddings[idx2]
        
        # Search for similar movies separately
        scores1, idxs1 = self.emb_manager.search_similar(emb1, k=40)
        scores2, idxs2 = self.emb_manager.search_similar(emb2, k=40)
        
        # Create candidate pool (union of both searches)
        set1 = set(idxs1)
        set2 = set(idxs2)
        combined = list(set1 | set2)
        
        # Hybrid scoring: use minimum of two similarities (encourages intersection)
        def blended_score(i):
            s1 = scores1[list(idxs1).index(i)] if i in set1 else 0
            s2 = scores2[list(idxs2).index(i)] if i in set2 else 0
            return min(s1, s2)
        
        # Rank candidates
        ranked = sorted(combined, key=blended_score, reverse=True)
        
        # Remove seed movies
        ranked = [i for i in ranked if i not in (idx1, idx2)]
        
        # Get top candidates
        recs = self.movies_df.iloc[ranked[:top_n]].copy()
        
        # Genre-aware reranking
        seed_genres = self._get_combined_genres(idx1, idx2)
        recs["genre_score"] = recs.apply(lambda row: self._genre_bonus(row, seed_genres), axis=1)
        
        return recs[['id', 'title', 'overview', 'vote_average', 'genres_str', 'genre_score']]
    
    def _get_combined_genres(self, idx1: int, idx2: int) -> Set[str]:
        """Get combined genres from two movies."""
        genres1 = str(self.movies_df.loc[idx1, "genres_str"])
        genres2 = str(self.movies_df.loc[idx2, "genres_str"])
        return set((genres1 + " " + genres2).split())
    
    def _genre_bonus(self, row: pd.Series, seed_genres: Set[str]) -> float:
        """Calculate genre overlap bonus."""
        movie_genres = set(str(row["genres_str"]).split())
        overlap = len(movie_genres & seed_genres)
        return overlap / (len(seed_genres) + 1e-9)
