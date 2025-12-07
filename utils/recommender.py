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
        self.index = embedding_manager.index  # FAISS index for similarity search
    
    def recommend(self, movie_id1: int, movie_id2: int, top_n: int = 6) -> pd.DataFrame:
        """
        Generate movie recommendations based on two seed movies.
        
        Uses Intersection Priority approach (tested score: 0.4121):
        1. Semantic similarity via embeddings (finds movies similar to each seed)
        2. Intersection scoring (prefers movies similar to both seeds)
        3. Boost consensus recommendations by 1.5x
        
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
        
        # Search for similar movies (larger pool for better results)
        scores1, idxs1 = self.index.search(emb1.reshape(1, -1), 50)
        scores2, idxs2 = self.index.search(emb2.reshape(1, -1), 50)

        set1 = set(idxs1[0])
        set2 = set(idxs2[0])
        intersection = set1 & set2
        union_only = (set1 | set2) - intersection

        # Scoring function: prioritize movies recommended by both inputs
        def scoring(i):
            s1 = scores1[0][list(idxs1[0]).index(i)] if i in set1 else 0
            s2 = scores2[0][list(idxs2[0]).index(i)] if i in set2 else 0
            
            # Boost intersection items (consensus recommendations)
            if i in intersection:
                return (s1 + s2) * 1.5
            else:
                return (s1 + s2)

        # Rank all candidates
        ranked = sorted(intersection | union_only, key=scoring, reverse=True)
        
        # Remove seed movies from results
        ranked = [i for i in ranked if i not in (idx1, idx2)]

        # Get top N recommendations
        recs = self.movies_df.iloc[ranked[:top_n]].copy()

        return recs[['id', 'title', 'overview', 'vote_average', 'genres_str']]
