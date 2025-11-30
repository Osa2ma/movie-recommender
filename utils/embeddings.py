"""
Embedding generation and FAISS index management.
"""
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Tuple, Dict


class EmbeddingManager:
    """Manages sentence embeddings and FAISS index for movie recommendations."""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2", cache_file: str = "models/movie_embeddings.npy"):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Sentence transformer model name
            cache_file: Path to cache embeddings file
        """
        self.model_name = model_name
        self.cache_file = cache_file
        self.embedder = SentenceTransformer(model_name)
        self.embeddings = None
        self.index = None
        self.id_to_index = None
    
    def build_embeddings(self, texts: list, movie_ids: list, force_rebuild: bool = False) -> None:
        """
        Build or load embeddings for movie texts.
        
        Args:
            texts: List of combined text features
            movie_ids: List of movie IDs corresponding to texts
            force_rebuild: Force rebuild even if cache exists
        """
        # Create id to index mapping
        self.id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}
        
        # Load cached embeddings if available
        if os.path.exists(self.cache_file) and not force_rebuild:
            print(f"Loading cached embeddings from {self.cache_file}")
            self.embeddings = np.load(self.cache_file)
        else:
            print(f"Generating embeddings for {len(texts)} movies...")
            self.embeddings = self.embedder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            # Save to cache
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            np.save(self.cache_file, self.embeddings)
            print(f"Embeddings saved to {self.cache_file}")
        
        # Build FAISS index
        self._build_index()
    
    def _build_index(self) -> None:
        """Build FAISS index for fast similarity search."""
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
        self.index.add(self.embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors")
    
    def get_embedding(self, movie_id: int) -> np.ndarray:
        """Get embedding vector for a movie ID."""
        if movie_id not in self.id_to_index:
            raise ValueError(f"Movie ID {movie_id} not found")
        idx = self.id_to_index[movie_id]
        return self.embeddings[idx]
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar movies using FAISS.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
        
        Returns:
            Tuple of (scores, indices)
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, k)
        return scores[0], indices[0]
