"""
Data loading and preprocessing utilities.
"""
import pandas as pd
import ast
from typing import Tuple


def clean_genres(genres: str) -> str:
    """
    Parse and clean genre JSON string.
    
    Args:
        genres: JSON string of genres
    
    Returns:
        Space-separated genre names
    """
    if pd.isna(genres) or genres == '[]':
        return ''
    if isinstance(genres, str):
        try:
            genre_list = ast.literal_eval(genres)
            return ' '.join([g['name'] for g in genre_list])
        except:
            return ''
    return ''


def clean_keywords(keywords: str) -> str:
    """
    Parse and clean keywords JSON string.
    
    Args:
        keywords: JSON string of keywords
    
    Returns:
        Space-separated keyword names
    """
    if pd.isna(keywords) or keywords == '[]':
        return ''
    if isinstance(keywords, str):
        try:
            kw_list = ast.literal_eval(keywords)
            return ' '.join([k['name'] for k in kw_list])
        except:
            return ''
    return ''


def load_movie_data(movies_path: str = "data/movies.csv", 
                   keywords_path: str = "data/keywords.csv",
                   min_votes: int = 1000) -> pd.DataFrame:
    """
    Load and preprocess movie data.
    
    Args:
        movies_path: Path to movies CSV
        keywords_path: Path to keywords CSV
        min_votes: Minimum vote count threshold
    
    Returns:
        Preprocessed DataFrame with combined features
    """
    # Load movies
    try:
        movies = pd.read_csv(movies_path, encoding='utf-8', low_memory=False)
    except:
        movies = pd.read_csv(movies_path, encoding='latin1', low_memory=False)
    
    # Load keywords
    try:
        keywords_df = pd.read_csv(keywords_path, encoding='utf-8')
    except:
        keywords_df = pd.read_csv(keywords_path, encoding='latin1')
    
    # Filter by vote count
    movies = movies[movies['vote_count'] >= min_votes].copy()
    
    # Clean IDs
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    movies['id'] = movies['id'].astype(int)
    movies = movies.reset_index(drop=True)
    
    keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce')
    keywords_df = keywords_df.dropna(subset=['id'])
    keywords_df['id'] = keywords_df['id'].astype(int)
    keywords_df['keywords'] = keywords_df['keywords'].fillna('[]')
    
    # Process genres and keywords
    movies['genres_str'] = movies['genres'].apply(clean_genres)
    keywords_df['keywords_str'] = keywords_df['keywords'].apply(clean_keywords)
    
    # Merge keywords
    movies = movies.merge(keywords_df[['id', 'keywords_str']], on='id', how='left')
    movies['keywords_str'] = movies['keywords_str'].fillna('')
    
    # Create combined feature for embeddings
    movies['combined'] = (
        movies['overview'].fillna('') + ' ' +
        movies['genres_str'].fillna('') + ' ' +
        movies['keywords_str']
    )
    
    return movies
