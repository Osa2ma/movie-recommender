"""
Movie Recommender Streamlit App
Semantic hybrid recommendation system using sentence transformers and FAISS.
"""
import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_loader import load_movie_data
from utils.embeddings import EmbeddingManager
from utils.recommender import MovieRecommender
from utils.tmdb import TMDBClient


# Page configuration
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def load_data():
    """Load and cache movie data."""
    return load_movie_data()


@st.cache_resource
def initialize_system(_movies_df):
    """Initialize embedding manager and recommender system."""
    # Initialize embedding manager
    emb_manager = EmbeddingManager()
    
    # Build embeddings
    texts = _movies_df['combined'].tolist()
    movie_ids = _movies_df['id'].tolist()
    
    with st.spinner('🎬 Building movie embeddings... This may take a minute on first run.'):
        emb_manager.build_embeddings(texts, movie_ids)
    
    # Initialize recommender
    recommender = MovieRecommender(emb_manager, _movies_df)
    
    # Initialize TMDB client
    try:
        tmdb_client = TMDBClient()
    except Exception as e:
        st.error(f"⚠️ Failed to initialize TMDB client: {str(e)}")
        st.info("Please set TMDB_TOKEN environment variable")
        st.stop()
    
    return recommender, tmdb_client


# Custom CSS
st.markdown("""
<style>
.movie-card {
    background: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.movie-title {font-size: 20px; font-weight: bold; margin-bottom: 8px;}
.movie-rating {color: #ff9800; font-weight: bold;}
.movie-overview {font-size: 14px; margin-top: 8px; line-height: 1.5;}
.movie-genres {font-size: 13px; font-style: italic; color: #666; margin-top: 6px;}
</style>
""", unsafe_allow_html=True)


# Main app
def main():
    """Main application logic."""
    # Load data
    movies_df = load_data()
    
    # Initialize systems
    recommender, tmdb_client = initialize_system(movies_df)
    
    # Header
    st.title("🎬 Movie Recommender")
    st.markdown("### Find movies that blend two favorites together")
    st.divider()
    
    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Settings")
        num_recommendations = st.slider("Number of recommendations", 3, 10, 6)
        st.info("💡 Select two movies to get personalized recommendations based on their combined themes and styles.")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This recommender uses:
        - **Sentence Transformers** for semantic understanding
        - **FAISS** for fast similarity search
        - **Hybrid scoring** combining semantic + genre signals
        """)
    
    # Movie selection
    movie_titles = movies_df['title'].tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_title1 = st.selectbox(
            "🎥 First Movie",
            options=movie_titles,
            help="Choose your first favorite movie"
        )
    with col2:
        selected_title2 = st.selectbox(
            "🎥 Second Movie",
            options=movie_titles,
            help="Choose your second favorite movie"
        )
    
    movie_id1 = movies_df[movies_df['title'] == selected_title1]['id'].values[0]
    movie_id2 = movies_df[movies_df['title'] == selected_title2]['id'].values[0]
    
    # Display selected movies
    st.divider()
    st.subheader("📽️ Your Selected Movies")
    
    cols = st.columns(2)
    for col, movie_id, title in zip(cols, [movie_id1, movie_id2], [selected_title1, selected_title2]):
        movie_row = movies_df[movies_df['id'] == movie_id].iloc[0]
        genres = movie_row['genres_str']
        
        with col:
            with st.container():
                col_img, col_text = st.columns([1, 2])
                with col_img:
                    poster_url = tmdb_client.get_movie_poster(movie_id)
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    else:
                        st.info("No poster")
                with col_text:
                    st.markdown(f"**{movie_row['title']}**")
                    st.markdown(f"⭐ {movie_row['vote_average']}/10")
                    st.markdown(f"🎭 {genres}")
            
            overview = str(movie_row['overview'])
            st.caption(overview[:200] + '...' if len(overview) > 200 else overview)
    
    # Get recommendations
    st.divider()
    if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
        try:
            with st.spinner('🔍 Finding perfect movies for you...'):
                recs = recommender.recommend(movie_id1, movie_id2, top_n=num_recommendations)
            
            st.success(f"✨ Found {len(recs)} great recommendations!")
            st.subheader("🍿 Recommended Movies")
            
            # Display in rows of 3
            for i in range(0, len(recs), 3):
                cols = st.columns(3)
                for col_idx, (_, row) in enumerate(list(recs.iterrows())[i:i+3]):
                    with cols[col_idx]:
                        with st.container():
                            poster_url = tmdb_client.get_movie_poster(row['id'])
                            if poster_url:
                                st.image(poster_url, use_container_width=True)
                            
                            st.markdown(f"**{row['title']}**")
                            st.markdown(f"⭐ {row['vote_average']}/10")
                            st.markdown(f"🎭 *{row['genres_str']}*")
                            
                            with st.expander("📖 Read overview"):
                                st.write(row['overview'])
        
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {str(e)}")


if __name__ == "__main__":
    main()
