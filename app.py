# ---------------------------
# 0. IMPORT STREAMLIT
# ---------------------------
import streamlit as st

# ---------------------------
# 1. IMPORT LIBRARIES
# ---------------------------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np
import ast
import requests

# ---------------------------
# 2. LOAD CSV FILE (CACHED)
# ---------------------------
@st.cache_data
def load_movie_data():
    movies_file = "movies.csv"
    keywords_file = "keywords.csv"
    
    try:
        movies = pd.read_csv(movies_file, encoding='utf-8', low_memory=False)
    except:
        movies = pd.read_csv(movies_file, encoding='latin1', low_memory=False)

    try:
        keywords_df = pd.read_csv(keywords_file, encoding='utf-8')
    except:
        keywords_df = pd.read_csv(keywords_file, encoding='latin1')

    # Keep movies with enough votes
    movies = movies[movies['vote_count'] >= 1000].copy()

    # Clean ID
    movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
    movies = movies.dropna(subset=['id'])
    movies['id'] = movies['id'].astype(int)
    movies = movies.reset_index(drop=True)

    keywords_df['id'] = pd.to_numeric(keywords_df['id'], errors='coerce')
    keywords_df = keywords_df.dropna(subset=['id'])
    keywords_df['id'] = keywords_df['id'].astype(int)
    keywords_df['keywords'] = keywords_df['keywords'].fillna('[]')
    
    return movies, keywords_df

movies, keywords_df = load_movie_data()

# ---------------------------
# 3. CREATE COMBINED FEATURE
# ---------------------------
def clean_genres(genres):
    if pd.isna(genres) or genres == '[]':
        return ''
    if isinstance(genres, str):
        try:
            genre_list = ast.literal_eval(genres)
            return ' '.join([g['name'] for g in genre_list])
        except:
            return ''
    return ''

def clean_keywords(kw):
    if pd.isna(kw) or kw == '[]':
        return ''
    if isinstance(kw, str):
        try:
            kw_list = ast.literal_eval(kw)
            return ' '.join([k['name'] for k in kw_list])
        except:
            return ''
    return ''

movies['genres_str'] = movies['genres'].apply(clean_genres)
keywords_df['keywords_str'] = keywords_df['keywords'].apply(clean_keywords)

# Merge keywords into movies
movies = movies.merge(keywords_df[['id', 'keywords_str']], on='id', how='left')
movies['keywords_str'] = movies['keywords_str'].fillna('')

# Combine overview, genres, and keywords for TF-IDF
movies['combined'] = (
    movies['overview'].fillna('') + ' ' +
    movies['genres_str'].fillna('') + ' ' +
    movies['keywords_str']
)


# ---------------------------
# 4. BUILD MODEL INDEX (CACHED)
# ---------------------------
@st.cache_resource
def load_embeddings_and_index():
    id_to_index = pd.Series(movies.index, index=movies['id']).to_dict()
    
    MODEL = "all-mpnet-base-v2"
    embedder = SentenceTransformer(MODEL)
    
    EMB_FILE = "movie_embeddings.npy"
    
    if os.path.exists(EMB_FILE):
        embeddings = np.load(EMB_FILE)
    else:
        with st.spinner('🎬 Building movie embeddings... This may take a minute on first run.'):
            embeddings = embedder.encode(
                movies['combined'].tolist(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            np.save(EMB_FILE, embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return embeddings, index, id_to_index

embeddings, index, id_to_index = load_embeddings_and_index()

# ---------------------------
# 5. TMDB POSTER FUNCTIONS
# ---------------------------
TMDB_TOKEN = os.getenv("TMDB_TOKEN")
if not TMDB_TOKEN:
    st.error(" TMDB_TOKEN environment variable not found. Please set it and restart the app.")
    st.stop()

headers = {"Authorization": f"Bearer {TMDB_TOKEN}", "accept": "application/json"}

def get_tmdb_config():
    url = "https://api.themoviedb.org/3/configuration"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        if "images" not in data:
            st.error(f" Invalid TMDB API response. Check your TMDB_TOKEN. Response: {data}")
            st.stop()
        base_url = data["images"]["secure_base_url"]
        poster_sizes = data["images"]["poster_sizes"]
        return base_url, poster_sizes
    except Exception as e:
        st.error(f" Failed to get TMDB configuration: {str(e)}")
        st.stop()

@st.cache_data
def get_config():
    return get_tmdb_config()

BASE_URL, POSTER_SIZES = get_config()

@st.cache_data
def get_movie_poster(tmdb_id, size="w500"):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/images?language=en"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if data.get("posters"):
            poster_path = data["posters"][0]["file_path"]
            return f"{BASE_URL}{size}{poster_path}"
    except:
        pass
    return None
# ---------------------------
# 6. RECOMMENDATION FUNCTION (Updated)
# ---------------------------
def recommend_movies(movie_id1, movie_id2, top_n=6):
    if movie_id1 not in id_to_index or movie_id2 not in id_to_index:
        return "Movie ID not found"

    idx1 = id_to_index[movie_id1]
    idx2 = id_to_index[movie_id2]

    emb1 = embeddings[idx1]
    emb2 = embeddings[idx2]

    # search separately
    scores1, idxs1 = index.search(emb1.reshape(1, -1), 40)
    scores2, idxs2 = index.search(emb2.reshape(1, -1), 40)

    set1 = set(idxs1[0])
    set2 = set(idxs2[0])

    # balanced candidate pool
    combined = list(set1 | set2)

    # scoring function
    def blended_score(i):
        s1 = scores1[0][list(idxs1[0]).index(i)] if i in set1 else 0
        s2 = scores2[0][list(idxs2[0]).index(i)] if i in set2 else 0
        return min(s1, s2)


    ranked = sorted(combined, key=blended_score, reverse=True)
    ranked = [i for i in ranked if i not in (idx1, idx2)]

    recs = movies.iloc[ranked[:top_n]].copy()

    # genre-aware rerank
    seed_genres = set(
        (movies.loc[idx1, "genres_str"] + " " + movies.loc[idx2, "genres_str"]).split()
    )

    def genre_bonus(row):
        g = set(str(row["genres_str"]).split())
        return len(g & seed_genres) / (len(seed_genres) + 1e-9)

    recs["score"] = recs.apply(genre_bonus, axis=1)


    return recs[['id','title','overview','vote_average','genres_str']]



# ---------------------------
# 7. STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.centered-text {text-align: center;}
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

st.title("🎬 Movie Recommender")
st.markdown("### Find movies that blend two favorites together")

st.divider()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    num_recommendations = st.slider("Number of recommendations", 3, 10, 6)
    st.info("💡 Select two movies to get personalized recommendations based on their combined themes and styles.")

movie_titles = movies['title'].tolist()

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

movie_id1 = movies[movies['title'] == selected_title1]['id'].values[0]
movie_id2 = movies[movies['title'] == selected_title2]['id'].values[0]

st.divider()
st.subheader("📽️ Your Selected Movies")

cols = st.columns(2)

for col, movie_id, title in zip(cols, [movie_id1, movie_id2], [selected_title1, selected_title2]):
    movie_row = movies[movies['id'] == movie_id].iloc[0]
    genres = movie_row['genres_str']

    with col:
        with st.container():
            col_img, col_text = st.columns([1, 2])
            with col_img:
                poster_url = get_movie_poster(movie_id)
                if poster_url:
                    st.image(poster_url, use_container_width=True)
                else:
                    st.info("No poster available")
            with col_text:
                st.markdown(f"**{movie_row['title']}**")
                st.markdown(f"⭐ {movie_row['vote_average']}/10")
                st.markdown(f"🎭 {genres}")
        st.caption(movie_row['overview'][:200] + '...' if len(str(movie_row['overview'])) > 200 else movie_row['overview'])

st.divider()
if st.button("🎯 Get Recommendations", type="primary", use_container_width=True):
    with st.spinner('🔍 Finding perfect movies for you...'):
        recs = recommend_movies(movie_id1, movie_id2, top_n=num_recommendations)
    
    if isinstance(recs, str):
        st.warning(recs)
    else:
        st.success(f"✨ Found {len(recs)} great recommendations!")
        st.subheader("🍿 Recommended Movies")
        
        # Display in rows of 3
        for i in range(0, len(recs), 3):
            cols = st.columns(3)
            for col_idx, (_, row) in enumerate(list(recs.iterrows())[i:i+3]):
                with cols[col_idx]:
                    with st.container():
                        poster_url = get_movie_poster(row['id'])
                        if poster_url:
                            st.image(poster_url, use_container_width=True)
                        
                        st.markdown(f"**{row['title']}**")
                        st.markdown(f"⭐ {row['vote_average']}/10")
                        st.markdown(f"🎭 *{row['genres_str']}*")
                        
                        with st.expander("📖 Read overview"):
                            st.write(row['overview'])
