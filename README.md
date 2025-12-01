# 🎬 Movie Recommender

A semantic hybrid movie recommendation system that finds films combining the themes and styles of two movies you love. Built with Sentence Transformers, FAISS, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **Semantic Understanding**: Uses state-of-the-art sentence transformers to understand movie plots and themes
- **Fast Search**: FAISS-powered vector similarity search for instant recommendations
- **Hybrid Scoring**: Combines semantic similarity with genre-aware reranking
- **Beautiful UI**: Modern Streamlit interface with movie posters from TMDB
- **Customizable**: Adjust number of recommendations on the fly

## 🏗️ Architecture

```
┌─────────────────┐
│  User Selects   │
│  Two Movies     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Sentence Transformers Model    │
│  (all-mpnet-base-v2)            │
│  Generates 768-dim embeddings   │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  FAISS Index                    │
│  Fast similarity search         │
│  Returns top-40 candidates each │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Hybrid Scoring                 │
│  • Min(sim1, sim2) for balance  │
│  • Genre overlap bonus          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Top-N Recommendations          │
│  Ranked by combined score       │
└─────────────────────────────────┘
```

## 📁 Project Structure

```
movie-recommender/
├── app/
│   └── app.py                 # Streamlit UI application
├── data/
│   ├── movies.csv            # Movie metadata
│   └── keywords.csv          # Movie keywords
├── models/
│   └── movie_embeddings.npy  # Cached embeddings (generated)
├── notebooks/
│   └── experiments.ipynb     # Evaluation and experiments
├── utils/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── embeddings.py         # Embedding generation and FAISS
│   ├── recommender.py        # Recommendation logic
│   └── tmdb.py              # TMDB API client
├── .env.example              # Environment variables template
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- TMDB API token ([Get one here](https://www.themoviedb.org/settings/api))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Osa2ma/movie-recommender.git
   cd movie-recommender
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Copy the example file
   cp .env
   
   # Edit .env and add your TMDB token
   # TMDB_TOKEN=your_actual_token_here
   ```

5. **Run the application**
   ```bash
   # Set environment variable (Windows PowerShell)
   $env:TMDB_TOKEN = "your_token_here"
   streamlit run app/app.py
   
   # Or on Unix/macOS
   export TMDB_TOKEN="your_token_here"
   streamlit run app/app.py
   ```

6. **Open your browser**
   
   Navigate to `http://localhost:8501`

## 💡 How It Works

### 1. Embedding Generation

Movies are represented as 768-dimensional vectors using the `all-mpnet-base-v2` sentence transformer model. Each movie's embedding captures:
- Plot overview
- Genre information
- Keyword themes

Embeddings are cached in `models/movie_embeddings.npy` for fast subsequent loads.

### 2. Similarity Search

FAISS (Facebook AI Similarity Search) enables lightning-fast cosine similarity search over ~8,000 movies. For each seed movie, we retrieve the top-40 most similar candidates.

### 3. Hybrid Scoring

The recommendation algorithm uses a hybrid approach:

**Semantic Similarity**: 
- Searches separately for each seed movie
- Uses intersection scoring: `min(similarity_to_movie1, similarity_to_movie2)`
- This ensures recommendations are relevant to BOTH movies

**Genre Bonus**:
- Calculates genre overlap with seed movies
- Adds 20% weight to final score
- Encourages genre consistency while allowing surprises

### 4. Result Ranking

Final recommendations are sorted by combined score, excluding the seed movies themselves.


## 📊 Dataset

Uses the TMDB 5000 Movie Dataset containing:
- ~8,000 movies (filtered to >1000 votes)
- Metadata: title, overview, genres, vote average
- Keywords for enhanced semantic understanding

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)**: Web framework for Python
- **[Sentence Transformers](https://www.sbert.net/)**: State-of-the-art text embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)**: Fast similarity search
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation
- **[TMDB API](https://www.themoviedb.org/documentation/api)**: Movie posters and metadata

## 🧪 Development

### Project Organization

- `utils/`: Core logic (modular, testable, reusable)
- `app/`: UI layer only
- `data/`: Raw datasets
- `models/`: Generated artifacts (cached embeddings)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Questions? Open an issue or reach out!

---

**Happy movie discovering! 🍿**
