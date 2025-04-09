import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import ast

app = Flask(__name__)

# Configure paths for PythonAnywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Preprocess the data
def preprocess_data(movie_path, credit_path):
    # Load data with error handling
    try:
        movies = pd.read_csv(movie_path)
        credits = pd.read_csv(credit_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Data file not found: {e}")

    # Merge data and select required features
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    # Convert stringified features to list
    def convert(obj):
        try:
            return [i['name'] for i in ast.literal_eval(obj)]
        except (ValueError, SyntaxError):
            return []

    def convert_cast(obj):  # Top 3 cast members.
        try:
            return [i['name'] for i in ast.literal_eval(obj)[:3]]
        except (ValueError, SyntaxError):
            return []

    def get_director(obj):  # Only Director.
        try:
            return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']
        except (ValueError, SyntaxError):
            return []

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

    # Remove spaces and combine tags
    def remove_spaces(words):
        return [str(word).replace(" ", "") for word in words]

    movies['genres'] = movies['genres'].apply(remove_spaces)
    movies['keywords'] = movies['keywords'].apply(remove_spaces)
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

    # Prepare features
    new_df = movies[['movie_id', 'title', 'tags']]

    # Vectorize and calculate similarity
    cv = CountVectorizer(max_features=5000, stop_words='english', binary=True)
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity

# Get recommendations with improved error handling
def recommend(movie_title, new_df, similarity):
    try:
        matches = new_df[new_df['title'].str.lower() == movie_title.lower()]
        if matches.empty:
            return {"error": "Movie not found in database"}, 404
            
        index = matches.index[0]
        distances = similarity[index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return {"recommendations": [new_df.iloc[i[0]].title for i in movie_list]}
    except Exception as e:
        return {"error": str(e)}, 500

# Load data with path handling for PythonAnywhere
try:
    movie_path = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
    credit_path = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
    new_df, similarity = preprocess_data(movie_path, credit_path)
except Exception as e:
    print(f"Failed to load data: {e}")
    new_df, similarity = None, None

# Endpoints with improved error handling
@app.route('/recommend', methods=['GET'])
def recommend_movies():
    if new_df is None:
        return jsonify({"error": "Service unavailable - data not loaded"}), 503
        
    movie_title = request.args.get('title')
    if not movie_title:
        return jsonify({"error": "Missing 'title' parameter"}), 400
        
    return jsonify(recommend(movie_title, new_df, similarity))

@app.route('/')
def index():
    return "Welcome to the Movie Recommendation API!"

# For PythonAnywhere deployment
application = app

if __name__ == "__main__":
    app.run(debug=True)
