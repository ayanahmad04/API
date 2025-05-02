import os
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import requests as rq
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import ast

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')


OMDB_API_KEY = "67554e89"
OMDB_BASE_URL = "http://www.omdbapi.com/"

DEFAULT_POSTER_URL = "https://bit.ly/3Wp9XhG"


def preprocess_data(movie_path, credit_path):
    try:
        movies = pd.read_csv(movie_path)
        credits = pd.read_csv(credit_path)
    except FileNotFoundError as e:
        raise RuntimeError(f"Data file not found: {e}")


    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    def safe_literal_eval(obj):
        try:
            return ast.literal_eval(obj)
        except (ValueError, SyntaxError):
            return []

    def convert(obj):
        return [i['name'] for i in safe_literal_eval(obj) if isinstance(i, dict) and 'name' in i]

    def convert_cast(obj):  
        return [i['name'] for i in safe_literal_eval(obj)[:3] if isinstance(i, dict) and 'name' in i]

    def get_director(obj): 
        return [i['name'] for i in safe_literal_eval(obj) if isinstance(i, dict) and 'job' in i and i['job'] == 'Director']

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])

    def remove_spaces(words):
        return [str(word).replace(" ", "") for word in words if isinstance(word, str)]

    movies['genres'] = movies['genres'].apply(remove_spaces)
    movies['keywords'] = movies['keywords'].apply(remove_spaces)
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower() if isinstance(x, list) else "")
    
    new_df = movies[['movie_id', 'title', 'tags']]


    cv = CountVectorizer(max_features=5000, stop_words='english', binary=True)
    vectors = cv.fit_transform(new_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return new_df, similarity

def recommend(movie_title, new_df, similarity):
    try:
        matches = new_df[new_df['title'].str.lower() == movie_title.lower()]
        if matches.empty:
            return {"error": f"Movie '{movie_title}' not found in database"}, 404

        index = matches.index[0]
        distances = similarity[index]
        movie_indices = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_titles = [new_df.iloc[i[0]]['title'] for i in movie_indices]  

        posters = get_movie_posters(recommended_titles)

        recommendations = [
            {
                "title": title,
                "poster_url": posters.get(title, DEFAULT_POSTER_URL) 
            }
            for title in recommended_titles
        ]

        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}, 500

try:
    movie_path = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
    credit_path = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")
    new_df, similarity = preprocess_data(movie_path, credit_path)
except Exception as e:
    print(f"Failed to load data: {e}")
    new_df, similarity = None, None

def get_movie_posters(movie_titles):
    """Fetch poster URLs for multiple movies from the OMDb API with error handling"""
    posters = {}
    for title in movie_titles:
        try:
            response = rq.get(
                f"{OMDB_BASE_URL}?apikey={OMDB_API_KEY}&t={title}&type=movie&r=json",
                timeout=5  
            )
            response.raise_for_status() 
            data = response.json()

            if data and data.get('Poster') and data['Poster'] != "N/A":
                posters[title] = data['Poster']
            else:
                posters[title] = DEFAULT_POSTER_URL
        except rq.exceptions.RequestException as e:
            print(f"Error fetching poster for {title} from OMDb: {e}")
            posters[title] = DEFAULT_POSTER_URL
        except (ValueError, KeyError) as e:
            print(f"Error parsing OMDb response for {title}: {e}")
            posters[title] = DEFAULT_POSTER_URL
        except Exception as e:
            print(f"Unexpected error fetching poster for {title} from OMDb: {e}")
            posters[title] = DEFAULT_POSTER_URL
    return posters

@app.route('/recommend', methods=['GET'])
def recommend_movies():
    if new_df is None or similarity is None:
        return jsonify({"error": "Service unavailable - data loading failed"}), 503

    movie_title = request.args.get('title')
    if not movie_title:
        return jsonify({"error": "Missing 'title' parameter"}), 400

    return jsonify(recommend(movie_title, new_df, similarity))

@app.route('/')
def index():
    return "Welcome to the Movie Recommendation API!"

application = app

if __name__ == "__main__":
    app.run(debug=True)
