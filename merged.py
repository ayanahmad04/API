from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import ast

app = Flask(__name__)

# Preprocess the data
def preprocess_data(movie_path, credit_path):
    # Load data
    movies = pd.read_csv(movie_path)
    credits = pd.read_csv(credit_path)

    # Merge data and select required features
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    # Convert stringified features to list
    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert_cast(obj):  # Top 3 cast members.
        return [i['name'] for i in ast.literal_eval(obj)[:3]]

    def get_director(obj):  # Only Director.
        return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(get_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    # Remove spaces and combine tags
    def remove_spaces(words):
        return [word.replace(" ", "") for word in words]

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

# Get recommendations
def recommend(movie_title, new_df, similarity):
    try:
        index = new_df[new_df['title'] == movie_title].index[0]
        distances = similarity[index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        return [new_df.iloc[i[0]].title for i in movie_list]
    except IndexError:
        return []

# Load data and preprocess
movie_path = "tmdb_5000_movies.csv"
credit_path = "tmdb_5000_credits.csv"
new_df, similarity = preprocess_data(movie_path, credit_path)

# Endpoints
@app.route('/recommend', methods=['GET'])
def recommend_movies():
    movie_title = request.args.get('title')
    results = recommend(movie_title, new_df, similarity)
    return jsonify(results)

@app.route('/')
def index():
    return "Welcome to the Movie Recommendation API!"

if __name__ == "__main__":
    app.run(debug=True)