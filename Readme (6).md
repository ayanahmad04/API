
# Movie Recommendation API

This project provides a Flask-based API that recommends similar movies using content-based filtering. It uses text data from movie overviews, genres, keywords, cast, and crew to calculate similarities and recommend related titles.

## 🚀 Features

- Recommend top 5 similar movies based on a given movie title
- Fetches movie posters from the OMDb API
- Returns structured JSON responses
- Robust error handling for data loading and API requests

## 📁 Files and Structure

- `merged.py`: Main Flask application script
- `data/tmdb_5000_movies.csv`: Movie metadata
- `data/tmdb_5000_credits.csv`: Cast and crew information

## 🛠 Technologies Used

- Python
- Flask
- Pandas & NumPy
- Scikit-learn
- OMDb API for movie posters

## 🔧 Setup Instructions

1. Install required packages:
    ```bash
    pip install flask pandas numpy scikit-learn nltk requests
    ```

2. Place the dataset files (`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`) in a `data/` directory relative to the script.

3. Run the Flask app:
    ```bash
    python merged.py
    ```

4. Access the API:
    - Home: `http://localhost:5000/`
    - Recommend: `http://localhost:5000/recommend?title=<movie_title>`

## 📥 Example Request

```
GET /recommend?title=Inception
```

### 📤 Example Response

```json
{
  "recommendations": [
    {
      "title": "Interstellar",
      "poster_url": "https://image-url"
    },
    ...
  ]
}
```

## 🧠 How it Works

- Textual data is combined into a "tags" field.
- CountVectorizer encodes the tags.
- Cosine similarity is computed to find similar movies.
- OMDb API fetches poster images for display.

## 📌 Notes

- Ensure a valid OMDb API key is set in the script.
- Works well with properly formatted and clean datasets.

## 📬 API Status

- `/` — Welcome message
- `/recommend?title=...` — Returns top 5 movie recommendations

---

Enjoy building smarter movie recommendations!
