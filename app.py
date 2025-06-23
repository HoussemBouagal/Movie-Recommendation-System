from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
import random

# === Détection automatique du chemin du dossier de base ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "hybrid_recommender_model.h5")
ENCODINGS_PATH = os.path.join(BASE_DIR, "model", "encodings.pkl")
MOVIES_PATH = os.path.join(BASE_DIR, "ml-1m", "movies.dat")
RATINGS_PATH = os.path.join(BASE_DIR, "ml-1m", "ratings.dat")

# === Charger le modèle et les encodages ===
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(ENCODINGS_PATH, "rb") as f:
    encodings = pickle.load(f)

user2user_encoded = encodings["user2user_encoded"]
movie2movie_encoded = encodings["movie2movie_encoded"]
movie_encoded2movie = encodings["movie_encoded2movie"]
mlb = encodings["mlb"]

# === Charger les données des films ===
movies_df = pd.read_csv(
    MOVIES_PATH,
    sep="::", engine="python",
    names=["movieId", "title", "genres"],
    encoding="latin-1"
)
movies_df.dropna(subset=["movieId", "title", "genres"], inplace=True)

# === Charger les scores réels (moyenne des notes) ===
ratings_df = pd.read_csv(
    RATINGS_PATH,
    sep="::", engine="python",
    names=["userId", "movieId", "rating", "timestamp"]
)
movie_scores = ratings_df.groupby("movieId")["rating"].mean().to_dict()

# === Créer l'application Flask ===
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def recommend():
    movies = []
    error = None

    try:
        user_id = random.choice(list(user2user_encoded.keys()))  # Auto-select user
        genres_input = ""
        if request.method == "POST":
            genres_input = request.form["genres"].strip()

        user_encoded = user2user_encoded[user_id]

        if genres_input:
            filtered_movies = movies_df[movies_df["genres"].str.contains(genres_input, case=False, na=False)].copy()
        else:
            filtered_movies = movies_df.copy()

        genres_list = genres_input.split('|') if genres_input else []
        genre_input = mlb.transform([genres_list]) if genres_list else np.zeros((1, len(mlb.classes_)))

        results = []
        for _, row in filtered_movies.iterrows():
            mid = int(row["movieId"])
            if mid not in movie2movie_encoded:
                continue
            movie_encoded = movie2movie_encoded[mid]

            pred = model.predict([
                np.array([user_encoded]),
                np.array([movie_encoded]),
                genre_input
            ], verbose=0)

            rating = float(pred[0][0]) * 5
            score = movie_scores.get(mid, 0.0)

            results.append({
                "movieId": mid,
                "title": row["title"],
                "genres": row["genres"],
                "score": f"{score:.2f}",
                "pred_rating": f"{rating:.2f}"
            })

        movies = sorted(
            results,
            key=lambda x: (-float(x["pred_rating"]), -float(x["score"]))
        )

    except Exception as e:
        error = f"⚠️ Une erreur s'est produite : {str(e)}"

    return render_template("index.html", movies=movies, error=error)

if __name__ == "__main__":
    app.run(debug=True)
