from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import joblib

app = Flask(__name__)

# Memuat data yang telah diproses
data = pd.read_csv("../../data_result/data_kendaraan_processed.csv")

# Memuat model
tfidf_model = joblib.load('../../model/tfidf_model.pkl')
content_based_model = joblib.load('../../model/content_based_model.pkl')
collaborative_model = joblib.load('../../model/collaborative_model.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    keyword = request.form.get("keyword")
    # Lakukan pencarian menggunakan model Content-Based Filtering
    idx = data[data['processed_text'].str.contains(keyword)].index
    scores = list(enumerate(content_based_model[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = scores[:10]
    content_recommendations = [data.iloc[idx[0]]['unique_id'] for idx, _ in top_scores]
    # Tampilkan hasil pencarian ke dalam template HTML
    results = data[data['unique_id'].isin(content_recommendations)]
    return render_template("search_results.html", results=results)

@app.route("/detail/<unique_id>")
def detail(unique_id):
    # Tampilkan detail kendaraan berdasarkan unique_id
    vehicle = data[data['unique_id'] == unique_id].iloc[0]
    # Berikan rekomendasi menggunakan model Collaborative Filtering
    idx = data[data['unique_id'] == unique_id].index
    scores = list(enumerate(collaborative_model[idx][0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = scores[:10]
    collaborative_recommendations = [data.iloc[idx[0]]['unique_id'] for idx, _ in top_scores]
    recommendations = data[data['unique_id'].isin(collaborative_recommendations)]
    return render_template("detail.html", vehicle=vehicle, recommendations=recommendations)

if __name
