import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load data yang telah diproses
data = pd.read_csv("../data_result/data_kendaraan_processed.csv")

# Membuat model Collaborative Filtering menggunakan Cosine Similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['processed_text'])
collaborative_model = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Menyimpan model
joblib.dump(collaborative_model, '../model/collaborative_model.pkl')
