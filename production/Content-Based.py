import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import joblib

# Load data yang telah diproses
data = pd.read_csv("../data_result/data_kendaraan_processed.csv")

# Membuat model Content-Based Filtering menggunakan TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(data['processed_text'])
content_based_model = linear_kernel(tfidf_matrix, tfidf_matrix)

# Menyimpan model
joblib.dump(tfidf, '../model/tfidf_model.pkl')
joblib.dump(content_based_model, '../model/content_based_model.pkl')
