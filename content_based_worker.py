import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the preprocessed data
data = pd.read_csv('./data_result/model_motor_preprocessed_content.csv')

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['text_content'])

# Calculate cosine similarity matrix
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save the TF-IDF vectorizer and cosine similarity matrix to a file
with open('./model/content_based_model.pkl', 'wb') as f:
    pickle.dump((tfidf_vectorizer, content_similarity), f)