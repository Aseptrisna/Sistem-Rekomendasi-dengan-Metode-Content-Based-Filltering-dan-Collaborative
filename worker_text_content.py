import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load raw data
raw_data = pd.read_csv('./data_result/model_motor_preprocessed.csv')

# Preprocessing
raw_data['text_content'] = raw_data['preprocessed_brand'] + ' ' + raw_data['preprocessed_brand'] + ' ' + raw_data['preprocessed_body_type'] + ' ' + \
                            raw_data['preprocessed_price'].astype(str) + ' ' + raw_data['preprocessed_km'].astype(str) + ' ' + \
                            raw_data['preprocessed_vehicle_engine'].astype(str)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(raw_data['text_content'])

# Save processed data
raw_data.to_csv('./data_result/model_motor_preprocessed_content.csv', index=False)
