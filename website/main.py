from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Function to get recommendations
def get_recommendations(query):
    # Load the preprocessed data
    data = pd.read_csv('../data_result/model_motor_preprocessed_content.csv')

    # Fitur Extraction
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['text_content'])

    # Similarity Calculation
    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Load model
    with open('../model/content_based_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Unpack the model tuple
    tfidf_vectorizer_model = model[0]
    content_similarity_model = model[1]

    # Transform the query using the TF-IDF vectorizer from the model
    query_tfidf = tfidf_vectorizer_model.transform([query])

    # Calculate cosine similarity between the query and the data
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)

    # Get indices of top recommendations
    top_indices = similarities.argsort()[0][::-1][:5]

    # Get the recommendations
    recommendations = data.iloc[top_indices]

    return recommendations

# Index route
@app.route('/')
def index():
    return render_template('index.html')

# Recommendation route
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get the search query from the form
    search_query = request.form['search_query']

    # Get recommendations
    recommendations = get_recommendations(search_query)
    print(recommendations)

    # Render the result template with recommendations
    return render_template('result.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
