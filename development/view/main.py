from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load preprocessed data
data = pd.read_csv("../../data_result/data_kendaraan_processed.csv")

# Debugging: print column names
# print(data.columns)

# Load models
tfidf_model = joblib.load('../../model/tfidf_model.pkl')
content_based_model = joblib.load('../../model/content_based_model.pkl')
collaborative_model = joblib.load('../../model/collaborative_model.pkl')

# Function to format currency
def format_currency(value):
    try:
        value = float(value)
        return "{:,.2f}".format(value)  # Format value as currency with two decimal places
    except (ValueError, TypeError):
        return "0.00"  # Return "0.00" if value is not valid

# Add the custom filter to Jinja
app.jinja_env.filters['format_currency'] = format_currency


# Function to handle NaN values in 'km' column
def handle_km(value):
    if pd.isnull(value):
        return "0"  # Return "0" if the value is NaN
    else:
        return str(int(value))  # Convert the value to integer and then string

# Add the custom filter to Jinja
app.jinja_env.filters['handle_km'] = handle_km


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    keyword = request.form.get("keyword")
    # Perform search using Content-Based Filtering model
    idx = data[data['processed_text'].str.contains(keyword, case=False, na=False)].index
    content_recommendations = []
    for i in idx:
        similarity_scores = content_based_model[i]
        scores = list(enumerate(similarity_scores))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_scores = scores[:10]
        content_recommendations.extend([data.iloc[j]['unique__id'] for j, _ in top_scores])
    
    content_recommendations = list(set(content_recommendations))  # Remove duplicates
    results = data[data['unique__id'].isin(content_recommendations)]
    # print(results)
    # return render_template("search_results.html", results=results)
    if results.empty:
        return render_template("no_results.html")
    else:
        return render_template("search_results.html", results=results)


@app.route("/detail/<unique__id>")
def detail(unique__id):
    # Display vehicle details based on unique__id
    vehicle = data[data['unique__id'] == unique__id].iloc[0]
    # Provide recommendations using Collaborative Filtering model
    idx = data[data['unique__id'] == unique__id].index[0]
    similarity_scores = collaborative_model[idx]
    scores = list(enumerate(similarity_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_scores = scores[:10]
    collaborative_recommendations = [data.iloc[j]['unique__id'] for j, _ in top_scores]
    recommendations = data[data['unique__id'].isin(collaborative_recommendations)]
    # print(recommendations)
    return render_template("detail.html", vehicle=vehicle, recommendations=recommendations)

# Route to handle Not Found error
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == "__main__":
    app.run(debug=True)

