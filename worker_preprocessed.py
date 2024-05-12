import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        # Tokenization
        tokens = word_tokenize(text.lower())

        # Stopword Removal
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        return ' '.join(stemmed_tokens)
    else:
        return ''  # Return empty string for NaN values

# Load data from CSV
data = pd.read_csv('./data/model_motor.csv')

# Define allowed categories for body_type
allowed_categories = ['scooters', 'trail', 'adventure touring', 'naked', 'sport', 'super sport', 'super touring']

# Remove rows with NaN values in body_type column
data = data.dropna(subset=['body_type'])

# Preprocess relevant columns
columns_to_preprocess = ['brand', 'model', 'body_type']
for column in columns_to_preprocess:
    data['preprocessed_' + column] = data[column].apply(preprocess_text)

# Filter data for body_type in allowed categories
data = data[data['body_type'].apply(lambda x: x.lower() in allowed_categories)]

# For numerical columns, keep them as they are
numerical_columns = ['price', 'km', 'vehicle_engine']
for column in numerical_columns:
    data['preprocessed_' + column] = data[column]

# Display preprocessed data
print(data.head())

# Save processed data to new CSV file
data.to_csv('./data_result/model_motor_preprocessed.csv', index=False)
