import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Baca data kendaraan
data = pd.read_csv("../data/model_motor.csv")

# Preprocessing data
def preprocess_text(text):
    print('Preprocessing text:', text)  # Penanda proses
    try:
        # Tokenisasi
        tokens = word_tokenize(text.lower())
        # Penghapusan stopwords
        stop_words = set(stopwords.words('indonesian'))
        tokens = [word for word in tokens if word not in stop_words]
        # Stemming
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        tokens = [stemmer.stem(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        print('Error:', e)  # Penanganan error
        return ""  # Kembalikan string kosong jika terjadi error

# Mengubah tipe data kolom yang diperlukan menjadi string
data['model'] = data['model'].astype(str)
data['name'] = data['name'].astype(str)
data['year'] = data['year'].astype(str)
data['body_type'] = data['body_type'].astype(str)
data['km'] = data['km'].astype(str)
data['vehicle_engine'] = data['vehicle_engine'].astype(str)

# Menggabungkan kolom menjadi satu teks yang diproses
data['processed_text'] = data['brand'] + ' ' + data['model'] + ' ' + data['name'] + ' ' + data['year'] + ' ' + data['body_type'] + ' ' + data['km'] + ' ' + data['vehicle_engine']

# Penerapan preprocessing dengan memanggil fungsi preprocess_text
data['processed_text'] = data['processed_text'].apply(preprocess_text)

# Simpan data yang telah diproses
data.to_csv("../data_result/data_kendaraan_processed.csv", index=False)
