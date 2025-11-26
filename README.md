# 🎬 IMDB Review Sentiment Analysis using RNN

This project performs **Sentiment Analysis** on IMDB movie reviews using a **Simple Recurrent Neural Network (SimpleRNN)**.  
A Streamlit web application allows users to input any movie review and get a real-time prediction of whether the sentiment is **Positive** or **Negative**.

---

## 🚀 Project Overview

This project uses:

- IMDB dataset (tokenized + integer-encoded)
- Embedding layer for vector representation
- SimpleRNN layer to process sequential text data
- Dense layer for binary classification
- Streamlit UI for user input and interactive predictions
- A trained model (`simple_rnn_imdb.h5`) loaded at runtime

---

## 🧠 Model Architecture

The model follows this structure:

```python
Embedding(input_dim=10000, output_dim=32, input_length=500)
SimpleRNN(32)
Dense(1, activation='sigmoid')
```

This architecture allows the RNN to learn sequential patterns from text and classify sentiment accurately.

---

## 📂 Repository Structure

```
IMDB-Review-Sentiment-Analysis-using-RNN/
│
├── simple_rnn_imdb.h5        # Trained model
├── main.py                   # Streamlit application
├── requirements.txt          # Dependencies for deployment
├── README.md                 # Project documentation
└── images/                   # Screenshots (optional)
```

---

## 📝 Features

✔ IMDB dataset with 50,000 labeled reviews  
✔ Automatic text preprocessing (tokenizing, encoding, padding)  
✔ Word index → word decoding for reverse understanding  
✔ RNN model for sequence learning  
✔ Streamlit UI for real-time sentiment analysis  
✔ Clean prediction interface with scores  

---

## 🧩 How Sentiment Prediction Works

1. User enters a movie review.
2. The text is processed:
   - Converted to lowercase  
   - Split into words  
   - Converted to numerical indices  
   - Padded to maximum length (500 tokens)  
3. The processed input is fed into the RNN model.
4. Model outputs:
   - Value > **0.5** → **Positive**
   - Value ≤ **0.5** → **Negative**

---

## ▶️ Running the Project Locally

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/IMDB-Review-Sentiment-Analysis-using-RNN.git
cd IMDB-Review-Sentiment-Analysis-using-RNN
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Streamlit app**
```bash
streamlit run main.py
```

---

## 📦 Requirements

```
tensorflow==2.12.0
streamlit
numpy
pandas
h5py
```

(Older TF version is required to load `.h5` models correctly.)

---

## 📊 Dataset Information

The IMDB dataset is loaded from:

```python
from tensorflow.keras.datasets import imdb
```

- 25,000 training reviews  
- 25,000 testing reviews  
- Binary sentiment labels (positive/negative)  
- Words are integer-encoded based on frequency  

---

## 🧠 Core Functions

### ✔ Decode Review
```python
def decode_review(encoded_review):
    return ' '.join([reverse_word_idx.get(i-3, '?') for i in encoded_review])
```

### ✔ Preprocess User Input
```python
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review
```

---

## 🌐 Live Demo

🚀 **Streamlit App:**  
https://imdb-review-sentiment-analysis-using-rnn-xp7uhxe8zg9cwmhfpjkw7.streamlit.app/

---

## 🤝 Contributing

Pull requests and improvements are welcome!  
If you find a bug, feel free to open an issue.

---

