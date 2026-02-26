import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin1")

# Fix common Kaggle spam dataset structure
if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

# Clean labels
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df["message"] = df["message"].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF with better parameters
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Stronger model than Naive Bayes
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

def predict_message(message):
    message = clean_text(message)
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)[0]
    return "Spam" if prediction == 1 else "Ham"

def get_accuracy():
    return round(accuracy * 100, 2)

def get_dataset():
    return df
