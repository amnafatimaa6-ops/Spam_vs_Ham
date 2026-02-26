import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin1")

# Keep only required columns (common Kaggle spam dataset format)
if "v1" in df.columns and "v2" in df.columns:
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]

# Convert labels to numeric
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

def predict_message(message):
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]
    return "Spam" if prediction == 1 else "Ham"
