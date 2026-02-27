import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("spam.csv", encoding="latin1")

# Keep only relevant columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Map labels to numeric
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# ----------------------------
# Text Cleaning
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df["message"] = df["message"].apply(clean_text)

# ----------------------------
# Train / Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label_num"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_num"]
)

# ----------------------------
# Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------
# Model (Logistic Regression with balanced classes)
# ----------------------------
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)
model.fit(X_train_vec, y_train)

# ----------------------------
# Evaluation Metrics
# ----------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# ----------------------------
# Prediction Function
# ----------------------------
def predict_message(message):
    """Predict if a message is spam or ham and return label + confidence"""
    message = clean_text(message)
    vec = vectorizer.transform([message])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec)[0][prediction]

    label = "Spam" if prediction == 1 else "Ham"
    confidence = round(probability * 100, 2)

    return label, confidence

# ----------------------------
# Getter Functions
# ----------------------------
def get_metrics():
    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2)
    }

def get_confusion_matrix():
    return conf_matrix

def get_dataset():
    return df

def get_test_data():
    return X_test, y_test
