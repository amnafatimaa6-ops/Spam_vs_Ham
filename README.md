#Spam_vs_Ham
Live Demo: Try it online

Overview

The Spam vs Ham Classifier is a machine learning project built with Python and Streamlit. It classifies text messages as Spam or Ham (Not Spam) using a Logistic Regression model and TF-IDF vectorization.

This tool is ideal for detecting unwanted messages and can be extended to email filtering, SMS apps, or any text-based spam detection system.

Features

✅ Predict whether a message is Spam or Ham.

✅ Input a message manually or select from the dataset (spam.csv).

✅ Displays model metrics: Accuracy, Precision, Recall, F1 Score.

✅ Visualizes dataset distribution (Ham vs Spam).

✅ Shows a confusion matrix for model predictions.

✅ Interactive and visually appealing UI with Streamlit.

Dataset

The project uses the SMS Spam Collection dataset (spam.csv) with two columns:

v1 → Label: "ham" or "spam".

v2 → Message text.

Preprocessing:

Converts text to lowercase.

Removes numbers and punctuation.

Removes extra whitespaces.

Model

Algorithm: Logistic Regression (balanced class weights)

Vectorization: TF-IDF with unigrams and bigrams

Train/Test Split: 80/20 split with stratification

Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
