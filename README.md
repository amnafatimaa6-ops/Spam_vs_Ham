# Spam_vs_Ham

## 🚀 Live Demo  
👉 https://spamvsham-o9hmruazeddpmfmmwcgwcc.streamlit.app/

---

## 🎥 Demo

### Demo 1
![Spam vs Ham Classifier Demo](Spam%20vs%20Ham%20Classifier%20demo.gif)

### Demo 2
![Spam vs Ham Classifier Demo 2](Spam%20vs%20Ham%20Classifier%20gif2.demo.gif)


## 📖 Overview

The **Spam vs Ham Classifier** is a machine learning project built using **Python** and **Streamlit**. It classifies text messages as **Spam** or **Ham (Not Spam)** using a **Logistic Regression model** with **TF-IDF vectorization**.

This tool is ideal for detecting unwanted messages and can be extended to:
- Email filtering  
- SMS applications  
- Any text-based spam detection system  

---

## ✨ Features

✅ Predict whether a message is Spam or Ham  
✅ Input custom messages manually  
✅ Select sample messages from dataset (`spam.csv`)  

---

## 📂 Dataset

The project uses the **SMS Spam Collection dataset (`spam.csv`)**:

- **v1** → Label (`ham` or `spam`)  
- **v2** → Message text  

---

## 🧹 Preprocessing

- Converts text to lowercase  
- Removes numbers and punctuation  
- Removes extra whitespaces  

---

## 🤖 Model

- **Algorithm:** Logistic Regression (with balanced class weights)  
- **Vectorization:** TF-IDF (unigrams + bigrams)  
- **Train/Test Split:** 80/20 with stratification  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- Pandas  
- NumPy  

---

