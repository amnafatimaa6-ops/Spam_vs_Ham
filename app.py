import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from spam_model import predict_message, get_metrics, get_dataset, get_confusion_matrix

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Advanced Spam vs Ham Classifier", layout="wide")
st.title(" Advanced Spam vs Ham Classifier")

# ----------------------------
# Load Data & Metrics
# ----------------------------
df = get_dataset()
metrics = get_metrics()
conf_matrix = get_confusion_matrix()

# ----------------------------
# Display Model Metrics
# ----------------------------
st.markdown("### Model Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']}%")
col2.metric("Precision", f"{metrics['precision']}%")
col3.metric("Recall", f"{metrics['recall']}%")
col4.metric("F1 Score", f"{metrics['f1_score']}%")

# ----------------------------
# Dataset Visualization
# ----------------------------
st.markdown("### Dataset Distribution")
fig, ax = plt.subplots(figsize=(6,3))
sns.countplot(x="label", data=df, palette=["#6c5ce7", "#ff4757"], ax=ax)
ax.set_title("Ham vs Spam Distribution")
ax.set_xlabel("Message Type")
ax.set_ylabel("Count")
st.pyplot(fig)

# ----------------------------
# User Input Section
# ----------------------------
st.markdown("###  Predict a Message")
mode = st.radio("Choose input method:", ["Type a Message", "Pick from Dataset"])

if mode == "Type a Message":
    user_input = st.text_area("Enter your message:")

    if st.button("Predict"):
        if user_input.strip():
            label, confidence = predict_message(user_input)
            color = "green" if label == "Ham" else "red"
            st.markdown(f"**Prediction:** <span style='color:{color}; font-weight:bold'>{label}</span>", unsafe_allow_html=True)
            st.info(f"Confidence: {confidence}%")
        else:
            st.warning("Please enter a message.")

else:
    selected_index = st.selectbox("Select a row number:", df.index)
    selected_message = df.loc[selected_index, "message"]

    st.markdown(f"**Selected Message:**")
    st.info(selected_message)

    if st.button("Predict Selected Message"):
        label, confidence = predict_message(selected_message)
        color = "green" if label == "Ham" else "red"
        st.markdown(f"**Prediction:** <span style='color:{color}; font-weight:bold'>{label}</span>", unsafe_allow_html=True)
        st.info(f"Confidence: {confidence}%")

# ----------------------------
# Confusion Matrix Visual
# ----------------------------
st.markdown("### Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(4,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham","Spam"], yticklabels=["Ham","Spam"], ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
    <div style='text-align:center; color:#888; font-size:0.9em;'>
        Spam vs Ham Classifier — Powered by Streamlit & Python
    </div>
    """,
    unsafe_allow_html=True
)
