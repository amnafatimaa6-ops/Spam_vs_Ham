import streamlit as st
from spam_model import predict_message, get_metrics, get_dataset

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Spam vs Ham Classifier", layout="centered")
st.title("Spam vs Ham Classifier")

# ----------------------------
# Load Dataset & Metrics
# ----------------------------
df = get_dataset()
metrics = get_metrics()

# ----------------------------
# Display Model Accuracy
# ----------------------------
st.markdown(f"### Model Accuracy: **{metrics['accuracy']}%**")

# ----------------------------
# Dataset Distribution (Simple Bars)
# ----------------------------
st.markdown("### Dataset Distribution")
ham_count = df[df['label'] == 'ham'].shape[0]
spam_count = df[df['label'] == 'spam'].shape[0]

st.progress(ham_count / (ham_count + spam_count))
st.write(f"Ham Messages: {ham_count}")
st.progress(spam_count / (ham_count + spam_count))
st.write(f"Spam Messages: {spam_count}")

# ----------------------------
# User Input Section
# ----------------------------
st.markdown("### Predict a Message")
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
    st.markdown("**Selected Message:**")
    st.info(selected_message)
    if st.button("Predict Selected Message"):
        label, confidence = predict_message(selected_message)
        color = "green" if label == "Ham" else "red"
        st.markdown(f"**Prediction:** <span style='color:{color}; font-weight:bold'>{label}</span>", unsafe_allow_html=True)
        st.info(f"Confidence: {confidence}%")
