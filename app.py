import streamlit as st
from spam_model import predict_message, get_accuracy, get_dataset

st.set_page_config(page_title="Spam vs Ham Classifier", layout="centered")
st.title("Advanced Spam vs Ham Classifier")

st.markdown(f"### Model Accuracy: **{get_accuracy()}%**")

df = get_dataset()

mode = st.radio("Choose input method:", ["Type a Message", "Pick from Dataset"])

if mode == "Type a Message":
    user_input = st.text_area("Enter your message:")

    if st.button("Predict"):
        if user_input.strip():
            result = predict_message(user_input)
            st.success(f"Prediction: {result}")
        else:
            st.warning("Please enter a message.")

else:
    selected_index = st.selectbox("Select a row number:", df.index)

    selected_message = df.loc[selected_index, "message"]

    st.info(selected_message)

    if st.button("Predict Selected Message"):
        result = predict_message(selected_message)
        st.success(f"Prediction: {result}")
