import streamlit as st
from spam_model import predict_message, get_accuracy, get_dataset

st.set_page_config(page_title="Spam vs Ham Classifier", layout="centered")
st.title("📩 Advanced Spam vs Ham Classifier")

st.markdown(f"### Model Accuracy: **{get_accuracy()}%**")

df = get_dataset()

mode = st.radio("Choose input method:", ["Type a Message", "Pick from Dataset"])

if mode == "Type a Message":
    user_input = st.text_area("Enter your message:")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter a message.")
        else:
            result = predict_message(user_input)
            if result == "Spam":
                st.error(" This message is SPAM")
            else:
                st.success(" This message is HAM")

else:
    selected_message = st.selectbox(
        "Select a message from dataset:",
        df["message"].values
    )

    if st.button("Predict Selected Message"):
        result = predict_message(selected_message)
        st.write("### Selected Message:")
        st.info(selected_message)

        if result == "Spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: HAM")
