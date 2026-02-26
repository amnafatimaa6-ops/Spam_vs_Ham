import streamlit as st
from spam_model import predict_message

st.set_page_config(page_title="Spam vs Ham Classifier", layout="centered")
st.title(" Spam vs Ham Classifier")

user_input = st.text_area("Enter your message:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict_message(user_input)
        if result == "Spam":
            st.error(" This message is SPAM")
        else:
            st.success(" This message is HAM (Not Spam)")
