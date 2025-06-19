import pickle as pk
import streamlit as st
import os

st.title("🎬 Movie Review Sentiment Classifier")
st.markdown("Hello! I am **Dhruv Gupta** and this was my first Machine Learning project.")

# Load vectorizer and model safely
try:
    with open("vectorizer.pkl", 'rb') as f:
        vectorizer = pk.load(f)
    with open("dhruv_svc.pkl", 'rb') as f:
        model = pk.load(f)
except FileNotFoundError as e:
    st.error("Model files not found! Please make sure `vectorizer.pkl` and `dhruv_svc.pkl` are in your repo.")
    st.stop()

# Input
review = st.text_input('Enter a movie review:')

# Predict button
if st.button('Predict'):
    if review:
        X_new = vectorizer.transform([review])
        result = model.predict(X_new)
        if result[0] == 0:
            st.error("❌ Negative Review")
        else:
            st.success("✅ Positive Review")
    else:
        st.warning("⚠️ Please enter a review before predicting.")
