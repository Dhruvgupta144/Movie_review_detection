import pickle as pk
import streamlit as st

st.title("🎬 Movie Review Sentiment Classifier")
st.write("Hello! I am Dhruv Gupta and this was my first Machine Learning project.")

# Load vectorizer and model from relative paths
with open("vectorizer.pkl", 'rb') as f:
    vectorizer = pk.load(f)

with open("dhruv_svc.pkl", 'rb') as f:
    model = pk.load(f)

# Input
review = st.text_input('Enter Movie Review:')

if st.button('Predict'):
    if review:
        X_new = vectorizer.transform([review])
        result = model.predict(X_new)
        if result[0] == 0:
            st.write("Negative Review")
        else:
            st.write("Positive Review")
    else:
        st.write("Please enter a review to predict.")
