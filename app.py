import pickle as pk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Streamlit UI
st.write("Hello! I am Dhruv gupta and this was my first Machine Learning project :")
review = st.text_input('Enter Movie Review:')

with open("vectorizer.pkl", 'rb') as f:
    vectorizer = pk.load(f)

with open("dhruv_svc.pkl", 'rb') as f:
    model = pk.load(f)

if st.button('Predict'):
    if review:  
        X_new = vectorizer.transform([review])
        result = model.predict(X_new)
        if result[0]==0:
            st.write("Negative Review")
        else:
            st.write("positive Review")

    else:
        st.write("Please enter a review to predict.")
