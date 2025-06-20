import pickle as pk
import streamlit as st
import base64

# Load GIF as base64 to embed it in the background
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(gif_file):
    bin_str = get_base64(gif_file)
    page_bg_img = f'''
    <style>
    .stApp {{
      background-image: url("data:image/gif;base64,{bin_str}");
      background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background
set_background('cinema.gif')

# Titles
st.markdown("<h1 style='color:rgb(255,50,50); white-space: nowrap; '>🎬 Movie Review Sentiment Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='color:rgb(0,0,0); font-size:20px; background-color:white;padding: 10px; border-radius: 10px;'>Hello! I am Dhruv Gupta and this was my first Machine Learning project.</h1>", unsafe_allow_html=True)
st.markdown("""
    <style>
    label {
        color: red !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    input[type="text"] {
        color: white !important;
        background: linear-gradient(135deg, #ff4d4d, #ff0000) !important;
        font-weight: bold !important;
        font-size: 15px !important;
        border: 2px solid #ff0000 !important;
        border-radius: 10px !important;
        padding: 10px !important;
        box-shadow: 0 0 10px #ff0000, 0 0 20px #ff0000, 0 0 30px #ff0000;
    }
    </style>
""", unsafe_allow_html=True)

review = st.text_input("Enter Movie Review:")

# Load model and vectorizer
with open("vectorizer.pkl", 'rb') as f:
    vectorizer = pk.load(f)

with open("dhruv_svc.pkl", 'rb') as f:
    model = pk.load(f)


# Prediction
if st.button('Predict'):
    if review:
        X_new = vectorizer.transform([review])
        result = model.predict(X_new)
        if result[0] == 0:
            st.error("Negative Review 😞")
        else:
            st.success("Positive Review 😄")
    else:
        st.warning("Please enter a review to predict.")
