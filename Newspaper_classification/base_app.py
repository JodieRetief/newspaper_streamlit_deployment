"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib
import os

# Data dependencies
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your raw data
# Example: raw = pd.read_csv("streamlit/train.csv")

# Define function to load models and vectorizers
def load_models():
    model_file = "list_best_model.pkl"
    vectorizer_file = "list_tfidf_vectorizer.pkl"
    
    model = joblib.load(open(os.path.join("Streamlit", model_file), "rb"))
    vectorizer = joblib.load(open(os.path.join("Streamlit", vectorizer_file), "rb"))
    
    return model, vectorizer

# The main function where we will build the actual app
def main():
    """News Classifier App with Streamlit """
    
    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Analyzing news articles")
    
    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)
    
    # Load models and vectorizers
    model, vectorizer = load_models()
    
    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")
    
    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")
        
        if st.button("Classify"):
            # Transform user input with vectorizer
            vect_text = vectorizer.transform([news_text])
            
            # Predict using model
            prediction = model.predict(vect_text)
            
            # Display prediction
            st.success(f"Text Categorized as: {prediction[0]}")
    
# Required to let Streamlit instantiate our web app  
if __name__ == '__main__':
    main()
