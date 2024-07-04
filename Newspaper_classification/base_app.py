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

def main():
    """News Classifier App with Streamlit """
    
    # Creates a main title and subheader on your page -
    # these are static across all pages
    st.title("News Classifier")
    st.subheader("Analyzing news articles")
    
    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information", "Prediction", "Contact Us"]
    selection = st.sidebar.selectbox("Choose Option", options, index=0)
    
    # Building out the "Information" page
    if selection == "Information":
        st.info("General Information")
        st.markdown("""
        This web application analyzes news articles by classifying them into predefined categories 
        such as Business, Technology, Sports, Education, and Entertainment. It was developed as a 
        project for a data science course with ExploreAI (now called Sand Technologies).

        **How to Use:**
            - Select the "Prediction" option from the sidebar to classify news articles based on their content.
            - Enter or paste the text of a news article into the text box provided (You can use the headline, description or content).
            - Select the model you would like to use to classify the news article.
            - Click the "Classify" button to see the predicted category of the article.
        
            **About the Models:**
            - **Logistic Regression:** A linear model that performs well with text data. Accuracy: 98%
            - **Multinomial Naive Bayes:** A probabilistic model suitable for text classification tasks. Accuracy: 97.7%
            - **Random Forest:** An ensemble learning method that combines multiple decision trees. Accuracy: 97.6%
        """)
    
    # Building out the prediction page
    elif selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")
        
        if st.button("Classify"):
            # Placeholder for model prediction
            st.success("Prediction Placeholder")
    
    # Building out the "Contact Us" page
    elif selection == "Contact Us":
        contact_us()

# Required to let Streamlit instantiate our web app  
if __name__ == '__main__':
    main()