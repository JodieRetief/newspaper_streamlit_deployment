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
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load best models and vectorizers
best_models = joblib.load('models/model_best_model.pkl')
best_vectorizers = joblib.load('models/model_tfidf_vectorizer.pkl')

# Load category mapping
category_df = pd.read_csv('category_df.csv')

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

        model_names = [model[0] for model in best_models]
        selected_model = st.selectbox('Select Model', model_names)

        if st.button("Classify"):
            try:
                # Find the selected model and vectorizer
                selected_model_info = next(model for model in best_models if model[0] == selected_model)
                selected_vectorizer_info = next(vectorizer for vectorizer in best_vectorizers if vectorizer[0] == selected_model_info[0])

                # Load the selected model and vectorizer
                model = joblib.load(selected_model_info[1])
                vectorizer = joblib.load(selected_vectorizer_info[1])

                # Transform user input with the selected vectorizer
                vect_text = vectorizer.transform([news_text]).toarray()

                # Make prediction using the selected model
                prediction = model.predict(vect_text)

                # Map prediction to category name using category_df
                predicted_category = category_df.loc[prediction[0], 'category']

                # Display prediction
                st.success(f"Text Categorized as: {predicted_category}")

            except Exception as e:
                st.error(f"Error: {e}")

# Required to let Streamlit instantiate our web app
if __name__ == '__main__':
    main()