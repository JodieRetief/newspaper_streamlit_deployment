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
import streamlit as st
import joblib
import pandas as pd

# Load the saved models
model_lr = joblib.load('LogisticRegression_best_model.pkl')
model_svc = joblib.load('SVC_best_model.pkl')
model_nb = joblib.load('MultinomialNB_best_model.pkl')

# Load category mapping dataframe
category_df = pd.read_csv('category_df.csv')  # Adjust the filename if necessary

# Function to predict category
def predict_category(model, text):
    category_id = model.predict([text])[0]
    category = category_df[category_df['category_id'] == category_id]['category'].values[0]
    return category

# Streamlit app
def main():
    st.title('News Article Category Prediction')

    # User input for article excerpt
    article_text = st.text_area('Enter the excerpt from the article:', height=200)

    # Model selection dropdown
    selected_model = st.selectbox('Select Model:', ('Logistic Regression', 'Support Vector Classifier', 'Multinomial Naive Bayes'))

    # Prediction and display
    if st.button('Predict'):
        if selected_model == 'Logistic Regression':
            prediction = predict_category(model_lr, article_text)
        elif selected_model == 'Support Vector Classifier':
            prediction = predict_category(model_svc, article_text)
        elif selected_model == 'Multinomial Naive Bayes':
            prediction = predict_category(model_nb, article_text)

        st.success(f'Predicted Category: {prediction}')

# Run the app
if __name__ == '__main__':
    main()

