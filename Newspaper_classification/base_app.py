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

# Function to display information page
def display_information():
    st.info("General Information")
    st.markdown("""
    This web application analyzes news articles by classifying them into predefined categories 
    such as Business, Technology, Sports, Education, and Entertainment. It was developed as a 
    project for a data science course with ExploreAI (now called Sand Technologies).

    **How to Use:**
    - Select the "Prediction" option from the sidebar to classify news articles based on their content.
    - Enter or paste the text of a news article into the text box provided (You can use the headline, description or content).
    - Select the model you would like to use to classify the news article.
    - Click the "Predict" button to see the predicted category of the article.

    **About the Models:**
    - **Logistic Regression:** A linear model that performs well with text data. Accuracy: 98%
    - **Multinomial Naive Bayes:** A probabilistic model suitable for text classification tasks. Accuracy: 97.7%
    - **Support Vector Classifier (SVC):** A model using support vector machines. Accuracy: 97.5%
    """)

# Function to display contact us page
def display_contact_us():
    st.title("Contact Us")
    st.subheader("Project Team Members")

    members = {
        "Jodie Retief": "mojo.retief@gmail.com",
        "Mahlatse Lelosa": "mahlatselelosa98@gmail.com",
        "Mmapaseka Makgatla": "moswazipaseka@gmail.com",
        "Adroit Masingita Hlungwani": "masingitasingita@gmail.com",
        "Melody Msimango": "melodymsimango@gmail.com",
        "Sakhumuzi Mchunu": "sakhumuzimchunu@gmail.com"
    }

    for member, email in members.items():
        st.write(f"**{member}:** {email}")

# Streamlit app
def main():
    st.title('News Article Category Prediction')

    # Sidebar navigation
    page = st.sidebar.selectbox("Choose Option", ["Prediction", "Information", "Contact Us"])

    # Routing based on selection
    if page == "Information":
        display_information()
    elif page == "Contact Us":
        display_contact_us()
    else:  # Default to Prediction page
        st.subheader("Predict the Category of a News Article")
        article_text = st.text_area("Enter the excerpt from the article:", height=200)

        model_choice = st.selectbox("Select Model:", ("Logistic Regression", "Support Vector Classifier", "Multinomial Naive Bayes"))

        if st.button("Predict"):
            if model_choice == "Logistic Regression":
                prediction = predict_category(model_lr, article_text)
            elif model_choice == "Support Vector Classifier":
                prediction = predict_category(model_svc, article_text)
            elif model_choice == "Multinomial Naive Bayes":
                prediction = predict_category(model_nb, article_text)

            st.success(f"Predicted Category: {prediction}")

# Run the app
if __name__ == '__main__':
    main()