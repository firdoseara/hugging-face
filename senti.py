import streamlit as st
from transformers import pipeline

# Initialize the sentiment analysis pipeline with the specific model
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Streamlit UI
st.title("Sentiment Analysis App")

# Add an input box for the user to enter text
user_input = st.text_area("Enter text for sentiment analysis:")

# Button to trigger sentiment analysis
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text for sentiment analysis.")
    else:
        # Perform sentiment analysis
        result = sentiment_analyzer(user_input)
        
        # Extract sentiment and score
        sentiment = result[0]['label']
        score = result[0]['score']
        
        # Display the result
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.2f}")

