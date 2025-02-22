import streamlit as st
import numpy as np
import joblib

# Load pre-trained model and vectorizer
nb_model_bow = joblib.load("nb_model_bow.pkl")  # Ensure the model is saved
bow_vectorizer = joblib.load("bow_vectorizer.pkl")  # Load BoW vectorizer

# Function to classify the input message
def classify_message(message, model, vectorizer):
    message = message.lower().strip()  # Preprocess message
    message_vectorized = vectorizer.transform([message])  # Convert to numerical form
    
    if message_vectorized.nnz == 0:  # Handle unknown words
        return "Unknown (Out-of-Vocabulary)", 0, 0
    
    probabilities = model.predict_proba(message_vectorized)[0]
    spam_prob, ham_prob = probabilities[1], probabilities[0]
    prediction = "Spam" if spam_prob > ham_prob else "Ham"

    return prediction, spam_prob, ham_prob

# Streamlit UI
st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check if it's **Spam or Ham**.")

# Input text box
user_message = st.text_area("Type your message here:")

# Predict button
if st.button("Check Message"):
    if user_message.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        prediction, spam_prob, ham_prob = classify_message(user_message, nb_model_bow, bow_vectorizer)
        
        # Display result
        st.subheader(f"Prediction: **{prediction}**")
        st.write(f"📊 **Spam Probability:** {spam_prob:.4f}")
        st.write(f"📊 **Ham Probability:** {ham_prob:.4f}")
        
        if prediction == "Spam":
            st.error("🚨 This message is classified as Spam!")
        else:
            st.success("✅ This message is classified as Ham.")

# About section
st.markdown("---")
st.write("🔍 **How It Works**")
st.write("""
- This app uses **Naïve Bayes with Bag of Words (BoW)** to classify messages.
- It analyzes the words in your message and assigns a probability of being **Spam or Ham**.
- If the model does not recognize any words, it may return **"Unknown (Out-of-Vocabulary)".**
""")
