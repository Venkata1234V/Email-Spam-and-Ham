import pickle 
import streamlit as st
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer




model = pickle.load(open("model.pkl","rb"))
with open("model1.pkl","rb") as f:
    bow = pickle.load(f)

st.image("th.jpeg")

st.header("EMAIL SPAM OR HAM")
#st.title("Email Spam/Ham Classifier")

# Input email text
Email = st.text_input("Paste the email here:")

# Check if the email input is not empty
if Email:
    # Transform the input email text to feature array
    data = bow.transform([Email]).toarray()

    # Predict if the email is spam or ham
    spam_ham = model.predict(data)[0]

    # Display the prediction when the button is pressed
    if st.button('Submit'):
        if spam_ham == 'spam' :
            
            st.write("The email is:", spam_ham )
            st.image("spam-folder.jpg",width=200)
        else :
            st.write("The email is:", spam_ham )
            st.image("do-not-spam-sign-vector-11313202.jpg",width=200)


