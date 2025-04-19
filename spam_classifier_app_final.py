
import streamlit as st
import joblib
import string
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from nltk.stem import PorterStemmer

# Load the trained model and TF-IDF vectorizer
try:
    model = joblib.load("naive_bayes_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

# Stopwords list
custom_stopwords = set([...])  # Shortened for brevity

# Text normalization function
stemmer = PorterStemmer()
def normalize_text(text):
    try:
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in custom_stopwords]
        return ' '.join(words)
    except:
        return ""

# Streamlit UI
st.title("📧 Spam Email Classifier")
st.header("✉️ Single Message Analysis")
user_input = st.text_area("Enter your email message:")

if st.button("Classify Message"):
    if user_input.strip():
        try:
            cleaned = normalize_text(user_input)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            st.success(f"🔍 Result: This message is classified as **{prediction.upper()}**")
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
    else:
        st.warning("Please enter a message to classify.")

st.markdown("---")
st.header("📂 Batch Email Analysis via CSV")
file = st.file_uploader("Upload a CSV file containing a 'Message' column", type=["csv"])

if file is not None:
    try:
        df = pd.read_csv(file)
        if 'Message' not in df.columns:
            st.error("⚠️ The file does not contain a 'Message' column.")
        else:
            df = df.dropna(subset=['Message'])
            df['Normalized'] = df['Message'].apply(normalize_text)
            vectors = vectorizer.transform(df['Normalized'])
            df['Prediction'] = model.predict(vectors)
            st.success("✅ Classification completed successfully")
            st.dataframe(df[['Message', 'Prediction']])

            st.subheader("📊 Prediction Distribution")
            fig, ax = plt.subplots()
            df['Prediction'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
            ax.set_title("Message Counts by Class")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            if 'Category' in df.columns:
                st.subheader("📉 Confusion Matrix")
                cm = confusion_matrix(df['Category'], df['Prediction'], labels=['ham', 'spam'])
                fig2, ax2 = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax2)
                ax2.set_xlabel('Predicted')
                ax2.set_ylabel('Actual')
                ax2.set_title('Confusion Matrix')
                st.pyplot(fig2)

            csv_download = df[['Message', 'Prediction']].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions as CSV", data=csv_download, file_name="classified_messages.csv", mime='text/csv')
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
