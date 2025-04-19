
# 📧 Spam Classifier Web App (Multinomial Naive Bayes)

A Streamlit-based web app for classifying email messages as spam or ham using Multinomial Naive Bayes.

## Features
- Classify a single message.
- Upload a CSV file for batch processing.
- Visualize prediction distribution and confusion matrix.
- Download results as CSV.

## Files
- `spam_classifier_app_final.py` - Main Streamlit application.
- `naive_bayes_model.pkl` - Pre-trained classification model.
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer used during training.
- `requirements.txt` - Required Python packages.
- `README.md` - Project description.

## To Run Locally
1. Install Streamlit and dependencies:
```bash
pip install -r requirements.txt
```
2. Run the app:
```bash
streamlit run spam_classifier_app_final.py
```
