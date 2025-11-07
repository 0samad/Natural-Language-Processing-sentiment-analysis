from preprocess import clean_text

def predict_sentiment(model, vectorizer, text):
    """
    Predict sentiment for a given text string.
    """
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Positive 😀" if prediction == 1 else "Negative 😞"
