from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def vectorize(train_texts, test_texts):
    """
    Convert text into TF-IDF features.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Train a Logistic Regression model and evaluate it.
    """
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\n✅ Model Performance:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    return model

def save_model(model, vectorizer, path="sentiment_model.pkl"):
    """
    Save both the model and the vectorizer.
    """
    joblib.dump({"model": model, "vectorizer": vectorizer}, path)
    print(f"\n💾 Model saved to {path}")

def load_model(path="sentiment_model.pkl"):
    """
    Load a saved model and vectorizer.
    """
    data = joblib.load(path)
    return data["model"], data["vectorizer"]
