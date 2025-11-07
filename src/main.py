import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import clean_text
from model import vectorize, train_and_evaluate, save_model
from predict import predict_sentiment

def load_imdb_dataset(data_dir):
    """
    Load IMDB dataset (pos/neg) from folder structure.
    """
    data = {'review': [], 'sentiment': []}
    for split in ['train', 'test']:
        for sentiment in ['pos', 'neg']:
            folder = os.path.join(data_dir, split, sentiment)
            for file in os.listdir(folder):
                if file.endswith(".txt"):
                    with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                        text = f.read()
                        data['review'].append(text)
                        data['sentiment'].append(1 if sentiment == 'pos' else 0)
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("📥 Loading IMDB dataset...")
    df = load_imdb_dataset("data/aclImdb")
    print(f"Loaded {len(df)} reviews.")

    print("\n🧹 Cleaning text data (this may take a minute)...")
    df['cleaned'] = df['review'].apply(clean_text)

    print("\n🔀 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['sentiment'], test_size=0.2, random_state=42
    )

    print("\n🧮 Vectorizing text...")
    X_train_vec, X_test_vec, vectorizer = vectorize(X_train, X_test)

    print("\n🤖 Training model...")
    model = train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test)

    save_model(model, vectorizer)

    # Test predictions
    print("\n🧾 Example predictions:")
    print(predict_sentiment(model, vectorizer, "I absolutely loved this movie, it was fantastic!"))
    print(predict_sentiment(model, vectorizer, "This was a terrible movie, I hated it."))
