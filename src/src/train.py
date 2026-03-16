import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import load_data, preprocess


def train():

    data = load_data("data/news.csv")

    X, y = preprocess(data)

    vectorizer = TfidfVectorizer(stop_words="english")

    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Model Accuracy:", accuracy)

    pickle.dump(model, open("model/model.pkl", "wb"))
    pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

    print("Model saved successfully")


if __name__ == "__main__":
    train()
