"""
logistic_model.py - Baseline Model
TF-IDF + Logistic Regression
Runs on CPU/Laptop, Fast inference
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load merged dataset (train.csv + labeled_data.csv)
df = pd.read_csv("full_mixed_dataset.csv")
print(f"Dataset loaded: {len(df)} rows")
print(df['label'].value_counts())

# Split
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
X_train = vectorizer.fit_transform(X_train_text.astype(str))
X_test = vectorizer.transform(X_test_text.astype(str))

# Train
model = LogisticRegression(max_iter=1000, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred))

# Save models
with open("gali_detector_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Saved: gali_detector_model.pkl, tfidf_vectorizer.pkl")
