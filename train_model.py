# import pandas as pd
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import IsolationForest
# from nlp.preprocess import clean_text
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score,r2_score
# df = pd.read_csv("dataset/transactions.csv")
# df["cleaned"] = df["message"].apply(clean_text)

# X_train, X_test, y_train, y_test = train_test_split(df["cleaned"], df["category"], test_size=0.2, random_state=42)
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df["cleaned"])
# y = df["category"]
# X_test_vec = vectorizer.transform(X_test)

# clf = MultinomialNB()
# clf.fit(X, y)
# y_pred = clf.predict(X)
# print("Classification Accuracy:", accuracy_score(y, y_pred))
# print("R2 Score:", r2_score(y, y_pred))
# iso = IsolationForest(contamination=0.2)
# iso.fit(df[["amount"]])

# pickle.dump((vectorizer, clf), open("ml_models/classifier.pkl", "wb"))
# pickle.dump(iso, open("ml_models/anomaly.pkl", "wb"))

# print("Models trained successfully")
  # train_model.py


import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/transactions.csv")

# ✅ Correct columns
X_text = df["message"].astype(str)
y = df["category"]

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# Train classifier
clf = MultinomialNB()
clf.fit(X, y)

# Evaluate
y_pred = clf.predict(X)
print("Accuracy:", accuracy_score(y, y_pred)*100)

# Train anomaly model
iso = IsolationForest(contamination=0.2, random_state=42)
iso.fit(df[["amount"]])

# Save models (IMPORTANT FIX)
with open("ml_models/classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, clf), f)

with open("ml_models/anomaly.pkl", "wb") as f:
    pickle.dump(iso, f)

print("Models trained and saved successfully!")