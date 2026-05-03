import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

from nlp.preprocess import clean_text

df = pd.read_csv("dataset/transactions.csv")
df = df.dropna(how='all')

# Fill missing values
df["message"] = df["message"].fillna("")
df["category"] = df["category"].fillna("unknown")
df["amount"] = df["amount"].fillna(0)

# Remove rows with empty message
df = df[df["message"].str.strip() != ""]

# convert amount to numeric safely
df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
df["message"] = df["message"].astype(str).apply(clean_text)


# FEATURES 
X_text = df["message"]
y = df["category"]
amounts = df["amount"]


#Debug check 
# print("NaN in message:", X_text.isnull().sum())
# print("NaN in category:", y.isnull().sum())
# print("NaN in amount:", amounts.isnull().sum())

# VECTORIZATION 
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words="english"   # improves model
)

X = vectorizer.fit_transform(X_text)
clf = MultinomialNB()
clf.fit(X, y)
y_pred = clf.predict(X)
print("Training Accuracy:", accuracy_score(y, y_pred) * 100)
iso = IsolationForest(
    contamination=0.15,
    random_state=42
)

iso.fit(amounts.values.reshape(-1, 1))
with open("ml_models/classifier.pkl", "wb") as f:
    pickle.dump((vectorizer, clf), f)

with open("ml_models/anomaly.pkl", "wb") as f:
    pickle.dump(iso, f)
print("✅ Models trained and saved successfully!")
