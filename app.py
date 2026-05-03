import matplotlib
matplotlib.use('Agg')  

from flask import Flask, render_template, request
import sqlite3, pickle, re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nlp.preprocess import clean_text

app = Flask(__name__)

# ---------- LOAD MODELS ----------
vectorizer, clf = pickle.load(open("ml_models/classifier.pkl", "rb"))
iso = pickle.load(open("ml_models/anomaly.pkl", "rb"))


# pip DATABASE
def get_db():
    return sqlite3.connect("database/expense.db")


def create_table():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT,
            amount INTEGER,
            category TEXT,
            anomaly TEXT
        )
    """)
    conn.commit()
    conn.close()


# -HOME
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["message"]

        numbers = re.findall(r'\d+', text)
        if not numbers:
            return "❌ Please enter a valid transaction with amount"

        amount = int(numbers[0])
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        category = clf.predict(X)[0]
        raw_anomaly = iso.predict([[amount]])[0]
        anomaly = "Anomaly" if raw_anomaly == -1 else "Normal"

        conn = get_db()
        conn.execute(
            "INSERT INTO expenses (message, amount, category, anomaly) VALUES (?,?,?,?)",
            (text, amount, category, anomaly)
        )
        conn.commit()
        conn.close()

    return render_template("index.html")


# DASHBOARD 
@app.route("/dashboard")
def dashboard():
    conn = get_db()
    df = pd.read_sql("SELECT * FROM expenses", conn)
    conn.close()

    if not df.empty:
        plt.figure(figsize=(12, 6))  # wider for spacing

        sns.countplot(
            x="category",
            data=df,
            order=df["category"].value_counts().index,
            palette="viridis"
        )

        plt.title("Spending by Category", fontsize=16, fontweight='bold')
        plt.xlabel("Category", fontsize=12)
        plt.ylabel("Number of Transactions", fontsize=12)

        # 🔥 rotate labels (VERY IMPORTANT)
        plt.xticks(rotation=40, ha='right')

        # 🔥 add values on top of bars
        counts = df["category"].value_counts()
        for i, v in enumerate(counts):
            plt.text(i, v + 0.2, str(v), ha='center')

        plt.tight_layout()
        plt.savefig("static/category.png")
        plt.close()

    return render_template(
        "dashboard.html",
        tables=df.to_html(classes="table table-striped"),
        img="category.png"
    )

# ---------- RUN ----------
if __name__ == "__main__":
    create_table()
    app.run(debug=True)
