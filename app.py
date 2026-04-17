import matplotlib
matplotlib.use('Agg')  # non-GUI backend
from flask import Flask, render_template, request
import sqlite3, pickle, re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nlp.preprocess import clean_text
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
app = Flask(__name__)

vectorizer, clf = pickle.load(open("ml_models/classifier.pkl", "rb"))
iso = pickle.load(open("ml_models/anomaly.pkl", "rb"))

def get_db():
    return sqlite3.connect("database/expense.db")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["message"]
        amount = int(re.findall(r'\d+', text)[0])

        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])
        category = clf.predict(X)[0]
        raw_anomaly = iso.predict(pd.DataFrame([[amount]], columns=["amount"]))[0]
        if raw_anomaly == -1:
            anomaly = "Anomaly"
        else:
            anomaly = "Normal"



        conn = get_db()
        conn.execute(
            "INSERT INTO expenses (message, amount, category, anomaly) VALUES (?,?,?,?)",
            (text, amount, category, anomaly)
        )
        conn.commit()
        conn.close()

    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    conn = get_db()
    df = pd.read_sql("SELECT * FROM expenses", conn)
    conn.close()

    plt.figure()
    sns.countplot(x="category", data=df)
    plt.savefig("static/category.png")
    plt.close()

    return render_template("dashboard.html", tables=df.to_html(), img="category.png")
if __name__ == "__main__":
    create_table()   # 👈 MUST be here
    app.run(debug=True)
