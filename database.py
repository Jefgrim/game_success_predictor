import sqlite3
from datetime import datetime

DB_NAME = "predictions.db"

def create_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            game_name TEXT,
            price_original REAL,
            discount REAL,
            win INTEGER,
            mac INTEGER,
            linux INTEGER,
            prediction INTEGER,
            probability REAL
        )
    """)

    conn.commit()
    conn.close()


def insert_prediction(game_name, price_original, discount, win, mac, linux, prediction, probability):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        INSERT INTO prediction_history 
        (timestamp, game_name, price_original, discount, win, mac, linux, prediction, probability)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        game_name,
        price_original,
        discount,
        win,
        mac,
        linux,
        prediction,
        probability
    ))

    conn.commit()
    conn.close()


def fetch_predictions():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT * FROM prediction_history ORDER BY id DESC")
    rows = c.fetchall()

    conn.close()
    return rows