import pandas as pd
import mysql.connector

df = pd.read_csv("datasets/six.csv", encoding="latin1")

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="restaurant_db"
)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS restaurants (
    name TEXT,
    address TEXT,
    rate FLOAT,
    review_text TEXT,
    reviewer_rating FLOAT
)
""")

for _, row in df.iterrows():
    cursor.execute("""
    INSERT INTO restaurants (name, address, rate, review_text, reviewer_rating)
    VALUES (%s, %s, %s, %s, %s)
    """, tuple(row))

conn.commit()
conn.close()

print("CSV stored successfully in MySQL!")
