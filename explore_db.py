import psycopg2
import pandas as pd

conn = psycopg2.connect(
    dbname="volatility_db",
    user="khalil",
    password="MyStrongPass123",
    host="localhost",  # or 'timescaledb' if inside Docker
    port=55432         # or 5432 if not exposed
)

df = pd.read_sql("SELECT * FROM spy_option_chain ORDER BY time DESC LIMIT 10;", conn)
print(df)
