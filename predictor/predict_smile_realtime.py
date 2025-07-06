import os
import mlflow.pytorch
import torch
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict, deque
import psycopg2
from psycopg2.extras import execute_values
from sklearn.preprocessing import StandardScaler

# --- Parameters ---
SEQ_LENGTH = 1
MODEL_URI = "mlruns/494802465030627392/models/m-5d52f37a35c442ce890f1be629b0222f/artifacts"
SCALER_PATH = os.path.join(MODEL_URI, "scaler.pkl")
FEATURES = ['log_moneyness', 'dte', 'hour_sin', 'hour_cos', 'minutes_since_open', 'spot_price', 'right_enc']
SOURCE_TABLE = 'option_features'
TARGET_TABLE = 'predicted_smile'

# --- Load model ---
print("📦 Loading model from:", MODEL_URI)
model = mlflow.pytorch.load_model(MODEL_URI)
model.eval()

# --- Load StandardScaler ---
print("📐 Loading trained StandardScaler from:", SCALER_PATH)
scaler = joblib.load(SCALER_PATH)

# --- DB Connection ---
PG_URI = "dbname=volatility_db user=khalil password=MyStrongPass123 host=timescaledb port=5432"
conn = psycopg2.connect(PG_URI)
cursor = conn.cursor()

# --- Get latest timestamp ---
cursor.execute("SELECT MAX(ts_utc) FROM option_features;")
latest_ts = cursor.fetchone()[0]
print(f"🕒 Using only data with ts_utc = {latest_ts}")

# --- Load data ---
df = pd.read_sql(
    f"SELECT * FROM {SOURCE_TABLE} WHERE ts_utc = %s ORDER BY ts_utc ASC",
    conn, params=(latest_ts,)
)

if df.empty:
    print("⚠️ No data found at that timestamp.")
    exit(0)

# --- Predict IVs ---
buffers = defaultdict(lambda: deque(maxlen=SEQ_LENGTH))
predictions = []

for _, row in df.iterrows():
    key = (row["maturity"], row["strike"], row["right"])
    x = [row.get(f, 0) for f in FEATURES]
    buffers[key].append(x)

    if len(buffers[key]) == SEQ_LENGTH:
        X_seq = np.array(buffers[key])
        X_scaled = scaler.transform(X_seq)
        X_tensor = torch.tensor([X_scaled], dtype=torch.float32)
        with torch.no_grad():
            pred_iv = model(X_tensor).item()

        predictions.append((
            row["ts_utc"],
            row["maturity"],
            row["strike"],
            row["right"],
            pred_iv
        ))
        print(f"[✔] Predicted IV for {key}: {pred_iv:.4f}")

# --- Insert to DB ---
if predictions:
    print(f"🗑️ Deleting old predictions for ts_utc = {latest_ts}")
    cursor.execute(f"DELETE FROM {TARGET_TABLE} WHERE ts_utc = %s;", (latest_ts,))
    conn.commit()

    print(f"📝 Inserting {len(predictions)} predictions into {TARGET_TABLE}")
    insert_query = f"""
        INSERT INTO {TARGET_TABLE} (ts_utc, maturity, strike, "right", predicted_iv)
        VALUES %s
    """
    execute_values(cursor, insert_query, predictions)
    conn.commit()
    print("✅ Insertion done.")
else:
    print("⚠️ No sequences reached SEQ_LENGTH; no predictions written.")

cursor.close()
conn.close()
