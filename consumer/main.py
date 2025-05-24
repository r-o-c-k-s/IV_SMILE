# consumer/main.py
import os
import psycopg2
from kafka import KafkaConsumer
import json
import time

# Kafka topic
TOPIC_NAME = "spy_option_ticks"

# Database connection config
DB_CONFIG = {
    "dbname": os.environ["DB_NAME"],
    "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASS"],
    "host": os.environ["DB_HOST"],
    "port": os.environ["DB_PORT"]
}

# Retry logic for TimescaleDB connection
def connect_with_retry(retries=10, delay=5):
    for attempt in range(retries):
        try:
            print("🔌 Connecting to TimescaleDB...")
            conn = psycopg2.connect(**DB_CONFIG)
            print("✅ Connected to TimescaleDB.")
            return conn
        except Exception as e:
            print(f"❌ DB connection failed: {e}")
            time.sleep(delay)
    print("❌ Gave up trying to connect to DB.")
    exit(1)

# Connect to DB
conn = connect_with_retry()
cursor = conn.cursor()

# Connect to Kafka
print("📥 Connecting to Kafka...")
try:
    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=os.environ["BOOTSTRAP_SERVERS"],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset='latest',
        enable_auto_commit=True
    )
    print(f"✅ Subscribed to Kafka topic: {TOPIC_NAME}")
except Exception as e:
    print("❌ Kafka connection failed:", e)
    exit(1)

# Process messages from Kafka
for message in consumer:
    data = message.value
    print(f"➡️ Inserting/Updating: {data}")
    try:
        # Use time from producer
        timestamp = data.get("time")
        if not timestamp:
            raise ValueError("Missing 'time' in Kafka message")

        cursor.execute('''
            INSERT INTO spy_option_chain (
                time, spot_price, maturity, strike,
                call_bid, call_ask, call_iv, call_delta, call_vega, call_theta,
                put_bid, put_ask, put_iv, put_delta, put_vega, put_theta
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (time, maturity, strike) DO UPDATE SET
                spot_price = EXCLUDED.spot_price,
                call_bid = EXCLUDED.call_bid,
                call_ask = EXCLUDED.call_ask,
                call_iv = EXCLUDED.call_iv,
                call_delta = EXCLUDED.call_delta,
                call_vega = EXCLUDED.call_vega,
                call_theta = EXCLUDED.call_theta,
                put_bid = EXCLUDED.put_bid,
                put_ask = EXCLUDED.put_ask,
                put_iv = EXCLUDED.put_iv,
                put_delta = EXCLUDED.put_delta,
                put_vega = EXCLUDED.put_vega,
                put_theta = EXCLUDED.put_theta;
        ''', (
            timestamp,
            data.get("spot_price"),
            data["maturity"],
            data["strike"],
            data.get("call_bid"), data.get("call_ask"), data.get("call_iv"),
            data.get("call_delta"), data.get("call_vega"), data.get("call_theta"),
            data.get("put_bid"), data.get("put_ask"), data.get("put_iv"),
            data.get("put_delta"), data.get("put_vega"), data.get("put_theta")
        ))
        conn.commit()
        print("✅ Inserted or updated row in DB.")
    except Exception as e:
        print("❌ Insert/update failed:", e)
        conn.rollback()
