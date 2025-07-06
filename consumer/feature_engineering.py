# feature_engineering.py
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import traceback

# 📦 DB connection
PG_URI = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)
engine = create_engine(PG_URI)

# ⚙️ Parameters
LOOKBACK_DAYS = 5
MIN_BID_FILTER = 0.01
TABLE_NAME = 'option_features'

print("🚀 Starting feature engineering...")

# 🕒 Get latest timestamp from option_features
try:
    latest_time = pd.read_sql(f"SELECT MAX(ts_utc) FROM {TABLE_NAME}", engine).iloc[0, 0]
    print("🔍 Latest processed time:", latest_time)
except Exception as e:
    print(f"⚠️ Could not fetch latest time from {TABLE_NAME}: {e}")
    latest_time = None

# 📥 EXTRACT
if latest_time is None:
    query = f"""
    SELECT
        time AT TIME ZONE 'UTC' AS ts_utc,
        spot_price,
        maturity,
        strike,
        call_bid, call_ask, call_iv,
        put_bid,  put_ask,  put_iv
    FROM spy_option_chain
    WHERE time > NOW() - INTERVAL '{LOOKBACK_DAYS} days'
      AND spot_price > 0
      AND (call_bid > {MIN_BID_FILTER} OR put_bid > {MIN_BID_FILTER});
    """
else:
    query = f"""
    SELECT
        time AT TIME ZONE 'UTC' AS ts_utc,
        spot_price,
        maturity,
        strike,
        call_bid, call_ask, call_iv,
        put_bid,  put_ask,  put_iv
    FROM spy_option_chain
    WHERE time > '{latest_time}'
      AND spot_price > 0
      AND (call_bid > {MIN_BID_FILTER} OR put_bid > {MIN_BID_FILTER});
    """

print("📥 Executing SQL query...")
raw_df = pd.read_sql(query, engine)
print(f"📊 Retrieved {len(raw_df):,} rows from spy_option_chain")

if raw_df.empty:
    print("⚠️ No new data to process.")
    exit(0)

# 🧼 PREPROCESS
raw_df['ts_utc'] = pd.to_datetime(raw_df['ts_utc'])
raw_df['maturity'] = pd.to_datetime(raw_df['maturity'])

# 🧠 TRANSFORM
def _calc_features(df):
    df = df.copy()
    df['dte'] = (df['maturity'] - df['ts_utc']).dt.total_seconds() / 86400
    df['moneyness'] = df['strike'] / df['spot_price']
    df['log_moneyness'] = np.log(df['moneyness'])

    df['hour'] = df['ts_utc'].dt.hour + df['ts_utc'].dt.minute / 60
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    market_open = df['ts_utc'].dt.normalize() + pd.Timedelta(hours=13.5)
    df['minutes_since_open'] = (df['ts_utc'] - market_open).dt.total_seconds() / 60

    calls = df[['ts_utc', 'spot_price', 'maturity', 'dte', 'strike',
                'moneyness', 'log_moneyness', 'hour_sin', 'hour_cos',
                'minutes_since_open', 'call_iv']].dropna()
    calls = calls.rename(columns={'call_iv': 'iv'})
    calls['right'] = 'C'

    puts = df[['ts_utc', 'spot_price', 'maturity', 'dte', 'strike',
               'moneyness', 'log_moneyness', 'hour_sin', 'hour_cos',
               'minutes_since_open', 'put_iv']].dropna()
    puts = puts.rename(columns={'put_iv': 'iv'})
    puts['right'] = 'P'

    return pd.concat([calls, puts], ignore_index=True)

print("🔧 Processing features...")
feat_df = _calc_features(raw_df)
print(f"🎯 Generated {len(feat_df):,} feature rows")

# 💾 LOAD with chunked insert and error logging
try:
    chunk_size = 10000
    print(f"💾 Inserting in chunks of {chunk_size}...")
    for i in range(0, len(feat_df), chunk_size):
        chunk = feat_df.iloc[i:i + chunk_size]
        chunk.to_sql(TABLE_NAME, engine, if_exists='append', index=False, method='multi')
        print(f"✅ Inserted chunk {i // chunk_size + 1} ({len(chunk)} rows)")
    print(f"✅ All {len(feat_df):,} rows inserted into {TABLE_NAME}")
except SQLAlchemyError as e:
    print("❌ SQLAlchemyError during insertion:")
    traceback.print_exc()
except Exception as e:
    print("❌ General error during insertion:")
    traceback.print_exc()
