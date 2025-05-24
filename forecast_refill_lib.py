#!/usr/bin/env python3
import pandas as pd
from datetime import datetime, timedelta
import joblib
import numpy as np

# ─── Hard‐coded dataset & model ───────────────────────────────────────────────
CSV_PATH = 'pet_feeder_dataset_progressive_restock.csv'
MODEL_PATH = 'interval_regressor.joblib'
SCALER_PATH = 'interval_scaler.joblib'

# Specify datetime parsing formats for consistency
DATE_FORMATS = [
    '%d.%m.%y %H.%M.%S',  # e.g. 25.03.01 00.00.00
    '%d/%m/%y %H:%M:%S',  # e.g. 01/03/25 00:04:00
    '%Y-%m-%d %H:%M:%S',  # ISO fallback
]

def parse_timestamp(ts):
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            continue
    # Last resort: let pandas infer format
    return pd.to_datetime(ts, dayfirst=True, infer_datetime_format=True, errors='coerce')


def load_and_clean():
    """Load CSV, parse timestamps, detect refill events."""
    df = pd.read_csv(CSV_PATH)

    # Robust timestamp parsing
    df['timestamp'] = df['timestamp'].astype(str).apply(parse_timestamp)

    # Numeric conversion
    df['weight_g'] = pd.to_numeric(df['weight_g'], errors='coerce')

    # Drop invalid rows and sort
    df = (
        df.dropna(subset=['timestamp', 'weight_g'])
          .sort_values('timestamp')
          .reset_index(drop=True)
    )

    # Identify refill events (full bowl resets)
    df['prev_w'] = df['weight_g'].shift(1)
    df['refill_event'] = (df['weight_g'] == df['weight_g'].max()) & (df['prev_w'] < df['weight_g'].max())

    return df.loc[df['refill_event'], 'timestamp'].reset_index(drop=True)


def get_next_refill(last_date_iso):
    """
    last_date_iso: str 'YYYY-MM-DD'  (e.g. '2025-05-01')
    Returns tuple(next_refill_datetime, predicted_interval_hours)
    """
    # Parse user-provided date at midnight
    user_dt = datetime.strptime(last_date_iso, '%Y-%m-%d')

    # Load past refill timestamps
    refill_times = load_and_clean()

    # Select those on or before the given date
    past = refill_times[refill_times <= user_dt]
    if len(past) < 2:
        raise ValueError("Not enough past refill events before given date.")

    # Compute previous interval in hours
    t1, t2 = past.iloc[-2], past.iloc[-1]
    interval_hours = (t2 - t1).total_seconds() / 3600.0
    if interval_hours <= 0:
        raise ValueError("Invalid interval between last two refills.")

    # Load trained model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Build feature vector: [interval_hours, hour, dayofweek, day, month]
    feat = np.array([[
        interval_hours,
        t2.hour,
        t2.weekday(),
        t2.day,
        t2.month
    ]])
    feat_scaled = scaler.transform(feat)

    # Predict next interval
    pred_interval = model.predict(feat_scaled)[0]

    # Calculate next refill datetime
    next_refill = user_dt + timedelta(hours=pred_interval)
    return next_refill, pred_interval


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: forecast_refill_lib.py YYYY-MM-DD")
        sys.exit(1)

    date_str = sys.argv[1]
    try:
        next_dt, hours = get_next_refill(date_str)
        print(f"Next refill predicted at {next_dt} (in {hours:.1f} hours)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)