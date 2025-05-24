#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import sys

# ─── Helpers ───────────────────────────────────────────────────────────────────
def parse_ts(x):
    for fmt in ('%d/%m/%y %H:%M:%S','%Y-%m-%d %H:%M:%S','%y.%m.%d %H.%M.%S'):
        try:    return datetime.strptime(x, fmt)
        except: continue
    return pd.NaT

def format_time(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1) Load model
    try:
        kmeans = joblib.load('meal_time_kmeans.pkl')
    except FileNotFoundError:
        print("Error: meal_time_kmeans.pkl not found", file=sys.stderr)
        sys.exit(1)

    # 2) Load data
    df = pd.read_csv('pet_feeder_near_mealtimes_varied_duration.csv')
    df['timestamp'] = pd.to_datetime(
        df['timestamp'].astype(str).apply(parse_ts)
    )
    df = df.dropna(subset=['timestamp'])
    df['distance_cm'] = pd.to_numeric(
        df['distance_cm'].astype(str).str.replace(',', '.'),
        errors='coerce'
    )
    df = df.dropna(subset=['distance_cm'])

    # 3) Seconds‐of‐day & clustering
    df['sec_of_day'] = (
        df['timestamp'].dt.hour * 3600 +
        df['timestamp'].dt.minute * 60 +
        df['timestamp'].dt.second
    )
    df['label'] = kmeans.predict(df[['sec_of_day']])

    # 4) Meal times
    meal_names  = ['Night','Morning','Evening']
    meal_emojis = ['🥐','🥗','🍽️']
    centers     = kmeans.cluster_centers_.flatten()  # in label order

    print("\n🍽️  Your pet’s average meal times  🍽️")
    print("======================================")
    for i, (name, emoji) in enumerate(zip(meal_names, meal_emojis)):
        ct = centers[i]
        print(f"{emoji}  Average {name} time: {format_time(ct)}")

    # 5) Visit durations
    df['date'] = df['timestamp'].dt.date
    durations = df.groupby(['label','date'])['timestamp'] \
                  .agg(lambda x: (x.max() - x.min()).total_seconds()/60.0)
    avg_durs   = durations.groupby(level=0).mean()
    overall    = durations.mean()

    print("\n🐾  Your pet’s average meal‐visit durations  🐾")
    print("==============================================")
    for i, (name, emoji) in enumerate(zip(meal_names, meal_emojis)):
        avg = avg_durs.get(i, np.nan)
        if not np.isnan(avg):
            print(f"{emoji}  Avg {name} visit: {avg:.1f} min")
        else:
            print(f"{emoji}  Avg {name} visit: n/a")

    print(f"\n🐾  Overall avg visit length: {overall:.1f} min\n")

if __name__ == '__main__':
    main()
