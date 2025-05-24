import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import sys

# ─── Helper to parse feeder timestamps (full format) ──────────────────────────
def parse_ts(x):
    if not isinstance(x, str):
        return None
    for fmt in (
        '%d/%m/%y %H:%M:%S',  # e.g. '01/03/25 00:00:00'
        '%Y-%m-%d %H:%M:%S',  # e.g. '2025-03-01 00:00:00'
        '%y.%m.%d %H.%M.%S',  # e.g. '25.03.01 00.00.00'
        '%y-%m-%d',          # e.g. '25-03-01'
        '%d-%m-%y'           # e.g. '01-03-25'
    ):
        try:
            dt = datetime.strptime(x, fmt)
            # if only date provided, default to midnight
            if fmt in ('%y-%m-%d', '%d-%m-%y'):
                dt = dt.replace(hour=0, minute=0, second=0)
            return dt
        except ValueError:
            continue
    return None

# ─── Helper to parse user-provided date (YYYY-MM-DD) ─────────────────────────
def parse_date(x):
    try:
        return datetime.strptime(x, '%Y-%m-%d')
    except ValueError:
        return None

# ─── Load refill timestamps from CSV ──────────────────────────────────────────
def load_refill_times(csv_path):
    df = pd.read_csv(csv_path)
    # parse all recorded timestamps
    df['timestamp'] = df['timestamp'].astype(str).apply(parse_ts)
    # convert weight, handling comma decimals
    df['weight_g'] = pd.to_numeric(df['weight_g'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['timestamp', 'weight_g'])
    # identify refill events (weight jumps to 1000g)
    prev = df['weight_g'].shift(1)
    df['refill_event'] = ((df['weight_g'] == 1000) & (prev < 1000))
    # return timestamps of each refill
    return df.loc[df['refill_event'], 'timestamp'].reset_index(drop=True)

# ─── Compute interval (hours) between last two refills on or before last_dt ───
def compute_last_interval(refill_times, last_dt):
    prev_times = refill_times[refill_times <= last_dt]
    if len(prev_times) < 2:
        print('Not enough refill history.', file=sys.stderr)
        sys.exit(1)
    last_two = prev_times.iloc[-2:]
    # interval in hours
    return (last_two.iloc[1] - last_two.iloc[0]).total_seconds() / 3600

# ─── Main forecast logic ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Forecast next pet feeder refill date based on past data.'
    )
    parser.add_argument(
        '--csv',   required=True, help='CSV dataset path'
    )
    parser.add_argument(
        '--model', required=True, help='Joblib model path'
    )
    parser.add_argument(
        '--last',  required=True, help='Last known refill date YYYY-MM-DD'
    )
    args = parser.parse_args()

    # parse user input date
    last_dt = parse_date(args.last)
    if last_dt is None:
        print(f"Invalid date format for --last: '{args.last}'. Use YYYY-MM-DD", file=sys.stderr)
        sys.exit(1)
    # set to midnight
    last_dt = last_dt.replace(hour=0, minute=0, second=0)

    # load and process refill history
    refill_times = load_refill_times(args.csv)
    # compute last interval
    interval = compute_last_interval(refill_times, last_dt)

    # load trained regressor and predict next interval
    reg = joblib.load(args.model)
    next_interval = reg.predict([[interval]])[0]

        # compute next refill datetime
    next_dt = last_dt + timedelta(hours=next_interval)
    # display with separators and emojis
    print("***************************************************************** 🐾")
    print(f"🔄 Predicted next interval: {next_interval:.2f} hours")
    print(f"🎉 Next predicted refill at: {next_dt.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == '__main__':
    main()