import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os

# --- Configuration ---
API_URL = "https://api.india.delta.exchange/v2/history/candles"
MAX_CANDLES_PER_REQUEST = 2000
REQUEST_DELAY_SECONDS = 0.5
MAX_REQUESTS = 2000  # Increased for very long date ranges

# --- Helper Functions ---

def get_resolution_seconds(resolution_str):
    """Converts the resolution string (e.g., '5m', '1h') into seconds."""
    multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
    unit = resolution_str[-1]
    value = int(resolution_str[:-1])
    if unit in multipliers:
        return value * multipliers[unit]
    else:
        print(f"Error: Invalid time unit '{unit}' in resolution.")
        sys.exit(1)

def get_user_input():
    """
    Collects necessary input from the user with an easier date selection menu.
    """
    valid_resolutions = [
        '1m', '3m', '5m', '15m', '30m', 
        '1h', '2h', '4h', '6h', 
        '1d', '1w'
    ]

    print("--- Delta Exchange OHLCV Data Collector ---")

    # Get Symbol
    symbol = input("Enter the symbol (e.g., BTCUSDT, ETHUSDT): ").strip()

    # Get Resolution
    print("\nAvailable resolutions:", ", ".join(valid_resolutions))
    resolution = input("Enter the timeframe/resolution: ").lower().strip()
    while resolution not in valid_resolutions:
        print(f"Invalid resolution. Please choose from the list above.")
        resolution = input("Enter the timeframe/resolution: ").lower().strip()

    # --- Date Range Selection Menu ---
    print("\n--- Select a Date Range ---")
    print(" 1: Last 24 hours")
    print(" 2: Last 7 days")
    print(" 3: Last 30 days")
    print(" 4: Last 90 days")
    print(" 5: Year to Date (YTD)")
    print(" 6: All available data (from 2020-01-01)")
    print(" 7: Custom date range")

    start_timestamp = None
    end_timestamp = None
    now = datetime.now()

    while True:
        choice = input("Enter your choice (1-7): ")
        
        if choice == '1':
            start_dt = now - timedelta(days=1)
            end_dt = now
            break
        elif choice == '2':
            start_dt = now - timedelta(days=7)
            end_dt = now
            break
        elif choice == '3':
            start_dt = now - timedelta(days=30)
            end_dt = now
            break
        elif choice == '4':
            start_dt = now - timedelta(days=90)
            end_dt = now
            break
        elif choice == '5':
            start_dt = datetime(now.year, 1, 1)
            end_dt = now
            break
        elif choice == '6':
            start_dt = datetime(2020, 1, 1)
            end_dt = now
            break
        elif choice == '7':
            print("\nEnter custom dates in 'YYYY-MM-DD HH:MM:SS' format (e.g., 2023-06-01 00:00:00)")
            while True:
                try:
                    start_str = input("Enter the start date and time: ")
                    start_dt = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
                    if start_dt > now:
                        print("\nError: Start date cannot be in the future. Please try again.")
                        continue
                    start_timestamp = int(start_dt.timestamp())
                    break
                except ValueError:
                    print("Invalid date format. Please use 'YYYY-MM-DD HH:MM:SS'.")
            
            while True:
                try:
                    end_str = input("Enter the end date and time: ")
                    end_dt = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
                    
                    if end_dt > now:
                        print(f"Warning: End date is in the future. Using current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
                        end_dt = now

                    end_timestamp = int(end_dt.timestamp())
                    if end_timestamp <= start_timestamp:
                        print("End date must be after the start date. Please try again.")
                        continue
                    break
                except ValueError:
                    print("Invalid date format. Please use 'YYYY-MM-DD HH:MM:SS'.")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")
    
    if choice in ['1', '2', '3', '4', '5', '6']:
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())
        print(f"Fetching data from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")

    return symbol, resolution, start_timestamp, end_timestamp

def fetch_historical_data(symbol, resolution, start_ts, end_ts):
    """
    Fetches historical OHLCV data from the Delta Exchange API.
    Works perfectly with ALL timeframes by using proper pagination logic.
    """
    all_candles_dict = {}
    current_start_ts = start_ts
    
    resolution_secs = get_resolution_seconds(resolution)
    # Calculate the time span that MAX_CANDLES would cover
    max_time_span = MAX_CANDLES_PER_REQUEST * resolution_secs

    print("\nFetching data from the API...")
    print(f"Symbol: {symbol}")
    print(f"Resolution: {resolution} ({resolution_secs} seconds per candle)")
    print(f"Date range: {datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Max candles per request: {MAX_CANDLES_PER_REQUEST}")
    print(f"Max time span per request: {max_time_span / 86400:.2f} days")
    print(f"\nSearching through entire date range...\n")
    
    request_count = 0
    data_found = False
    api_error_occurred = False
    last_progress_update = 0
    
    while current_start_ts < end_ts and request_count < MAX_REQUESTS:
        request_count += 1
        
        # Calculate the end timestamp for this request
        # Don't exceed the user's requested end time
        current_end_ts = min(current_start_ts + max_time_span, end_ts)
        
        headers = {'Accept': 'application/json'}
        params = {
            'resolution': resolution,
            'symbol': symbol,
            'start': str(current_start_ts),
            'end': str(current_end_ts)
        }

        try:
            response = requests.get(API_URL, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()

            # Check if API returned an error
            if not data.get("success"):
                if not api_error_occurred:
                    api_error_occurred = True
                    error = data.get("error", {})
                    error_msg = error.get("message", "Unknown error")
                    error_code = error.get("code", "N/A")
                    
                    print(f"\n⚠️  API Error (Code: {error_code}): {error_msg}")
                    print(f"Symbol: {symbol} | Resolution: {resolution}")
                    print(f"Will continue searching through remaining date range...\n")
                
                # Move to next window and continue
                current_start_ts = current_end_ts
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            # Process the result
            if data.get("result") and len(data["result"]) > 0:
                candles = data["result"]
                
                # Count new candles before adding
                new_count = 0
                for candle in candles:
                    if candle['time'] not in all_candles_dict:
                        all_candles_dict[candle['time']] = candle
                        new_count += 1
                
                if new_count > 0:
                    data_found = True
                    
                    # Get time range of received data
                    times = [c['time'] for c in candles]
                    first_time = min(times)
                    last_time = max(times)
                    
                    # Show progress (but not too frequently for small timeframes)
                    if len(all_candles_dict) - last_progress_update >= 1000 or len(candles) >= 100:
                        print(f"  Request #{request_count:4d}: Fetched {len(candles):4d} candles, "
                              f"{new_count:4d} new | Total: {len(all_candles_dict):7d} | "
                              f"Up to {datetime.fromtimestamp(last_time).strftime('%Y-%m-%d %H:%M:%S')}")
                        last_progress_update = len(all_candles_dict)
                    
                    # CRITICAL FIX: Always move forward by the full window
                    # This ensures we don't miss any data even if API returns less than requested
                    # The dictionary will handle any duplicates automatically
                    current_start_ts = current_end_ts
                else:
                    # Got data but all duplicates, move forward
                    current_start_ts = current_end_ts
            else:
                # No data in this window, move to next
                current_start_ts = current_end_ts

        except requests.exceptions.RequestException as e:
            print(f"\n⚠️  Network error on request #{request_count}: {e}")
            print(f"Retrying in {REQUEST_DELAY_SECONDS * 2} seconds...\n")
            time.sleep(REQUEST_DELAY_SECONDS * 2)
            continue
            
        except Exception as e:
            print(f"\n⚠️  Unexpected error on request #{request_count}: {e}")
            print(f"Continuing to next window...\n")
            current_start_ts = current_end_ts
            time.sleep(REQUEST_DELAY_SECONDS)
            continue
            
        time.sleep(REQUEST_DELAY_SECONDS)
    
    # Final summary
    print(f"\n{'='*70}")
    if request_count >= MAX_REQUESTS:
        print(f"⚠️  Reached maximum request limit ({MAX_REQUESTS}).")
        print(f"Consider using a shorter date range or increase MAX_REQUESTS.")
    else:
        print(f"✓ Completed searching through entire date range.")
    
    print(f"\nSummary:")
    print(f"  Total requests made: {request_count}")
    print(f"  Total unique candles found: {len(all_candles_dict)}")
    
    if not data_found and api_error_occurred:
        print(f"\n⚠️  No data was found. Possible reasons:")
        print(f"  • The symbol '{symbol}' doesn't exist on Delta Exchange")
        print(f"  • The symbol name is incorrect (try: BTCUSDT, ETHUSDT, SOLUSDT)")
        print(f"  • No trading data available for the selected date range")
    elif not data_found:
        print(f"\n⚠️  No data found for this date range.")
        print(f"  Try a more recent date range or verify the symbol exists.")
    
    print(f"{'='*70}\n")
    
    return list(all_candles_dict.values())

def main():
    """Main function to run the script."""
    symbol, resolution, start_ts, end_ts = get_user_input()

    if start_ts is None or end_ts is None:
        print("\n❌ Could not determine start or end time. Exiting.")
        return
    
    candles = fetch_historical_data(symbol, resolution, start_ts, end_ts)

    if not candles:
        print("❌ No data was fetched after searching the entire date range.")
        print(f"\nTroubleshooting:")
        print(f"  1. Verify symbol name (try: BTCUSDT, ETHUSDT, SOLUSDT)")
        print(f"  2. Try a recent date range (e.g., Last 7 days)")
        print(f"  3. Check Delta Exchange India website for available symbols")
        return

    print(f"✓ Total unique candles fetched: {len(candles)}")

    df = pd.DataFrame(candles)

    df['datetime'] = pd.to_datetime(df['time'], unit='s')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('datetime').drop_duplicates(subset=['datetime']).reset_index(drop=True) #type: ignore

    # Display sample data
    print("\n--- First 5 rows ---")
    print(df.head().to_string(index=False))
    print("\n--- Last 5 rows ---")
    print(df.tail().to_string(index=False))

    # Check for gaps in data
    print(f"\n--- Data Quality Check ---")
    expected_interval = pd.Timedelta(seconds=get_resolution_seconds(resolution))
    df['time_diff'] = df['datetime'].diff()
    
    # Allow some tolerance for missing candles (e.g., market closed, low liquidity)
    significant_gaps = df[df['time_diff'] > expected_interval * 1.5] #type: ignore
    
    if len(significant_gaps) > 0:
        print(f"⚠️  Found {len(significant_gaps)} time gaps in data (may be normal for low liquidity periods)")
    else:
        print(f"✓ No significant gaps detected - data appears continuous")
    
    df = df.drop('time_diff', axis=1)

    start_date_str = datetime.fromtimestamp(start_ts).strftime('%Y%m%d')
    end_date_str = datetime.fromtimestamp(end_ts).strftime('%Y%m%d')

    # Create directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)

    # Save to data/raw/ directory
    filename = f"data/raw/{symbol}_{resolution}_{start_date_str}_to_{end_date_str}.csv"

    try:
        df.to_csv(filename, index=False)
        file_size = os.path.getsize(filename) / 1024  # Size in KB
        print(f"\n✓ Successfully saved data to '{filename}'")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Rows: {len(df):,}")
        print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    except Exception as e:
        print(f"\n❌ Could not save data to file. Error: {e}")

if __name__ == "__main__":
    main()