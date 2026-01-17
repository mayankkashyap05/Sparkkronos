"""
🎯 Quality Data Pipeline - User Friendly Edition
================================================

Easily convert raw OHLCV data into high-quality, forecast-ready data.

"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.processes.preprocessing import preprocess, PreprocessorConfig

# ========================================
# DATA LOADER
# ========================================

class DataLoader:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path
    
    def select_csv_file(self):
        """Show available CSV files and let user select one"""
        raw_dir = Path("./data/raw")
        
        if not raw_dir.exists():
            raise FileNotFoundError("❌ data/raw directory not found!")
        
        # Find all CSV files
        csv_files = list(raw_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("❌ No CSV files found in data/raw/")
        
        print("\n" + "="*50)
        print("📁 AVAILABLE CSV FILES")
        print("="*50)
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"{i}. {csv_file.name}")
        
        print("="*50)
        
        while True:
            try:
                choice = input(f"Select file (1-{len(csv_files)}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(csv_files):
                    selected_file = csv_files[choice_num - 1]
                    print(f"✓ Selected: {selected_file.name}")
                    return str(selected_file)
                else:
                    print(f"❌ Please enter a number between 1 and {len(csv_files)}")
            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled")
                sys.exit(1)
        
    def ask_timeframe(self):
        """Ask user for timeframe"""
        print("\n" + "="*50)
        print("Select Timeframe:")
        print("="*50)
        print("1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w")
        print("="*50)
        
        timeframe = input("Enter timeframe (default: 1h): ").strip()
        return timeframe if timeframe else '1h'
    
    def _parse_timestamps(self, timestamp_series):
        """
        Intelligently parse timestamps from various formats.
        
        Handles:
        - Unix timestamps (seconds)
        - Unix timestamps (milliseconds)
        - ISO date strings
        - Excel serial dates
        """
        # Get sample value
        sample = timestamp_series.iloc[0]
        
        print(f"\n🔍 TIMESTAMP PARSING:")
        print(f"  • Sample value: {sample}")
        print(f"  • Type: {type(sample)}")
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(timestamp_series):
            sample_val = float(sample)
            print(f"  • Numeric value: {sample_val}")
            
            # Milliseconds (13 digits, > 1e12)
            if sample_val > 1e12:
                print(f"  • Detected: Unix MILLISECONDS")
                return pd.to_datetime(timestamp_series, unit='ms')
            
            # Seconds (10 digits, > 1e9)
            elif sample_val > 1e9:
                print(f"  • Detected: Unix SECONDS")
                return pd.to_datetime(timestamp_series, unit='s')
            
            # Excel serial date (typically 5-6 digits for recent dates)
            elif 40000 < sample_val < 60000:
                print(f"  • Detected: Excel serial date")
                # Excel epoch is 1899-12-30
                return pd.to_datetime('1899-12-30') + pd.to_timedelta(timestamp_series, unit='D')
            
            # Too small - likely an error
            else:
                raise ValueError(
                    f"❌ INVALID TIMESTAMP FORMAT!\n"
                    f"   Sample value: {sample_val}\n"
                    f"   This is too small to be a valid timestamp.\n\n"
                    f"   Expected formats:\n"
                    f"   • Unix seconds (10 digits): 1609459200\n"
                    f"   • Unix milliseconds (13 digits): 1609459200000\n"
                    f"   • ISO string: '2024-01-01 12:00:00'\n\n"
                    f"  ⚠️ CHECK YOUR CSV FILE!\n"
                    f"   Your CSV might have:\n"
                    f"   - Row indices instead of timestamps\n"
                    f"   - Incorrectly formatted dates\n"
                    f"   - Missing timestamp data"
                )
        
        # String format
        else:
            print(f"  • Detected: String datetime")
            try:
                return pd.to_datetime(timestamp_series)
            except Exception as e:
                raise ValueError(
                    f"❌ Could not parse timestamp strings!\n"
                    f"   Error: {e}\n"
                    f"   Sample: {sample}\n"
                    f"   Please check your CSV file format."
                )
    
    def _detect_columns(self, df):
        """
        Ask user to select OHLCV columns for processing.
        """
        columns = df.columns.tolist()
        
        print(f"\n🔍 OHLCV COLUMN SELECTION:")
        print(f"  • Available columns: {columns}")
        print("="*50)
        
        # Show numbered list of columns
        print("Available columns:")
        for i, col in enumerate(columns, 1):
            print(f"  {i}. {col}")
        print("="*50)
        
        detected = {}
        required = ['open', 'high', 'low', 'close', 'volume']
        
        for col_type in required:
            while True:
                try:
                    choice = input(f"Select column for {col_type.upper()} (1-{len(columns)}): ").strip()
                    choice_num = int(choice)
                    
                    if 1 <= choice_num <= len(columns):
                        selected_col = columns[choice_num - 1]
                        detected[col_type] = selected_col
                        print(f"  ✓ {col_type.upper()}: '{selected_col}'")
                        break
                    else:
                        print(f"  ❌ Please enter a number between 1 and {len(columns)}")
                except ValueError:
                    print("  ❌ Please enter a valid number")
                except KeyboardInterrupt:
                    print("\n❌ Operation cancelled")
                    sys.exit(1)
        
        return detected
    
    def _detect_timestamp_column(self, df):
        """
        Ask user to select timestamp column.
        """
        print(f"\n🔍 TIMESTAMP COLUMN SELECTION:")
        print("="*50)
        
        # Show options
        print("Timestamp options:")
        print("  0. Use DataFrame index as timestamp")
        
        columns = df.columns.tolist()
        for i, col in enumerate(columns, 1):
            print(f"  {i}. {col}")
        
        print("="*50)
        
        while True:
            try:
                choice = input(f"Select timestamp column (0-{len(columns)}): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    print(f"  ✓ Using index as timestamp")
                    return None
                elif 1 <= choice_num <= len(columns):
                    selected_col = columns[choice_num - 1]
                    print(f"  ✓ Using column '{selected_col}' as timestamp")
                    return selected_col
                else:
                    print(f"  ❌ Please enter a number between 0 and {len(columns)}")
            except ValueError:
                print("  ❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Operation cancelled")
                sys.exit(1)
    
    def load(self):
        """
        Main method to load data and return (raw_data, timeframe).
        """
        print("\n" + "="*70)
        print("🔧 STEP 1: DATA LOADING")
        print("="*70)
        
        # Select CSV file if not provided
        if self.csv_path is None:
            self.csv_path = self.select_csv_file()
        
        # Check if file exists
        csv_path = Path(self.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"❌ CSV file not found: {csv_path}\n"
                f"   Please check the path and try again."
            )
        
        print(f"\n📂 Loading: {csv_path}")
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            raise ValueError(f"❌ Error loading CSV: {e}")
        
        # Detect timestamp column
        timestamp_col = self._detect_timestamp_column(df)
        
        # Parse timestamps
        if timestamp_col is None:
            # Use index
            df.index = self._parse_timestamps(df.index.to_series())
            df.index.name = 'timestamp'
        else:
            # Use column
            df.index = self._parse_timestamps(df[timestamp_col])
            df.index.name = 'timestamp'
            df = df.drop(columns=[timestamp_col])
        
        # Detect OHLCV columns
        column_mapping = self._detect_columns(df)
        
        # Rename columns to standard names
        df = df.rename(columns={
            column_mapping['open']: 'open',
            column_mapping['high']: 'high',
            column_mapping['low']: 'low',
            column_mapping['close']: 'close',
            column_mapping['volume']: 'volume'
        })
        
        # Keep only OHLCV columns
        raw_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # Sort by timestamp
        raw_data = raw_data.sort_index()
        
        # Basic validation
        print(f"\n✅ DATA LOADED SUCCESSFULLY:")
        print(f"  • Shape: {raw_data.shape}")
        print(f"  • Date range: {raw_data.index[0]} to {raw_data.index[-1]}")
        print(f"  • Columns: {raw_data.columns.tolist()}")
        
        # Display sample
        print(f"\n📊 Data Sample:")
        print(raw_data.head(3))
        
        # Ask for timeframe
        timeframe = self.ask_timeframe()
        
        return raw_data, timeframe


# ========================================
# LOAD DATA
# ========================================

# Initialize DataLoader
loader = DataLoader()  # Will show file selection menu
# Or specify custom path:
# loader = DataLoader(csv_path="./path/to/your/data.csv")

# Load data
raw_data, timeframe = loader.load()
# Ensure raw_data is a DataFrame for type safety
assert isinstance(raw_data, pd.DataFrame), "raw_data must be a DataFrame"

# ========================================
# PREPROCESSING
# ========================================

print("\n🔧 STEP 2: PREPROCESSING")
print("="*70)

# ========================================
# OPTION A: Simple Preprocessing (Default)
# ========================================
clean_data, freq = preprocess(raw_data, timeframe)

# ========================================
# OPTION B: Advanced Preprocessing Configuration
# ========================================
# prep_config = PreprocessorConfig(
#     # Timeline regularization
#     ensure_regular_timeline=True,      # CRITICAL for Time Series
#     fill_gaps=True,                    # Fill weekends/holidays
#     
#     # OHLCV validation
#     validate_ohlcv_logic=True,
#     fix_ohlcv_violations=True,
#     
#     # Volume validation
#     ensure_positive_volume=True,
#     
#     # Missing value handling
#     price_fill_method='time_interpolate',  # 'time_interpolate', 'linear', 'ffill'
#     volume_fill_method='forward_mean',      # 'forward_mean', 'mean', 'zero'
#     
#     # Outlier handling
#     detect_outliers=True,
#     max_price_change_pct=50.0,  # Maximum % change between candles
#     
#     # Invalid values
#     remove_invalid_values=True
# )
# clean_data, freq = preprocess(raw_data, timeframe, config=prep_config)

# ========================================
# OPTION C: Minimal Preprocessing (for clean data)
# ========================================
# prep_config = PreprocessorConfig(
#     ensure_regular_timeline=True,
#     fill_gaps=True,
#     validate_ohlcv_logic=False,  # Skip validation if data is clean
#     detect_outliers=False,        # Skip outlier detection
#     fix_ohlcv_violations=False
# )
# clean_data, freq = preprocess(raw_data, timeframe, config=prep_config)

# ========================================
# OPTION D: Aggressive Cleaning (for noisy data)
# ========================================
# prep_config = PreprocessorConfig(
#     ensure_regular_timeline=True,
#     fill_gaps=True,
#     validate_ohlcv_logic=True,
#     fix_ohlcv_violations=True,
#     detect_outliers=True,
#     max_price_change_pct=20.0,  # Stricter outlier detection (20% instead of 50%)
#     remove_invalid_values=True
# )
# clean_data, freq = preprocess(raw_data, timeframe, config=prep_config)

print(f"✓ Cleaned data: {clean_data.shape}")
print(f"✓ Frequency: {freq}")
print(f"✓ Missing values: {clean_data.isna().sum().sum()}")

# Save preprocessed data
output_dir = Path("./data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Get the input filename and create output filename
input_path = Path(loader.csv_path or "")
output_filename = input_path.stem + "_processed.csv"
cleaned_file = output_dir / output_filename

clean_data.to_csv(cleaned_file, index=True, index_label='timestamps')
print(f"✓ Saved cleaned data: {cleaned_file}")

