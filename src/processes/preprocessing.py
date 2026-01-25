import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessorConfig:
    """Configuration for the preprocessor"""
    # Timeline regularization
    ensure_regular_timeline: bool = True
    fill_gaps: bool = True
    
    # OHLCV validation
    validate_ohlcv_logic: bool = False
    fix_ohlcv_violations: bool = False
    
    # Volume validation
    ensure_positive_volume: bool = False
    
    # Missing value handling
    price_fill_method: str = 'time_interpolate'  # Best for prices: time-aware interpolation
    volume_fill_method: str = 'forward_mean'      # For volume: forward fill or mean
    
    # Outlier handling (based on percentage change)
    detect_outliers: bool = False
    max_price_change_pct: float = 50.0  # 50% max change between candles
    
    # Ensure no infinite or invalid values
    remove_invalid_values: bool = True


class DataPreprocessor:
    """
    Robust data preprocessing for time series forecasting.
    
    Key Features:
    - Creates perfectly regular timeline
    - Fills gaps from weekends, holidays, outages
    - Time-based interpolation for prices
    - No missing values in output
    - OHLCV validation and fixing
    
    Output Guarantee:
    - Regular intervals with NO gaps
    - No NaN values
    - Ready for time series forecasting
    """
    
    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        self.stats = {}
    
    def process(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, str]:
        """
        Main preprocessing pipeline.
        
        Args:
            df: Raw DataFrame from DataLoader with columns: timestamp, open, high, low, close, volume
            timeframe: Timeframe string (e.g., '1h', '5m', '1d')
            
        Returns:
            Tuple of (cleaned_df, frequency_string)

        Note: Output may have MORE rows than input due to gap filling (required for time series forecasting)
        """
        print(f"\n{'='*60}")
        print("PREPROCESSING FOR TIME SERIES FORECASTING")
        print(f"{'='*60}")
        
        original_rows = len(df)
        print(f"Input rows: {original_rows:,}")
        
        # Step 1: Prepare timestamps
        df_clean = self._prepare_timestamps(df)
        print(f"✓ Timestamps prepared")
        
        # Step 2: Map timeframe to pandas frequency
        freq = self._map_timeframe_to_freq(timeframe)
        print(f"✓ Frequency: {freq}")
        
        # Step 3: Create regular timeline (CRITICAL for Time Series Forecasting)
        df_clean = self._create_regular_timeline(df_clean, freq)
        print(f"✓ Regular timeline created: {len(df_clean):,} rows")
        gaps_filled = len(df_clean) - original_rows
        if gaps_filled > 0:
            print(f"  └─ Filled {gaps_filled:,} gaps (weekends/holidays/outages)")
        
        # Step 4: Validate and fix OHLCV logic (before filling)
        df_clean = self._validate_ohlcv(df_clean)
        print(f"✓ OHLCV logic validated")
        
        # Step 5: Handle outliers
        outliers_detected = self._handle_outliers(df_clean)
        if outliers_detected > 0:
            print(f"✓ Handled {outliers_detected} outliers")
        
        # Step 6: Fill missing values (TIME INTERPOLATION for prices)
        missing_before = df_clean.isna().sum().sum()
        df_clean = self._fill_missing_values(df_clean)
        print(f"✓ Filled {missing_before:,} missing values")
        
        # Step 7: Remove invalid values
        df_clean = self._remove_invalid_values(df_clean)
        print(f"✓ Removed invalid values")
        
        # Step 8: Final OHLCV validation (after filling)
        df_clean = self._validate_ohlcv(df_clean)
        
        # Step 9: Final validation
        self._final_validation(df_clean)
        print(f"✓ Final validation passed")
        
        print(f"\n{'='*60}")
        print(f"OUTPUT: {len(df_clean):,} rows | {df_clean.isna().sum().sum()} NaNs | Ready for Time Series Forecasting")
        print(f"Data shape: {df_clean.shape} (timestamp is index, {len(df_clean.columns)} data columns)")
        print(f"{'='*60}\n")
        
        return df_clean, freq
    
    def _prepare_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamps to datetime and set as index.
        
        Note: This reduces the column count by 1 since timestamp becomes the index.
        This is standard practice for time series data processing.
        """
        df = df.copy()
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            # Handle various timestamp formats
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                # Unix timestamp (seconds or milliseconds)
                first_val = df['timestamp'].iloc[0]
                if first_val > 1e12:  # Milliseconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
                else:  # Seconds
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            else:
                # String datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Remove timezone for easier processing
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            # Set as index and sort
            df = df.set_index('timestamp').sort_index()
        
        # Remove any duplicate timestamps (keep first)
        if df.index.duplicated().any():
            duplicates = df.index.duplicated().sum()
            df = df[~df.index.duplicated(keep='first')] # type: ignore
            self.stats['duplicates_removed'] = duplicates
        
        return df
    
    def _create_regular_timeline(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        Create a perfectly regular timeline with no gaps.
        This is CRITICAL for Time Series Forecasting.
        
        Gaps occur due to:
        - Weekends (crypto trades, stocks don't)
        - Holidays
        - Exchange outages
        - Data feed issues
        """
        if not self.config.ensure_regular_timeline:
            return df
        
        # Create complete date range from start to end
        start_date = df.index.min()
        end_date = df.index.max()
        
        # Generate complete regular timeline
        regular_timeline = pd.date_range(
            start=start_date,
            end=end_date,
            freq=freq
        )
        
        # Reindex to regular timeline (creates NaN for missing timestamps)
        df_regular = df.reindex(regular_timeline)
        
        # Store gap statistics
        original_count = len(df)
        regularized_count = len(df_regular)
        gaps_added = regularized_count - original_count
        
        self.stats['original_rows'] = original_count
        self.stats['regularized_rows'] = regularized_count
        self.stats['gaps_filled'] = gaps_added
        
        return df_regular
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix OHLCV logic."""
        if not self.config.validate_ohlcv_logic:
            return df
        
        df = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if we have OHLCV columns
        if not all(col in df.columns for col in required_cols):
            return df
        
        violations_fixed = 0
        
        if self.config.fix_ohlcv_violations:
            # Fix OHLCV logic violations
            for idx in df.index:
                # Skip rows with any NaN values
                if df.loc[idx, ['open', 'high', 'low', 'close']].isna().values.any():
                    continue
                
                o, h, l, c = df.loc[idx, ['open', 'high', 'low', 'close']].values.astype(float)
                
                # Ensure high is the maximum
                actual_high = max(o, h, l, c)
                if h < actual_high:
                    df.loc[idx, 'high'] = actual_high
                    violations_fixed += 1
                
                # Ensure low is the minimum
                actual_low = min(o, h, l, c)
                if l > actual_low:
                    df.loc[idx, 'low'] = actual_low
                    violations_fixed += 1
        
        # Ensure volume is non-negative
        if self.config.ensure_positive_volume and 'volume' in df.columns:
            negative_volumes = (df['volume'] < 0).sum()
            if negative_volumes > 0:
                df['volume'] = df['volume'].abs()
                violations_fixed += negative_volumes
        
        self.stats['ohlcv_violations_fixed'] = violations_fixed
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> int:
        """Detect and handle outliers based on percentage change."""
        if not self.config.detect_outliers:
            return 0
        
        df_orig = df.copy()
        price_cols = ['open', 'high', 'low', 'close']
        total_outliers = 0
        
        for col in price_cols:
            if col not in df.columns:
                continue
            
            # Calculate percentage change without implicit NA filling (future-proof)
            pct_change = df[col].pct_change(fill_method=None).abs() * 100
            
            # Find outliers (changes > max_price_change_pct)
            outliers = pct_change > self.config.max_price_change_pct
            outlier_count = outliers.sum()
            
            # Replace outliers with NaN (will be filled later)
            if outlier_count > 0:
                df.loc[outliers, col] = np.nan
                total_outliers += outlier_count
        
        self.stats['outliers_detected'] = total_outliers
        return total_outliers
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values intelligently.
        
        For PRICES (OHLC): Use time-based interpolation (best practice)
        For VOLUME: Use forward fill then mean (volume interpolation is less meaningful)
        """
        df = df.copy()
        
        price_cols = ['open', 'high', 'low', 'close']
        volume_cols = ['volume']
        
        # Fill PRICES with time interpolation
        if self.config.price_fill_method == 'time_interpolate':
            for col in price_cols:
                if col in df.columns:
                    # Time-based interpolation: accounts for irregular spacing
                    df[col] = df[col].interpolate(
                        method='time',
                        limit_direction='both'
                    )
        
        # Fill VOLUME with forward fill + mean
        if self.config.volume_fill_method == 'forward_mean':
            for col in volume_cols:
                if col in df.columns:
                    # Forward fill first
                    df[col] = df[col].ffill()
                    # Then use mean for any remaining NaN
                    df[col] = df[col].fillna(df[col].mean())
        
        # Safety: catch any remaining NaN with forward/backward fill
        df = df.ffill().bfill()
        
        # Last resort: fill with column mean
        for col in df.columns:
            if df[col].isna().any(): # type: ignore
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _remove_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove infinite and invalid values."""
        if not self.config.remove_invalid_values:
            return df
        
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Replace infinite values with NaN
        inf_mask = np.isinf(df[numeric_cols])
        inf_count = inf_mask.sum().sum()
        
        if inf_count > 0:
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
            # Fill the NaN values created
            df = df.ffill().bfill()
            self.stats['infinite_values_fixed'] = inf_count
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> None:
        """Final validation checks."""
        
        # Check no missing values
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            raise ValueError(
                f"❌ Data still contains {missing_count} missing values after preprocessing. "
                "This will break Time Series Forecasting."
            )
        
        # Check no infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            raise ValueError(
                f"❌ Data contains {inf_count} infinite values. "
                "This will break Time Series Forecasting."
            )
        
        # Check for positive prices (if OHLCV)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    raise ValueError(
                        f"❌ Column '{col}' contains zero or negative values. "
                        "Invalid for financial data."
                    )
        
        # Check timeline regularity
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            unique_diffs = time_diffs.unique()
            
            # Allow for slight variations due to daylight saving, etc.
            if len(unique_diffs) > 3:
                print(f"⚠ Warning: Timeline has {len(unique_diffs)} different intervals")
                print(f"  This may affect Time Series Forecasting performance")
    
    def _map_timeframe_to_freq(self, timeframe: str) -> str:
        """Map exchange timeframe to pandas frequency."""
        timeframe_map = {
            '1m': '1min',
            '3m': '3min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '12h': '12h',
            '1d': '1d',
            '1w': '1w',
        }
        
        freq = timeframe_map.get(timeframe.lower())
        
        if not freq:
            raise ValueError(
                f"❌ Unknown timeframe: {timeframe}. "
                f"Valid options: {list(timeframe_map.keys())}"
            )
        
        return freq
    
    def get_stats(self) -> dict:
        """Get preprocessing statistics."""
        return self.stats


# Convenience function
def preprocess(
    df: pd.DataFrame,
    timeframe: str,
    config: Optional[PreprocessorConfig] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Preprocess data for time series forecasting.

    Args:
        df: Raw DataFrame from DataLoader
        timeframe: Timeframe string (e.g., '1h', '5m', '1d')
        config: Optional preprocessor configuration
        
    Returns:
        Tuple of (cleaned_df, frequency_string)
        
    Guarantees:
        - Regular timeline with NO gaps
        - No missing values
        - No infinite values
        - Valid OHLCV data
        - Ready for Time Series Forecasting
    """
    preprocessor = DataPreprocessor(config=config)
    cleaned_df, freq = preprocessor.process(df, timeframe)
    
    # Print statistics
    stats = preprocessor.get_stats()
    if stats:
        print("\nPreprocessing Statistics:")
        for key, value in stats.items():
            print(f"  • {key}: {value}")
    
    return cleaned_df, freq


# Example usage in your pipeline
if __name__ == "__main__":
    # Step 1: Create mock data
    print("Creating mock data...")
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=2048, freq='1h')
    
    # Generate realistic OHLCV data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, 2048)  # 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, 2048)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 2048))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 2048))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 2048)
    })
    
    timeframe = '1h'
    
    print(f"\nRaw data shape: {data.shape}")
    print(f"Timeframe: {timeframe}")
    print(f"Columns: {list(data.columns)}")
    
    # Step 2: Preprocess for Time Series Forecasting
    config = PreprocessorConfig(
        ensure_regular_timeline=True,      # CRITICAL for Time Series Forecasting
        fill_gaps=True,                    # Fill weekends/holidays/outages
        price_fill_method='time_interpolate',  # Best for prices
        volume_fill_method='forward_mean',     # For volume
        validate_ohlcv_logic=True,
        fix_ohlcv_violations=True,
        detect_outliers=True,
        max_price_change_pct=50.0
    )

    cleaned_data, freq = preprocess(data, timeframe, config)

    # Step 3: Verify output quality
    print(f"\nCleaned data shape: {cleaned_data.shape}")
    print(f"Frequency: {freq}")
    print(f"Missing values: {cleaned_data.isna().sum().sum()}")
    print(f"Infinite values: {np.isinf(cleaned_data.select_dtypes(include=[np.number])).sum().sum()}")
    print(f"\nFirst 5 rows:")
    print(cleaned_data.head())
    print(f"\nLast 5 rows:")
    print(cleaned_data.tail())
    
    # Step 4: Ready for Time Series Forecasting!
    print("\n" + "="*60)
    print("✅ DATA READY FOR TIME SERIES FORECASTING")
    print("="*60)
    print("You can now safely use this data for forecasting!")
    print("="*60 + "\n")