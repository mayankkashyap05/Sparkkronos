# data_loader.py

"""
Data Loader Module
Handles loading CSV files from the processed data directory
"""

import pandas as pd
import os
from pathlib import Path


class DataLoader:
    def __init__(self, data_dir="data/processed"):
        """
        Initialize the DataLoader
        
        Args:
            data_dir (str): Path to the directory containing processed CSV files
        """
        self.data_dir = Path(data_dir)
        
    def list_csv_files(self):
        """
        List all CSV files in the processed data directory
        
        Returns:
            list: List of CSV file paths
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory '{self.data_dir}' not found!")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in '{self.data_dir}'")
        
        return csv_files
    
    def select_file(self):
        """
        Prompt user to select a CSV file from available files
        
        Returns:
            Path: Selected file path
        """
        csv_files = self.list_csv_files()
        
        print("\n" + "="*50)
        print("Available CSV files:")
        print("="*50)
        
        for idx, file_path in enumerate(csv_files, 1):
            file_size = file_path.stat().st_size / 1024  # Size in KB
            print(f"{idx}. {file_path.name} ({file_size:.2f} KB)")
        
        print("="*50)
        
        while True:
            try:
                choice = int(input(f"\nSelect a file (1-{len(csv_files)}): "))
                if 1 <= choice <= len(csv_files):
                    selected_file = csv_files[choice - 1]
                    print(f"\n✓ Selected: {selected_file.name}\n")
                    return selected_file
                else:
                    print(f"Please enter a number between 1 and {len(csv_files)}")
            except ValueError:
                print("Invalid input! Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                exit(0)
    
    def load_data(self, file_path=None):
        """
        Load CSV data into a pandas DataFrame
        
        Args:
            file_path (str or Path, optional): Path to CSV file. 
                                              If None, prompts user to select.
        
        Returns:
            pd.DataFrame: Loaded data with parsed timestamps
        """
        if file_path is None:
            file_path = self.select_file()
        else:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File '{file_path}' not found!")
        
        print(f"Loading data from {file_path.name}...")
        
        # Load CSV with timestamp parsing
        df = pd.read_csv(
            file_path,
            parse_dates=['timestamps'],
            index_col='timestamps'
        )
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"✓ Data loaded successfully!")
        print(f"  - Shape: {df.shape}")
        print(f"  - Date range: {df.index.min()} to {df.index.max()}")
        print(f"  - Columns: {list(df.columns)}\n")
        
        return df


def load_csv_data(data_dir="data/processed", file_path=None):
    """
    Convenience function to load CSV data
    
    Args:
        data_dir (str): Directory containing processed CSV files
        file_path (str, optional): Specific file to load. If None, prompts user.
    
    Returns:
        pd.DataFrame: Loaded data
    
    Example:
        >>> from src.data.data_loader import load_csv_data
        >>> df = load_csv_data()  # Interactive selection
        >>> df = load_csv_data(file_path="data/processed/test_processed.csv")  # Direct load
    """
    loader = DataLoader(data_dir)
    return loader.load_data(file_path)


if __name__ == "__main__":
    # Test the data loader
    df = load_csv_data()
    print("\nDataFrame Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())