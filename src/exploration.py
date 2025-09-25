# QCEW Data Exploration
# This script loads QCEW CSV files and examines the data structure, columns, and data types.

import pandas as pd
import os
from pathlib import Path
from data_download import main as download_data

# Set up data directory
data_dir = Path(__file__).parent.parent / "data" / "raw"

# List all CSV files
csv_files = list(data_dir.glob('*.csv'))
if not csv_files:
    print("No CSV files found. Downloading data...")
    download_data()
    csv_files = list(data_dir.glob('*.csv'))

print(f"Found {len(csv_files)} CSV files:")
for file in sorted(csv_files):
    print(f"  {file.name}")

# Load the first file to examine structure
if csv_files:
    sample_file = sorted(csv_files)[0]
    print(f"\nLoading sample file: {sample_file.name}")
    df = pd.read_csv(sample_file)

    print(f"\nData shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")

    print(f"\nFirst 5 rows:")
    print(df.head())

    print(f"\nSummary statistics:")
    print(df.describe())
else:
    print("No CSV files found!")