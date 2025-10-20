"""
Quick test script to validate T033 quarterly filtering fix
"""
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from feature_engineering import handle_annual_vs_quarterly

# Load validated data
print("Loading validated data...")
df = pd.read_csv('data/validated/qcew_validated.csv', nrows=10000)
print(f"Loaded {len(df):,} records (sample)")

# Check quarter values
print("\nQuarter values in data:")
print(df['quarter'].unique())

# Test filtering
print("\nTesting quarterly filtering...")
result = handle_annual_vs_quarterly(df)

print(f"\nResults:")
print(f"  Input: {len(df):,} records")
print(f"  Output: {len(result):,} records")
print(f"  Removed: {len(df) - len(result):,} records")

if len(result) > 0:
    print("\n✓ SUCCESS - Filtering worked!")
    print(f"Quarter values in output: {result['quarter'].unique()}")
else:
    print("\n✗ FAILED - All records filtered out!")
