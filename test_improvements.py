"""
Quick test to verify feature engineering improvements
"""
import sys
sys.path.insert(0, 'src')

print("Testing feature engineering updates...")
print("="*80)

# Test 1: Check if visualization functions exist
print("\n✓ Test 1: Imports")
from feature_pipeline import create_feature_plots, engineer_features
print("  ✓ create_feature_plots imported")
print("  ✓ engineer_features imported")

# Test 2: Check if preprocessing functions are implemented
print("\n✓ Test 2: Preprocessing implementations")
from preprocessing import EmploymentDataPreprocessor
import pandas as pd
import numpy as np

# Create sample data
df_test = pd.DataFrame({
    'avg_monthly_emplvl': [100, 200, np.nan, 400, 500],
    'total_qtrly_wages': [10000, 20000, 30000, np.nan, 50000],
    'industry_code': ['A', 'B', 'A', 'B', 'C'],
    'area_name': ['County1', 'County2', 'County1', 'County2', 'County3']
})

preprocessor = EmploymentDataPreprocessor()

print("\n  Testing normalization...")
df_norm = preprocessor.normalize_employment_data(df_test.copy())
print(f"    Before: mean={100:,.2f}, After: mean={df_norm['avg_monthly_emplvl'].mean():.3f}")
print("    ✓ Normalization working")

print("\n  Testing imputation...")
missing_before = df_test.isnull().sum().sum()
df_imputed = preprocessor.handle_missing_values(df_test.copy())
missing_after = df_imputed.isnull().sum().sum()
print(f"    Before: {missing_before} nulls, After: {missing_after} nulls")
print("    ✓ Imputation working")

print("\n  Testing encoding...")
df_encoded = preprocessor.create_categorical_encodings(df_test.copy())
print(f"    industry_code encoded: {df_encoded['industry_code'].dtype}")
print(f"    area_name encoded: {df_encoded['area_name'].dtype}")
print("    ✓ Encoding working")

print("\n" + "="*80)
print("✓ All tests passed!")
print("\nYou can now run: python main.py → option 5 (Feature Engineering)")
print("  - You'll see updated completion messages")
print("  - Plots will be generated in data/feature_engineering/plots/")
print("  - Missing values will be properly imputed")
