"""
Automated data acquisition and aggregation script for QCEW data.

This script handles the complete data acquisition pipeline:
1. Downloads QCEW CSV files from BLS
2. Validates downloaded data
3. Performs initial data aggregation and cleaning
4. Saves processed data to appropriate directories
"""

import os
import pandas as pd
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import from our data_download module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from data_download import download_qcew_data, START_YEAR, END_YEAR, QUARTERS, AREA_CODE, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROCESSED_DIR = Path("data/processed")
VALIDATED_DIR = Path("data/validated")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class DataAcquisitionError(Exception):
    """Custom exception for data acquisition errors."""
    pass

def validate_csv_file(filepath: Path) -> bool:
    """
    Validate that a CSV file has the expected structure and data.

    Args:
        filepath: Path to the CSV file

    Returns:
        bool: True if file is valid
    """
    try:
        # Check if file exists and has content
        if not filepath.exists() or filepath.stat().st_size == 0:
            logger.error(f"File {filepath} does not exist or is empty")
            return False

        # Try to read the CSV
        df = pd.read_csv(filepath)

        # Check for required columns (based on QCEW data structure)
        required_columns = ['area_fips', 'year', 'qtr', 'qtrly_estabs', 'total_qtrly_wages']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"File {filepath} missing required columns: {missing_columns}")
            return False

        # Check for reasonable data ranges
        if len(df) == 0:
            logger.error(f"File {filepath} contains no data rows")
            return False

        # Check year and quarter ranges
        if not (df['year'].between(START_YEAR, END_YEAR).all()):
            logger.warning(f"File {filepath} contains unexpected years")

        if not (df['qtr'].between(1, 4).all()):
            logger.warning(f"File {filepath} contains unexpected quarters")

        logger.info(f"✓ File {filepath} validated successfully ({len(df)} rows)")
        return True

    except Exception as e:
        logger.error(f"Error validating {filepath}: {e}")
        return False

def download_with_retry(year: int, quarter: int, max_retries: int = MAX_RETRIES) -> bool:
    """
    Download QCEW data with retry logic.

    Args:
        year: Year to download
        quarter: Quarter to download
        max_retries: Maximum number of retry attempts

    Returns:
        bool: True if download successful
    """
    for attempt in range(max_retries):
        try:
            if download_qcew_data(year, quarter):
                filepath = DATA_DIR / f"qcew_{AREA_CODE}_{year}_q{quarter}.csv"
                if validate_csv_file(filepath):
                    return True
                else:
                    logger.warning(f"Downloaded file failed validation, retrying...")
            else:
                logger.warning(f"Download failed, attempt {attempt + 1}/{max_retries}")

        except Exception as e:
            logger.error(f"Error during download attempt {attempt + 1}: {e}")

        if attempt < max_retries - 1:
            time.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff

    return False

def aggregate_quarterly_data() -> Optional[pd.DataFrame]:
    """
    Aggregate all quarterly CSV files into a single DataFrame.

    Returns:
        pd.DataFrame: Combined quarterly data, or None if failed
    """
    logger.info("Aggregating quarterly data...")

    all_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        for quarter in QUARTERS:
            filename = f"qcew_{AREA_CODE}_{year}_q{quarter}.csv"
            filepath = DATA_DIR / filename

            if not filepath.exists():
                logger.warning(f"File {filename} not found, skipping")
                continue

            try:
                df = pd.read_csv(filepath)

                # Add source file info for tracking
                df['source_file'] = filename

                all_data.append(df)
                logger.info(f"Loaded {filename}: {len(df)} rows")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                continue

    if not all_data:
        logger.error("No data files could be loaded")
        return None

    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)

    # Sort by area, year, quarter for consistency
    combined_df = combined_df.sort_values(['area_fips', 'year', 'qtr'])

    logger.info(f"Combined data: {len(combined_df)} total rows from {len(all_data)} files")
    return combined_df

def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the aggregated data.

    Args:
        df: Raw aggregated DataFrame

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame
    """
    logger.info("Cleaning and validating data...")

    original_rows = len(df)

    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"Removed {original_rows - len(df)} duplicate rows")

    # Handle missing values in critical columns
    critical_columns = ['area_fips', 'year', 'qtr', 'qtrly_estabs', 'total_qtrly_wages']
    df = df.dropna(subset=critical_columns)

    # Convert data types
    df['area_fips'] = df['area_fips'].astype(str)
    df['year'] = df['year'].astype(int)
    df['qtr'] = df['qtr'].astype(int)

    # Convert numeric columns
    numeric_columns = ['qtrly_estabs', 'month1_emplvl', 'month2_emplvl', 'month3_emplvl',
                      'total_qtrly_wages', 'taxable_qtrly_wages', 'qtrly_contributions', 'avg_wkly_wage']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with invalid employment counts (negative values)
    df = df[df['qtrly_estabs'] >= 0]

    # Add derived columns
    df['quarter_start'] = pd.to_datetime(df['year'].astype(str) + 'Q' + df['qtr'].astype(str))
    df['avg_monthly_emplvl'] = df[['month1_emplvl', 'month2_emplvl', 'month3_emplvl']].mean(axis=1)

    logger.info(f"Data cleaning complete: {len(df)} rows remaining")
    return df

def save_processed_data(df: pd.DataFrame) -> bool:
    """
    Save processed data to the appropriate directories.

    Args:
        df: Processed DataFrame

    Returns:
        bool: True if save successful
    """
    try:
        # Ensure directories exist
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        VALIDATED_DIR.mkdir(parents=True, exist_ok=True)

        # Save processed data
        processed_file = PROCESSED_DIR / f"qcew_california_processed_{START_YEAR}_{END_YEAR}.csv"
        df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed data to {processed_file}")

        # Save validated data (same as processed for now, but could add more validation)
        validated_file = VALIDATED_DIR / f"qcew_california_validated_{START_YEAR}_{END_YEAR}.csv"
        df.to_csv(validated_file, index=False)
        logger.info(f"Saved validated data to {validated_file}")

        # Save summary statistics
        summary_file = PROCESSED_DIR / f"qcew_california_summary_{START_YEAR}_{END_YEAR}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"QCEW California Data Summary\n")
            f.write(f"Date Range: {START_YEAR} Q1 - {END_YEAR} Q4\n")
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Unique Areas: {df['area_fips'].nunique()}\n")
            f.write(f"Years Covered: {sorted(df['year'].unique())}\n")
            f.write(f"Data Columns: {list(df.columns)}\n")

            # Basic statistics
            f.write("\nEmployment Statistics:\n")
            f.write(f"Total Establishments: {df['qtrly_estabs'].sum():,.0f}\n")
            f.write(f"Average Quarterly Wages: ${df['total_qtrly_wages'].mean():,.0f}\n")
            f.write(f"Average Weekly Wage: ${df['avg_wkly_wage'].mean():,.2f}\n")

        logger.info(f"Saved summary to {summary_file}")

        return True

    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        return False

def run_data_acquisition_pipeline(use_parallel: bool = False) -> bool:
    """
    Run the complete data acquisition pipeline.

    Args:
        use_parallel: Whether to download files in parallel

    Returns:
        bool: True if pipeline completed successfully
    """
    logger.info("Starting QCEW data acquisition pipeline...")

    # Ensure data directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Download all required files
    logger.info("Step 1: Downloading QCEW data files...")

    download_tasks = []
    for year in range(START_YEAR, END_YEAR + 1):
        for quarter in QUARTERS:
            download_tasks.append((year, quarter))

    successful_downloads = 0

    if use_parallel:
        # Parallel downloads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download_with_retry, year, quarter)
                      for year, quarter in download_tasks]

            for future in as_completed(futures):
                if future.result():
                    successful_downloads += 1
    else:
        # Sequential downloads
        for year, quarter in download_tasks:
            if download_with_retry(year, quarter):
                successful_downloads += 1

    total_expected = len(download_tasks)
    logger.info(f"Download phase complete: {successful_downloads}/{total_expected} files downloaded")

    if successful_downloads < total_expected * 0.8:  # Require at least 80% success
        logger.error("Too many downloads failed, aborting pipeline")
        return False

    # Step 2: Aggregate data
    logger.info("Step 2: Aggregating quarterly data...")
    raw_data = aggregate_quarterly_data()
    if raw_data is None:
        logger.error("Data aggregation failed")
        return False

    # Step 3: Clean and validate
    logger.info("Step 3: Cleaning and validating data...")
    cleaned_data = clean_and_validate_data(raw_data)

    # Step 4: Save processed data
    logger.info("Step 4: Saving processed data...")
    if not save_processed_data(cleaned_data):
        logger.error("Failed to save processed data")
        return False

    logger.info("✓ Data acquisition pipeline completed successfully")
    return True

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="QCEW Data Acquisition Pipeline")
    parser.add_argument("--parallel", action="store_true",
                       help="Download files in parallel")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate existing data without downloading")

    args = parser.parse_args()

    if args.validate_only:
        # Just validate existing files
        logger.info("Validating existing data files...")
        all_valid = True
        for year in range(START_YEAR, END_YEAR + 1):
            for quarter in QUARTERS:
                filename = f"qcew_{AREA_CODE}_{year}_q{quarter}.csv"
                filepath = DATA_DIR / filename
                if not validate_csv_file(filepath):
                    all_valid = False

        if all_valid:
            logger.info("✓ All existing files are valid")
            return True
        else:
            logger.error("✗ Some files failed validation")
            return False
    else:
        # Run full pipeline
        return run_data_acquisition_pipeline(use_parallel=args.parallel)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)