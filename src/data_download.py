"""
Automated data download script for QCEW CSV files from Bureau of Labor Statistics.

This script downloads Quarterly Census of Employment and Wages (QCEW) data
for California from the BLS open data API.
"""

import os
import requests
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://data.bls.gov/cew/data/api"
AREA_CODE = "06000"  # California statewide
DATA_DIR = Path("data/raw")

# Download parameters
START_YEAR = 2020
END_YEAR = 2024
QUARTERS = [1, 2, 3, 4]

def download_qcew_data(year: int, quarter: int, area_code: str = AREA_CODE) -> bool:
    """
    Download QCEW data for a specific year and quarter.

    Args:
        year: The year to download data for
        quarter: The quarter to download data for (1-4)
        area_code: The area code (default: California statewide)

    Returns:
        bool: True if download successful, False otherwise
    """
    url = f"{BASE_URL}/{year}/{quarter}/area/{area_code}.csv"
    filename = f"qcew_{area_code}_{year}_q{quarter}.csv"
    filepath = DATA_DIR / filename

    # Skip if file already exists
    if filepath.exists():
        logger.info(f"File {filename} already exists, skipping download")
        return True

    try:
        logger.info(f"Downloading {filename} from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save the file
        with open(filepath, 'wb') as f:
            f.write(response.content)

        logger.info(f"Successfully downloaded {filename}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {filename}: {e}")
        return False

def main():
    """Main function to download all QCEW data."""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting QCEW data download for California")
    logger.info(f"Downloading data from {START_YEAR} to {END_YEAR}")

    total_files = len(QUARTERS) * (END_YEAR - START_YEAR + 1)
    successful_downloads = 0

    for year in range(START_YEAR, END_YEAR + 1):
        for quarter in QUARTERS:
            if download_qcew_data(year, quarter):
                successful_downloads += 1

            # Add a small delay to be respectful to the server
            time.sleep(0.5)

    logger.info(f"Download complete: {successful_downloads}/{total_files} files downloaded successfully")

    if successful_downloads == total_files:
        logger.info("All downloads completed successfully")
        return True
    else:
        logger.warning(f"Some downloads failed: {total_files - successful_downloads} files missing")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)