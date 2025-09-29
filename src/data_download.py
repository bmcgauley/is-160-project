"""
Automated data download script for QCEW CSV files from California Open Data Portal.

This script downloads Quarterly Census of Employment and Wages (QCEW) data
for California from the California Open Data Portal, which provides historical
data from 2004 onwards.
"""

import os
import requests
import time
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BASE_URL = "https://data.ca.gov/dataset/3f08b68e-1d1a-4ba4-a07d-1ec3392ed191/resource"
AREA_CODE = "06000"  # California statewide
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

# Download parameters - Extended historical range
START_YEAR = 2004
END_YEAR = 2025
QUARTERS = [1, 2, 3, 4]

# California Open Data resource IDs for different year ranges
RESOURCE_MAP = {
    (2004, 2007): "577beabf-3f53-4848-807f-adfd0551831c",
    (2008, 2011): "0f196f1d-4447-479b-bef5-fb16a9093be9",
    (2012, 2015): "51b27b21-bbd2-48ba-9f6d-997a282c5fce",
    (2016, 2019): "78a9d6ee-ec9a-4c25-ae34-5bac44010cb2",
    (2020, 2022): "ca165fad-4f16-48c7-808e-8b5222bc4182",
    (2023, 2025): "119eef38-3b59-499f-8f7c-9bea4768469d",
}

def convert_to_bls_format(df_ca: pd.DataFrame, year: int, quarter: int, area_code: str) -> pd.DataFrame:
    """
    Convert California Open Data format to BLS-compatible format.

    Args:
        df_ca: DataFrame from California Open Data
        year: Year of the data
        quarter: Quarter of the data
        area_code: Area FIPS code

    Returns:
        pd.DataFrame: DataFrame in BLS format
    """
    if df_ca.empty:
        # Return empty dataframe with correct columns
        return pd.DataFrame(columns=[
            'area_fips', 'own_code', 'industry_code', 'agglvl_code', 'size_code',
            'year', 'qtr', 'disclosure_code', 'qtrly_estabs', 'month1_emplvl',
            'month2_emplvl', 'month3_emplvl', 'total_qtrly_wages', 'taxable_qtrly_wages',
            'qtrly_contributions', 'avg_wkly_wage'
        ])

    # Map ownership codes
    ownership_map = {
        'Private': 1,
        'State Government': 2,
        'Local Government': 3,
        'Federal Government': 4
    }

    # Create BLS format dataframe with the same number of rows
    df_bls = pd.DataFrame(index=df_ca.index)

    # Basic fields
    df_bls['area_fips'] = area_code
    df_bls['own_code'] = df_ca['Ownership'].map(ownership_map).fillna(1).astype(int)
    df_bls['industry_code'] = df_ca['NAICS Code'].astype(str)
    df_bls['agglvl_code'] = df_ca['NAICS Level'].map({
        2: 50,  # Total by ownership
        3: 51,  # Total by ownership - supersector
        4: 52,  # Total by ownership - sector
        5: 53,  # Total by ownership - 3-digit NAICS
        6: 54   # Total by ownership - 4-6 digit NAICS
    }).fillna(54).astype(int)
    df_bls['size_code'] = 0  # All sizes
    df_bls['year'] = year
    df_bls['qtr'] = quarter
    df_bls['disclosure_code'] = ''  # No disclosure info in CA data

    # Employment and wage fields
    df_bls['qtrly_estabs'] = df_ca['Establishments'].fillna(0).astype(int)
    df_bls['month1_emplvl'] = df_ca['1st Month Emp'].fillna(0).astype(int)
    df_bls['month2_emplvl'] = df_ca['2nd Month Emp'].fillna(0).astype(int)
    df_bls['month3_emplvl'] = df_ca['3rd Month Emp'].fillna(0).astype(int)
    df_bls['total_qtrly_wages'] = df_ca['Total Wages (All Workers)'].fillna(0).astype(int)
    df_bls['taxable_qtrly_wages'] = 0  # Not available in CA data
    df_bls['qtrly_contributions'] = 0  # Not available in CA data
    df_bls['avg_wkly_wage'] = df_ca['Average Weekly Wages'].fillna(0).astype(int)

    return df_bls

def download_qcew_data(year: int, quarter: int, area_code: str = AREA_CODE) -> bool:
    """
    Download QCEW data for a specific year and quarter.

    For years 2004+, downloads from California Open Data Portal and converts to BLS format.
    For earlier years, would need alternative sources.

    Args:
        year: The year to download data for
        quarter: The quarter to download data for (1-4)
        area_code: The area code (default: California statewide)

    Returns:
        bool: True if download successful, False otherwise
    """
    filename = f"qcew_{area_code}_{year}_q{quarter}.csv"
    filepath = DATA_DIR / filename

    # Skip if file already exists
    if filepath.exists():
        logger.info(f"File {filename} already exists, skipping download")
        return True

    # Find the appropriate resource for this year
    resource_id = None
    for (start, end), rid in RESOURCE_MAP.items():
        if start <= year <= end:
            resource_id = rid
            break

    if not resource_id:
        logger.error(f"No data source available for year {year}")
        return False

    url = f"{BASE_URL}/{resource_id}/download/qcew_{start}-{end}.csv"

    try:
        logger.info(f"Downloading data for years {start}-{end} from {url}")

        # Download the consolidated file
        response = requests.get(url, timeout=300)  # 5 minute timeout for large files
        response.raise_for_status()

        # Parse the CSV data
        # Note: California data uses different encoding, try multiple
        try:
            df = pd.read_csv(pd.io.common.StringIO(response.text), low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(pd.io.common.StringIO(response.text), encoding='latin1', low_memory=False)

        # Filter for California statewide data only
        statewide_mask = (df['Area Type'] == 'California - Statewide') & (df['Area Name'] == 'California')
        df_statewide = df[statewide_mask].copy()

        if df_statewide.empty:
            logger.error(f"No statewide California data found in downloaded file")
            return False

        # Filter for the specific year and quarter
        year_mask = df_statewide['Year'] == year
        quarter_map = {1: '1st Qtr', 2: '2nd Qtr', 3: '3rd Qtr', 4: '4th Qtr'}
        quarter_mask = df_statewide['Quarter'] == quarter_map[quarter]
        df_filtered = df_statewide[year_mask & quarter_mask]

        if df_filtered.empty:
            logger.warning(f"No data found for {year} Q{quarter} in downloaded file")
            # Create empty dataframe with correct structure
            df_filtered = pd.DataFrame(columns=[
                'area_fips', 'own_code', 'industry_code', 'agglvl_code', 'size_code',
                'year', 'qtr', 'disclosure_code', 'qtrly_estabs', 'month1_emplvl',
                'month2_emplvl', 'month3_emplvl', 'total_qtrly_wages', 'taxable_qtrly_wages',
                'qtrly_contributions', 'avg_wkly_wage'
            ])

        # Convert to BLS-compatible format
        df_bls = convert_to_bls_format(df_filtered, year, quarter, area_code)

        # Save the converted data
        df_bls.to_csv(filepath, index=False)

        logger.info(f"Successfully downloaded and converted {filename} ({len(df_bls)} rows)")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download data for years {start}-{end}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing data for {year} Q{quarter}: {e}")
        return False

def main():
    """Main function to download all QCEW data."""
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Starting QCEW data download for California from California Open Data Portal")
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