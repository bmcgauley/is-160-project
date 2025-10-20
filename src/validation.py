"""
QCEW Data Validation Module

This module provides comprehensive validation functions for QCEW employment data,
including range checks, consistency validation, statistical anomaly detection,
and data quality assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QCEWValidator:
    """
    Comprehensive validator for QCEW employment data quality and consistency.
    """

    def __init__(self, data_path: Optional[str] = None, df: pd.DataFrame = None):
        """
        Initialize validator with optional data path or DataFrame.

        Args:
            data_path: Path to consolidated QCEW data file
            df: DataFrame to validate directly (takes precedence over data_path)
        """
        self.data_path = data_path or Path(__file__).parent.parent / "data" / "processed" / "qcew_consolidated.csv"
        self.df = df  # If df is provided, use it directly
        self.validation_results = {}

    def load_data(self) -> pd.DataFrame:
        """Load consolidated QCEW data for validation."""
        if self.df is None:
            logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
        return self.df

    def validate_employment_ranges(self) -> Dict[str, any]:
        """
        Validate employment count ranges and reasonableness.

        Returns:
            Dictionary with validation results
        """
        df = self.load_data()
        results = {
            'total_records': len(df),
            'negative_employment': {},
            'zero_employment_with_estabs': 0,
            'extreme_employment_outliers': {},
            'range_validation': {}
        }

        # Check for negative employment
        for month in ['month1_emplvl', 'month2_emplvl', 'month3_emplvl']:
            negative_count = (df[month] < 0).sum()
            results['negative_employment'][month] = negative_count

        # Check establishments with zero employment
        results['zero_employment_with_estabs'] = (
            (df['qtrly_estabs'] > 0) & (df['month1_emplvl'] == 0)
        ).sum()

        # Check for extreme outliers (beyond reasonable bounds)
        for month in ['month1_emplvl', 'month2_emplvl', 'month3_emplvl']:
            # Define reasonable bounds (0 to 1M employees per industry)
            extreme_high = (df[month] > 1_000_000).sum()
            extreme_low = (df[month] < 0).sum()  # Already checked, but for completeness
            results['extreme_employment_outliers'][month] = {
                'extreme_high': extreme_high,
                'extreme_low': extreme_low
            }

        # Range validation summary
        # Note: Total employment sum is across ALL records including duplicates at different
        # aggregation levels (county/state/national), so it's not a meaningful statistic.
        # We report it but note it's for reference only.
        total_employment = df['month1_emplvl'].sum()
        results['range_validation'] = {
            'total_employment': total_employment,
            'total_employment_note': 'Sum across all records - includes multiple aggregation levels',
            'min_employment': df['month1_emplvl'].min(),
            'max_employment': df['month1_emplvl'].max(),
            'mean_employment': df['month1_emplvl'].mean(),
            'median_employment': df['month1_emplvl'].median()
        }

        self.validation_results['employment_ranges'] = results
        return results

    def validate_wage_consistency(self) -> Dict[str, any]:
        """
        Validate wage data consistency and reasonableness.

        Returns:
            Dictionary with wage validation results
        """
        df = self.load_data()
        results = {
            'zero_wages_with_employment': 0,
            'negative_wages': 0,
            'extreme_wage_outliers': {},
            'wage_distribution': {},
            'wage_consistency_checks': {}
        }

        # Check for zero wages with positive employment
        results['zero_wages_with_employment'] = (
            (df['month1_emplvl'] > 0) & (df['avg_wkly_wage'] == 0)
        ).sum()

        # Check for negative wages
        results['negative_wages'] = (df['avg_wkly_wage'] < 0).sum()

        # Wage outlier detection
        wage_q1 = df['avg_wkly_wage'].quantile(0.25)
        wage_q3 = df['avg_wkly_wage'].quantile(0.75)
        wage_iqr = wage_q3 - wage_q1
        wage_lower_bound = wage_q1 - 3 * wage_iqr  # More lenient for wages
        wage_upper_bound = wage_q3 + 3 * wage_iqr

        results['extreme_wage_outliers'] = {
            'lower_bound': wage_lower_bound,
            'upper_bound': wage_upper_bound,
            'outliers_below': (df['avg_wkly_wage'] < wage_lower_bound).sum(),
            'outliers_above': (df['avg_wkly_wage'] > wage_upper_bound).sum(),
            'total_outliers': ((df['avg_wkly_wage'] < wage_lower_bound) | (df['avg_wkly_wage'] > wage_upper_bound)).sum()
        }

        # Wage distribution statistics
        results['wage_distribution'] = {
            'mean': df['avg_wkly_wage'].mean(),
            'median': df['avg_wkly_wage'].median(),
            'std': df['avg_wkly_wage'].std(),
            'min': df['avg_wkly_wage'].min(),
            'max': df['avg_wkly_wage'].max(),
            'q25': wage_q1,
            'q75': wage_q3
        }

        # Consistency checks
        results['wage_consistency_checks'] = {
            'wages_scale_with_employment': self._check_wage_employment_correlation(df),
            'wage_changes_reasonable': self._check_wage_change_reasonableness(df)
        }

        self.validation_results['wage_consistency'] = results
        return results

    def _check_wage_employment_correlation(self, df: pd.DataFrame) -> Dict[str, float]:
        """Check correlation between wages and employment levels."""
        # Sample a subset to avoid memory issues
        sample_df = df.sample(min(10000, len(df)), random_state=42)

        corr_employment = sample_df['avg_wkly_wage'].corr(sample_df['month1_emplvl'])
        corr_estabs = sample_df['avg_wkly_wage'].corr(sample_df['qtrly_estabs'])

        return {
            'wage_employment_correlation': corr_employment,
            'wage_establishments_correlation': corr_estabs
        }

    def _check_wage_change_reasonableness(self, df: pd.DataFrame) -> Dict[str, int]:
        """Check for unreasonable year-over-year wage changes."""
        # Check if the column exists (may not exist in older data)
        if 'oty_avg_wkly_wage_pct_chg' not in df.columns:
            return {
                'extreme_wage_increases': 0,
                'extreme_wage_decreases': 0,
                'total_extreme_changes': 0,
                'note': 'oty_avg_wkly_wage_pct_chg column not available in data'
            }
        
        # Flag wage changes > 50% or < -50% as potentially problematic
        extreme_increases = (df['oty_avg_wkly_wage_pct_chg'] > 50).sum()
        extreme_decreases = (df['oty_avg_wkly_wage_pct_chg'] < -50).sum()

        return {
            'extreme_wage_increases': extreme_increases,
            'extreme_wage_decreases': extreme_decreases,
            'total_extreme_changes': extreme_increases + extreme_decreases
        }

    def detect_statistical_anomalies(self) -> Dict[str, any]:
        """
        Implement statistical tests for detecting anomalies in quarterly employment changes.

        Returns:
            Dictionary with anomaly detection results
        """
        df = self.load_data()
        results = {
            'employment_change_anomalies': {},
            'seasonal_anomalies': {},
            'industry_anomalies': {},
            'temporal_anomalies': {}
        }

        # Employment change anomalies using Z-score
        for change_col in ['oty_month1_emplvl_pct_chg', 'oty_month2_emplvl_pct_chg', 'oty_month3_emplvl_pct_chg']:
            if change_col in df.columns:
                # Remove infinite and NaN values for analysis
                clean_changes = df[change_col].replace([np.inf, -np.inf], np.nan).dropna()

                if len(clean_changes) > 0:
                    z_scores = np.abs(stats.zscore(clean_changes))
                    anomalies = (z_scores > 3).sum()  # Z-score > 3 is typically anomalous

                    results['employment_change_anomalies'][change_col] = {
                        'total_records': len(clean_changes),
                        'anomalies_detected': anomalies,
                        'anomaly_percentage': (anomalies / len(clean_changes)) * 100,
                        'mean_change': clean_changes.mean(),
                        'std_change': clean_changes.std()
                    }

        # Seasonal pattern analysis
        # Use 'quarter' column if it exists, otherwise fallback to 'qtr'
        quarter_col = 'quarter' if 'quarter' in df.columns else 'qtr'
        df['quarter_label'] = df['year'].astype(str) + 'Q' + df[quarter_col].astype(str)
        quarterly_avg = df.groupby(quarter_col)['month1_emplvl'].mean()
        seasonal_variation = quarterly_avg.std() / quarterly_avg.mean()

        results['seasonal_anomalies'] = {
            'quarterly_averages': quarterly_avg.to_dict(),
            'seasonal_coefficient_variation': seasonal_variation,
            'seasonal_pattern_stable': seasonal_variation < 0.1  # Less than 10% variation
        }

        # Industry-specific anomalies
        industry_stats = df.groupby('industry_code')['month1_emplvl'].agg(['mean', 'std', 'count'])
        industry_cv = industry_stats['std'] / industry_stats['mean']
        high_variation_industries = industry_cv[industry_cv > 2].index.tolist()

        results['industry_anomalies'] = {
            'high_variation_industries': len(high_variation_industries),
            'most_variable_industries': high_variation_industries[:10],
            'industry_coefficient_variation': industry_cv.mean()
        }

        # Temporal continuity check
        expected_quarters = set()
        for year in range(df['year'].min(), df['year'].max() + 1):
            for qtr in range(1, 5):
                expected_quarters.add(f"{year}Q{qtr}")

        actual_quarters = set(df['quarter'].unique())
        missing_quarters = expected_quarters - actual_quarters

        results['temporal_anomalies'] = {
            'expected_quarters': len(expected_quarters),
            'actual_quarters': len(actual_quarters),
            'missing_quarters': list(missing_quarters),
            'continuity_score': (len(actual_quarters) / len(expected_quarters)) * 100
        }

        self.validation_results['statistical_anomalies'] = results
        return results

    def build_data_quality_scorecards(self) -> Dict[str, any]:
        """
        Build data quality scorecards for geographic areas and industry sectors.

        Returns:
            Dictionary with quality scorecards
        """
        df = self.load_data()
        results = {
            'overall_quality_score': 0,
            'industry_quality_scores': {},
            'ownership_quality_scores': {},
            'temporal_quality_scores': {},
            'quality_dimensions': {}
        }

        # Industry quality scores
        industry_groups = df.groupby('industry_code')
        for industry, group in industry_groups:
            if len(group) > 4:  # Need minimum records for meaningful analysis
                completeness = (group[['month1_emplvl', 'avg_wkly_wage']].notna().all(axis=1)).mean()
                consistency = 1 - (group['month1_emplvl'] == 0).mean()  # Lower zero employment is better
                
                # Calculate stability only if oty column exists
                if 'oty_month1_emplvl_pct_chg' in group.columns:
                    stability = 1 / (1 + group['oty_month1_emplvl_pct_chg'].std())  # Lower variation is better
                else:
                    stability = 0.5  # Default mid-range score if data not available

                quality_score = (completeness * 0.4 + consistency * 0.3 + stability * 0.3)
                results['industry_quality_scores'][industry] = {
                    'score': quality_score,
                    'completeness': completeness,
                    'consistency': consistency,
                    'stability': stability,
                    'record_count': len(group)
                }

        # Ownership quality scores
        ownership_col = 'ownership' if 'ownership' in df.columns else 'own_code'
        ownership_groups = df.groupby(ownership_col)
        for ownership, group in ownership_groups:
            completeness = (group[['month1_emplvl', 'avg_wkly_wage']].notna().all(axis=1)).mean()
            coverage = len(group) / len(df)  # Representation in dataset

            quality_score = (completeness * 0.7 + coverage * 0.3)
            results['ownership_quality_scores'][ownership] = {
                'score': quality_score,
                'completeness': completeness,
                'coverage': coverage,
                'record_count': len(group)
            }

        # Temporal quality scores
        year_groups = df.groupby('year')
        for year, group in year_groups:
            # More realistic quarterly completeness: check if we have data for 4 quarters
            quarters_present = group['quarter'].nunique() if 'quarter' in group.columns else group.get('qtr', pd.Series()).nunique()
            quarterly_completeness = min(quarters_present / 4.0, 1.0)  # Cap at 1.0
            data_quality = (group[['month1_emplvl', 'avg_wkly_wage']].notna().all(axis=1)).mean()

            quality_score = (quarterly_completeness * 0.6 + data_quality * 0.4)
            results['temporal_quality_scores'][year] = {
                'score': quality_score,
                'quarterly_completeness': quarterly_completeness,
                'data_quality': data_quality,
                'record_count': len(group),
                'quarters_present': quarters_present
            }

        # Overall quality dimensions
        # Calculate statistical stability only if oty column exists
        if 'oty_month1_emplvl_pct_chg' in df.columns:
            oty_std = df['oty_month1_emplvl_pct_chg'].std()
            stat_stability = 1 / (1 + oty_std) if not np.isnan(oty_std) else 0.5
        else:
            stat_stability = 0.5  # Default mid-range score if data not available
        
        # Temporal coverage: check if we have reasonable quarterly coverage
        # Instead of complex calculation, check average quarters per year
        avg_quarters_per_year = df.groupby('year')['quarter'].nunique().mean() if 'quarter' in df.columns else 0
        temporal_coverage_score = min(avg_quarters_per_year / 4.0, 1.0)  # Cap at 1.0
            
        results['quality_dimensions'] = {
            'data_completeness': (df[['month1_emplvl', 'avg_wkly_wage']].notna().all(axis=1)).mean(),
            'temporal_coverage': temporal_coverage_score,
            'value_consistency': 1 - ((df['month1_emplvl'] == 0) & (df['qtrly_estabs'] > 0)).mean(),
            'statistical_stability': stat_stability
        }

        # Overall quality score (weighted average)
        weights = {'data_completeness': 0.3, 'temporal_coverage': 0.3, 'value_consistency': 0.2, 'statistical_stability': 0.2}
        results['overall_quality_score'] = sum(
            results['quality_dimensions'][dim] * weight
            for dim, weight in weights.items()
        )

        self.validation_results['quality_scorecards'] = results
        return results

    def generate_validation_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report with flagged records and quality metrics.

        Args:
            output_path: Optional path to save detailed report

        Returns:
            Summary report string
        """
        # Run all validations
        self.validate_employment_ranges()
        self.validate_wage_consistency()
        self.detect_statistical_anomalies()
        self.build_data_quality_scorecards()

        # Generate report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("QCEW DATA VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Data file: {self.data_path}")
        report_lines.append("")

        # Overall quality score
        overall_score = self.validation_results.get('quality_scorecards', {}).get('overall_quality_score', 0)
        report_lines.append(f"OVERALL DATA QUALITY SCORE: {overall_score:.3f} ({overall_score*100:.1f}%)")
        report_lines.append("")

        # Employment validation summary
        emp_results = self.validation_results.get('employment_ranges', {})
        report_lines.append("EMPLOYMENT VALIDATION SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Total records: {emp_results.get('total_records', 0):,}")
        report_lines.append(f"Zero employment with establishments: {emp_results.get('zero_employment_with_estabs', 0):,}")

        for month, outliers in emp_results.get('extreme_employment_outliers', {}).items():
            report_lines.append(f"{month} extreme outliers: {outliers.get('extreme_high', 0):,}")

        range_val = emp_results.get('range_validation', {})
        # Note: Don't report total employment as it's misleading with multiple aggregation levels
        report_lines.append(f"Mean employment per record: {range_val.get('mean_employment', 0):,.0f}")
        report_lines.append(f"Median employment per record: {range_val.get('median_employment', 0):,.0f}")
        report_lines.append(f"Employment range: {range_val.get('min_employment', 0):,.0f} - {range_val.get('max_employment', 0):,.0f}")
        report_lines.append("")

        # Wage validation summary
        wage_results = self.validation_results.get('wage_consistency', {})
        report_lines.append("WAGE VALIDATION SUMMARY:")
        report_lines.append("-" * 30)
        report_lines.append(f"Zero wages with employment: {wage_results.get('zero_wages_with_employment', 0):,}")
        report_lines.append(f"Negative wages: {wage_results.get('negative_wages', 0):,}")

        wage_dist = wage_results.get('wage_distribution', {})
        report_lines.append(f"Average weekly wage: ${wage_dist.get('mean', 0):,.0f}")
        report_lines.append(f"Wage range: ${wage_dist.get('min', 0):,.0f} - ${wage_dist.get('max', 0):,.0f}")
        report_lines.append("")

        # Anomaly detection summary
        anomaly_results = self.validation_results.get('statistical_anomalies', {})
        report_lines.append("ANOMALY DETECTION SUMMARY:")
        report_lines.append("-" * 35)

        for change_col, stats in anomaly_results.get('employment_change_anomalies', {}).items():
            report_lines.append(f"{change_col} anomalies: {stats.get('anomalies_detected', 0):,} ({stats.get('anomaly_percentage', 0):.1f}%)")

        temporal_anom = anomaly_results.get('temporal_anomalies', {})
        report_lines.append(f"Temporal continuity: {temporal_anom.get('continuity_score', 0):.1f}%")
        if temporal_anom.get('missing_quarters'):
            report_lines.append(f"Missing quarters: {', '.join(temporal_anom['missing_quarters'])}")
        report_lines.append("")

        # Quality dimensions
        quality_dims = self.validation_results.get('quality_scorecards', {}).get('quality_dimensions', {})
        report_lines.append("DATA QUALITY DIMENSIONS:")
        report_lines.append("-" * 30)
        for dim, score in quality_dims.items():
            report_lines.append(f"{dim.replace('_', ' ').title()}: {score:.3f} ({score*100:.1f}%)")
        report_lines.append("")

        # Recommendations
        report_lines.append("VALIDATION RECOMMENDATIONS:")
        report_lines.append("-" * 35)

        issues = []
        if emp_results.get('zero_employment_with_estabs', 0) > 0:
            issues.append("Review records with establishments but zero employment")
        if wage_results.get('zero_wages_with_employment', 0) > 0:
            issues.append("Investigate records with employment but zero wages")
        if overall_score < 0.8:
            issues.append("Overall data quality needs improvement")
        if temporal_anom.get('missing_quarters'):
            issues.append("Address temporal gaps in data coverage")

        if issues:
            for issue in issues:
                report_lines.append(f"  - {issue}")
        else:
            report_lines.append("  - Data quality is acceptable for analysis")
        report_lines.append("")

        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Save detailed report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Detailed validation report saved to {output_path}")

        return report_text

def validate_data_quality(df: pd.DataFrame, output_file) -> Dict[str, any]:
    """
    Wrapper function to validate data quality and save validated data.
    
    Args:
        df: DataFrame to validate
        output_file: Path to save validated data (string or Path object)
        
    Returns:
        Dictionary with validation results
    """
    logger.info(f"Validating data quality for {len(df):,} records")
    
    # Convert output_file to Path object if it's a string
    output_file = Path(output_file) if isinstance(output_file, str) else output_file
    
    # Initialize validator with the DataFrame
    validator = QCEWValidator(df=df)
    
    # Run all validation checks
    logger.info("Running employment range validation...")
    emp_results = validator.validate_employment_ranges()
    
    logger.info("Running wage consistency validation...")
    wage_results = validator.validate_wage_consistency()
    
    logger.info("Running statistical anomaly detection...")
    anomaly_results = validator.detect_statistical_anomalies()
    
    logger.info("Generating quality scorecards...")
    quality_results = validator.build_data_quality_scorecards()
    
    # Generate validation report
    report_dir = output_file.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "validation_report.txt"
    report = validator.generate_validation_report(report_path)
    
    # Save validated data (original data passes through)
    logger.info(f"Saving validated data to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    # Extract temporal info from anomaly results
    temporal_results = anomaly_results.get('temporal_anomalies', {})
    
    # Combine all results
    validation_summary = {
        'employment_ranges': emp_results,
        'wage_consistency': wage_results,
        'statistical_anomalies': anomaly_results,
        'temporal_continuity': temporal_results,
        'quality_scorecards': quality_results,
        'overall_score': quality_results.get('overall_quality_score', 0),
        'validated_file': str(output_file),
        'report_file': str(report_path)
    }
    
    logger.info(f"\n[OK] Validation complete!")
    logger.info(f"Overall Quality Score: {validation_summary['overall_score']:.3f}")
    logger.info(f"Validated data saved to: {output_file}")
    logger.info(f"Report saved to: {report_path}")
    
    return validation_summary


def main():
    """Main function to run complete validation suite."""
    validator = QCEWValidator()

    # Generate validation report
    report_path = Path(__file__).parent.parent / "data" / "processed" / "validation_report.txt"
    report = validator.generate_validation_report(report_path)

    print("QCEW Data Validation Complete")
    print("=" * 50)
    print(report)

if __name__ == "__main__":
    main()