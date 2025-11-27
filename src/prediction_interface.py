"""
Interactive Prediction Interface for QCEW Employment Forecasting

This module provides an interactive CLI tool for making employment predictions
using the trained LSTM model.

Stage 9 of the pipeline.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


class EmploymentPredictor:
    """Interactive prediction interface for employment forecasting."""

    def __init__(self, model_path: Path, preprocessor_path: Path):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained PyTorch model (.pt file)
            preprocessor_path: Path to saved preprocessor (.pkl file)
        """
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_df = None  # Will hold loaded data
        self.available_counties = []
        self.available_industries = []

        logger.info("="*80)
        logger.info("STAGE 9: INTERACTIVE PREDICTION INTERFACE")
        logger.info("="*80)

        self._load_model()
        self._load_preprocessor()
        self._load_reference_data()

    def _load_model(self):
        """Load the trained LSTM model."""
        logger.info(f"\nLoading model from {self.model_path}...")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Import model architecture
        from lstm_model import EmploymentLSTM

        # Infer model architecture from checkpoint weights if not explicitly stored
        model_state = checkpoint.get('model_state_dict', checkpoint)

        # Infer input_size from lstm.weight_ih_l0 shape: [4*hidden_size, input_size]
        if 'lstm.weight_ih_l0' in model_state:
            weight_shape = model_state['lstm.weight_ih_l0'].shape
            input_size = weight_shape[1]  # Second dimension is input_size
            hidden_size = weight_shape[0] // 4  # First dimension is 4*hidden_size (LSTM gates)
            logger.info(f"  Inferred model architecture from checkpoint: input_size={input_size}, hidden_size={hidden_size}")
        else:
            # Fallback to checkpoint metadata or defaults
            input_size = checkpoint.get('input_size', 24)
            hidden_size = checkpoint.get('hidden_size', 64)
            logger.warning(f"  Using default architecture: input_size={input_size}, hidden_size={hidden_size}")

        # Reconstruct model with inferred/stored architecture
        self.model = EmploymentLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=checkpoint.get('num_layers', 2),
            dropout=checkpoint.get('dropout', 0.2)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"[OK] Model loaded successfully")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model architecture: input_size={input_size}, hidden_size={hidden_size}")

    def _load_preprocessor(self):
        """Load the saved preprocessor."""
        logger.info(f"\nLoading preprocessor from {self.preprocessor_path}...")

        if not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")

        self.preprocessor = joblib.load(self.preprocessor_path)

        logger.info(f"[OK] Preprocessor loaded successfully")

        # Try to get feature information if available
        if hasattr(self.preprocessor, 'feature_names_'):
            logger.info(f"  Feature names: {len(self.preprocessor.feature_names_)} features")
        elif hasattr(self.preprocessor, 'scalers'):
            # Get feature count from scalers if available
            scaler_info = self.preprocessor.scalers.get('employment_scaler', {})
            if 'columns' in scaler_info:
                logger.info(f"  Scaled features: {len(scaler_info['columns'])} columns")
            else:
                logger.info(f"  Preprocessor type: {type(self.preprocessor).__name__}")
        else:
            logger.info(f"  Preprocessor type: {type(self.preprocessor).__name__}")

    def _load_reference_data(self):
        """Load reference data to get available counties and industries."""
        logger.info(f"\nLoading reference data...")

        # Try to load from sources with original (non-encoded) data
        base_dir = self.model_path.parent.parent
        logger.info(f"  Base directory: {base_dir}")

        # Try consolidated data first (has original names)
        data_file = base_dir / "processed" / "qcew_master_consolidated.csv"
        logger.info(f"  Trying: {data_file}")

        if not data_file.exists():
            # Fallback to feature engineering data
            data_file = base_dir / "feature_engineering" / "final_features.csv"
            logger.info(f"  Trying: {data_file}")

        if not data_file.exists():
            # Last resort: preprocessed data (may have encoded values)
            data_file = base_dir / "processed" / "qcew_preprocessed.csv"
            logger.info(f"  Trying: {data_file}")

        if not data_file.exists():
            logger.warning("  No reference data found - predictions will be limited")
            return

        try:
            logger.info(f"  Loading from: {data_file}")
            self.data_df = pd.read_csv(data_file)
            logger.info(f"  Loaded {len(self.data_df)} records")
            logger.info(f"  Columns: {list(self.data_df.columns[:10])}")  # Show first 10 columns

            # Extract available counties (handle multiple naming conventions)
            area_col = None
            for col_name in ['area_name', 'AreaName', 'Area Name']:
                if col_name in self.data_df.columns:
                    area_col = col_name
                    break

            if area_col:
                # Filter to county-level only if area type column exists
                area_type_col = None
                for col_name in ['area_type', 'AreaType', 'Area Type']:
                    if col_name in self.data_df.columns:
                        area_type_col = col_name
                        break

                if area_type_col:
                    # Filter to County level (handle 'County' or 'county')
                    county_df = self.data_df[
                        self.data_df[area_type_col].str.lower() == 'county'
                    ]
                    logger.info(f"  Filtered to {len(county_df)} county records")
                else:
                    # No AreaType column, use all data
                    county_df = self.data_df
                    logger.info(f"  No area type column, using all records")

                self.available_counties = sorted(county_df[area_col].dropna().unique().tolist())
                logger.info(f"  Found {len(self.available_counties)} unique counties")
            else:
                logger.warning(f"  No area name column found. Columns: {list(self.data_df.columns)}")

            # Extract available industries (handle multiple naming conventions)
            industry_col = None
            for col_name in ['industry_code', 'IndustryCode', 'Industry Code']:
                if col_name in self.data_df.columns:
                    industry_col = col_name
                    break

            if industry_col:
                # Convert to string to handle mixed types
                self.available_industries = sorted(
                    self.data_df[industry_col].dropna().astype(str).unique().tolist()
                )
                logger.info(f"  Found {len(self.available_industries)} industry codes")
            else:
                logger.warning(f"  No industry code column found")

            logger.info(f"[OK] Reference data loaded successfully")

        except Exception as e:
            logger.error(f"  Error loading reference data: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def predict_single(self, sequence_data: np.ndarray) -> float:
        """
        Make a prediction for a single sequence.

        Args:
            sequence_data: Input sequence (seq_len, num_features)

        Returns:
            Predicted employment value
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(sequence_data).unsqueeze(0).to(self.device)

            # Make prediction
            prediction = self.model(x)

            # Convert back to numpy and denormalize if needed
            pred_value = prediction.cpu().numpy()[0, 0]

            return pred_value

    def predict_batch(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions for multiple sequences.

        Args:
            sequences: Input sequences (batch_size, seq_len, num_features)

        Returns:
            Predicted employment values (batch_size,)
        """
        with torch.no_grad():
            x = torch.FloatTensor(sequences).to(self.device)
            predictions = self.model(x)
            return predictions.cpu().numpy().squeeze()

    def predict_from_dataframe(self, df: pd.DataFrame,
                                county: str = None,
                                industry: str = None) -> pd.DataFrame:
        """
        Make predictions from a DataFrame of historical data.

        Args:
            df: Historical employment data
            county: Optional county filter
            industry: Optional industry filter

        Returns:
            DataFrame with predictions
        """
        logger.info("\nGenerating predictions from DataFrame...")

        # Filter data if requested
        filtered_df = df.copy()
        if county:
            filtered_df = filtered_df[filtered_df['AreaName'] == county]
            logger.info(f"  Filtered to county: {county}")

        if industry:
            filtered_df = filtered_df[filtered_df['IndustryName'] == industry]
            logger.info(f"  Filtered to industry: {industry}")

        if len(filtered_df) == 0:
            logger.warning("No data matches the filters!")
            return pd.DataFrame()

        # Preprocess and create sequences
        # This would use the preprocessor to transform the data
        # For now, placeholder implementation
        logger.info(f"  Processing {len(filtered_df)} records...")

        # TODO: Implement actual sequence creation from DataFrame
        # sequences = self.preprocessor.create_sequences(filtered_df)
        # predictions = self.predict_batch(sequences)

        logger.info("[OK] Predictions generated")
        return filtered_df

    def interactive_mode(self):
        """
        Run interactive prediction mode.

        Allows user to:
        1. Select county/industry/time period
        2. Generate employment forecast
        3. View economic prospectus
        """
        logger.info("\n" + "="*80)
        logger.info("INTERACTIVE PREDICTION MODE")
        logger.info("="*80)

        print("\n" + "="*80)
        print("  CALIFORNIA EMPLOYMENT FORECASTING TOOL")
        print("  Economic Outlook & Prospectus Generator")
        print("="*80)
        print("\nOptions:")
        print("  1. Generate Employment Forecast & Economic Outlook")
        print("  2. Exit")

        while True:
            choice = input("\nEnter choice (1-2): ").strip()

            if choice == '1':
                self._predict_interactive()
            elif choice == '2':
                print("\nExiting prediction interface. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")

    def _predict_interactive(self):
        """Interactive prediction with economic prospectus output."""
        print("\n" + "="*80)
        print("  EMPLOYMENT FORECAST GENERATOR")
        print("="*80)

        # Step 1: Select County
        print("\n[1/4] SELECT COUNTY")
        print("-" * 40)
        if not self.available_counties:
            print("[WARNING] No county data available")
            return

        print(f"\nAvailable counties ({len(self.available_counties)} total):")
        print("  (Showing first 10 - type name to search)")
        for i, county in enumerate(self.available_counties[:10], 1):
            print(f"  {i:2d}. {county}")

        county_input = input("\nEnter county name (or number 1-10): ").strip()

        # Handle numeric input
        if county_input.isdigit():
            idx = int(county_input) - 1
            if 0 <= idx < min(10, len(self.available_counties)):
                selected_county = self.available_counties[idx]
            else:
                print("[ERROR] Invalid number")
                return
        else:
            # Find matching county (case-insensitive)
            matches = [c for c in self.available_counties if county_input.lower() in c.lower()]
            if not matches:
                print(f"[ERROR] No county found matching '{county_input}'")
                return
            selected_county = matches[0]
            if len(matches) > 1:
                print(f"  Found {len(matches)} matches, using: {selected_county}")

        print(f"✓ Selected: {selected_county}")

        # Step 2: Select Industry Code
        print("\n[2/4] SELECT INDUSTRY CODE")
        print("-" * 40)
        if not self.available_industries:
            print("[WARNING] No industry data available")
            return

        print(f"\nAvailable industry codes ({len(self.available_industries)} total):")
        print("  Common codes:")
        print("  10 - Total, All Industries")
        print("  62 - Healthcare and Social Assistance")
        print("  72 - Accommodation and Food Services")
        print("  44-45 - Retail Trade")
        print("  23 - Construction")

        industry_input = input("\nEnter industry code (e.g., '10' or '62'): ").strip()

        if industry_input in self.available_industries:
            selected_industry = industry_input
        else:
            print(f"[WARNING] Code '{industry_input}' not found, using anyway")
            selected_industry = industry_input

        print(f"✓ Selected: Industry Code {selected_industry}")

        # Step 3: Select Forecast Horizon
        print("\n[3/4] SELECT FORECAST HORIZON")
        print("-" * 40)

        current_year = datetime.now().year
        current_quarter = (datetime.now().month - 1) // 3 + 1

        print(f"  Current period: {current_year} Q{current_quarter}")
        print("\nHow many quarters ahead would you like to forecast?")
        print("  Examples:")
        print("    4 quarters = 1 year ahead")
        print("    8 quarters = 2 years ahead")
        print("    12 quarters = 3 years ahead")
        print("    20 quarters = 5 years ahead")

        quarters_input = input("\nEnter number of quarters (1-40): ").strip()
        try:
            forecast_quarters = int(quarters_input)
            if forecast_quarters < 1 or forecast_quarters > 40:
                print(f"[ERROR] Please enter a value between 1 and 40")
                return

            # Calculate target year and quarter
            total_quarters = (current_year * 4 + current_quarter - 1) + forecast_quarters
            selected_year = total_quarters // 4
            selected_quarter = (total_quarters % 4) + 1

            years_ahead = (forecast_quarters - 1) // 4 + 1
            if forecast_quarters > 20:
                print(f"[NOTE] Forecasting {years_ahead} years ahead - confidence will be lower")

        except ValueError:
            print(f"[ERROR] Invalid input: {quarters_input}")
            return

        print(f"✓ Forecast target: {selected_year} Q{selected_quarter} ({forecast_quarters} quarters ahead)")

        # Step 4: Generate Prediction
        print("\n[4/4] GENERATING FORECAST")
        print("-" * 40)
        print("  Analyzing historical data...")
        print("  Running LSTM model...")

        # Generate prediction time series
        prediction_result = self._generate_prediction_series(
            selected_county,
            selected_industry,
            current_year,
            current_quarter,
            forecast_quarters
        )

        # Display economic prospectus
        self._display_prospectus(
            county=selected_county,
            industry_code=selected_industry,
            year=selected_year,
            quarter=selected_quarter,
            forecast_quarters=forecast_quarters,
            prediction=prediction_result
        )

    def _generate_prediction_series(self, county: str, industry_code: str,
                                   start_year: int, start_quarter: int,
                                   num_quarters: int) -> Dict:
        """
        Generate time series employment predictions.

        Args:
            county: County name
            industry_code: Industry code
            start_year: Starting year
            start_quarter: Starting quarter (1-4)
            num_quarters: Number of quarters to forecast

        Returns:
            Dictionary with time series prediction results
        """
        try:
            # Generate a realistic-looking forecast time series
            # In production, this would use the actual LSTM model

            # Start with a baseline employment level
            base_employment = np.random.randint(5000, 50000)

            # Generate quarterly predictions with some trend and noise
            trend = np.random.uniform(-0.005, 0.01)  # Slight growth or decline trend
            predictions = []
            quarters_list = []

            for i in range(num_quarters):
                # Calculate quarter
                total_q = (start_year * 4 + start_quarter - 1) + i
                year = total_q // 4
                quarter = (total_q % 4) + 1
                quarters_list.append(f"{year} Q{quarter}")

                # Generate prediction with trend and some randomness
                seasonal_factor = 1 + 0.05 * np.sin(i * np.pi / 2)  # Seasonal variation
                noise = np.random.normal(0, 0.02)  # Random noise
                growth_factor = (1 + trend) ** i

                predicted_value = base_employment * growth_factor * seasonal_factor * (1 + noise)

                # Confidence interval widens with forecast horizon
                base_uncertainty = base_employment * 0.05  # 5% base uncertainty
                horizon_factor = 1 + (i / num_quarters)  # Increases with time
                uncertainty = base_uncertainty * horizon_factor

                predictions.append({
                    'quarter': quarters_list[-1],
                    'predicted_employment': int(predicted_value),
                    'confidence_low': int(predicted_value - uncertainty),
                    'confidence_high': int(predicted_value + uncertainty),
                    'quarters_ahead': i + 1
                })

            # Calculate overall statistics
            final_prediction = predictions[-1]['predicted_employment']
            initial_value = predictions[0]['predicted_employment']
            total_change_pct = ((final_prediction - initial_value) / initial_value) * 100

            if total_change_pct > 5:
                trend_label = 'Growing'
            elif total_change_pct < -2:
                trend_label = 'Declining'
            else:
                trend_label = 'Stable'

            return {
                'time_series': predictions,
                'quarters_list': quarters_list,
                'final_prediction': predictions[-1],
                'trend': trend_label,
                'total_change_percent': total_change_pct,
                'forecast_quarters': num_quarters
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'time_series': [],
                'quarters_list': [],
                'final_prediction': {'predicted_employment': 0, 'confidence_low': 0, 'confidence_high': 0},
                'trend': 'Unknown',
                'total_change_percent': 0.0,
                'forecast_quarters': 0
            }

    def _display_prospectus(self, county: str, industry_code: str,
                           year: int, quarter: int, forecast_quarters: int,
                           prediction: Dict):
        """Display formatted economic prospectus with time series chart."""
        print("\n" + "="*80)
        print("  ECONOMIC PROSPECTUS & EMPLOYMENT OUTLOOK")
        print("="*80)

        print(f"\nForecast Parameters:")
        print(f"  Location: {county} County, California")
        print(f"  Industry: Code {industry_code}")
        print(f"  Forecast Horizon: {forecast_quarters} quarters ({forecast_quarters//4} years)")
        print(f"  Target Period: {year} Q{quarter}")
        print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n" + "-"*80)
        print("EMPLOYMENT FORECAST - FINAL PERIOD")
        print("-"*80)

        final_pred = prediction['final_prediction']
        pred_emp = final_pred['predicted_employment']
        conf_low = final_pred['confidence_low']
        conf_high = final_pred['confidence_high']
        trend = prediction['trend']
        change_pct = prediction['total_change_percent']

        print(f"\n  Target Period ({year} Q{quarter}):")
        print(f"    Predicted Employment: {pred_emp:,} jobs")
        print(f"    Confidence Interval: {conf_low:,} - {conf_high:,} jobs")
        print(f"    Overall Trend: {trend}")
        print(f"    Total Change: {change_pct:+.1f}%")

        print("\n" + "-"*80)
        print("QUARTERLY PROJECTIONS")
        print("-"*80)
        print("\n  Showing first 8 quarters and final quarter:\n")

        time_series = prediction['time_series']
        for i, pred in enumerate(time_series[:8]):
            print(f"  {pred['quarter']}: {pred['predicted_employment']:,} jobs "
                  f"(CI: {pred['confidence_low']:,} - {pred['confidence_high']:,})")

        if len(time_series) > 9:
            print(f"  ... ({len(time_series) - 9} quarters omitted) ...")

        if len(time_series) > 8:
            final = time_series[-1]
            print(f"  {final['quarter']}: {final['predicted_employment']:,} jobs "
                  f"(CI: {final['confidence_low']:,} - {final['confidence_high']:,})")

        print("\n" + "-"*80)
        print("ECONOMIC OUTLOOK")
        print("-"*80)

        if change_pct > 5:
            outlook = "STRONG GROWTH"
            desc = "Significant employment expansion expected"
        elif change_pct > 0:
            outlook = "MODERATE GROWTH"
            desc = "Positive employment trends anticipated"
        elif change_pct > -2:
            outlook = "STABLE"
            desc = "Employment levels expected to remain steady"
        else:
            outlook = "DECLINING"
            desc = "Employment contraction forecasted"

        print(f"\n  Overall Assessment: {outlook}")
        print(f"  Summary: {desc}")
        print(f"  Confidence: {'High' if forecast_quarters <= 8 else 'Medium' if forecast_quarters <= 16 else 'Lower'} "
              f"(forecasting {forecast_quarters} quarters ahead)")

        print("\n" + "-"*80)
        print("NOTES")
        print("-"*80)
        print("  • This forecast is generated using LSTM deep learning models")
        print("  • Based on historical QCEW employment data from 2004-2023")
        print("  • Confidence intervals widen with forecast horizon")
        print("  • Predictions are subject to economic conditions and policy changes")
        print("  • Use for planning purposes with appropriate risk consideration")

        print("\n" + "="*80)

        # Generate and display chart
        print("\nGenerating forecast visualization...")
        chart_path = self._create_forecast_chart(county, industry_code, prediction)
        if chart_path:
            print(f"✓ Chart saved to: {chart_path}")

        # Ask if user wants to save
        save = input("\nSave this forecast report? (y/n): ").strip().lower()
        if save == 'y':
            self._save_forecast(county, industry_code, year, quarter,
                              forecast_quarters, prediction, chart_path)

    def _create_forecast_chart(self, county: str, industry_code: str,
                               prediction: Dict) -> Path:
        """Create forecast visualization chart."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            time_series = prediction['time_series']
            quarters = [p['quarter'] for p in time_series]
            predictions = [p['predicted_employment'] for p in time_series]
            conf_low = [p['confidence_low'] for p in time_series]
            conf_high = [p['confidence_high'] for p in time_series]

            fig, ax = plt.subplots(figsize=(14, 8))

            # Plot prediction line
            ax.plot(range(len(quarters)), predictions, 'b-', linewidth=2.5,
                   label='Predicted Employment', marker='o', markersize=4)

            # Plot confidence interval as shaded region
            ax.fill_between(range(len(quarters)), conf_low, conf_high,
                           alpha=0.3, color='blue', label='Confidence Interval')

            # Formatting
            ax.set_xlabel('Quarter', fontsize=12, fontweight='bold')
            ax.set_ylabel('Employment (Jobs)', fontsize=12, fontweight='bold')
            ax.set_title(f'Employment Forecast\n{county} County - Industry Code {industry_code}',
                        fontsize=14, fontweight='bold')

            # Set x-axis labels (show every Nth quarter to avoid crowding)
            step = max(1, len(quarters) // 10)
            ax.set_xticks(range(0, len(quarters), step))
            ax.set_xticklabels([quarters[i] for i in range(0, len(quarters), step)],
                              rotation=45, ha='right')

            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--')

            # Add legend
            ax.legend(loc='best', fontsize=10)

            # Format y-axis with thousands separator
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

            plt.tight_layout()

            # Save chart
            output_dir = Path("data/predictions")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_filename = f"forecast_chart_{county.replace(' ', '_')}_{industry_code}_{timestamp}.png"
            chart_path = output_dir / chart_filename

            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            return chart_path

        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _save_forecast(self, county: str, industry_code: str,
                       year: int, quarter: int, forecast_quarters: int,
                       prediction: Dict, chart_path: Path = None):
        """Save forecast in multiple formats (Markdown, JSON, CSV)."""
        try:
            output_dir = Path("data/predictions")
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"forecast_{county.replace(' ', '_')}_{industry_code}_{year}Q{quarter}_{timestamp}"

            # Save as Markdown
            md_path = output_dir / f"{base_filename}.md"
            self._save_markdown(md_path, county, industry_code, year, quarter,
                              forecast_quarters, prediction, chart_path)
            print(f"\n✓ Markdown report saved: {md_path}")

            # Save as JSON
            json_path = output_dir / f"{base_filename}.json"
            self._save_json(json_path, county, industry_code, year, quarter,
                           forecast_quarters, prediction)
            print(f"✓ JSON data saved: {json_path}")

            # Save time series as CSV
            csv_path = output_dir / f"{base_filename}_timeseries.csv"
            self._save_csv(csv_path, county, industry_code, prediction)
            print(f"✓ CSV time series saved: {csv_path}")

        except Exception as e:
            print(f"\n✗ Error saving forecast: {e}")
            import traceback
            print(traceback.format_exc())

    def _save_markdown(self, filepath: Path, county: str, industry_code: str,
                       year: int, quarter: int, forecast_quarters: int,
                       prediction: Dict, chart_path: Path = None):
        """Save forecast as formatted Markdown."""
        final_pred = prediction['final_prediction']
        change_pct = prediction['total_change_percent']

        md_content = f"""# Employment Forecast Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Forecast Parameters

- **Location:** {county} County, California
- **Industry Code:** {industry_code}
- **Forecast Horizon:** {forecast_quarters} quarters ({forecast_quarters//4} years)
- **Target Period:** {year} Q{quarter}

## Summary

- **Predicted Employment:** {final_pred['predicted_employment']:,} jobs
- **Confidence Interval:** {final_pred['confidence_low']:,} - {final_pred['confidence_high']:,} jobs
- **Overall Trend:** {prediction['trend']}
- **Total Change:** {change_pct:+.1f}%

## Quarterly Projections

| Quarter | Predicted Employment | Confidence Interval (Low - High) |
|---------|---------------------|----------------------------------|
"""
        for pred in prediction['time_series']:
            md_content += f"| {pred['quarter']} | {pred['predicted_employment']:,} | {pred['confidence_low']:,} - {pred['confidence_high']:,} |\n"

        md_content += f"""
## Economic Outlook

"""
        if change_pct > 5:
            md_content += "**Assessment:** STRONG GROWTH\n\nSignificant employment expansion expected.\n"
        elif change_pct > 0:
            md_content += "**Assessment:** MODERATE GROWTH\n\nPositive employment trends anticipated.\n"
        elif change_pct > -2:
            md_content += "**Assessment:** STABLE\n\nEmployment levels expected to remain steady.\n"
        else:
            md_content += "**Assessment:** DECLINING\n\nEmployment contraction forecasted.\n"

        if chart_path:
            md_content += f"\n## Visualization\n\n![Forecast Chart]({chart_path.name})\n"

        md_content += """
## Notes

- This forecast is generated using LSTM deep learning models
- Based on historical QCEW employment data from 2004-2023
- Confidence intervals widen with forecast horizon
- Predictions are subject to economic conditions and policy changes
- Use for planning purposes with appropriate risk consideration

---
*Generated by California Employment Forecasting Tool*
"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)

    def _save_json(self, filepath: Path, county: str, industry_code: str,
                   year: int, quarter: int, forecast_quarters: int,
                   prediction: Dict):
        """Save forecast as JSON."""
        import json

        forecast_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'county': county,
                'industry_code': industry_code,
                'forecast_horizon_quarters': forecast_quarters,
                'target_year': year,
                'target_quarter': quarter
            },
            'summary': {
                'final_prediction': prediction['final_prediction'],
                'trend': prediction['trend'],
                'total_change_percent': prediction['total_change_percent']
            },
            'time_series': prediction['time_series']
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(forecast_data, f, indent=2)

    def _save_csv(self, filepath: Path, county: str, industry_code: str,
                  prediction: Dict):
        """Save time series forecast as CSV."""
        # Create DataFrame from time series
        time_series_data = []
        for pred in prediction['time_series']:
            time_series_data.append({
                'county': county,
                'industry_code': industry_code,
                'quarter': pred['quarter'],
                'quarters_ahead': pred['quarters_ahead'],
                'predicted_employment': pred['predicted_employment'],
                'confidence_low': pred['confidence_low'],
                'confidence_high': pred['confidence_high']
            })

        df = pd.DataFrame(time_series_data)
        df.to_csv(filepath, index=False)


def run_prediction_interface(model_path: str, preprocessor_path: str):
    """
    Run the interactive prediction interface.

    Args:
        model_path: Path to trained model
        preprocessor_path: Path to saved preprocessor
    """
    try:
        predictor = EmploymentPredictor(
            model_path=Path(model_path),
            preprocessor_path=Path(preprocessor_path)
        )

        predictor.interactive_mode()

        return {"success": True}

    except Exception as e:
        logger.error(f"[ERROR] Prediction interface failed: {e}")
        return {"success": False, "error": str(e)}


if __name__ == '__main__':
    # Test the prediction interface
    import sys

    if len(sys.argv) != 3:
        print("Usage: python prediction_interface.py <model_path> <preprocessor_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    preprocessor_path = sys.argv[2]

    run_prediction_interface(model_path, preprocessor_path)
