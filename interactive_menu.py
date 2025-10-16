"""
Interactive Menu Interface for QCEW Employment Forecasting Pipeline

Provides a user-friendly menu-driven interface instead of command-line arguments.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from logging_config import setup_logging
from pipeline_orchestrator import QCEWPipeline

logger = setup_logging()


def print_banner():
    """Print application banner."""
    print("\n" + "="*80)
    print(" " * 15 + "QCEW EMPLOYMENT FORECASTING PIPELINE")
    print(" " * 20 + "RNN/LSTM Time Series Analysis")
    print("="*80 + "\n")


def print_menu():
    """Print main menu options."""
    print("\n" + "-"*80)
    print("MAIN MENU")
    print("-"*80)
    print("  1. Run Full Pipeline (All Stages)")
    print("  2. Data Consolidation")
    print("  3. Data Exploration & Visualization")
    print("  4. Data Validation")
    print("  5. Feature Engineering")
    print("  6. Data Preprocessing")
    print("  7. Train LSTM Model")
    print("  8. Evaluate Model")
    print("  9. Interactive Prediction Interface")
    print(" 10. View Pipeline Status")
    print("  0. Exit")
    print("-"*80)


def check_pipeline_status(pipeline: QCEWPipeline):
    """Check and display which pipeline stages have been completed."""
    print("\n" + "-"*80)
    print("PIPELINE STATUS")
    print("-"*80)
    
    status = []
    status.append(("1. Data Consolidation", pipeline.consolidated_file.exists()))
    status.append(("2. Data Exploration", (pipeline.plots_dir / "employment_trends.png").exists()))
    status.append(("3. Data Validation", pipeline.validated_file.exists()))
    status.append(("4. Feature Engineering", pipeline.features_file.exists()))
    status.append(("5. Data Preprocessing", pipeline.preprocessed_file.exists()))
    status.append(("6. Model Training", pipeline.model_file.exists()))
    status.append(("7. Model Evaluation", (pipeline.processed_dir / "evaluation_results.json").exists()))
    
    for stage, completed in status:
        status_str = "[COMPLETE]" if completed else "[PENDING]"
        print(f"  {status_str} {stage}")
    
    print("-"*80)
    
    completed_count = sum(1 for _, completed in status if completed)
    total = len(status)
    progress = (completed_count / total) * 100
    print(f"\nOverall Progress: {completed_count}/{total} stages ({progress:.0f}%)")
    print()


def run_interactive_menu():
    """Run the interactive menu interface."""
    print_banner()
    
    print("Initializing pipeline...")
    pipeline = QCEWPipeline()
    
    while True:
        print_menu()
        
        try:
            choice = input("\nSelect an option (0-10): ").strip()
            
            if choice == '0':
                print("\nExiting pipeline. Goodbye!")
                break
            
            elif choice == '1':
                print("\n[INFO] Starting full pipeline execution...")
                confirm = input("This will run all stages. Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    pipeline.run_full_pipeline()
                    input("\nPress Enter to continue...")
                else:
                    print("Operation cancelled.")
            
            elif choice == '2':
                print("\n[INFO] Running Data Consolidation...")
                pipeline.stage_1_consolidate_data()
                input("\nPress Enter to continue...")
            
            elif choice == '3':
                print("\n[INFO] Running Data Exploration...")
                if not pipeline.consolidated_file.exists():
                    print("[ERROR] Please run Data Consolidation first (option 2)")
                else:
                    import pandas as pd
                    df = pd.read_csv(pipeline.consolidated_file)
                    pipeline.stage_2_explore_data(df)
                input("\nPress Enter to continue...")
            
            elif choice == '4':
                print("\n[INFO] Running Data Validation...")
                if not pipeline.consolidated_file.exists():
                    print("[ERROR] Please run Data Consolidation first (option 2)")
                else:
                    import pandas as pd
                    df = pd.read_csv(pipeline.consolidated_file)
                    pipeline.stage_3_validate_data(df)
                input("\nPress Enter to continue...")
            
            elif choice == '5':
                print("\n[INFO] Running Feature Engineering...")
                if not pipeline.validated_file.exists():
                    print("[ERROR] Please run Data Validation first (option 4)")
                else:
                    import pandas as pd
                    df = pd.read_csv(pipeline.validated_file)
                    pipeline.stage_4_feature_engineering(df)
                input("\nPress Enter to continue...")
            
            elif choice == '6':
                print("\n[INFO] Running Data Preprocessing...")
                if not pipeline.features_file.exists():
                    print("[ERROR] Please run Feature Engineering first (option 5)")
                else:
                    import pandas as pd
                    df = pd.read_csv(pipeline.features_file)
                    pipeline.stage_5_preprocessing(df)
                input("\nPress Enter to continue...")
            
            elif choice == '7':
                print("\n[INFO] Running Model Training...")
                if not pipeline.preprocessed_file.exists():
                    print("[ERROR] Please run Data Preprocessing first (option 6)")
                else:
                    import pandas as pd
                    df = pd.read_csv(pipeline.preprocessed_file)
                    pipeline.stage_6_train_model(df)
                input("\nPress Enter to continue...")
            
            elif choice == '8':
                print("\n[INFO] Running Model Evaluation...")
                if not pipeline.model_file.exists():
                    print("[ERROR] Please train the model first (option 7)")
                else:
                    pipeline.stage_7_evaluate_model()
                input("\nPress Enter to continue...")
            
            elif choice == '9':
                print("\n[INFO] Launching Prediction Interface...")
                if not pipeline.model_file.exists():
                    print("[ERROR] Please train the model first (option 7)")
                else:
                    pipeline.stage_8_prediction_interface()
                input("\nPress Enter to continue...")
            
            elif choice == '10':
                check_pipeline_status(pipeline)
                input("Press Enter to continue...")
            
            else:
                print("\n[ERROR] Invalid option. Please select 0-10.")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            import traceback
            traceback.print_exc()
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    run_interactive_menu()
