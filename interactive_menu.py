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
    print("  0. Run Full Pipeline (All Stages)")
    print("  1. Stage 1: Data Consolidation")
    print("  2. Stage 2: Data Exploration & Visualization")
    print("  3. Stage 3: Data Validation")
    print("  4. Stage 4: Feature Engineering")
    print("  5. Stage 5: Data Preprocessing")
    print("  6. Stage 6: Train LSTM Model")
    print("  7. Stage 7: Evaluate Model")
    print("  8. Stage 8: Interactive Prediction Interface")
    print("  9. View Pipeline Status")
    print(" 10. Exit")
    print("-"*80)


def check_pipeline_status(pipeline: QCEWPipeline):
    """Check and display which pipeline stages have been completed."""
    print("\n" + "-"*80)
    print("PIPELINE STATUS")
    print("-"*80)

    status = []
    status.append(("Stage 1: Data Consolidation", pipeline.consolidated_file.exists()))
    status.append(("Stage 2: Data Exploration", (pipeline.plots_dir / "employment_trends.png").exists()))
    status.append(("Stage 3: Data Validation", pipeline.validated_file.exists()))
    status.append(("Stage 4: Feature Engineering", pipeline.features_file.exists()))
    status.append(("Stage 5: Data Preprocessing", pipeline.preprocessed_file.exists()))
    status.append(("Stage 6: Model Training", pipeline.model_file.exists()))
    status.append(("Stage 7: Model Evaluation", (pipeline.processed_dir / "evaluation_results.json").exists()))
    status.append(("Stage 8: Prediction Interface", pipeline.preprocessor_file.exists()))

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
                print("\n[INFO] Starting full pipeline execution...")
                confirm = input("This will run all stages. Continue? (y/n): ").strip().lower()
                if confirm == 'y':
                    pipeline.run_full_pipeline()
                    input("\nPress Enter to continue...")
                else:
                    print("Operation cancelled.")

            elif choice == '1':
                print("\n[INFO] Running Stage 1: Data Consolidation...")
                pipeline.stage_1_consolidate_data()
                input("\nPress Enter to continue...")

            elif choice == '2':
                print("\n[INFO] Running Stage 2: Data Exploration...")
                if not pipeline.consolidated_file.exists():
                    print("[ERROR] Please run Stage 1 first (option 1)")
                else:
                    import pandas as pd
                    df = pd.read_csv(pipeline.consolidated_file)
                    pipeline.stage_2_explore_data(df)
                input("\nPress Enter to continue...")

            elif choice == '3':
                print("\n[INFO] Running Stage 3: Data Validation...")
                if not pipeline.consolidated_file.exists():
                    print("[ERROR] Please run Stage 1 first (option 1)")
                else:
                    import pandas as pd
                    df = pd.read_csv(pipeline.consolidated_file)
                    pipeline.stage_3_validate_data(df)
                input("\nPress Enter to continue...")

            elif choice == '4':
                print("\n[INFO] Running Stage 4: Feature Engineering...")
                if not pipeline.validated_file.exists():
                    print("[ERROR] Please run Stage 3 first (option 3)")
                else:
                    import pandas as pd
                    df = pd.read_csv(pipeline.validated_file)
                    pipeline.stage_4_feature_engineering(df)
                input("\nPress Enter to continue...")

            elif choice == '5':
                print("\n[INFO] Running Stage 5: Data Preprocessing...")
                if not pipeline.features_file.exists():
                    print("[ERROR] Please run Stage 4 first (option 4)")
                else:
                    import pandas as pd
                    df = pd.read_csv(pipeline.features_file)
                    pipeline.stage_5_preprocessing(df)
                input("\nPress Enter to continue...")

            elif choice == '6':
                print("\n[INFO] Running Stage 6: Model Training...")
                # Check if sequences file exists
                sequences_file = pipeline.processed_dir / "qcew_preprocessed_sequences.npz"
                if not sequences_file.exists():
                    print("[ERROR] Please run Stage 5 first (option 5)")
                    print(f"[ERROR] Sequences file not found: {sequences_file}")
                else:
                    import numpy as np
                    import torch
                    from preprocessing import EmploymentDataPreprocessor

                    # Load sequences from .npz file
                    print(f"[INFO] Loading sequences from: {sequences_file.name}")
                    data = np.load(sequences_file)
                    X_sequences = data['X']
                    y_targets = data['y']

                    # Convert to PyTorch tensors
                    X_tensor = torch.FloatTensor(X_sequences)
                    y_tensor = torch.FloatTensor(y_targets)

                    # Create preprocessor instance (needed for metadata)
                    preprocessor = EmploymentDataPreprocessor(scaler_type='robust')

                    print(f"[INFO] Loaded {len(X_sequences):,} sequences")
                    print(f"[INFO] Input shape: {X_tensor.shape}")
                    print(f"[INFO] Target shape: {y_tensor.shape}")

                    # Run training
                    pipeline.stage_6_train_model(X_tensor, y_tensor, preprocessor)
                input("\nPress Enter to continue...")

            elif choice == '7':
                print("\n[INFO] Running Stage 7: Model Evaluation...")
                if not pipeline.model_file.exists():
                    print("[ERROR] Please train the model first (option 6)")
                else:
                    # Load model and create training results for evaluation
                    import numpy as np
                    import torch
                    from sklearn.metrics import mean_squared_error

                    # Load test data (from sequences file)
                    sequences_file = pipeline.processed_dir / "qcew_preprocessed_sequences.npz"
                    if not sequences_file.exists():
                        print("[ERROR] Sequences file not found. Please run preprocessing first.")
                    else:
                        # Load sequences
                        data = np.load(sequences_file)
                        X_sequences = data['X']
                        y_targets = data['y']

                        # Use last 20% as test set
                        split_idx = int(len(X_sequences) * 0.8)
                        X_test = X_sequences[split_idx:]
                        y_test = y_targets[split_idx:]

                        # Load model and make predictions
                        from lstm_model import EmploymentLSTM
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'

                        checkpoint = torch.load(pipeline.model_file, map_location=device)
                        model = EmploymentLSTM(
                            input_size=X_sequences.shape[2],
                            hidden_size=64,
                            num_layers=2,
                            dropout=0.2
                        ).to(device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()

                        # Make predictions in batches to avoid memory issues
                        with torch.no_grad():
                            batch_size = 1024
                            predictions_list = []
                            num_batches = (len(X_test) + batch_size - 1) // batch_size

                            for i in range(num_batches):
                                start_idx = i * batch_size
                                end_idx = min((i + 1) * batch_size, len(X_test))
                                batch = torch.FloatTensor(X_test[start_idx:end_idx]).to(device)
                                batch_preds = model(batch).cpu().numpy().squeeze()
                                predictions_list.append(batch_preds)

                                if (i + 1) % 100 == 0:
                                    print(f"  Processing batch {i + 1}/{num_batches}...")

                            predictions = np.concatenate(predictions_list)

                        # Create training_results dict
                        # Try to get history from checkpoint, provide empty defaults if missing
                        checkpoint_history = checkpoint.get('history', {})
                        training_results = {
                            "success": True,
                            "history": {
                                'train_loss': checkpoint_history.get('train_loss', []),
                                'val_loss': checkpoint_history.get('val_loss', []),
                                'learning_rates': checkpoint_history.get('learning_rates', [])
                            },
                            "test_loss": mean_squared_error(y_test, predictions),
                            "test_rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                            "test_mape": 0.0,  # Will be calculated in evaluation
                            "directional_accuracy": 0.0,  # Will be calculated in evaluation
                            "model_path": str(pipeline.model_file),
                            "num_epochs": checkpoint.get('epoch', 0),
                            "best_val_loss": checkpoint.get('val_loss', 0),
                            "test_predictions": predictions,
                            "test_targets": y_test
                        }

                        pipeline.stage_7_evaluate_model(training_results)
                input("\nPress Enter to continue...")

            elif choice == '8':
                print("\n[INFO] Running Stage 8: Interactive Prediction Interface...")
                if not pipeline.model_file.exists():
                    print("[ERROR] Please train the model first (option 6)")
                else:
                    pipeline.stage_8_prediction_interface()
                input("\nPress Enter to continue...")

            elif choice == '9':
                check_pipeline_status(pipeline)
                input("Press Enter to continue...")

            elif choice == '10':
                print("\nExiting pipeline. Goodbye!")
                break

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
