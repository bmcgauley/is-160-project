#!/usr/bin/env python3
"""
Create GitHub labels for the project
Run this BEFORE creating issues
"""

import subprocess
import json

# Label definitions: (name, color, description)
LABELS = [
    # Phase labels
    ("phase-3.1", "0E8A16", "Phase 3.1: Setup"),
    ("phase-3.2", "0E8A16", "Phase 3.2: Data Exploration & Validation"),
    ("phase-3.3", "0E8A16", "Phase 3.3: Feature Engineering"),
    ("phase-3.4", "0E8A16", "Phase 3.4: Data Preprocessing & Model Architecture"),
    ("phase-3.5", "0E8A16", "Phase 3.5: Training Infrastructure"),
    ("phase-3.6", "0E8A16", "Phase 3.6: Loss Functions & Evaluation"),
    ("phase-3.7", "0E8A16", "Phase 3.7: Visualization & Comparison"),
    ("phase-3.8", "0E8A16", "Phase 3.8: Documentation & Reporting"),
    ("unified-pipeline", "0E8A16", "Unified Pipeline Development"),

    # Type labels
    ("setup", "1D76DB", "Setup and infrastructure tasks"),
    ("exploration", "1D76DB", "Data exploration and analysis"),
    ("consolidation", "1D76DB", "Data consolidation"),
    ("validation", "1D76DB", "Data validation and quality checks"),
    ("feature-engineering", "1D76DB", "Feature engineering tasks"),
    ("preprocessing", "1D76DB", "Data preprocessing"),
    ("model-architecture", "1D76DB", "Model architecture design"),
    ("training", "1D76DB", "Model training"),
    ("training-infrastructure", "1D76DB", "Training infrastructure"),
    ("loss-functions", "1D76DB", "Loss function implementation"),
    ("metrics", "1D76DB", "Metrics and evaluation"),
    ("evaluation", "1D76DB", "Model evaluation"),
    ("visualization", "1D76DB", "Data/result visualization"),
    ("documentation", "1D76DB", "Documentation tasks"),
    ("testing", "1D76DB", "Testing tasks"),
    ("baseline", "1D76DB", "Baseline model implementation"),
    ("forecasting", "1D76DB", "Forecasting tasks"),
    ("analysis", "1D76DB", "Analysis tasks"),
    ("orchestration", "1D76DB", "Pipeline orchestration"),
    ("cli", "1D76DB", "Command-line interface"),
    ("interface", "1D76DB", "User interface"),
    ("architecture", "1D76DB", "System architecture"),

    # Status labels
    ("enhancement", "A2EEEF", "Enhancement or optional feature"),
    ("available", "7057FF", "Available for team members to claim"),
    ("deferred", "FBCA04", "Deferred for later"),
    ("advanced", "FBCA04", "Advanced optional feature"),
    ("blocked", "D93F0B", "Blocked by other issues"),
    ("bug", "D93F0B", "Bug or issue"),
    ("bug-fix", "0E8A16", "Bug fix (completed)"),
    ("in-progress", "FEF2C0", "Work in progress"),
    ("completed", "0E8A16", "Completed task"),
    ("investigation", "D4C5F9", "Investigation task"),
]

def run_gh_command(args):
    """Run gh CLI command"""
    try:
        result = subprocess.run(
            ['gh'] + args,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr.strip()}")
        return None

def label_exists(name):
    """Check if label already exists"""
    result = run_gh_command(['api', 'repos/:owner/:repo/labels'])
    if result:
        labels = json.loads(result)
        return any(label['name'] == name for label in labels)
    return False

def create_label(name, color, description):
    """Create a label"""
    if label_exists(name):
        print(f"  ⏭️  {name} (already exists)")
        return True

    print(f"  Creating: {name}...", end=" ", flush=True)
    result = run_gh_command([
        'api',
        'repos/:owner/:repo/labels',
        '-X', 'POST',
        '-f', f'name={name}',
        '-f', f'color={color}',
        '-f', f'description={description}'
    ])

    if result:
        print("✓")
        return True
    else:
        print("✗")
        return False

def main():
    print("=" * 80)
    print("Creating GitHub Labels")
    print("=" * 80)
    print()

    success_count = 0
    skip_count = 0
    fail_count = 0

    for name, color, description in LABELS:
        if label_exists(name):
            skip_count += 1
            print(f"  ⏭️  {name} (already exists)")
        else:
            if create_label(name, color, description):
                success_count += 1
            else:
                fail_count += 1

    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Created: {success_count}")
    print(f"Skipped (already exist): {skip_count}")
    print(f"Failed: {fail_count}")
    print()

    if fail_count == 0:
        print("✅ All labels ready!")
        print("Next step: Run create_github_issues_simple.py")
    else:
        print("⚠️  Some labels failed to create")

if __name__ == "__main__":
    main()
