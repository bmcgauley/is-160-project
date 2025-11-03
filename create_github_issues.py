#!/usr/bin/env python3
"""
Script to create GitHub issues from tasks.md
Creates milestones for each phase and issues for each task
"""

import subprocess
import json
import re
from typing import Dict, List, Tuple

# Milestone definitions matching phases in tasks.md
MILESTONES = [
    {
        "title": "Phase 3.1: Setup",
        "description": "Project setup, environment configuration, data directory structure, logging system",
        "tasks": range(1, 6)  # T001-T005
    },
    {
        "title": "Phase 3.2: Data Exploration & Validation",
        "description": "Load QCEW data, perform EDA, identify data quality issues, implement validation functions",
        "tasks": range(6, 32)  # T006-T031
    },
    {
        "title": "Phase 3.3: Feature Engineering",
        "description": "Data filtering, feature calculations, temporal features, geographic features",
        "tasks": list(range(32, 43)) + list(range(43, 53))  # T032-T042, T043-T052
    },
    {
        "title": "Phase 3.4: Data Preprocessing & Model Architecture",
        "description": "Normalization, imputation, encoding, sequence transformation, LSTM/RNN architectures",
        "tasks": range(54, 64)  # T054-T063
    },
    {
        "title": "Phase 3.5: Training Infrastructure",
        "description": "PyTorch Dataset, DataLoader, training loop, validation, checkpointing, early stopping",
        "tasks": range(65, 75)  # T065-T074
    },
    {
        "title": "Phase 3.6: Loss Functions & Evaluation",
        "description": "Custom loss functions, evaluation metrics, accuracy calculations, performance validation",
        "tasks": range(76, 86)  # T076-T085
    },
    {
        "title": "Phase 3.7: Visualization & Comparison",
        "description": "Feature attribution, LSTM visualization, employment trend plots, baseline models",
        "tasks": range(87, 104)  # T087-T103
    },
    {
        "title": "Phase 3.8: Documentation & Reporting",
        "description": "LSTM methodology documentation, results analysis, reproducibility validation",
        "tasks": range(105, 110)  # T105-T109
    },
    {
        "title": "Unified Pipeline",
        "description": "Master orchestrator, CLI interface, end-to-end testing, prediction interface",
        "tasks": range(111, 120)  # T111-T119
    },
    {
        "title": "Enhancement Tasks",
        "description": "Optional visualization enhancements and advanced features",
        "tasks": [120, 121]  # T120-NEW, T121-NEW
    }
]

# Task definitions with status
TASKS = {
    # Phase 3.1: Setup (T001-T005) - All complete
    1: {"title": "Implement automated data download script", "status": "completed", "assignee": "Brian", "labels": ["phase-3.1", "setup"]},
    2: {"title": "Set up PyTorch environment with dependencies", "status": "completed", "assignee": "Brian", "labels": ["phase-3.1", "setup"]},
    3: {"title": "Create data directory structure", "status": "completed", "assignee": "Brian", "labels": ["phase-3.1", "setup"]},
    4: {"title": "Implement automated data fetching/aggregation scripts", "status": "completed", "assignee": "Brian", "labels": ["phase-3.1", "setup"]},
    5: {"title": "Set up logging system", "status": "completed", "assignee": "Brian", "labels": ["phase-3.1", "setup"]},

    # Phase 3.2: Data Exploration & Validation (T006-T031) - All complete
    6: {"title": "Load QCEW CSV files and examine data structure", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "exploration"]},
    7: {"title": "Merge all QCEW CSV files into single consolidated dataset", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "consolidation"]},
    8: {"title": "Perform exploratory data analysis on employment counts and wages", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "exploration"]},
    9: {"title": "Identify missing values, outliers, and data quality issues", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "exploration"]},
    10: {"title": "Create summary statistics and visualizations for employment trends", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "visualization"]},
    11: {"title": "Document data schema and create data dictionary", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "documentation"]},
    12: {"title": "Create automated validation functions for employment ranges", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    13: {"title": "Implement statistical tests for detecting anomalies", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    14: {"title": "Build data quality scorecards", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    15: {"title": "Validate temporal continuity and identify gaps", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    16: {"title": "Create validation reports with flagged records", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    17: {"title": "Investigate NaN values in oty_month1_emplvl_pct_chg (2004-2019)", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "bug-fix"]},
    18: {"title": "Compare data schema between older and newer CSV files", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "investigation"]},
    19: {"title": "Fix year-over-year percentage calculations for early data", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "bug-fix"]},
    20: {"title": "Fix duplicate record detection logic", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "bug-fix"]},
    21: {"title": "Investigate records with establishments but zero employment", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "investigation"]},
    22: {"title": "Investigate records with positive employment but zero wages", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "investigation"]},
    23: {"title": "Review employment outliers outside IQR bounds", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "investigation"]},
    24: {"title": "Review wage outliers with extreme values", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "investigation"]},
    25: {"title": "Improve data quality checks to handle missing data gracefully", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "enhancement"]},
    26: {"title": "Verify row counts match between raw and consolidated data", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    27: {"title": "Sample and verify random records in consolidated dataset", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    28: {"title": "Validate aggregation level distribution", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    29: {"title": "Verify wage statistics at county level are reasonable", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    30: {"title": "Check consolidation preserves unique combinations", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "validation"]},
    31: {"title": "Document aggregation levels and implications for modeling", "status": "completed", "assignee": "Brian", "labels": ["phase-3.2", "documentation"]},

    # Phase 3.3: Feature Engineering (T032-T052) - Partially complete
    32: {"title": "Filter data to county-level records only", "status": "completed", "assignee": "Brian", "labels": ["phase-3.3", "feature-engineering"]},
    33: {"title": "Handle Annual vs quarterly records (keep only quarterly)", "status": "completed", "assignee": "Brian", "labels": ["phase-3.3", "feature-engineering"]},
    34: {"title": "Create data quality filter to remove incomplete records", "status": "completed", "assignee": "Brian", "labels": ["phase-3.3", "feature-engineering"]},
    35: {"title": "Create Central Valley counties reference file", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "deferred"]},
    36: {"title": "Generate all CA counties and Central Valley subset datasets", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "deferred"]},
    37: {"title": "Validate filtered datasets have consistent temporal coverage", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "validation"]},
    38: {"title": "Calculate quarter-over-quarter employment growth rates", "status": "completed", "assignee": "Brian", "labels": ["phase-3.3", "feature-engineering"]},
    39: {"title": "Create seasonal adjustment factors", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "deferred", "advanced"]},
    40: {"title": "Engineer industry concentration metrics and diversity indices", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "deferred", "advanced"]},
    41: {"title": "Build geographic clustering features", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "deferred", "advanced"]},
    42: {"title": "Generate lag features for temporal dependencies", "status": "completed", "assignee": "Brian", "labels": ["phase-3.3", "feature-engineering"]},
    43: {"title": "Create rolling window statistics (3, 6, 12 quarter averages)", "status": "open", "assignee": "", "labels": ["phase-3.3", "enhancement", "available"]},
    44: {"title": "Engineer cyclical features and economic cycle indicators", "status": "open", "assignee": "", "labels": ["phase-3.3", "enhancement", "available"]},
    45: {"title": "Calculate employment volatility measures", "status": "open", "assignee": "", "labels": ["phase-3.3", "enhancement", "available"]},
    46: {"title": "Validate temporal features for consistency", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "validation"]},
    47: {"title": "Create time-based train/validation/test splits", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "preprocessing"]},
    48: {"title": "Create geographic feature maps for counties/regions", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "advanced"]},
    49: {"title": "Engineer industry classification features", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "advanced"]},
    50: {"title": "Build regional economic indicators", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "advanced"]},
    51: {"title": "Calculate spatial autocorrelation features", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "advanced"]},
    52: {"title": "Validate geographic features against known patterns", "status": "open", "assignee": "Alejo", "labels": ["phase-3.3", "validation"]},
    53: {"title": "Set up feature engineering structure and initial files", "status": "completed", "assignee": "Brian", "labels": ["phase-3.3", "setup"]},

    # Phase 3.4: Preprocessing & Model Architecture (T054-T063) - All complete
    54: {"title": "Normalize employment counts and wage data using robust scaling", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "preprocessing"]},
    55: {"title": "Handle missing values with domain-appropriate imputation", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "preprocessing"]},
    56: {"title": "Create categorical encodings for industry codes and geographic identifiers", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "preprocessing"]},
    57: {"title": "Transform tabular data into sequence format for RNN/LSTM", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "preprocessing"]},
    58: {"title": "Validate preprocessing steps maintain data distribution properties", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "validation"]},
    59: {"title": "Design LSTM layers for temporal employment sequence processing", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "model-architecture"]},
    60: {"title": "Implement RNN architecture for sequential pattern recognition", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "model-architecture"]},
    61: {"title": "Create custom LSTM architecture combining temporal and spatial features", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "model-architecture"]},
    62: {"title": "Add batch normalization and dropout layers", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "model-architecture"]},
    63: {"title": "Validate LSTM architecture dimensions match sequence shapes", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "validation"]},
    64: {"title": "Set up preprocessing and model architecture structure", "status": "completed", "assignee": "Brian", "labels": ["phase-3.4", "setup"]},

    # Phase 3.5: Training Infrastructure (T065-T074) - All complete (but has runtime issues)
    65: {"title": "Create PyTorch Dataset class for efficient data loading", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "training-infrastructure"]},
    66: {"title": "Implement data augmentation techniques for employment time series", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "training-infrastructure"]},
    67: {"title": "Build DataLoader with proper batch sizes", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "training-infrastructure"]},
    68: {"title": "Create train/validation data splits preserving temporal order", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "training-infrastructure"]},
    69: {"title": "Validate batch processing maintains data integrity", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "validation"]},
    70: {"title": "Implement training loop with employment-specific loss functions", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "training"]},
    71: {"title": "Create validation loop with forecasting accuracy metrics", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "training"]},
    72: {"title": "Add model checkpointing for best performance", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "training"]},
    73: {"title": "Implement early stopping based on validation loss", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "training"]},
    74: {"title": "Build learning rate scheduling", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "training"]},
    75: {"title": "Set up training infrastructure structure", "status": "completed", "assignee": "Brian", "labels": ["phase-3.5", "setup"]},

    # Phase 3.6: Loss Functions & Evaluation (T076-T085) - Partially complete
    76: {"title": "Implement weighted loss functions emphasizing recent trends", "status": "completed", "assignee": "Brian", "labels": ["phase-3.6", "loss-functions"]},
    77: {"title": "Create custom metrics (MAPE, directional accuracy)", "status": "completed", "assignee": "Brian", "labels": ["phase-3.6", "metrics"]},
    78: {"title": "Add employment volatility prediction loss", "status": "completed", "assignee": "Brian", "labels": ["phase-3.6", "loss-functions"]},
    79: {"title": "Build industry-weighted loss functions", "status": "completed", "assignee": "Brian", "labels": ["phase-3.6", "loss-functions"]},
    80: {"title": "Validate loss functions align with forecasting standards", "status": "completed", "assignee": "Brian", "labels": ["phase-3.6", "validation"]},
    81: {"title": "Calculate employment prediction accuracy across time horizons", "status": "blocked", "assignee": "Andrew", "labels": ["phase-3.6", "evaluation", "blocked"]},
    82: {"title": "Create confusion matrices for employment growth/decline classification", "status": "blocked", "assignee": "Andrew", "labels": ["phase-3.6", "evaluation", "blocked"]},
    83: {"title": "Plot predicted vs actual employment trends by industry and region", "status": "blocked", "assignee": "Andrew", "labels": ["phase-3.6", "evaluation", "blocked"]},
    84: {"title": "Generate employment volatility prediction accuracy assessments", "status": "blocked", "assignee": "Andrew", "labels": ["phase-3.6", "evaluation", "blocked"]},
    85: {"title": "Validate model performance against forecasting benchmarks", "status": "blocked", "assignee": "Andrew", "labels": ["phase-3.6", "evaluation", "blocked"]},
    86: {"title": "Set up loss functions and evaluation structure", "status": "completed", "assignee": "Brian", "labels": ["phase-3.6", "setup"]},

    # Phase 3.7: Visualization & Comparison (T087-T103) - Not started
    87: {"title": "Implement feature attribution techniques", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "visualization"]},
    88: {"title": "Visualize LSTM learned patterns", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "visualization"]},
    89: {"title": "Create employment trend visualizations (model vs reality)", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "visualization"]},
    90: {"title": "Generate geographic heat maps of prediction accuracy", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "visualization"]},
    91: {"title": "Validate feature importance aligns with known economic factors", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "validation"]},
    92: {"title": "Implement traditional ARIMA forecasting model", "status": "open", "assignee": "", "labels": ["phase-3.7", "baseline", "enhancement", "available"]},
    93: {"title": "Implement exponential smoothing model", "status": "open", "assignee": "", "labels": ["phase-3.7", "baseline", "enhancement", "available"]},
    94: {"title": "Create performance comparison tables (LSTM vs ARIMA vs Exponential Smoothing)", "status": "open", "assignee": "", "labels": ["phase-3.7", "baseline", "enhancement", "available"]},
    95: {"title": "Benchmark computational efficiency of forecasting approaches", "status": "open", "assignee": "", "labels": ["phase-3.7", "baseline", "enhancement", "available"]},
    96: {"title": "Validate LSTM provides meaningful improvement over baselines", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "validation"]},
    97: {"title": "Create visual predictions vs actuals plots", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "visualization"]},
    98: {"title": "Implement multi-step ahead forecasts with uncertainty bands", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "forecasting"]},
    99: {"title": "Build industry risk dashboard", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "visualization"]},
    100: {"title": "Develop county-level comparison visualizations", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "visualization"]},
    101: {"title": "Create early warning system for employment decline", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "forecasting"]},
    102: {"title": "Generate wage growth predictions", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "forecasting"]},
    103: {"title": "Produce policy insights with actionable recommendations", "status": "open", "assignee": "Brian", "labels": ["phase-3.7", "analysis"]},
    104: {"title": "Set up visualization and comparison structure", "status": "completed", "assignee": "Brian", "labels": ["phase-3.7", "setup"]},

    # Phase 3.8: Documentation & Reporting (T105-T109) - Not started
    105: {"title": "Document LSTM methodology for employment data analysis", "status": "open", "assignee": "Alejo", "labels": ["phase-3.8", "documentation"]},
    106: {"title": "Create comprehensive results analysis with insights", "status": "open", "assignee": "Alejo", "labels": ["phase-3.8", "documentation"]},
    107: {"title": "Build reproducible experiment scripts", "status": "open", "assignee": "", "labels": ["phase-3.8", "documentation", "enhancement", "available"]},
    108: {"title": "Generate academic-style report on LSTM applications", "status": "open", "assignee": "Alejo", "labels": ["phase-3.8", "documentation"]},
    109: {"title": "Validate all results are reproducible", "status": "open", "assignee": "Alejo", "labels": ["phase-3.8", "validation"]},
    110: {"title": "Set up documentation and reporting structure", "status": "completed", "assignee": "Brian", "labels": ["phase-3.8", "setup"]},

    # Unified Pipeline (T111-T119)
    111: {"title": "Develop modular components in separate files", "status": "completed", "assignee": "Brian", "labels": ["unified-pipeline", "architecture"]},
    112: {"title": "Create integration functions to combine modules", "status": "completed", "assignee": "Brian", "labels": ["unified-pipeline", "architecture"]},
    113: {"title": "Build unified script (main.py) for entire pipeline", "status": "completed", "assignee": "Brian", "labels": ["unified-pipeline", "orchestration"]},
    114: {"title": "Add command-line interface and configuration options", "status": "completed", "assignee": "Brian", "labels": ["unified-pipeline", "cli"]},
    115: {"title": "Test unified script end-to-end", "status": "in-progress", "assignee": "Brian", "labels": ["unified-pipeline", "testing", "bug"]},
    116: {"title": "Document unified pipeline usage and deployment", "status": "open", "assignee": "Brian", "labels": ["unified-pipeline", "documentation"]},
    117: {"title": "Build interactive prediction interface", "status": "blocked", "assignee": "Brian", "labels": ["unified-pipeline", "interface", "blocked"]},
    118: {"title": "Integrate maps, charts, and confidence bands into interface", "status": "blocked", "assignee": "Brian", "labels": ["unified-pipeline", "visualization", "blocked"]},
    119: {"title": "Add uncertainty estimation and error bands to visualizations", "status": "blocked", "assignee": "Brian", "labels": ["unified-pipeline", "visualization", "blocked"]},

    # Enhancement Tasks
    120: {"title": "Create video walkthrough of pipeline execution", "status": "open", "assignee": "", "labels": ["enhancement", "documentation", "available"]},
    121: {"title": "Build interactive Jupyter notebook demonstrating pipeline stages", "status": "open", "assignee": "", "labels": ["enhancement", "documentation", "available"]},
}

def run_gh_command(args: List[str]) -> str:
    """Run gh CLI command and return output"""
    try:
        result = subprocess.run(
            ['gh'] + args,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running gh command: {e.stderr}")
        return ""

def create_milestone(title: str, description: str) -> str:
    """Create a milestone and return its number"""
    print(f"Creating milestone: {title}")
    result = run_gh_command([
        'api',
        'repos/:owner/:repo/milestones',
        '-X', 'POST',
        '-f', f'title={title}',
        '-f', f'description={description}',
        '-f', 'state=open'
    ])

    if result:
        milestone_data = json.loads(result)
        return str(milestone_data['number'])
    return ""

def create_issue(task_num: int, task_data: Dict, milestone: str = None) -> str:
    """Create a GitHub issue for a task"""
    title = f"T{task_num:03d}: {task_data['title']}"

    # Build body with additional context
    body_parts = [
        f"**Task ID**: T{task_num:03d}",
        f"**Status**: {task_data['status'].title()}",
        f"**Original Assignee**: {task_data['assignee']}" if task_data['assignee'] else "",
        "",
        "---",
        "",
        f"See [tasks.md](../specs/001/tasks.md) for full context and implementation details."
    ]
    body = "\n".join([p for p in body_parts if p])

    # Build gh issue create command
    cmd = [
        'issue', 'create',
        '--title', title,
        '--body', body
    ]

    # Add labels
    for label in task_data['labels']:
        cmd.extend(['--label', label])

    # Add milestone if provided
    if milestone:
        cmd.extend(['--milestone', milestone])

    # Add assignee if specified and not empty
    if task_data['assignee'] and task_data['assignee'] != "":
        cmd.extend(['--assignee', f"@{task_data['assignee'].lower()}"])

    print(f"Creating issue: T{task_num:03d} - {task_data['title'][:50]}...")
    result = run_gh_command(cmd)

    return result

def close_issue(issue_number: str, task_num: int):
    """Close a completed issue"""
    print(f"Closing completed issue T{task_num:03d} (#{issue_number})")
    run_gh_command([
        'issue', 'close', issue_number,
        '--comment', f"Task T{task_num:03d} completed. See commit history for implementation details."
    ])

def main():
    print("=" * 80)
    print("Creating GitHub Issues from tasks.md")
    print("=" * 80)
    print()

    # Create milestones
    milestone_map = {}
    for milestone_def in MILESTONES:
        milestone_num = create_milestone(milestone_def['title'], milestone_def['description'])
        if milestone_num:
            for task in milestone_def['tasks']:
                milestone_map[task] = milestone_num
        print()

    # Create issues for each task
    issue_map = {}
    for task_num, task_data in TASKS.items():
        milestone = milestone_map.get(task_num)
        issue_url = create_issue(task_num, task_data, milestone)

        if issue_url:
            # Extract issue number from URL
            issue_number = issue_url.split('/')[-1]
            issue_map[task_num] = issue_number

            # Close if completed
            if task_data['status'] == 'completed':
                close_issue(issue_number, task_num)

        print()

    print("=" * 80)
    print("GitHub Issues Creation Complete!")
    print("=" * 80)
    print(f"Total issues created: {len(issue_map)}")
    print(f"Completed issues closed: {sum(1 for t in TASKS.values() if t['status'] == 'completed')}")
    print(f"Open issues: {sum(1 for t in TASKS.values() if t['status'] == 'open')}")
    print(f"In-progress issues: {sum(1 for t in TASKS.values() if t['status'] == 'in-progress')}")
    print(f"Blocked issues: {sum(1 for t in TASKS.values() if t['status'] == 'blocked')}")

if __name__ == "__main__":
    main()
