# GitHub Issues Setup Guide

This document explains how to create GitHub issues and milestones from the tasks defined in `specs/001/tasks.md`.

## Overview

The `create_github_issues.py` script automates the creation of:
- **Milestones** for each project phase (3.1 through 3.8, Unified Pipeline, Enhancement Tasks)
- **GitHub Issues** for every task (T001-T121)
- Proper **labels**, **assignments**, and **milestone associations**
- **Closes completed issues** automatically

## Prerequisites

1. **GitHub CLI (`gh`) installed and authenticated**:
   ```bash
   # Install gh (if not already installed)
   # Windows: winget install GitHub.cli
   # Mac: brew install gh
   # Linux: See https://github.com/cli/cli#installation

   # Authenticate
   gh auth login
   ```

2. **Python 3.7+** installed

3. **Repository access**: You must have write access to the repository

## Team Member Emails

Make sure team members are added as collaborators:
- **Andrew Lackey**: (GitHub username from existing access)
- **Alejo Landeros**:
  - alejolanderos1995@gmail.com
  - gentlemen_4ever@mail.fresnostate.edu

## Running the Script

### From Project Root

```bash
# Navigate to project root
cd c:\GitHub\is-160-project

# Run the script
python create_github_issues.py
```

### What It Does

1. **Creates 10 Milestones**:
   - Phase 3.1: Setup (T001-T005)
   - Phase 3.2: Data Exploration & Validation (T006-T031)
   - Phase 3.3: Feature Engineering (T032-T052)
   - Phase 3.4: Data Preprocessing & Model Architecture (T054-T063)
   - Phase 3.5: Training Infrastructure (T065-T074)
   - Phase 3.6: Loss Functions & Evaluation (T076-T085)
   - Phase 3.7: Visualization & Comparison (T087-T103)
   - Phase 3.8: Documentation & Reporting (T105-T109)
   - Unified Pipeline (T111-T119)
   - Enhancement Tasks (T120-T121)

2. **Creates 121 GitHub Issues** (one per task):
   - Proper title format: "T001: Task description"
   - Body includes task ID, status, original assignee
   - Links back to tasks.md for full context
   - Assigned to appropriate team member (where applicable)
   - Labeled by phase, type (setup, feature-engineering, etc.)
   - Associated with correct milestone

3. **Closes Completed Issues**:
   - All tasks marked as "completed" in tasks.md are automatically closed
   - Adds a comment noting completion
   - Approximately 80 tasks will be closed (all completed work through Phase 3.6)

## Issue Labels

The script creates issues with the following label categories:

**By Phase:**
- `phase-3.1`, `phase-3.2`, ..., `phase-3.8`, `unified-pipeline`

**By Type:**
- `setup`, `exploration`, `consolidation`, `validation`, `visualization`
- `feature-engineering`, `preprocessing`, `model-architecture`
- `training`, `training-infrastructure`, `loss-functions`, `metrics`, `evaluation`
- `documentation`, `testing`

**By Status:**
- `completed` (for closed issues - historical record)
- `in-progress` (e.g., T115 - end-to-end testing)
- `blocked` (e.g., T081-T085, T117-T119 - waiting on training fix)
- `enhancement` (optional tasks for team contribution)
- `available` (open for any team member to claim)
- `deferred` (T035-T041 - advanced features)
- `advanced` (complex optional features)

**Special:**
- `bug` (e.g., T115 - training stage has runtime issues)
- `bug-fix` (e.g., T017, T019 - historical bug fixes)

## Expected Output

```
================================================================================
Creating GitHub Issues from tasks.md
================================================================================

Creating milestone: Phase 3.1: Setup
Creating milestone: Phase 3.2: Data Exploration & Validation
...

Creating issue: T001 - Implement automated data download script...
Closing completed issue T001 (#1)

Creating issue: T002 - Set up PyTorch environment with dependencies...
Closing completed issue T002 (#2)
...

================================================================================
GitHub Issues Creation Complete!
================================================================================
Total issues created: 121
Completed issues closed: 80
Open issues: 31
In-progress issues: 1
Blocked issues: 9
```

## Viewing Issues

After running the script:

```bash
# View all issues
gh issue list

# View open issues only
gh issue list --state open

# View issues by milestone
gh issue list --milestone "Phase 3.3: Feature Engineering"

# View issues assigned to you
gh issue list --assignee @me

# View enhancement opportunities
gh issue list --label enhancement --label available
```

## GitHub Web Interface

After creation, you can view:
- **Issues tab**: See all issues organized by status
- **Milestones tab**: See progress by phase (percentage completion)
- **Projects tab**: Can create project boards to organize work

## Troubleshooting

**Error: "gh: command not found"**
- Install GitHub CLI: https://cli.github.com/

**Error: "authentication required"**
- Run: `gh auth login` and follow prompts

**Error: "permission denied"**
- Ensure you have write access to the repository
- Check with repository admin (Brian)

**Error: "milestone already exists"**
- Script doesn't handle existing milestones
- Either delete existing milestones first, or modify script to skip creation

**Warning: Assignee not found**
- Ensure team members have been added as collaborators
- Verify GitHub usernames are correct
- Assignees may need to accept repository invitation first

## Integration with PR Workflow

Once issues are created:

1. **Reference issues in commits**:
   ```bash
   git commit -m "feat(T092): implement ARIMA baseline model

   Related to #92"
   ```

2. **Link PRs to issues**:
   ```bash
   gh pr create --title "Feature T092" --body "Implementation

   Closes #92"
   ```

3. **Automatic closure**: When PR merges with "Closes #92", issue #92 automatically closes

## Manual Issue Management

If needed, you can also create issues manually:

```bash
# Create new issue
gh issue create --title "Bug: Training stage crashes" --label bug --assignee @me

# Edit existing issue
gh issue edit 92 --add-label "in-progress"

# Close issue
gh issue close 92 --comment "Completed and tested"
```

## Notes for Team Members

- **All existing tasks are already in GitHub**: No need to create new issues for tasks.md items
- **Find available tasks**: Look for issues labeled `enhancement` + `available`
- **Claim a task**: Assign yourself: `gh issue edit <issue-num> --add-assignee @me`
- **Update task status**: Comment on issue or add labels to track progress
- **Link your work**: Reference issue number in commits and PRs

## Updating This Setup

If new tasks are added to tasks.md:

1. Update the `TASKS` dictionary in `create_github_issues.py`
2. Add to appropriate `MILESTONES` entry
3. Re-run script (it will skip existing issues)
4. Or create individual issues manually using gh CLI

## Contact

For issues with script or GitHub integration:
- Contact: Brian (Project Lead)
- Reference: `specs/001/tasks.md` and this README
