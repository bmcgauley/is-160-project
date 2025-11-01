# Git Workflow Guide: Feature Branches and Pull Requests

## Table of Contents
1. [Overview](#overview)
2. [Feature Branch Workflow](#feature-branch-workflow)
3. [GitHub CLI (gh) Commands](#github-cli-gh-commands)
4. [Pull Request Process](#pull-request-process)
5. [Branch Protection and Best Practices](#branch-protection-and-best-practices)
6. [Common Scenarios](#common-scenarios)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers the **feature branch workflow** with GitHub Pull Requests (PRs). This workflow ensures:
- **Code review** before merging to master
- **Clean history** with focused, reviewable changes
- **Safety** through branch protection rules
- **Collaboration** with team members

**Core Principle**: Never commit directly to `master`. Always work on feature branches and merge via Pull Requests.

---

## Feature Branch Workflow

### 1. Create a Feature Branch

Always start from an up-to-date master branch:

```bash
# Ensure you're on master
git checkout master

# Pull latest changes
git pull origin master

# Create and switch to new feature branch
git checkout -b feature/descriptive-name

# Examples:
git checkout -b feature/add-lstm-improvements
git checkout -b fix/preprocessing-bug
git checkout -b docs/update-readme
```

**Branch Naming Conventions**:
- `feature/` - New features (e.g., `feature/quarterly-aggregation`)
- `fix/` - Bug fixes (e.g., `fix/validation-error`)
- `refactor/` - Code refactoring (e.g., `refactor/pipeline-cleanup`)
- `docs/` - Documentation updates (e.g., `docs/add-api-guide`)
- `test/` - Test additions (e.g., `test/add-unit-tests`)

### 2. Push Feature Branch to Remote

After creating your branch locally, push it to GitHub:

```bash
# Push and set upstream tracking
git push -u origin feature/descriptive-name

# Shorter form (for current branch)
git push -u origin HEAD
```

The `-u` flag sets the upstream tracking, so future `git push` commands automatically push to the correct remote branch.

### 3. Work on Your Feature

Make changes, commit regularly with descriptive messages:

```bash
# Stage changes
git add .

# Commit with meaningful message
git commit -m "feat: implement quarterly data aggregation"

# Push to your feature branch
git push
```

**Commit Message Conventions**:
- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `docs:` - Documentation changes
- `test:` - Test additions
- `chore:` - Maintenance tasks

### 4. Keep Your Branch Updated

While working, periodically sync with master:

```bash
# Fetch latest changes
git fetch origin

# Rebase your branch on top of master (cleaner history)
git rebase origin/master

# OR merge master into your branch (preserves history)
git merge origin/master

# Push updated branch (may need --force-with-lease after rebase)
git push --force-with-lease
```

**When to use rebase vs merge**:
- **Rebase**: Creates linear history, cleaner for feature branches
- **Merge**: Preserves all history, safer if multiple people work on the branch

---

## GitHub CLI (gh) Commands

### Installation

If you don't have `gh` installed:

```bash
# Windows (via winget)
winget install GitHub.cli

# Verify installation
gh --version

# Authenticate (first time only)
gh auth login
```

### Essential gh Commands

#### View Repository Status

```bash
# View current repository info
gh repo view

# View PR list
gh pr list

# View issues
gh issue list

# View your open PRs
gh pr list --author "@me"
```

#### Create a Pull Request

```bash
# Create PR interactively (prompts for title, body)
gh pr create

# Create PR with inline title and body
gh pr create --title "Add quarterly aggregation" --body "Implements T042 quarterly aggregation feature"

# Create draft PR (not ready for review)
gh pr create --draft --title "WIP: LSTM improvements"

# Create PR with reviewers and labels
gh pr create --title "Fix preprocessing bug" --reviewer @username --label bug
```

#### View Pull Request Details

```bash
# View PR in browser
gh pr view --web

# View PR details in terminal
gh pr view 123

# View PR diff
gh pr diff 123

# View PR checks/status
gh pr checks
```

#### Review Pull Requests

```bash
# Checkout PR locally for testing
gh pr checkout 123

# Comment on PR
gh pr comment 123 --body "Looks good, just one suggestion..."

# Review PR (approve, request changes, comment)
gh pr review 123 --approve
gh pr review 123 --request-changes --body "Please fix the validation logic"
gh pr review 123 --comment --body "This looks promising"
```

#### Merge Pull Requests

```bash
# Merge PR with squash (recommended - creates single commit)
gh pr merge 123 --squash

# Merge PR with merge commit (preserves all commits)
gh pr merge 123 --merge

# Merge PR with rebase (linear history)
gh pr merge 123 --rebase

# Delete branch after merge
gh pr merge 123 --squash --delete-branch

# Auto-merge when checks pass
gh pr merge 123 --squash --auto
```

#### Update or Close Pull Requests

```bash
# Edit PR title or description
gh pr edit 123 --title "New title" --body "New description"

# Mark PR as ready (remove draft status)
gh pr ready 123

# Close PR without merging
gh pr close 123

# Reopen closed PR
gh pr reopen 123
```

---

## Pull Request Process

### Step-by-Step PR Workflow

#### 1. **Complete Your Work**

```bash
# Ensure all changes are committed
git status

# Run tests/validation
python main.py --stage validate
pytest

# Ensure branch is up to date with master
git fetch origin
git rebase origin/master
git push --force-with-lease
```

#### 2. **Create Pull Request**

```bash
# Create PR with descriptive title and body
gh pr create --title "feat: Implement quarterly aggregation (T042)" --body "$(cat <<'EOF'
## Summary
- Implemented quarterly data aggregation (T042)
- Added tests for aggregation functions
- Updated pipeline to include quarterly features

## Changes
- Modified `feature_engineering.py` to add `aggregate_quarterly_data()`
- Added unit tests in `test_T042.py`
- Updated `feature_pipeline.py` to call aggregation

## Testing
- All existing tests pass
- New tests added for quarterly aggregation
- Manual testing with sample data confirms correct output

## Checklist
- [x] Code follows project conventions
- [x] Tests added and passing
- [x] Documentation updated
- [ ] Security audit pending

Closes #42
EOF
)"
```

**PR Description Best Practices**:
- **Summary**: Brief overview of changes
- **Changes**: Detailed list of modifications
- **Testing**: How changes were tested
- **Checklist**: Ensure all requirements met
- **References**: Link related issues (`Closes #42`, `Relates to #38`)

#### 3. **Request Review**

```bash
# Add reviewers
gh pr edit --add-reviewer @teammate1,@teammate2

# Add labels
gh pr edit --add-label enhancement,feature

# Add to project board (if using)
gh pr edit --add-project "IS-160 Project"
```

#### 4. **Address Review Feedback**

Make requested changes on your feature branch:

```bash
# Make changes based on feedback
git add .
git commit -m "fix: address review feedback on validation logic"
git push

# Comment on PR to notify reviewers
gh pr comment --body "Updated validation logic as requested. Ready for re-review."
```

#### 5. **Run Security Audit** (per project requirements)

```bash
# Before merging to master, run security audit
# (This is a custom project command from CLAUDE.md)
# /security-audit

# Ensure all checks pass
gh pr checks
```

#### 6. **Merge Pull Request**

Once approved and all checks pass:

```bash
# Squash merge (recommended - clean history)
gh pr merge --squash --delete-branch

# The squash commit message will combine all commits into one
# GitHub will prompt for final commit message
```

After merging:

```bash
# Switch back to master and pull latest
git checkout master
git pull origin master

# Your feature branch is now merged and deleted
```

---

## Branch Protection and Best Practices

### Setting Up Branch Protection Rules

To enforce PR workflow and prevent direct pushes to `master`, configure branch protection on GitHub:

#### Via GitHub Web UI:

1. Go to repository **Settings** → **Branches**
2. Click **Add rule** under "Branch protection rules"
3. Branch name pattern: `master`
4. Configure protection:
   - ✅ **Require a pull request before merging**
     - ✅ Require approvals: 1 (or more)
     - ✅ Dismiss stale reviews when new commits are pushed
   - ✅ **Require status checks to pass before merging**
     - Add required checks (CI/tests if configured)
   - ✅ **Require conversation resolution before merging**
   - ✅ **Include administrators** (applies rules to admins too)
   - ✅ **Restrict who can push to matching branches** (optional)
5. Click **Create** or **Save changes**

#### Via GitHub CLI:

```bash
# Enable branch protection (requires admin access)
gh api repos/:owner/:repo/branches/master/protection \
  --method PUT \
  --field required_pull_request_reviews[required_approving_review_count]=1 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field enforce_admins=true
```

### Best Practices

#### Branch Management
- ✅ **Always branch from master**: Ensure master is up-to-date first
- ✅ **One feature per branch**: Keep branches focused and reviewable
- ✅ **Keep branches short-lived**: Merge frequently to avoid conflicts
- ✅ **Delete merged branches**: Clean up after merging
- ✅ **Sync regularly**: Rebase/merge master into your branch often

#### Commit Practices
- ✅ **Atomic commits**: Each commit should represent a logical unit
- ✅ **Descriptive messages**: Explain *why*, not just *what*
- ✅ **Conventional commits**: Use prefixes (feat, fix, docs, etc.)
- ✅ **Small commits**: Easier to review and revert if needed

#### Pull Request Practices
- ✅ **Clear titles**: Summarize the change in one line
- ✅ **Detailed descriptions**: Include summary, changes, testing, checklist
- ✅ **Link issues**: Use "Closes #123" to auto-close related issues
- ✅ **Request reviews**: Tag appropriate reviewers
- ✅ **Respond to feedback**: Address all comments
- ✅ **Keep PRs small**: Large PRs are hard to review (aim for < 400 lines)
- ✅ **Run tests**: Ensure all tests pass before requesting review
- ✅ **Security audit**: Run `/security-audit` before merging (project requirement)

#### Code Review Practices
- ✅ **Review promptly**: Don't block teammates
- ✅ **Be constructive**: Suggest improvements, explain reasoning
- ✅ **Ask questions**: Clarify unclear code
- ✅ **Approve when satisfied**: Don't be overly critical
- ✅ **Test locally**: Check out PR and test if needed

---

## Common Scenarios

### Scenario 1: Start Working on a New Feature

```bash
# 1. Switch to master and update
git checkout master
git pull origin master

# 2. Create feature branch
git checkout -b feature/add-lstm-dropout

# 3. Push to remote
git push -u origin feature/add-lstm-dropout

# 4. Make changes, commit, push
git add src/lstm_model.py
git commit -m "feat: add dropout to LSTM layers"
git push

# 5. Create PR when ready
gh pr create --title "feat: Add dropout to LSTM for better generalization" \
  --body "Adds configurable dropout to LSTM layers to prevent overfitting"
```

### Scenario 2: Your Branch is Behind Master

```bash
# Fetch latest changes
git fetch origin

# Option A: Rebase (cleaner history)
git rebase origin/master
git push --force-with-lease

# Option B: Merge (preserves history)
git merge origin/master
git push
```

### Scenario 3: Resolve Merge Conflicts

```bash
# During rebase or merge, conflicts may occur
# Git will mark conflicted files

# 1. View conflicted files
git status

# 2. Open conflicted files, resolve conflicts
#    Look for <<<<<<< HEAD, =======, >>>>>>> markers
#    Edit to keep desired changes

# 3. Stage resolved files
git add path/to/resolved/file.py

# 4. Continue rebase (if rebasing)
git rebase --continue

# OR commit merge (if merging)
git commit -m "merge: resolve conflicts with master"

# 5. Push changes
git push --force-with-lease  # if rebased
git push                      # if merged
```

### Scenario 4: Update PR After Review Feedback

```bash
# Make requested changes
git add .
git commit -m "fix: address code review comments"
git push

# Add comment to PR
gh pr comment --body "Updated per review feedback - ready for re-review"
```

### Scenario 5: Abandon Feature Branch

```bash
# If you decide not to proceed with a branch

# 1. Switch to master
git checkout master

# 2. Delete local branch
git branch -D feature/abandoned-feature

# 3. Delete remote branch
git push origin --delete feature/abandoned-feature

# OR close PR without merging
gh pr close
```

### Scenario 6: Check Out Someone Else's PR for Testing

```bash
# View open PRs
gh pr list

# Check out PR #42 locally
gh pr checkout 42

# Test the changes
python main.py --stage preprocess

# Leave review
gh pr review 42 --comment --body "Tested locally, works great!"

# Return to your branch
git checkout feature/your-branch
```

### Scenario 7: Emergency Hotfix

```bash
# For urgent fixes, use a hotfix branch
git checkout master
git pull origin master
git checkout -b hotfix/critical-bug

# Make fix
git add .
git commit -m "fix: resolve critical data validation bug"
git push -u origin hotfix/critical-bug

# Create PR with urgent priority
gh pr create --title "URGENT: Fix critical validation bug" \
  --label "priority:high,bug" \
  --body "Critical fix for production data validation issue"

# After review, merge immediately
gh pr merge --squash --delete-branch
```

---

## Troubleshooting

### Problem: "Cannot push to protected branch master"

**Cause**: Branch protection rules prevent direct pushes to master.

**Solution**: This is expected behavior! Create a feature branch and use a PR:
```bash
git checkout -b feature/my-changes
git push -u origin feature/my-changes
gh pr create
```

### Problem: "Diverged branches" error when pushing

**Cause**: Remote branch has commits your local branch doesn't have.

**Solution**:
```bash
# Pull latest changes
git pull --rebase origin feature/my-branch

# Resolve conflicts if any, then push
git push
```

### Problem: Accidentally committed to master

**Cause**: Forgot to create feature branch before committing.

**Solution**:
```bash
# Don't push! Create feature branch with your changes
git checkout -b feature/my-changes

# Reset master to match remote
git checkout master
git reset --hard origin/master

# Switch back to feature branch (changes are preserved)
git checkout feature/my-changes
git push -u origin feature/my-changes
```

### Problem: PR has conflicts with master

**Cause**: Master has changed since your branch was created.

**Solution**:
```bash
# Update your branch with latest master
git fetch origin
git rebase origin/master

# Resolve conflicts, then
git rebase --continue
git push --force-with-lease
```

### Problem: Want to undo last commit

**Cause**: Made a mistake in your last commit.

**Solution**:
```bash
# Undo commit but keep changes (can re-commit)
git reset --soft HEAD~1

# Undo commit and discard changes (CAREFUL!)
git reset --hard HEAD~1
```

### Problem: gh CLI not authenticated

**Cause**: GitHub CLI needs authentication.

**Solution**:
```bash
gh auth login
# Follow prompts to authenticate
```

### Problem: Can't find PR number

**Solution**:
```bash
# List all PRs
gh pr list

# List your PRs
gh pr list --author "@me"

# View current branch's PR
gh pr view
```

---

## Quick Reference

### Most Common Commands

```bash
# Start new feature
git checkout master && git pull && git checkout -b feature/name && git push -u origin HEAD

# Commit and push changes
git add . && git commit -m "feat: description" && git push

# Create PR
gh pr create --title "Title" --body "Description"

# Update branch with master
git fetch origin && git rebase origin/master && git push --force-with-lease

# Merge PR (after approval)
gh pr merge --squash --delete-branch

# Return to master after merge
git checkout master && git pull
```

### Workflow Cheat Sheet

```
1. CREATE BRANCH
   git checkout -b feature/name

2. PUSH BRANCH
   git push -u origin HEAD

3. WORK & COMMIT
   git add . && git commit -m "message" && git push

4. CREATE PR
   gh pr create

5. REVIEW & ITERATE
   (address feedback, push updates)

6. MERGE PR
   gh pr merge --squash --delete-branch

7. BACK TO MASTER
   git checkout master && git pull
```

---

## Additional Resources

- [GitHub Pull Request Documentation](https://docs.github.com/en/pull-requests)
- [GitHub CLI Manual](https://cli.github.com/manual/)
- [Git Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)

---

## Project-Specific Notes

Per the project's `CLAUDE.md`:

1. **Always use feature branches** for development work
2. **Create PRs via gh CLI** following the examples in this guide
3. **Run `/security-audit`** before merging PRs to master (project requirement)
4. **Use descriptive branch names** following conventions (feature/, fix/, etc.)
5. **Link PRs to issues** using "Closes #issue-number" in PR description
6. **Delete branches after merge** to keep repository clean

---

**Current Status**: You are now on branch `feature/development-workflow`. This guide was created on this branch. Follow the steps above to create a PR and merge this documentation back to master!
