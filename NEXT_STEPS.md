# Next Steps: GitHub Issues & PR Workflow Setup

## What's Been Done

✅ **Created `create_github_issues.py`**
- Automated script to create all 121 GitHub issues from tasks.md
- Organizes issues into 10 milestones by phase
- Automatically closes 80+ completed tasks
- Assigns labels, assignees, and milestone associations

✅ **Updated `specs/001/tasks.md`**
- Added comprehensive PR workflow requirements section
- Added mandatory branch workflow documentation
- Added example workflows for creating PRs
- Added GitHub issues integration notes
- Added warning at top about PR requirements for this file

✅ **Created `README_GITHUB_ISSUES.md`**
- Complete guide for running the issues creation script
- Troubleshooting section
- Team member email references
- Integration with PR workflow

## What You Need to Do Next

### Step 1: Commit These Changes (Using New PR Workflow!)

Since we just enabled PR protection, even **these** files need to go through a PR:

```bash
# 1. Make sure you're on master and up to date
git checkout master
git pull origin master

# 2. Create a branch for this documentation update
git checkout -b docs/github-issues-setup

# 3. Stage the new/modified files
git add create_github_issues.py
git add README_GITHUB_ISSUES.md
git add NEXT_STEPS.md
git add specs/001/tasks.md

# 4. Commit with descriptive message
git commit -m "docs: add GitHub issues automation and PR workflow documentation

- Add create_github_issues.py script for automating issue creation
- Add README_GITHUB_ISSUES.md with usage instructions
- Update tasks.md with mandatory PR workflow requirements
- Add comprehensive examples and troubleshooting

This establishes the new workflow where all changes require PRs
and documents how team members should contribute going forward."

# 5. Push the branch
git push -u origin docs/github-issues-setup

# 6. Create a Pull Request
gh pr create \
  --title "Setup: GitHub Issues Automation & PR Workflow Documentation" \
  --body "## Overview
This PR establishes the new GitHub issues tracking system and mandatory PR workflow.

## Changes
- **create_github_issues.py**: Automated script to create all 121 issues from tasks.md
- **README_GITHUB_ISSUES.md**: Complete documentation for using the script
- **NEXT_STEPS.md**: Step-by-step guide for setup process
- **specs/001/tasks.md**: Updated with PR workflow requirements and examples

## Why This Matters
- Provides clear tracking of all project tasks via GitHub issues
- Establishes mandatory code review process
- Creates accountability and audit trail
- Enables better collaboration through structured PRs

## Next Steps After Merge
1. Run \`python create_github_issues.py\` to populate GitHub with issues
2. Team members can start claiming enhancement tasks
3. All future work follows PR workflow documented in tasks.md"
```

### Step 2: Get PR Approved and Merge

Since you're the project lead and repository admin:
1. You can approve your own PR (as admin)
2. Or ask team members to review it at the Nov 9 meeting
3. Merge once approved: `gh pr merge --squash`

### Step 3: Run the Issues Creation Script

```bash
# After PR is merged, switch back to master
git checkout master
git pull origin master

# Run the script to create all GitHub issues
python create_github_issues.py
```

**Expected output:**
- 10 milestones created (one per phase)
- 121 GitHub issues created
- ~80 issues automatically closed (completed tasks)
- ~40 issues remain open (pending work and enhancements)

### Step 4: Add Team Member Emails to GitHub

Make sure both email addresses for Alejo are linked to his GitHub account:
1. Go to: https://github.com/settings/emails
2. Add both emails:
   - alejolanderos1995@gmail.com
   - gentlemen_4ever@mail.fresnostate.edu
3. Have him verify both emails

For Andrew, ensure he's added as a collaborator and has accepted the invitation.

### Step 5: Review GitHub Issues

```bash
# View all milestones and their progress
gh issue list --milestone "Phase 3.3: Feature Engineering"

# View open enhancement tasks available for team
gh issue list --label enhancement --label available --state open

# View your assigned tasks
gh issue list --assignee @me
```

### Step 6: Share with Team at Nov 9 Meeting

At the checkpoint meeting:

1. **Show them the GitHub Issues**:
   - Navigate to repository → Issues tab
   - Show milestones and their completion percentages
   - Show how closed issues document completed work

2. **Walk through the PR workflow**:
   - Show them the tasks.md section on PR requirements
   - Do a live demo of creating a branch, making a change, creating a PR
   - Emphasize: NO direct commits to master

3. **Help them claim tasks**:
   - Show enhancement opportunities in issues
   - Help them assign themselves to issues
   - Guide them through first PR creation

## Verification Checklist

After completing all steps, verify:

- [ ] Documentation PR created and merged
- [ ] `create_github_issues.py` script has run successfully
- [ ] GitHub shows 10 milestones in repository
- [ ] GitHub shows 121 issues total
- [ ] ~80 issues are closed (completed tasks)
- [ ] ~40 issues are open (pending work)
- [ ] Team members can see the issues
- [ ] Team members understand the PR workflow
- [ ] Repository settings enforce PR requirements

## Troubleshooting

**Q: Script fails with authentication error**
```bash
gh auth login
# Follow prompts to authenticate
```

**Q: Can't push to master (expected!)**
```bash
# This is correct! Create a branch instead:
git checkout -b feature/your-work
# Then create a PR
```

**Q: Team member can't see issues**
```bash
# Ensure they're added as collaborators:
gh api repos/:owner/:repo/collaborators/:username -X PUT
```

**Q: Milestone percentages don't match expectations**
- GitHub calculates automatically based on open/closed issues
- Check if issues are assigned to correct milestone
- Refresh page or wait a moment for GitHub to update

## Files Created/Modified Summary

```
New Files:
├── create_github_issues.py          (Python script - 400+ lines)
├── README_GITHUB_ISSUES.md          (Documentation - complete guide)
└── NEXT_STEPS.md                    (This file - action items)

Modified Files:
└── specs/001/tasks.md               (Added PR workflow section, ~100 lines added)
```

## Timeline

- **Now**: Commit documentation changes via PR
- **After merge**: Run issues creation script
- **Nov 9 Meeting**: Show team members the new system
- **Nov 9-15**: Team works on claimed enhancement tasks
- **Nov 15 Deadline**: Evaluate completed enhancements

## Key Points to Remember

1. ✅ **Everything** requires a PR now - even documentation
2. ✅ GitHub issues provide single source of truth for all tasks
3. ✅ Milestones show progress by phase automatically
4. ✅ Team can claim enhancement tasks via issue assignment
5. ✅ PRs automatically close linked issues when merged
6. ✅ This creates full audit trail for the professor

## Questions?

Refer to:
- `README_GITHUB_ISSUES.md` for script details
- `specs/001/tasks.md` for workflow requirements
- `CLAUDE.md` for project architecture
- GitHub Issues tab for current task status

---

**Remember**: The first PR (this documentation) sets the example for all future PRs! Make sure it follows best practices.
