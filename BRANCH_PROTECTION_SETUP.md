# Branch Protection Setup Guide

## Current Status
⚠️ **Branch protection is NOT currently enabled on master branch**

## Why You Need Branch Protection

Without branch protection, anyone (including you) can accidentally:
- Push commits directly to master
- Delete the master branch
- Force push and overwrite history
- Merge without review

Branch protection enforces the PR workflow and prevents these accidents.

## Quick Setup (Recommended Settings)

### Option 1: Via GitHub Web UI (Easiest)

1. Go to: https://github.com/bmcgauley/is-160-project/settings/branches

2. Click **"Add rule"** under "Branch protection rules"

3. Configure these settings:
   - **Branch name pattern**: `master`
   - ✅ **Require a pull request before merging**
     - Required approvals: `1` (or `0` if working solo)
     - ✅ Dismiss stale reviews when new commits pushed
   - ✅ **Require status checks to pass** (if you have CI/CD)
   - ✅ **Require conversation resolution before merging**
   - ✅ **Include administrators** (applies rules to you too!)
   - ✅ **Do not allow bypassing the above settings**

4. Click **"Create"** to save

### Option 2: Via GitHub CLI

```bash
# Basic protection (requires PRs, no direct pushes)
gh api repos/bmcgauley/is-160-project/branches/master/protection \
  --method PUT \
  --field required_pull_request_reviews[required_approving_review_count]=0 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field enforce_admins=true \
  --field required_conversation_resolution=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false

# With required approvals (if working with team)
gh api repos/bmcgauley/is-160-project/branches/master/protection \
  --method PUT \
  --field required_pull_request_reviews[required_approving_review_count]=1 \
  --field required_pull_request_reviews[dismiss_stale_reviews]=true \
  --field enforce_admins=true \
  --field required_conversation_resolution=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false
```

## What Happens After Setup

### ✅ Allowed Actions
- Create feature branches
- Push to feature branches
- Create pull requests
- Merge via approved PRs
- Review and approve PRs

### ❌ Blocked Actions
- `git push origin master` - Direct pushes blocked
- `git push --force origin master` - Force pushes blocked
- Merging without PR - Must use PR workflow
- Deleting master branch - Deletion blocked

## Solo Developer Settings

If you're working alone and don't need approvals:
- Set **required approvals** to `0`
- Still keep **"Require a pull request before merging"** enabled
- This allows you to approve and merge your own PRs
- Still enforces PR workflow and code review discipline

## Team Settings

If working with others:
- Set **required approvals** to `1` or more
- Enable **"Dismiss stale reviews when new commits pushed"**
- Consider adding **CODEOWNERS** file for automatic reviewer assignment

## Testing Branch Protection

After setup, try this (it should fail):

```bash
# This should be blocked
git checkout master
echo "test" >> README.md
git add README.md
git commit -m "test"
git push origin master
# ❌ Should see: "error: failed to push some refs"
```

If you see the error, protection is working! ✅

## Emergency Bypass (Use Sparingly)

If you absolutely need to push directly to master:
1. Temporarily disable branch protection on GitHub
2. Make your push
3. **Immediately re-enable protection**

**Better approach**: Even for emergencies, use a PR with the `--auto` merge flag:
```bash
gh pr create --title "HOTFIX: Critical bug" --body "Emergency fix"
gh pr merge --squash --auto  # Merges when checks pass
```

## Verification

Check if protection is active:
```bash
gh api repos/bmcgauley/is-160-project/branches/master/protection
```

If configured, you'll see JSON with protection settings.
If not configured, you'll see: "Branch not protected"

## Next Steps

1. **Set up branch protection now** using Option 1 or 2 above
2. **Test the protection** by trying to push directly to master
3. **Continue working** on feature branches with PR workflow
4. **Create a PR** for the Git Workflow Guide to practice the workflow!

---

**Remember**: Branch protection is your safety net. Once enabled, you'll never accidentally push to master again!
