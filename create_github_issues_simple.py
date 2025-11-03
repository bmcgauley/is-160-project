#!/usr/bin/env python3
"""
Simplified GitHub issues creation script (without assignees)
This version skips assignees to avoid permission issues
"""

import subprocess
from create_github_issues import MILESTONES, TASKS

def run_gh_command(args):
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
        print(f"Error: {e.stderr}")
        return ""

def get_milestone_title(task_num):
    """Get milestone title for a task number"""
    for milestone_def in MILESTONES:
        if task_num in milestone_def['tasks']:
            return milestone_def['title']
    return None

def create_issue_simple(task_num, task_data, milestone=None):
    """Create issue without assignee"""
    title = f"T{task_num:03d}: {task_data['title']}"

    # Build body
    body_parts = [
        f"**Task ID**: T{task_num:03d}",
        f"**Status**: {task_data['status'].title()}",
        f"**Suggested Assignee**: {task_data['assignee']}" if task_data['assignee'] else "**Available for**: Any team member",
        "",
        "---",
        "",
        f"See [tasks.md](../specs/001/tasks.md) for full context and implementation details.",
        "",
        f"**Labels**: {', '.join(task_data['labels'])}"
    ]
    body = "\n".join([p for p in body_parts if p])

    # Build command (NO assignee)
    cmd = [
        'issue', 'create',
        '--title', title,
        '--body', body
    ]

    # Add labels
    for label in task_data['labels']:
        cmd.extend(['--label', label])

    # Add milestone
    if milestone:
        cmd.extend(['--milestone', milestone])

    print(f"Creating T{task_num:03d}... ", end='', flush=True)
    result = run_gh_command(cmd)

    if result:
        print(f"✓ {result}")
        return result.split('/')[-1]
    else:
        print("✗ Failed")
        return None

def close_issue(issue_number, task_num):
    """Close a completed issue"""
    print(f"Closing T{task_num:03d} (#{issue_number})... ", end='', flush=True)
    result = run_gh_command([
        'issue', 'close', issue_number,
        '--comment', f"Task T{task_num:03d} completed. See commit history for implementation details."
    ])
    if result:
        print("✓")
    else:
        print("✗")

def main():
    print("=" * 80)
    print("Creating GitHub Issues (Simplified - No Assignees)")
    print("=" * 80)
    print()

    # Create issues
    print("Creating issues...")
    print()

    issue_map = {}
    completed_count = 0
    open_count = 0

    for task_num, task_data in sorted(TASKS.items()):
        milestone = get_milestone_title(task_num)
        issue_number = create_issue_simple(task_num, task_data, milestone)

        if issue_number:
            issue_map[task_num] = issue_number

            # Close if completed
            if task_data['status'] == 'completed':
                close_issue(issue_number, task_num)
                completed_count += 1
            else:
                open_count += 1

    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Total issues created: {len(issue_map)}")
    print(f"Completed (closed): {completed_count}")
    print(f"Open: {open_count}")
    print()
    print("Next steps:")
    print("1. Go to GitHub Issues tab to verify")
    print("2. Team members can manually assign themselves to issues")
    print("3. Run: gh issue list --milestone 'Phase 3.3: Feature Engineering'")

if __name__ == "__main__":
    main()
