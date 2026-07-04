---
name: new-pr
description: Use when creating a new pull request, opening a PR, pushing a branch for review, or submitting changes for merge. Triggers on: new PR, open PR, create pull request, submit for review, push branch, 创建PR, 提交PR, 新建PR
---

# New Pull Request

**Core principle:** A PR tells a story. Title = what changed. Body = why + how to verify.

---

## The Iron Law

**Run `code-review` BEFORE creating the PR.**

Create PR before review? Run review first. Fix Critical and Important issues. Then create.

---

## Workflow

```
[1] PRE-FLIGHT — ensure branch is ready
      │  ↳ Tests pass, no uncommitted changes
      ▼
[2] REVIEW — run code-review skill
      │  ↳ Fix Critical + Important issues
      │  ↳ Note Minor issues in PR body
      ▼
[3] GATHER CONTEXT — understand the diff
      │  ↳ Commits, files changed, scope
      ▼
[4] WRITE PR — title + body
      │  ↳ Follow format below
      ▼
[5] CREATE — gh pr create
      │  ↳ Set base branch, reviewers, labels
      ▼
[6] CONFIRM — verify PR is correct
```

---

## Step 1: Pre-Flight Checks

```bash
# Ensure working tree is clean
git status

# Ensure all tests pass
cargo test

# Ensure branch is up to date with base
git fetch origin
git log --oneline origin/main..HEAD  # commits to be merged
```

**If tests fail → fix before continuing. Do not create a PR with broken tests.**

---

## Step 2: Run Code Review First

Invoke `code-review` skill before writing the PR.

```bash
BASE_SHA=$(git merge-base HEAD origin/main)
HEAD_SHA=$(git rev-parse HEAD)
```

Fix all Critical and Important issues from the 5-lens review.
Minor issues → document in PR body under "Known limitations".

---

## Step 3: Gather Context

```bash
# Summary of what changed
git log --oneline origin/main..HEAD

# Files and line counts
git diff --stat origin/main..HEAD

# Full diff (for writing the description)
git diff origin/main..HEAD
```

From this, identify:
- **What** changed (feature / bugfix / refactor / docs)
- **Why** it was needed (links to issue, motivation)
- **Risk level** (small isolated change vs. cross-cutting refactor)

---

## Step 4: Write the PR

### Title Format

```
<type>: <short imperative summary>  (≤ 72 chars)
```

| Type | When |
|------|------|
| `feat` | New feature or behaviour |
| `fix` | Bug fix |
| `refactor` | Code change with no behaviour change |
| `perf` | Performance improvement |
| `test` | Tests only |
| `docs` | Documentation only |
| `chore` | Build, deps, tooling |

**Good titles:**
- `feat: add order rejection when balance is insufficient`
- `fix: prevent panic on empty config file`
- `refactor: extract order validation into domain layer`

**Bad titles:**
- `Update code` — too vague
- `WIP` — not a PR title
- `Fix bug in order.rs` — what bug? why?

### Body Template

```markdown
## Summary

<!-- 2-4 bullet points: what changed and why -->
-
-

## Changes

<!-- Files/modules changed and what each does -->
| Area | Change |
|------|--------|
|  |  |

## How to Test

<!-- Concrete steps to verify the change works -->
1.
2.

## Known Limitations / Follow-ups

<!-- Minor issues from code-review, out-of-scope items -->
-

## Review Confidence

<!-- From code-review synthesis step -->
| Lens | Score |
|------|-------|
| Correctness | X% |
| Architecture | X% |
| Performance | X% |
| Security | X% |
| Tests | X% |
| **Overall** | **X%** |
```

---

## Step 5: Create the PR

```bash
gh pr create \
  --base main \
  --title "<type>: <summary>" \
  --body "$(cat <<'EOF'
## Summary
...

## Changes
...

## How to Test
...

## Known Limitations / Follow-ups
...

## Review Confidence
...
EOF
)"
```

**Optional flags:**

```bash
--reviewer <github-username>   # request specific reviewer
--label <label>                # feat, bug, refactor, etc.
--draft                        # not ready for merge yet
--assignee @me                 # assign to yourself
```

---

## Step 6: Confirm

```bash
# Verify PR was created correctly
gh pr view --web

# Check CI status
gh pr checks
```

If CI fails → fix and push to the same branch (PR updates automatically).

---

## PR Size Guidelines

| Lines changed | Guidance |
|---------------|---------|
| < 200 | Ideal — easy to review |
| 200–500 | Acceptable — add extra context in body |
| 500–1000 | Split if possible; explain why it's large |
| > 1000 | Almost always should be split |

If the PR is large, add to the body:
```markdown
## Why This PR Is Large
<explanation — e.g., necessary coupled refactor, migration>
```

---

## Red Flags — STOP

| Situation | Action |
|-----------|--------|
| Tests failing | Fix first |
| Critical issues from code-review unfixed | Fix first |
| Title is "WIP" or "fix stuff" | Write a real title |
| Body is empty | Fill the template |
| PR touches unrelated areas | Split into separate PRs |
| Pushing directly to main | Create branch + PR instead |

---

## Related Skills

| Need | Skill |
|------|-------|
| Review changes before PR | code-review |
| Implement feature with TDD | rust-implement |
| Refactor safely | rust-refactor-helper |
