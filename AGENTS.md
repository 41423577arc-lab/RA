# Codex Parallel Development Rules

## Repository workflow

- `main` is the integration branch. Feature tasks must never commit, force-push, or merge directly into `main`.
- Keep each task in its own Codex-managed worktree and feature branch.
- A feature task may commit and push only its own branch. The integration task owns merges and pushes `main` explicitly to both `origin` and `vftci`.
- Do not use `git reset --hard`, overwrite another task's changes, or delete branches that belong to another task.
- Before editing, confirm the current branch/worktree and keep unrelated changes untouched.

## Functional ownership

The task prompt assigns exactly one role. Stay within that role's owned files.

### Identity resolution

Owns identity extraction, normalization, completeness, and deterministic identity matching:

- `backend/app/services/intake_agent.py`
- `backend/app/services/intake_completeness.py`
- `backend/app/services/entity_resolver.py`
- `backend/prompts/intake_chat_v1.txt`
- `backend/prompts/intake_followup_v1.txt`
- `backend/prompts/intake_identity_normalize_v1.txt`
- New focused tests in `backend/tests/test_intake_identity.py`

Do not change Tavily orchestration, research pipeline behavior, or frontend presentation.

### Intake web lookup

Owns key-person identity lookup, Tavily access, candidate evidence validation, and lookup failure behavior:

- `backend/app/services/intake_entity_candidates.py`
- `backend/app/services/tavily_client.py`
- New focused tests in `backend/tests/test_intake_web_lookup.py`

Preserve these invariants: internal lookup runs first; web lookup is limited to unresolved identity completion; accepted external candidates must be supported by exact source-page evidence.

### Intake activity UI

Owns intake activity storage, polling, progress presentation, and intake interaction UI:

- `backend/app/services/intake_activity.py`
- `frontend/src/app/page.tsx`
- `frontend/src/app/globals.css`
- New focused tests in `backend/tests/test_intake_activity.py`

The frontend displays server state and must not infer identity or research outcomes independently.

## Shared integration files

These files are integration-owned and must not be edited by feature tasks without explicit user or integration-task authorization:

- `backend/app/api/intake.py`
- `backend/app/schemas/intake.py`
- `backend/app/tasks/pipeline.py`
- `backend/app/models/database.py`
- `backend/app/database.py`
- `backend/tests/test_intake.py`
- Project-wide configuration, dependency, Docker, and documentation files

When a feature needs a shared change, stop before editing it and report a compact contract proposal: affected file, new or changed field/function, compatibility impact, and required tests. The integration task applies cross-module wiring after reviewing all proposals.

## Verification

- Add or update focused tests for changed behavior.
- Backend changes: run the focused test file, then `python -m pytest backend/tests -q` when practical.
- Frontend changes: run `npm run build` from `frontend`.
- Before handoff, run `git diff --check`, review `git status`, and summarize changed files, tests, and any shared-contract proposal.
- The integration task merges in this order unless dependencies require otherwise: identity resolution, intake web lookup, intake activity UI. It then runs the complete backend suite and frontend production build before updating `main` on both remotes.
