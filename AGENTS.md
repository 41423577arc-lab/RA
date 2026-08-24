# Codex Development Rules

## Repository workflow

- `main` is the only long-lived branch and `C:\Users\86139\code\resource-agent-demo` is the canonical worktree.
- Routine work happens in the canonical worktree on `main`. Before editing, confirm the branch and worktree are correct and preserve unrelated changes.
- Do not force-push `main`, use `git reset --hard`, overwrite another task's changes, or delete unmerged work.
- Create a temporary feature branch or worktree only when the user requests isolation or concurrent work makes it necessary. Merge it into `main` after verification, then remove the temporary worktree and delete the merged branch.
- Push approved `main` updates explicitly to both `origin` and `vftci`.
- Run the `resource-agent-demo` Compose project only from the canonical `main` worktree. Always pass `-p resource-agent-demo` and preserve the existing named volumes; never use `down -v` when live data is in scope.
- Dockerfiles use `COPY`, so source changes require rebuilding the affected application services from the canonical worktree. Container labels must identify the canonical worktree as `com.docker.compose.project.working_dir`.
- If a temporary worktree is needed, run `powershell -ExecutionPolicy Bypass -File scripts/setup_worktree.ps1` once before testing. Do not install or upgrade packages through its shared `.venv` or `frontend/node_modules` links. Use Webpack for frontend commands in that worktree, for example `npm run build -- --webpack`.

## Functional ownership

The task prompt assigns exactly one role. Stay within that role's owned files.

### Identity resolution

Owns identity extraction, normalization, completeness, and deterministic identity matching:

- `backend/app/services/intake/completeness.py`
- `backend/app/services/intake/entity_resolver.py`
- `backend/app/services/intake/identity_loop.py`
- `backend/prompts/intake_chat_v1.txt`
- `backend/prompts/intake_followup_v1.txt`
- `backend/prompts/intake_identity_initialize_v1.txt`
- `backend/prompts/intake_identity_normalize_v1.txt`
- `backend/prompts/intake_identity_update_v1.txt`
- `backend/prompts/intake_readiness_v1.txt`
- New focused tests in `backend/tests/test_intake_identity.py`

Do not change Tavily orchestration, research pipeline behavior, or frontend presentation.

### Intake web lookup

Owns key-person identity lookup, Tavily access, candidate evidence validation, and lookup failure behavior:

- `backend/app/services/intake/entity_candidates.py`
- `backend/app/services/integrations/tavily_client.py`
- New focused tests in `backend/tests/test_intake_web_lookup.py`

Preserve these invariants: internal lookup runs first; web lookup is limited to unresolved identity completion; accepted external candidates must be supported by exact source-page evidence.

### Intake agent orchestration

Owns the runtime-only Intake Agent state wrapper, turn validation, state reduction, structured internal query execution, decision-loop orchestration, and the legacy-to-V2 feature switch:

- `backend/app/services/intake/agent_loop.py`
- `backend/app/services/intake/state_reducer.py`
- `backend/app/services/intake/query_executor.py`
- `backend/app/services/intake/runner.py`
- `backend/prompts/intake_agent_v1.txt`
- `backend/prompts/intake_skills/`
- New focused tests in `backend/tests/test_intake_agent_loop.py`

`IntakeStructuredContext` remains the only persisted Intake business context. Runtime `AgentState` must wrap it without duplicating or separately persisting people, organizations, resolutions, or other business facts. Do not change Tavily evidence validation, Research Pipeline behavior, new-customer persistence, or frontend presentation in this role.

### Intake activity UI

Owns intake activity storage, polling, progress presentation, and intake interaction UI:

- `backend/app/services/intake/activity.py`
- `frontend/src/app/page.tsx`
- `frontend/src/app/globals.css`
- New focused tests in `backend/tests/test_intake_activity.py`

The frontend displays server state and must not infer identity or research outcomes independently.

## Shared cross-module files

Changes to these files can affect multiple functional areas and require explicit scope review before editing:

- `backend/app/api/intake.py`
- `backend/app/config.py`
- `backend/app/schemas/intake.py`
- `backend/app/services/intake/agent.py`
- `backend/app/services/integrations/llm_client.py`
- `backend/app/services/integrations/mcp_client.py`
- `backend/app/tasks/pipeline.py`
- `backend/app/models/database.py`
- `backend/app/database.py`
- `backend/tests/test_intake.py`
- `mcp_server/server.py`
- `mcp_server/project_repository.py`
- Project-wide configuration, dependency, Docker, and documentation files

For cross-module changes, report a compact contract proposal before editing: affected file, new or changed field/function, compatibility impact, and required tests. Apply the wiring on `main` only after the scope is understood.

## Verification

- Add or update focused tests for changed behavior.
- Backend changes: run the focused test file, then `.\.venv\Scripts\python -m pytest backend\tests -q` when practical.
- Frontend changes in a temporary feature worktree: run `npm run build -- --webpack` from `frontend`. Run the normal `npm run build` from the canonical main worktree.
- Before handoff, run `git diff --check`, review `git status`, and summarize changed files, tests, and any shared-contract proposal.
- Before pushing or deploying `main`, run the complete backend suite and frontend production build when practical, then verify the live Compose services after rebuilding from the canonical worktree.
