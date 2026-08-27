# Agent Configuration Contract

Status: phase 0 baseline
Configuration schema version: `1`

## Purpose and scope

The first configurable product version keeps the current Python-controlled workflow. It makes the resources used by that workflow configurable: model profiles, prompt revisions, logical tools, loop limits, and output settings. Arbitrary graph editing and user-defined Python execution are explicitly out of scope.

This document separates stable product identifiers from Python function names. A refactor may move a call site, but it must not silently rename a published `node_key` or `logical_tool_key`.

## Current runtime topology

The current topology is not a single linear chain. Intake has legacy and V2 paths, identity resolution may call internal and public lookup tools, and public evidence verification only calls an LLM for candidates that deterministic rules mark as ambiguous.

```text
Intake
  |-- legacy intake extraction (`intake_chat`)
  |-- or V2 decision loop (`intake_agent`)
  |-- identity context initialize/update
  |-- internal identity lookup
  |-- optional public identity lookup and normalization
  `-- final confirmation

Research pipeline
  |-- public search and page extraction
  |     `-- conditional ambiguous evidence LLM verification
  |-- internal project search
  |-- deterministic project ranking
  |-- deterministic association building
  |-- final LLM synthesis with deterministic fallback/validation
  `-- deterministic report rendering

Post-run
  `-- optional analysis chat
```

## Node Registry

The code scan found eleven real LLM call identifiers. These strings already select prompt files and are the initial stable `node_key` values.

| `node_key` | Runtime purpose | Output contract | Invocation |
| --- | --- | --- | --- |
| `intake_chat` | Legacy intake extraction and response | `IntakeChatResult` | Legacy path, per intake turn |
| `intake_agent` | V2 skill/action decision | `AgentTurn` | V2 loop, up to configured loop limit |
| `intake_identity_initialize` | Initialize normalized intake context | `IntakeStructuredContext` | Identity loop initialization |
| `intake_identity_update` | Reduce user/tool observations into context | `IntakeStructuredContext` | Identity loop updates |
| `intake_followup` | Interpret internal lookup observations | `IntakeFollowupResult` | Conditional legacy follow-up |
| `intake_identity_normalize` | Normalize externally sourced identity evidence | `ExternalIdentityNormalizationResult` | Conditional public identity lookup |
| `intake_readiness` | Assess intake readiness after identity work | `IntakeReadinessResult` | Conditional identity/readiness flow |
| `intake_final_confirmation` | Produce the user-facing confirmation summary | `IntakeFinalConfirmationResult` | When deterministic readiness gates pass |
| `evidence_verify` | Resolve ambiguous public evidence candidates | `WebEvidenceDecision` | Only when deterministic routing has ambiguous candidates |
| `final_synthesis` | Generate evidence-backed report content | `GeneratedReportContent` | Once per research run, with fallback |
| `analysis_chat` | Answer follow-up questions about a task | `TaskChatResult` | On explicit post-run chat request |

The output contract named above remains code-owned. Prompt editing cannot replace it. The Node Registry must also declare capabilities such as `uses_model`, `uses_prompt`, `allows_tools`, `conditional`, and the code-owned output schema identifier.

Current model routing is only a compatibility default: `evidence_verify`, `intake_identity_normalize`, and `intake_readiness` use the review model; other nodes use the default model. `final_synthesis` receives the larger token limit. Phase 2 replaces these buckets with explicit node bindings.

## Prompt inventory

Every LLM node has a matching `backend/prompts/<node_key>_v1.txt` file. `intake_agent` additionally assembles all sorted files in `backend/prompts/intake_skills/` into its system prompt. A future published prompt binding therefore references either one immutable prompt revision or an immutable prompt bundle containing the main revision and ordered skill revisions.

The following remain code-owned and are appended or enforced at runtime:

- untrusted dynamic-context boundaries;
- Pydantic JSON Schema output contracts;
- retry formatting instructions;
- deterministic validation and fallback behavior.

## Tool Registry baseline

Stable logical tool keys must hide customer-specific MCP names and payload shapes.

| Proposed `logical_tool_key` | Current provider/call | Current use |
| --- | --- | --- |
| `identity.find_candidates` | MCP `find_entity_candidates` | Internal identity completion |
| `projects.search` | MCP `search_projects` | Internal project research |
| `identity.search_public` | Tavily search/extract plus normalization | Unresolved identity completion only |
| `public.search` | Tavily search | Public research |
| `public.extract_pages` | Tavily extract | Exact source-page evidence acquisition |

The current MCP server also exposes `get_project_details` and `get_sales_portfolio`, and the client has wrappers for them, but the Agent runtime does not currently invoke them. They must not be enabled for an Agent merely because discovery finds them.

Simple customer differences may use a restricted declarative input/output mapping. Complex joins, stored procedures, internal APIs, or authorization rules require a code-owned `DomainToolAdapter`. Configuration must never execute arbitrary Python or JavaScript.

## Loop and output configuration baseline

The configurable Intake loop values are currently `agent_max_loops`, `agent_max_tool_calls`, and `agent_max_repeated_actions`. Compatibility feature switches and deterministic gates include Intake V1/V2 selection, entity-resolution enablement, ReAct enablement, and the automatic identity threshold. Phase 1 snapshots all values that can change run behavior; later phases decide which are tenant-editable versus platform-controlled.

The current Research pipeline is a deterministic sequence, not a user-editable graph. Output consists of structured `GeneratedReportContent`, deterministic validation, and Jinja rendering to detailed-report and action-brief Markdown. Output configuration may select immutable template revisions and enabled formats, but cannot disable evidence validation.

## Version and publication model

```text
Tenant
  `-- AgentDefinition (stable identity)
        `-- AgentVersion (draft or immutable published version)
              `-- NodeBinding[]
                    |-- ModelProfile revision
                    |-- Prompt revision/bundle
                    `-- allowed Logical Tool mapping revisions
```

- Saving a draft has no runtime effect.
- Publishing validates the complete aggregate, calculates its hash, and makes it immutable.
- A new Intake run resolves the selected published version once.
- Publishing another version affects only later runs.
- Referenced published revisions cannot be mutated or deleted while retained runs depend on them.
- Rollback means selecting or republishing a prior valid aggregate for new runs; it never rewrites old runs.

## AgentRun boundary

An `AgentRun` is created on the first Intake turn, because Intake itself consumes configurable models and prompts. The same run is linked to the later `IntakeSession`, `ResearchTask`, report, and analysis chat. A future Conversation may contain multiple runs.

Phase 1 creates the minimal run fields:

```text
id
tenant_id NOT NULL
agent_definition_id
agent_version_id
config_schema_version
resolved_config_snapshot
config_hash
status
intake_session_id
research_task_id
created_at / updated_at
```

`owner_id`, `started_by`, and `conversation_id` are added with User and Conversation in phase 5. Phase 1 does not create or depend on a system user. Existing and anonymous data belongs to a non-null `system-tenant`.

## Resolved snapshot contract

The resolved snapshot is canonical JSON and contains all behavior-affecting configuration:

```json
{
  "config_schema_version": 1,
  "agent_definition_id": "...",
  "agent_version_id": "...",
  "nodes": {
    "<node_key>": {
      "model_profile_revision_id": "...",
      "model": {},
      "prompt_revision_id": "...",
      "prompt_content_hash": "...",
      "prompt_bundle_revision_ids": [],
      "allowed_tools": []
    }
  },
  "tool_mappings": [],
  "mcp_server_revision_ids": [],
  "loop": {},
  "output": {}
}
```

Each allowed tool entry identifies the logical tool, immutable mapping revision, remote tool name, MCP server revision, node permissions, timeout, and validated mapping configuration. The output section identifies immutable template revisions and enabled formats.

Canonicalization uses UTF-8 JSON with sorted object keys, no insignificant whitespace, stable list ordering defined by each schema, and no runtime timestamps. `config_hash` is lowercase SHA-256 over those canonical bytes. Secret values and other runtime-only data are excluded. The resolver must reject unknown schema versions rather than guess.

## Secret contract

Agent configuration versioning and secret-content versioning are separate concerns.

- Snapshots, hashes, API responses, execution events, and errors never contain secret plaintext.
- Configuration stores a stable `secret_ref` only.
- Rotating the value behind the same `secret_ref` does not create an AgentVersion.
- Changing provider, Base URL, authentication mode, or `secret_ref` requires a new published AgentVersion.
- Historical traceability proves which reference was used, not the historical secret value.
- Later calls and retries resolve the currently active value behind the reference.
- A secret store revision label may be logged only when it is non-sensitive; it does not permit restoring plaintext.

## Phase 1 compatibility contract

Phase 1 must seed `system-tenant` and `default-agent` from current settings, prompt files, skill files, report templates, and the current MCP URL. Calls that do not explicitly select an Agent use the published default version. Existing API response shapes and current Intake/Research behavior remain compatible while the V2 resolver is introduced.

Production schema evolution moves to Alembic. Existing databases require a documented baseline/stamp path before applying the first new migration. Application `create_all()` may remain temporarily for isolated tests, but it must not be the production migration mechanism for the new configuration tables.

## Planned shared-file contracts

Before phase 1 edits, each batch must restate its exact scope. The expected shared changes are:

| File/area | Planned contract change | Compatibility requirement | Focused verification |
| --- | --- | --- | --- |
| `backend/app/models/database.py` | Add Tenant, AgentDefinition, AgentVersion, NodeBinding, AgentRun and links | Existing columns and records preserved | model and migration tests |
| `backend/app/database.py` | Add repositories/resolver access; retire new ad-hoc DDL | Existing repositories remain callable | repository tests |
| `backend/app/config.py` | Add bootstrap/feature settings only | Existing `.env` remains valid | settings tests |
| `backend/app/api/intake.py` | Create/resolve AgentRun at first turn and carry its ID | Existing clients may omit Agent selection | intake API tests |
| `backend/app/services/intake/agent.py` | Resolve LLM node configuration through run context | Existing node outputs unchanged | intake node tests |
| `backend/app/services/integrations/llm_client.py` | Accept resolved node config with settings fallback | Existing direct test construction works during migration | LLM integration tests |
| `backend/app/tasks/pipeline.py` | Load the task's run snapshot instead of latest config | Existing tasks without a run use compatibility fallback | pipeline tests |
| project migrations/Compose | Run Alembic upgrade before application services | Preserve named volumes and current data | clean and existing DB migration tests |

## Phase 0 completion evidence

Phase 0 is complete when repository scans show that:

1. all production LLM parse call identifiers are present in the Node Registry;
2. every registered LLM node has a prompt source;
3. conditional evidence verification is represented as conditional;
4. runtime-used MCP tools are distinguished from merely exposed tools;
5. loop and output configuration sources are documented;
6. Tenant, publication, Snapshot, hash, and Secret semantics are unambiguous.
