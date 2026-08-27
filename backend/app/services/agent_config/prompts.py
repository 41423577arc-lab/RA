import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models.database import PromptDefinition, PromptRevision
from app.services.agent_config.registry import NODE_REGISTRY
from app.services.agent_config.snapshot import canonical_hash


SLUG_PATTERN = re.compile(r"^[a-z][a-z0-9-]{1,62}[a-z0-9]$")
PLACEHOLDER_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}|\$\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}")
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
INTAKE_SKILLS = {
    "01_identity_resolution": "identity_resolution",
    "02_internal_lookup": "internal_lookup",
    "03_public_lookup": "public_lookup",
    "04_intake_readiness": "intake_readiness",
}
RESERVED_SECTIONS = (
    "<dynamic_context",
    "</dynamic_context>",
    "## 最终输出契约",
)


class PromptConfigService:
    def __init__(self, session: Session, settings: Settings):
        self.session = session
        self.settings = settings

    def ensure_defaults(self, tenant_id: str) -> dict[str, PromptRevision]:
        revisions: dict[str, PromptRevision] = {}
        prompt_dir = Path(self.settings.prompt_dir)
        for node_key, node_spec in NODE_REGISTRY.items():
            slug = f"default-{node_key.replace('_', '-')}"
            definition = self.session.scalar(
                select(PromptDefinition).where(
                    PromptDefinition.tenant_id == tenant_id,
                    PromptDefinition.slug == slug,
                )
            )
            if definition is None:
                definition = PromptDefinition(
                    id=str(uuid4()),
                    tenant_id=tenant_id,
                    name=f"Default {node_key}",
                    slug=slug,
                    node_key=node_key,
                )
                self.session.add(definition)
                self.session.flush()
            elif definition.node_key != node_key:
                raise ValueError(f"Default Prompt node mismatch: {slug}")

            revision = (
                self.session.get(PromptRevision, definition.active_revision_id)
                if definition.active_revision_id
                else None
            )
            if revision is None:
                content = (prompt_dir / f"{node_key}_v1.txt").read_text(encoding="utf-8")
                skills = self._disk_skills(prompt_dir) if node_key == "intake_agent" else []
                revision = self._build_revision(
                    definition,
                    content=content,
                    skills=skills,
                    source=f"backend/prompts/{node_key}_v1.txt",
                    version=1,
                )
                self.session.add(revision)
                self.session.flush()
                definition.active_revision_id = revision.id
            self.resolve_revision(revision.id, expected_node_key=node_spec.node_key)
            revisions[node_key] = revision
        self.session.flush()
        return revisions

    def create_definition(
        self,
        *,
        tenant_id: str,
        name: str,
        slug: str,
        node_key: str,
        content: str,
        skills: list[dict] | None = None,
    ) -> tuple[PromptDefinition, PromptRevision]:
        self._validate_slug(slug)
        self._node_spec(node_key)
        if self.session.scalar(
            select(PromptDefinition).where(
                PromptDefinition.tenant_id == tenant_id,
                PromptDefinition.slug == slug,
            )
        ):
            raise ValueError(f"Prompt slug already exists: {slug}")
        definition = PromptDefinition(
            id=str(uuid4()),
            tenant_id=tenant_id,
            name=name.strip(),
            slug=slug,
            node_key=node_key,
        )
        revision = self._build_revision(
            definition,
            content=content,
            skills=skills or [],
            source="admin",
            version=1,
        )
        self.session.add_all([definition, revision])
        self.session.flush()
        definition.active_revision_id = revision.id
        self.session.commit()
        self.session.refresh(definition)
        self.session.refresh(revision)
        return definition, revision

    def revise_definition(
        self,
        prompt_definition_id: str,
        *,
        content: str,
        skills: list[dict] | None = None,
    ) -> PromptRevision:
        definition = self._definition(prompt_definition_id)
        active = self.session.get(PromptRevision, definition.active_revision_id)
        if active is None:
            raise ValueError("Prompt definition has no active revision")
        revision = self._build_revision(
            definition,
            content=content,
            skills=active.skill_bundle if skills is None else skills,
            source="admin",
            version=self._next_version(definition.id),
        )
        self.session.add(revision)
        self.session.flush()
        definition.active_revision_id = revision.id
        self.session.commit()
        self.session.refresh(revision)
        return revision

    def list_definitions(
        self, tenant_id: str, *, node_key: str | None = None
    ) -> list[tuple[PromptDefinition, PromptRevision]]:
        query = select(PromptDefinition).where(
            PromptDefinition.tenant_id == tenant_id,
            PromptDefinition.status == "ACTIVE",
        )
        if node_key:
            self._node_spec(node_key)
            query = query.where(PromptDefinition.node_key == node_key)
        definitions = list(self.session.scalars(query.order_by(PromptDefinition.name)))
        output = []
        for definition in definitions:
            revision = self.session.get(PromptRevision, definition.active_revision_id)
            if revision is not None:
                output.append((definition, revision))
        return output

    def list_revisions(self, prompt_definition_id: str) -> list[PromptRevision]:
        definition = self._definition(prompt_definition_id)
        return list(
            self.session.scalars(
                select(PromptRevision)
                .where(PromptRevision.prompt_definition_id == definition.id)
                .order_by(PromptRevision.version.desc())
            )
        )

    def resolve_revision(
        self,
        revision_id: str,
        *,
        expected_node_key: str | None = None,
    ) -> dict:
        revision = self.session.get(PromptRevision, revision_id)
        if revision is None or revision.status != "PUBLISHED":
            raise KeyError(f"Prompt revision not found: {revision_id}")
        definition = self._definition(revision.prompt_definition_id)
        if expected_node_key and definition.node_key != expected_node_key:
            raise ValueError(
                f"Prompt revision belongs to {definition.node_key}, not {expected_node_key}"
            )
        skills, report = self.validate_prompt(
            node_key=definition.node_key,
            content=revision.content,
            skills=revision.skill_bundle,
        )
        payload = self._revision_payload(
            node_key=definition.node_key,
            content=revision.content,
            skills=skills,
            required_variables=report["required_variables"],
        )
        if canonical_hash(revision.content) != revision.content_hash:
            raise ValueError(f"Prompt revision content failed integrity validation: {revision.id}")
        if canonical_hash(payload) != revision.config_hash:
            raise ValueError(f"Prompt revision failed integrity validation: {revision.id}")
        if report != revision.validation_report:
            raise ValueError(f"Prompt validation report has changed: {revision.id}")
        return {
            "prompt_definition_id": definition.id,
            "revision_id": revision.id,
            "version": revision.version,
            "node_key": definition.node_key,
            "content_hash": revision.content_hash,
            "content": revision.content,
            "source": revision.source,
            "required_variables": revision.required_variables,
            "skills": skills,
            "validation_report": report,
            "smoke_test_status": revision.smoke_test_status,
        }

    def validate_revision(self, revision_id: str) -> dict:
        resolved = self.resolve_revision(revision_id)
        return resolved["validation_report"]

    def validate_prompt(
        self,
        *,
        node_key: str,
        content: str,
        skills: list[dict] | None,
    ) -> tuple[list[dict], dict]:
        node_spec = self._node_spec(node_key)
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("Prompt content cannot be empty")
        if not normalized_content.startswith("# "):
            raise ValueError("Prompt must start with a Markdown level-one heading")
        if "\x00" in normalized_content:
            raise ValueError("Prompt contains unsupported control characters")
        for reserved in RESERVED_SECTIONS:
            if reserved in normalized_content:
                raise ValueError(f"Prompt contains code-owned section: {reserved}")
        variables = sorted(
            {
                first or second
                for first, second in PLACEHOLDER_PATTERN.findall(normalized_content)
            }
        )
        unknown = set(variables) - set(node_spec.prompt_variables)
        if unknown:
            raise ValueError(
                f"Unsupported Prompt placeholders: {', '.join(sorted(unknown))}"
            )
        normalized_skills = self._normalize_skills(skills or [])
        if node_key == "intake_agent":
            names = [item["name"] for item in normalized_skills]
            if set(names) != set(INTAKE_SKILLS) or len(names) != len(INTAKE_SKILLS):
                raise ValueError("Intake Agent must contain the complete code-owned Skill set")
            for item in normalized_skills:
                expected_skill = INTAKE_SKILLS[item["name"]]
                if f"`{expected_skill}`" not in item["content"].splitlines()[0]:
                    raise ValueError(f"Skill content does not match {item['name']}")
        elif normalized_skills:
            raise ValueError(f"Node {node_key} does not accept a Skill bundle")
        report = {
            "valid": True,
            "node_key": node_key,
            "output_schema": node_spec.output_schema,
            "output_schema_boundary": "code_owned",
            "required_variables": variables,
            "skill_names": [item["name"] for item in normalized_skills],
        }
        return normalized_skills, report

    def _build_revision(
        self,
        definition: PromptDefinition,
        *,
        content: str,
        skills: list[dict],
        source: str,
        version: int,
    ) -> PromptRevision:
        normalized_content = content.strip() + "\n"
        normalized_skills, report = self.validate_prompt(
            node_key=definition.node_key,
            content=normalized_content,
            skills=skills,
        )
        payload = self._revision_payload(
            node_key=definition.node_key,
            content=normalized_content,
            skills=normalized_skills,
            required_variables=report["required_variables"],
        )
        return PromptRevision(
            id=str(uuid4()),
            prompt_definition_id=definition.id,
            version=version,
            content=normalized_content,
            content_hash=canonical_hash(normalized_content),
            required_variables=report["required_variables"],
            skill_bundle=normalized_skills,
            validation_report=report,
            source=source,
            config_hash=canonical_hash(payload),
            published_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _revision_payload(
        *, node_key: str, content: str, skills: list[dict], required_variables: list[str]
    ) -> dict:
        return {
            "node_key": node_key,
            "content": content,
            "skills": skills,
            "required_variables": required_variables,
        }

    @staticmethod
    def _normalize_skills(skills: list[dict]) -> list[dict]:
        normalized = []
        seen: set[str] = set()
        for item in skills:
            name = str(item.get("name", "")).strip()
            content = str(item.get("content", "")).strip()
            if not SKILL_NAME_PATTERN.fullmatch(name):
                raise ValueError(f"Invalid Skill name: {name}")
            if name in seen:
                raise ValueError(f"Duplicate Skill name: {name}")
            if not content.startswith("# Skill:"):
                raise ValueError(f"Skill {name} must start with a Skill heading")
            seen.add(name)
            content = content + "\n"
            digest = canonical_hash(content)
            normalized.append(
                {
                    "revision_id": f"skill:{name}:{digest}",
                    "name": name,
                    "content_hash": digest,
                    "content": content,
                }
            )
        return normalized

    @staticmethod
    def _disk_skills(prompt_dir: Path) -> list[dict]:
        return [
            {"name": path.stem, "content": path.read_text(encoding="utf-8")}
            for path in sorted((prompt_dir / "intake_skills").glob("*.txt"))
        ]

    @staticmethod
    def _validate_slug(slug: str) -> None:
        if not SLUG_PATTERN.fullmatch(slug):
            raise ValueError("Slug must contain 3-64 lowercase letters, numbers, or hyphens")

    @staticmethod
    def _node_spec(node_key: str):
        node = NODE_REGISTRY.get(node_key)
        if node is None:
            raise ValueError(f"Unknown Agent node: {node_key}")
        return node

    def _definition(self, prompt_definition_id: str) -> PromptDefinition:
        definition = self.session.get(PromptDefinition, prompt_definition_id)
        if definition is None or definition.status != "ACTIVE":
            raise KeyError(f"Prompt definition not found: {prompt_definition_id}")
        return definition

    def _next_version(self, prompt_definition_id: str) -> int:
        return (
            self.session.scalar(
                select(func.max(PromptRevision.version)).where(
                    PromptRevision.prompt_definition_id == prompt_definition_id
                )
            )
            or 0
        ) + 1
