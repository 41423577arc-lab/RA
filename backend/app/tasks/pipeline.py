import asyncio
from pathlib import Path
from typing import Protocol

from app.config import settings
from app.database import SessionLocal, TaskRepository
from app.schemas.task import (
    ConfirmedContext,
    ConfirmedEntity,
    ExtractedInfo,
    IntentUnderstanding,
    ProjectQueryPlan,
    ProjectResult,
    PublicClaim,
)
from app.services.agent_nodes import (
    AgentNodes,
    deterministic_rankings,
    fallback_association,
    fallback_project_query,
    fallback_report_content,
    fallback_understanding,
    validate_analysis,
    validate_rankings,
    validate_report_content,
)
from app.services.entity_resolver import EntityResolver
from app.services.extractor import RuleExtractor
from app.services.llm_client import StructuredLLM
from app.services.mcp_client import ProjectMcpClient
from app.services.report_renderer import ReportRenderer
from app.services.tavily_client import TavilyClient
from app.services.transcriber import LocalWhisperTranscriber
from app.tasks.celery_app import celery_app


class Repository(Protocol):
    def get(self, task_id: str): ...

    def update(self, task_id: str, **values: object): ...


class Transcriber(Protocol):
    def transcribe(self, webm_path: Path) -> str: ...


class ProjectService(Protocol):
    async def search_projects(
        self, person_names: list[str], organization_names: list[str], keywords: list[str]
    ) -> list[ProjectResult]: ...


class ResearchPipeline:
    def __init__(
        self,
        repository: Repository,
        transcriber: Transcriber,
        extractor: RuleExtractor,
        web: object,
        projects: ProjectService,
        renderer: ReportRenderer,
        agents: AgentNodes | None = None,
        entity_resolver: EntityResolver | None = None,
    ):
        self.repository = repository
        self.transcriber = transcriber
        self.extractor = extractor
        self.projects = projects
        self.renderer = renderer
        self.agents = agents
        self.entity_resolver = entity_resolver

    def run(self, task_id: str) -> None:
        if self.agents is None or self.entity_resolver is None:
            self._run_legacy(task_id)
            return
        task = self.repository.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        audio_path = Path(task.audio_path) if task.audio_path else None
        degraded = list(task.degraded_nodes or [])
        try:
            input_text = task.input_text or ""
            if task.input_type == "audio" and not input_text:
                self.repository.update(task_id, status="TRANSCRIBING")
                input_text = self.transcriber.transcribe(audio_path)
                if not input_text:
                    raise ValueError("未识别到有效语音，请重新录制")
                self.repository.update(task_id, input_text=input_text)

            if task.confirmed_context:
                extracted = ExtractedInfo.model_validate(task.extracted_info)
                understanding = IntentUnderstanding.model_validate(task.llm_understanding)
                context = ConfirmedContext.model_validate(task.confirmed_context)
            else:
                self.repository.update(task_id, status="CONTEXT_EXTRACTING")
                extracted = self.extractor.extract(input_text)
                self.repository.update(task_id, extracted_info=extracted.model_dump(mode="json"))

                understanding = self._with_fallback(
                    "understanding",
                    degraded,
                    lambda: self.agents.understanding(task_id, input_text, extracted),
                    lambda: fallback_understanding(extracted),
                )
                self.repository.update(
                    task_id, llm_understanding=understanding.model_dump(mode="json")
                )

                context = context_from_intake_snapshot(
                    getattr(task, "input_snapshot", None), understanding
                )
                confirmation = None
                if context is None:
                    version = int(task.confirmation_version or 0) + 1
                    context, confirmation = self.entity_resolver.resolve(
                        input_text, understanding, version
                    )
                if confirmation:
                    self.repository.update(
                        task_id,
                        status="NEEDS_CONFIRMATION",
                        confirmation_version=confirmation.version,
                        confirmation_request=confirmation.model_dump(mode="json"),
                        degraded_nodes=degraded,
                    )
                    return
                self.repository.update(
                    task_id,
                    confirmed_context=context.model_dump(mode="json"),
                    confirmation_request=None,
                )

            claims = identity_claims_from_intake_snapshot(
                getattr(task, "input_snapshot", None)
            )
            web_search_status = "REUSED_INTAKE" if claims else "SKIPPED"
            web_fetch_status = web_search_status
            self.repository.update(
                task_id,
                web_search_plan=None,
                web_results=[],
                web_pages=[],
                web_search_status=web_search_status,
                web_fetch_status=web_fetch_status,
                verified_web_results=[],
                public_claims=[item.model_dump(mode="json") for item in claims],
            )

            self.repository.update(task_id, status="PLANNING_PROJECT_SEARCH")
            project_plan = self._with_fallback(
                "project_query",
                degraded,
                lambda: self.agents.project_query(task_id, context),
                lambda: fallback_project_query(context),
            )
            project_plan = sanitize_project_plan(project_plan, context)
            self.repository.update(
                task_id, project_query_plan=project_plan.model_dump(mode="json")
            )

            self.repository.update(task_id, status="PROJECT_SEARCHING")
            project_results: list[ProjectResult] = []
            internal_search_status = "SUCCESS"
            try:
                project_results = asyncio.run(
                    self.projects.search_projects(
                        project_plan.person_names,
                        project_plan.organization_names,
                        unique_non_empty([*project_plan.project_names, *project_plan.business_terms]),
                    )
                )
                project_results = [
                    item for item in project_results if item.status in project_plan.statuses
                ]
            except Exception:
                internal_search_status = "FAILED"
            self.repository.update(
                task_id,
                internal_results=[item.model_dump(mode="json") for item in project_results],
                internal_search_status=internal_search_status,
            )

            self.repository.update(task_id, status="RERANKING_PROJECTS")
            rankings = self._with_fallback(
                "project_rerank",
                degraded,
                lambda: validate_rankings(
                    self.agents.project_rerank(task_id, context, project_results).rankings,
                    project_results,
                    settings.llm_project_confidence_threshold,
                ),
                lambda: deterministic_rankings(project_results, context),
            )
            self.repository.update(
                task_id,
                ranked_internal_results=[item.model_dump(mode="json") for item in rankings],
            )

            self.repository.update(task_id, status="ANALYZING_ASSOCIATIONS")
            analysis = self._with_fallback(
                "association",
                degraded,
                lambda: validate_analysis(
                    self.agents.association(task_id, context, claims, project_results, rankings),
                    claims,
                    project_results,
                    settings.llm_analysis_confidence_threshold,
                ),
                lambda: fallback_association(claims, project_results, rankings),
            )
            self.repository.update(
                task_id, association_analysis=analysis.model_dump(mode="json")
            )

            self.repository.update(task_id, status="GENERATING_REPORT_CONTENT")
            report_content = self._with_fallback(
                "report_content",
                degraded,
                lambda: validate_report_content(
                    self.agents.report_content(
                        task_id, input_text, context, claims, project_results, analysis
                    ),
                    claims,
                    project_results,
                    context,
                ),
                lambda: validate_report_content(
                    fallback_report_content(
                        input_text, context, analysis, claims, project_results
                    ),
                    claims,
                    project_results,
                    context,
                ),
            )
            self.repository.update(
                task_id, generated_report_content=report_content.model_dump(mode="json")
            )

            self.repository.update(task_id, status="RENDERING_REPORT")
            detailed, action = self.renderer.render_generated(
                report_content,
                claims,
                project_results,
                web_search_status,
                web_fetch_status,
                internal_search_status,
            )
            self.repository.update(
                task_id,
                status="COMPLETED",
                detailed_report_markdown=detailed,
                action_brief_markdown=action,
                report_markdown=detailed,
                degraded_nodes=degraded,
                error_message=None,
            )
        except Exception as exc:
            self.repository.update(task_id, status="FAILED", error_message=str(exc), degraded_nodes=degraded)
        finally:
            if audio_path:
                audio_path.unlink(missing_ok=True)
                audio_path.with_suffix(".wav").unlink(missing_ok=True)

    def _with_fallback(self, node_name: str, degraded: list[str], call, fallback):
        try:
            return call()
        except Exception:
            if node_name not in degraded:
                degraded.append(node_name)
            return fallback()

    def _run_legacy(self, task_id: str) -> None:
        task = self.repository.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        audio_path = Path(task.audio_path) if task.audio_path else None
        try:
            input_text = task.input_text or ""
            if task.input_type == "audio":
                self.repository.update(task_id, status="TRANSCRIBING")
                input_text = self.transcriber.transcribe(audio_path)
                if not input_text:
                    raise ValueError("未识别到有效语音，请重新录制")
                self.repository.update(task_id, input_text=input_text)
            self.repository.update(task_id, status="EXTRACTING")
            extracted = self.extractor.extract(input_text)
            self.repository.update(task_id, extracted_info=extracted.model_dump(mode="json"))
            claims = identity_claims_from_intake_snapshot(
                getattr(task, "input_snapshot", None)
            )
            web_search_status = "REUSED_INTAKE" if claims else "SKIPPED"
            web_fetch_status = web_search_status
            self.repository.update(
                task_id,
                web_search_plan=None,
                web_results=[],
                web_pages=[],
                web_search_status=web_search_status,
                web_fetch_status=web_fetch_status,
                verified_web_results=[],
                public_claims=[item.model_dump(mode="json") for item in claims],
            )
            self.repository.update(task_id, status="PROJECT_SEARCHING")
            person_names = unique_non_empty(person.name for person in extracted.people)
            organization_names = unique_non_empty(person.organization for person in extracted.people)
            project_results: list[ProjectResult] = []
            internal_search_status = "SUCCESS"
            try:
                project_results = asyncio.run(
                    self.projects.search_projects(person_names, organization_names, extracted.keywords)
                )
            except Exception:
                internal_search_status = "FAILED"
            self.repository.update(
                task_id,
                internal_results=[item.model_dump(mode="json") for item in project_results],
                internal_search_status=internal_search_status,
            )
            self.repository.update(task_id, status="GENERATING")
            report = self.renderer.render(
                input_text,
                extracted,
                claims,
                project_results,
                web_search_status,
                web_fetch_status,
                internal_search_status,
            )
            self.repository.update(task_id, status="COMPLETED", report_markdown=report)
        except Exception as exc:
            self.repository.update(task_id, status="FAILED", error_message=str(exc))
        finally:
            if audio_path:
                audio_path.unlink(missing_ok=True)
                audio_path.with_suffix(".wav").unlink(missing_ok=True)


def sanitize_project_plan(
    plan: ProjectQueryPlan, context: ConfirmedContext
) -> ProjectQueryPlan:
    base = fallback_project_query(context)
    return plan.model_copy(
        update={
            "person_names": unique_non_empty([*base.person_names, *plan.person_names]),
            "organization_names": unique_non_empty(
                [*base.organization_names, *plan.organization_names]
            ),
            "project_names": unique_non_empty([*base.project_names, *plan.project_names]),
            "business_terms": unique_non_empty(
                [*base.business_terms, *plan.business_terms]
            ),
            "statuses": plan.statuses or ["ACTIVE", "COMPLETED"],
        }
    )


def unique_non_empty(values) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def context_from_intake_snapshot(
    snapshot: dict | None, understanding: IntentUnderstanding
) -> ConfirmedContext | None:
    structured = (snapshot or {}).get("structured_context", {})
    resolutions = structured.get("entity_resolutions", [])
    entities: list[ConfirmedEntity] = []
    for item in resolutions:
        entity_type = item.get("entity_type")
        canonical_name = (item.get("canonical_name") or "").strip()
        if entity_type not in {"PERSON", "ORGANIZATION", "PROJECT"} or not canonical_name:
            continue
        entities.append(
            ConfirmedEntity(
                candidate_id=item.get("candidate_id"),
                entity_type=entity_type,
                canonical_name=canonical_name,
                aliases=item.get("aliases") or [],
                organization=item.get("organization"),
                title=item.get("title"),
                region=item.get("region"),
                confirmed_by="AUTO"
                if item.get("confirmed_by") in {"INTERNAL", "EXTERNAL_AUTO", "AUTO"}
                else "USER",
            )
        )
    if not entities:
        return None
    return ConfirmedContext(
        intents=understanding.intents,
        entities=entities,
        event_type=understanding.event_type,
        event_time=structured.get("event_time") or understanding.event_time,
        event_location=structured.get("event_location") or understanding.event_location,
        business_directions=understanding.business_directions,
        focus_questions=understanding.focus_questions,
    )


def identity_claims_from_intake_snapshot(snapshot: dict | None) -> list[PublicClaim]:
    structured = (snapshot or {}).get("structured_context", {})
    resolutions = structured.get("entity_resolutions", [])
    claims: list[PublicClaim] = []
    for item in resolutions:
        source_url = (item.get("source_url") or "").strip()
        evidence_quote = (item.get("evidence_quote") or "").strip()
        canonical_name = (item.get("canonical_name") or "").strip()
        if not source_url or not evidence_quote or not canonical_name:
            continue
        organization = (item.get("organization") or "").strip()
        title = (item.get("title") or "").strip()
        details = "、".join(value for value in (organization, title) if value)
        claim = f"{canonical_name}（{details}）" if details else canonical_name
        index = len(claims) + 1
        claims.append(
            PublicClaim(
                web_result_id=f"INTAKE{index:03d}",
                evidence_id="IDENTITY",
                subject=canonical_name,
                claim=claim,
                evidence_quote=evidence_quote,
                source_title="关键人身份核验来源",
                source_url=source_url,
                matched_keywords=[],
                confidence=float(item.get("confidence") or 1),
            )
        )
    return claims




@celery_app.task(name="run_research_pipeline")
def run_research_pipeline(task_id: str) -> None:
    with SessionLocal() as session:
        repository = TaskRepository(session)
        llm = StructuredLLM(settings, repository)
        pipeline = ResearchPipeline(
            repository=repository,
            transcriber=LocalWhisperTranscriber(settings.whisper_model_path),
            extractor=RuleExtractor(settings.seed_dir),
            web=TavilyClient(settings.tavily_api_key),
            projects=ProjectMcpClient(settings.mcp_server_url),
            renderer=ReportRenderer(
                settings.report_template,
                settings.detailed_report_template,
                settings.action_brief_template,
            ),
            agents=AgentNodes(llm),
            entity_resolver=EntityResolver(),
        )
        pipeline.run(task_id)
