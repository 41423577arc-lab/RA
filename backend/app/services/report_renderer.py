from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from app.schemas.task import ExtractedInfo, GeneratedReportContent, ProjectResult, PublicClaim


_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _superscript(value: int) -> str:
    return str(value).translate(_SUPERSCRIPT_DIGITS)


def _markdown_table_cell(value: object) -> str:
    return " ".join(str(value).splitlines()).replace("|", "\\|")


class ReportRenderer:
    def __init__(
        self,
        template_path: Path,
        detailed_template_path: Path | None = None,
        action_template_path: Path | None = None,
    ):
        self.environment = Environment(
            loader=FileSystemLoader(str(template_path.parent)),
            undefined=StrictUndefined,
            autoescape=select_autoescape(default_for_string=True, default=True),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template_name = template_path.name
        self.detailed_template_name = (detailed_template_path or template_path).name
        self.action_template_name = action_template_path.name if action_template_path else None
        self.environment.filters["table_cell"] = _markdown_table_cell

    def render(
        self,
        input_text: str,
        extracted: ExtractedInfo,
        claims: list[PublicClaim],
        projects: list[ProjectResult],
        web_search_status: str | None,
        web_fetch_status: str | None,
        internal_search_status: str | None,
    ) -> str:
        active = [project for project in projects if project.status == "ACTIVE"]
        completed = [project for project in projects if project.status == "COMPLETED"]
        return self.environment.get_template(self.template_name).render(
            input_text=input_text,
            info=extracted,
            claims=claims,
            active_projects=active,
            completed_projects=completed,
            web_search_failed=web_search_status == "FAILED",
            web_fetch_failed=web_fetch_status == "FAILED",
            internal_search_failed=internal_search_status == "FAILED",
        )

    def render_generated(
        self,
        content: GeneratedReportContent,
        claims: list[PublicClaim],
        projects: list[ProjectResult],
        web_search_status: str | None,
        web_fetch_status: str | None,
        internal_search_status: str | None,
    ) -> tuple[str, str]:
        source_numbers: dict[str, int] = {}
        evidence_links: dict[str, dict[str, str | int]] = {}
        for claim in claims:
            if not claim.source_url:
                continue
            if claim.source_url not in source_numbers:
                source_numbers[claim.source_url] = len(source_numbers) + 1
            number = source_numbers[claim.source_url]
            evidence_links[f"WEB:{claim.web_result_id}:{claim.evidence_id}"] = {
                "label": f"来源{number}",
                "marker": _superscript(number),
                "number": number,
                "url": claim.source_url,
            }
        project_refs = {f"PROJECT:{project.project_id}": project for project in projects}

        def source_refs(refs: list[str]) -> dict[str, list[object]]:
            web: list[dict[str, str | int]] = []
            internal: list[ProjectResult] = []
            seen_web: set[int] = set()
            seen_projects: set[str] = set()
            for ref in refs:
                link = evidence_links.get(ref)
                if link and link["number"] not in seen_web:
                    seen_web.add(int(link["number"]))
                    web.append(link)
                project = project_refs.get(ref)
                if project and project.project_id not in seen_projects:
                    seen_projects.add(project.project_id)
                    internal.append(project)
            return {"web": web, "projects": internal}

        context = {
            "content": content,
            "claims": claims,
            "projects": projects,
            "evidence_links": evidence_links,
            "project_refs": project_refs,
            "source_refs": source_refs,
            "web_search_failed": web_search_status == "FAILED",
            "web_fetch_failed": web_fetch_status == "FAILED",
            "internal_search_failed": internal_search_status == "FAILED",
        }
        detailed = self.environment.get_template(self.detailed_template_name).render(**context)
        if self.action_template_name:
            action = self.environment.get_template(self.action_template_name).render(**context)
        else:
            action = detailed
        return detailed, action
