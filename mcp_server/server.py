from mcp.server.fastmcp import FastMCP

from app.config import settings
from mcp_server.project_repository import ProjectRepository


mcp = FastMCP(
    "Internal Project Search",
    host="0.0.0.0",
    port=8001,
    stateless_http=True,
    json_response=True,
)
repository = ProjectRepository(
    settings.database_readonly_url,
    settings.vector_similarity_threshold,
)


@mcp.tool()
def find_entity_candidates(
    person_mention: str | None = None,
    organization_mention: str | None = None,
) -> list[dict]:
    """Find read-only internal customer and contact candidates."""
    return repository.find_entity_candidates(person_mention, organization_mention)


@mcp.tool()
def search_projects(
    person_names: list[str], organization_names: list[str], keywords: list[str]
) -> list[dict]:
    """Search read-only active and completed internal projects."""
    return [
        project.model_dump(mode="json")
        for project in repository.search(person_names, organization_names, keywords)
    ]


@mcp.tool()
def get_project_details(project_id: str) -> dict:
    """Get one project with its customer lead, internal salesperson, manager, and current status."""
    project = repository.get_project_details(project_id)
    return project or {"error": "PROJECT_NOT_FOUND", "project_id": project_id}


@mcp.tool()
def get_sales_portfolio(
    manager_name: str | None = None, sales_rep_name: str | None = None
) -> list[dict]:
    """List project counts and values by salesperson, project status, and project stage."""
    return repository.get_sales_portfolio(manager_name, sales_rep_name)

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
