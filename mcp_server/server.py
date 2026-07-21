from mcp.server.fastmcp import FastMCP

from app.config import settings
from mcp_server.entity_repository import EntityRepository
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
entity_repository = EntityRepository(settings.database_readonly_url)


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
def resolve_entities(mentions: list[str], organization_names: list[str]) -> list[dict]:
    """Resolve people and organizations through the read-only internal alias table."""
    return [
        candidate.model_dump(mode="json")
        for candidate in entity_repository.resolve(mentions, organization_names)
    ]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
