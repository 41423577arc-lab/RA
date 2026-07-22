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
def search_projects(
    person_names: list[str], organization_names: list[str], keywords: list[str]
) -> list[dict]:
    """Search read-only active and completed internal projects."""
    return [
        project.model_dump(mode="json")
        for project in repository.search(person_names, organization_names, keywords)
    ]

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
