import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.mcp_client import ProjectMcpClient  # noqa: E402


async def main() -> None:
    client = ProjectMcpClient("http://localhost:8001/mcp")
    ambiguous_person_results = await client.search_projects(["王总"], [], [])
    relation_keyword_results = await client.search_projects([], [], ["中建二局", "王总"])

    person_ids = {project.project_id for project in ambiguous_person_results}
    relation_ids = {project.project_id for project in relation_keyword_results}
    assert {"P021", "P023", "P025"} <= person_ids
    assert {"P021", "P022", "P023", "P024", "P025", "P026"} <= relation_ids

    print(
        json.dumps(
            {
                "ambiguous_person_project_ids": sorted(person_ids),
                "relation_project_ids": sorted(relation_ids),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
