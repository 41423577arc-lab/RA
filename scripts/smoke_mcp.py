import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.mcp_client import ProjectMcpClient  # noqa: E402


async def main() -> None:
    client = ProjectMcpClient("http://localhost:8001/mcp")
    projects = await client.search_projects(
        ["王传福"], ["比亚迪股份有限公司"], ["新能源", "储能"]
    )
    print(json.dumps([project.model_dump(mode="json") for project in projects], ensure_ascii=True))


if __name__ == "__main__":
    asyncio.run(main())
