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
    project_details = await client.get_project_details("P001")
    sales_portfolio = await client.get_sales_portfolio(sales_rep_name="张伟")
    print(
        json.dumps(
            {
                "search_projects": [project.model_dump(mode="json") for project in projects],
                "get_project_details": project_details,
                "get_sales_portfolio": sales_portfolio,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
