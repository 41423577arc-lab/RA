import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
os.environ["DATABASE_URL"] = "sqlite:///./test_resource_agent.db"
os.environ["SEED_DIR"] = str(ROOT / "seed")
os.environ["REPORT_TEMPLATE"] = str(ROOT / "backend/templates/report.md.j2")
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
