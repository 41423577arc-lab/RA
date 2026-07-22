from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.intake import router as intake_router
from app.api.tasks import router as tasks_router
from app.database import init_database


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_database()
    yield


app = FastAPI(title="资源推动 Agent Demo", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(tasks_router)
app.include_router(intake_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
