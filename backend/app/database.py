from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.models.database import Base, IntakeSession, LlmCallLog, ResearchTask


engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_database() -> None:
    if inspect(engine).has_table("research_tasks"):
        _migrate_research_tasks()
    Base.metadata.create_all(engine)
    _migrate_intake_sessions()


def _migrate_intake_sessions() -> None:
    if engine.dialect.name != "postgresql":
        return
    columns = ("messages", "structured_context", "missing_information", "confirmation_request")
    with engine.begin() as connection:
        for column in columns:
            connection.execute(
                text(
                    f"ALTER TABLE intake_sessions ALTER COLUMN {column} "
                    f"TYPE JSONB USING {column}::jsonb"
                )
            )


def _migrate_research_tasks() -> None:
    """Add V0.5 columns to databases created by the V0.4 demo."""
    columns = {column["name"] for column in inspect(engine).get_columns("research_tasks")}
    json_type = "JSONB" if engine.dialect.name == "postgresql" else "JSON"
    additions = {
        "llm_understanding": json_type,
        "confirmation_request": json_type,
        "confirmed_context": json_type,
        "confirmation_version": "INTEGER NOT NULL DEFAULT 0",
        "confirmed_at": "TIMESTAMP",
        "web_search_plan": json_type,
        "verified_web_results": json_type,
        "project_query_plan": json_type,
        "ranked_internal_results": json_type,
        "association_analysis": json_type,
        "generated_report_content": json_type,
        "detailed_report_markdown": "TEXT",
        "action_brief_markdown": "TEXT",
        "degraded_nodes": json_type,
        "prompt_versions": json_type,
        "intake_session_id": "VARCHAR(36)",
        "input_snapshot": json_type,
    }
    with engine.begin() as connection:
        for name, sql_type in additions.items():
            if name not in columns:
                connection.execute(text(f"ALTER TABLE research_tasks ADD COLUMN {name} {sql_type}"))
        connection.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_research_tasks_intake_session_id "
                "ON research_tasks (intake_session_id) "
                "WHERE intake_session_id IS NOT NULL"
            )
        )


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session


class TaskRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, task: ResearchTask) -> ResearchTask:
        self.session.add(task)
        self.session.commit()
        self.session.refresh(task)
        return task

    def get(self, task_id: str) -> ResearchTask | None:
        return self.session.get(ResearchTask, task_id)

    def update(self, task_id: str, **values: object) -> ResearchTask:
        task = self.get(task_id)
        if task is None:
            raise KeyError(f"Task {task_id} not found")
        for key, value in values.items():
            setattr(task, key, value)
        self.session.commit()
        self.session.refresh(task)
        return task

    def log_llm_call(self, task_id: str, **values: object) -> None:
        self.session.add(LlmCallLog(task_id=task_id, **values))
        self.session.commit()


class IntakeSessionRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, session_id: str, *, for_update: bool = False) -> IntakeSession | None:
        statement = select(IntakeSession).where(IntakeSession.id == session_id)
        if for_update:
            statement = statement.with_for_update()
        return self.session.scalar(statement)

    def add(self, intake_session: IntakeSession) -> IntakeSession:
        self.session.add(intake_session)
        self.session.commit()
        self.session.refresh(intake_session)
        return intake_session

    def update(self, session_id: str, **values: object) -> IntakeSession:
        intake_session = self.get(session_id)
        if intake_session is None:
            raise KeyError(f"Intake session {session_id} not found")
        for key, value in values.items():
            setattr(intake_session, key, value)
        self.session.commit()
        self.session.refresh(intake_session)
        return intake_session
