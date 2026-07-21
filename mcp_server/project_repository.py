from collections.abc import Iterable

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, String, Text, create_engine, or_, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.schemas.task import ProjectResult
from app.services.text_vectorizer import vectorize_text


class Base(DeclarativeBase):
    pass


class InternalProject(Base):
    __tablename__ = "internal_projects"

    project_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_name: Mapped[str] = mapped_column(String(255))
    customer_name: Mapped[str] = mapped_column(String(255))
    contact_name: Mapped[str | None] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(16))
    owner_name: Mapped[str] = mapped_column(String(100))
    start_date: Mapped[object] = mapped_column(Date)
    end_date: Mapped[object | None] = mapped_column(Date)
    description: Mapped[str] = mapped_column(Text)
    project_embedding: Mapped[list[float]] = mapped_column(Vector(512))


PRIORITY = {"PERSON_EXACT": 0, "ORG_EXACT": 1, "TEXT_MATCH": 2, "VECTOR_MATCH": 3}


class ProjectRepository:
    def __init__(self, database_url: str, threshold: float):
        self.session_factory = sessionmaker(
            bind=create_engine(database_url, pool_pre_ping=True), expire_on_commit=False
        )
        self.threshold = threshold

    def search(
        self,
        person_names: list[str],
        organization_names: list[str],
        keywords: list[str],
    ) -> list[ProjectResult]:
        matches: dict[str, tuple[str, float | None, InternalProject]] = {}
        with self.session_factory() as session:
            if person_names:
                self._add_exact_matches(session, matches, person_names, [])
                self._add_text_matches(session, matches, unique_non_empty(person_names))
            else:
                self._add_exact_matches(session, matches, [], organization_names)
                terms = unique_non_empty([*organization_names, *keywords])
                self._add_text_matches(session, matches, terms)
                if keywords:
                    self._add_vector_matches(session, matches, keywords)

        ordered = sorted(
            matches.values(),
            key=lambda item: (PRIORITY[item[0]], -(item[1] or 0.0), item[2].project_id),
        )[:10]
        return [self._to_result(project, match_type, similarity) for match_type, similarity, project in ordered]

    def _add_exact_matches(
        self,
        session: Session,
        matches: dict[str, tuple[str, float | None, InternalProject]],
        people: list[str],
        organizations: list[str],
    ) -> None:
        if people:
            for project in session.scalars(
                select(InternalProject).where(InternalProject.contact_name.in_(people))
            ):
                matches[project.project_id] = ("PERSON_EXACT", None, project)
        if organizations:
            for project in session.scalars(
                select(InternalProject).where(InternalProject.customer_name.in_(organizations))
            ):
                if project.project_id not in matches:
                    matches[project.project_id] = ("ORG_EXACT", None, project)

    def _add_text_matches(
        self,
        session: Session,
        matches: dict[str, tuple[str, float | None, InternalProject]],
        terms: list[str],
    ) -> None:
        conditions = []
        for term in terms:
            pattern = f"%{term}%"
            conditions.extend(
                (
                    InternalProject.contact_name.ilike(pattern),
                    InternalProject.customer_name.ilike(pattern),
                    InternalProject.project_name.ilike(pattern),
                    InternalProject.description.ilike(pattern),
                )
            )
        if not conditions:
            return
        for project in session.scalars(select(InternalProject).where(or_(*conditions))):
            if project.project_id not in matches:
                matches[project.project_id] = ("TEXT_MATCH", None, project)

    def _add_vector_matches(
        self,
        session: Session,
        matches: dict[str, tuple[str, float | None, InternalProject]],
        keywords: list[str],
    ) -> None:
        vector = vectorize_text(" ".join(keywords))
        distance = InternalProject.project_embedding.cosine_distance(vector)
        rows = session.execute(
            select(InternalProject, distance.label("distance"))
            .where(distance <= 1 - self.threshold)
            .order_by(distance)
            .limit(10)
        )
        for project, cosine_distance in rows:
            if project.project_id not in matches:
                matches[project.project_id] = (
                    "VECTOR_MATCH",
                    round(1 - float(cosine_distance), 4),
                    project,
                )

    @staticmethod
    def _to_result(
        project: InternalProject, match_type: str, similarity: float | None
    ) -> ProjectResult:
        return ProjectResult(
            project_id=project.project_id,
            project_name=project.project_name,
            customer_name=project.customer_name,
            contact_name=project.contact_name,
            status=project.status,
            owner_name=project.owner_name,
            start_date=project.start_date,
            end_date=project.end_date,
            description=project.description,
            match_type=match_type,
            similarity=similarity,
        )


def unique_non_empty(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value and len(value.strip()) >= 2))
