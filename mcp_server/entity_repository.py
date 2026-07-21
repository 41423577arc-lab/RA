from sqlalchemy import String, create_engine, or_, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.schemas.task import CandidateOption


class Base(DeclarativeBase):
    pass


class EntityAlias(Base):
    __tablename__ = "entity_aliases"

    candidate_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(16))
    canonical_name: Mapped[str] = mapped_column(String(255))
    alias: Mapped[str] = mapped_column(String(255))
    organization_name: Mapped[str | None] = mapped_column(String(255))
    title: Mapped[str | None] = mapped_column(String(100))
    region: Mapped[str | None] = mapped_column(String(100))


class EntityRepository:
    def __init__(self, database_url: str):
        self.session_factory = sessionmaker(
            bind=create_engine(database_url, pool_pre_ping=True), expire_on_commit=False
        )

    def resolve(
        self, mentions: list[str], organization_names: list[str]
    ) -> list[CandidateOption]:
        terms = list(dict.fromkeys(term.strip() for term in mentions if term.strip()))
        if not terms:
            return []
        conditions = []
        for term in terms:
            conditions.extend(
                (EntityAlias.canonical_name.ilike(f"%{term}%"), EntityAlias.alias.ilike(f"%{term}%"))
            )
        statement = select(EntityAlias).where(or_(*conditions))
        if organization_names:
            org_conditions = [
                EntityAlias.organization_name.ilike(f"%{name}%")
                for name in organization_names
                if name
            ]
            if org_conditions:
                statement = statement.where(or_(*org_conditions))
        with self.session_factory() as session:
            rows = list(session.scalars(statement.limit(20)))
        return [
            CandidateOption(
                candidate_id=row.candidate_id,
                entity_type=row.entity_type,
                canonical_name=row.canonical_name,
                aliases=[row.alias],
                organization=row.organization_name,
                title=row.title,
                region=row.region,
                reason="内部实体别名库匹配",
                confidence=0.95 if row.canonical_name in terms else 0.80,
            )
            for row in rows
        ]
