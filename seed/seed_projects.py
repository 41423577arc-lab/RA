import csv
import os
from pathlib import Path

from sqlalchemy import create_engine, text

from app.services.text_vectorizer import vectorize_texts


ROOT = Path(__file__).resolve().parent
DATABASE_URL = os.environ["DATABASE_URL"]
def main() -> None:
    engine = create_engine(DATABASE_URL)
    with (ROOT / "internal_projects.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    contents = [
        f"{row['customer_name']} {row['project_name']} {row['description']}" for row in rows
    ]
    vectors = vectorize_texts(contents)
    statement = text(
        """
        INSERT INTO internal_projects (
            project_id, project_name, customer_name, contact_name, status, owner_name,
            start_date, end_date, description, project_embedding
        ) VALUES (
            :project_id, :project_name, :customer_name, :contact_name, :status, :owner_name,
            :start_date, :end_date, :description, CAST(:project_embedding AS vector)
        )
        ON CONFLICT (project_id) DO UPDATE SET
            project_name = EXCLUDED.project_name,
            customer_name = EXCLUDED.customer_name,
            contact_name = EXCLUDED.contact_name,
            status = EXCLUDED.status,
            owner_name = EXCLUDED.owner_name,
            start_date = EXCLUDED.start_date,
            end_date = EXCLUDED.end_date,
            description = EXCLUDED.description,
            project_embedding = EXCLUDED.project_embedding
        """
    )
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE internal_projects ADD COLUMN IF NOT EXISTS project_aliases TEXT[] NOT NULL DEFAULT '{}'") )
        connection.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS entity_aliases (
                    candidate_id VARCHAR(64) PRIMARY KEY,
                    entity_type VARCHAR(16) NOT NULL,
                    canonical_name VARCHAR(255) NOT NULL,
                    alias VARCHAR(255) NOT NULL,
                    organization_name VARCHAR(255),
                    title VARCHAR(100),
                    region VARCHAR(100)
                )
                """
            )
        )
        for row, vector in zip(rows, vectors, strict=True):
            values = dict(row)
            values["end_date"] = values["end_date"] or None
            values["project_embedding"] = "[" + ",".join(map(str, vector)) + "]"
            connection.execute(statement, values)
        with (ROOT / "entities.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            entities = list(csv.DictReader(handle))
        entity_statement = text(
            """
            INSERT INTO entity_aliases (
                candidate_id, entity_type, canonical_name, alias,
                organization_name, title, region
            ) VALUES (
                :candidate_id, 'PERSON', :person_name, :person_alias,
                :organization_name, :title, :region
            ) ON CONFLICT (candidate_id) DO UPDATE SET
                canonical_name = EXCLUDED.canonical_name,
                alias = EXCLUDED.alias,
                organization_name = EXCLUDED.organization_name,
                title = EXCLUDED.title,
                region = EXCLUDED.region
            """
        )
        for entity in entities:
            connection.execute(entity_statement, entity)
        connection.execute(text("GRANT SELECT ON internal_projects, entity_aliases TO resource_reader"))


if __name__ == "__main__":
    main()
