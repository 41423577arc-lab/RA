import csv
import os
from datetime import datetime, time, timezone
from pathlib import Path

from sqlalchemy import create_engine, text

from app.services.text_vectorizer import vectorize_texts


ROOT = Path(__file__).resolve().parent
DATABASE_URL = os.environ["DATABASE_URL"]


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    engine = create_engine(DATABASE_URL)
    projects = read_csv("internal_projects.csv")
    organization = read_csv("sales_organization.csv")
    contact_rows = read_csv("customer_contacts.csv")
    project_details = read_csv("project_details.csv")
    details_by_project = {row["project_id"]: row for row in project_details}

    contents = [
        f"{row['customer_name']} {row['project_name']} {row['description']}" for row in projects
    ]
    vectors = vectorize_texts(contents)

    with engine.begin() as connection:
        connection.exec_driver_sql("SET LOCAL search_path = public, pg_catalog")
        connection.exec_driver_sql((ROOT / "init.sql").read_text(encoding="utf-8"))
        seed_sales_organization(connection, organization)
        seed_customers_and_contacts(connection, contact_rows)
        seed_projects(connection, projects, vectors, details_by_project)
        seed_status_history(connection, projects, details_by_project)
        enforce_project_constraints(connection)
        connection.execute(text("GRANT SELECT ON ALL TABLES IN SCHEMA public TO resource_reader"))


def seed_sales_organization(connection, rows: list[dict[str, str]]) -> None:
    manager_statement = text(
        """
        INSERT INTO sales_managers (
            manager_id, manager_name, region_name, phone, email
        ) VALUES (
            :id, :name, :region_or_territory, :phone, :email
        )
        ON CONFLICT (manager_id) DO UPDATE SET
            manager_name = EXCLUDED.manager_name,
            region_name = EXCLUDED.region_name,
            phone = EXCLUDED.phone,
            email = EXCLUDED.email,
            active = TRUE
        """
    )
    representative_statement = text(
        """
        INSERT INTO sales_representatives (
            sales_rep_id, manager_id, sales_rep_name, phone, email, territory, hired_on
        ) VALUES (
            :id, :parent_id, :name, :phone, :email, :region_or_territory, :hired_on
        )
        ON CONFLICT (sales_rep_id) DO UPDATE SET
            manager_id = EXCLUDED.manager_id,
            sales_rep_name = EXCLUDED.sales_rep_name,
            phone = EXCLUDED.phone,
            email = EXCLUDED.email,
            territory = EXCLUDED.territory,
            hired_on = EXCLUDED.hired_on,
            active = TRUE
        """
    )
    for row in rows:
        statement = manager_statement if row["record_type"] == "MANAGER" else representative_statement
        connection.execute(statement, row)


def seed_customers_and_contacts(connection, rows: list[dict[str, str]]) -> None:
    customer_statement = text(
        """
        INSERT INTO customers (
            customer_id, customer_name, industry, region_name, account_tier
        ) VALUES (
            :customer_id, :customer_name, :industry, :region_name, :account_tier
        )
        ON CONFLICT (customer_id) DO UPDATE SET
            customer_name = EXCLUDED.customer_name,
            industry = EXCLUDED.industry,
            region_name = EXCLUDED.region_name,
            account_tier = EXCLUDED.account_tier
        """
    )
    contact_statement = text(
        """
        INSERT INTO customer_contacts (
            contact_id, customer_id, contact_name, job_title, phone, email
        ) VALUES (
            :contact_id, :customer_id, :contact_name, :job_title, :phone, :email
        )
        ON CONFLICT (contact_id) DO UPDATE SET
            customer_id = EXCLUDED.customer_id,
            contact_name = EXCLUDED.contact_name,
            job_title = EXCLUDED.job_title,
            phone = EXCLUDED.phone,
            email = EXCLUDED.email,
            is_primary = TRUE
        """
    )
    for row in rows:
        connection.execute(customer_statement, row)
        connection.execute(contact_statement, row)


def seed_projects(connection, rows, vectors, details_by_project) -> None:
    statement = text(
        """
        INSERT INTO internal_projects (
            project_id, project_name, customer_name, contact_name, status, owner_name,
            start_date, end_date, description, project_embedding, customer_id,
            customer_contact_id, sales_rep_id, project_stage, health_status, priority,
            contract_value, win_probability, last_activity_date, next_followup_date
        ) VALUES (
            :project_id, :project_name, :customer_name, :contact_name, :status, :owner_name,
            :start_date, :end_date, :description, CAST(:project_embedding AS vector), :customer_id,
            :customer_contact_id, :sales_rep_id, :project_stage, :health_status, :priority,
            :contract_value, :win_probability, :last_activity_date, :next_followup_date
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
            project_embedding = EXCLUDED.project_embedding,
            customer_id = EXCLUDED.customer_id,
            customer_contact_id = EXCLUDED.customer_contact_id,
            sales_rep_id = EXCLUDED.sales_rep_id,
            project_stage = EXCLUDED.project_stage,
            health_status = EXCLUDED.health_status,
            priority = EXCLUDED.priority,
            contract_value = EXCLUDED.contract_value,
            win_probability = EXCLUDED.win_probability,
            last_activity_date = EXCLUDED.last_activity_date,
            next_followup_date = EXCLUDED.next_followup_date
        """
    )
    for row, vector in zip(rows, vectors, strict=True):
        values = {**row, **details_by_project[row["project_id"]]}
        values["end_date"] = values["end_date"] or None
        values["next_followup_date"] = values["next_followup_date"] or None
        values["project_embedding"] = "[" + ",".join(map(str, vector)) + "]"
        connection.execute(statement, values)


def seed_status_history(connection, projects, details_by_project) -> None:
    statement = text(
        """
        INSERT INTO project_status_history (
            project_id, status, project_stage, health_status, changed_at, changed_by, change_note
        ) VALUES (
            :project_id, :status, :project_stage, :health_status, :changed_at, :changed_by,
            :change_note
        )
        ON CONFLICT (project_id, changed_at) DO UPDATE SET
            status = EXCLUDED.status,
            project_stage = EXCLUDED.project_stage,
            health_status = EXCLUDED.health_status,
            changed_by = EXCLUDED.changed_by,
            change_note = EXCLUDED.change_note
        """
    )
    for project in projects:
        details = details_by_project[project["project_id"]]
        initial_at = datetime.combine(
            datetime.fromisoformat(project["start_date"]).date(), time(9), timezone.utc
        )
        current_date = project["end_date"] or details["last_activity_date"]
        current_at = datetime.combine(
            datetime.fromisoformat(current_date).date(), time(18), timezone.utc
        )
        connection.execute(
            statement,
            {
                "project_id": project["project_id"],
                "status": "ACTIVE",
                "project_stage": "DISCOVERY",
                "health_status": "GREEN",
                "changed_at": initial_at,
                "changed_by": project["owner_name"],
                "change_note": "项目立项并进入商机发现阶段",
            },
        )
        connection.execute(
            statement,
            {
                "project_id": project["project_id"],
                "status": project["status"],
                "project_stage": details["project_stage"],
                "health_status": details["health_status"],
                "changed_at": current_at,
                "changed_by": project["owner_name"],
                "change_note": "种子数据同步的最新项目状态",
            },
        )


def enforce_project_constraints(connection) -> None:
    connection.exec_driver_sql(
        """
        ALTER TABLE internal_projects
            ALTER COLUMN customer_id SET NOT NULL,
            ALTER COLUMN customer_contact_id SET NOT NULL,
            ALTER COLUMN sales_rep_id SET NOT NULL,
            ALTER COLUMN project_stage SET NOT NULL,
            ALTER COLUMN health_status SET NOT NULL,
            ALTER COLUMN priority SET NOT NULL,
            ALTER COLUMN contract_value SET NOT NULL,
            ALTER COLUMN win_probability SET NOT NULL,
            ALTER COLUMN last_activity_date SET NOT NULL;

        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_project_health_status') THEN
                ALTER TABLE internal_projects ADD CONSTRAINT ck_project_health_status
                    CHECK (health_status IN ('GREEN', 'AMBER', 'RED'));
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_project_priority') THEN
                ALTER TABLE internal_projects ADD CONSTRAINT ck_project_priority
                    CHECK (priority IN ('P0', 'P1', 'P2', 'P3'));
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_project_probability') THEN
                ALTER TABLE internal_projects ADD CONSTRAINT ck_project_probability
                    CHECK (win_probability BETWEEN 0 AND 100);
            END IF;
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_project_contract_value') THEN
                ALTER TABLE internal_projects ADD CONSTRAINT ck_project_contract_value
                    CHECK (contract_value >= 0);
            END IF;
        END
        $$;
        """
    )


if __name__ == "__main__":
    main()
