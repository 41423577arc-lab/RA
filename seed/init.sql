CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS internal_projects (
    project_id VARCHAR(64) PRIMARY KEY,
    project_name VARCHAR(255) NOT NULL,
    customer_name VARCHAR(255) NOT NULL,
    contact_name VARCHAR(100),
    status VARCHAR(16) NOT NULL CHECK (status IN ('ACTIVE', 'COMPLETED')),
    owner_name VARCHAR(100) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    description TEXT NOT NULL,
    project_embedding VECTOR(512) NOT NULL
);

ALTER TABLE internal_projects
    ADD COLUMN IF NOT EXISTS project_aliases TEXT[] NOT NULL DEFAULT '{}';

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'resource_reader') THEN
        CREATE ROLE resource_reader LOGIN PASSWORD 'resource_reader';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE resource_agent TO resource_reader;
GRANT USAGE ON SCHEMA public TO resource_reader;
GRANT SELECT ON internal_projects TO resource_reader;
