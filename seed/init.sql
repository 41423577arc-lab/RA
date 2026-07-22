CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sales_managers (
    manager_id VARCHAR(16) PRIMARY KEY,
    manager_name VARCHAR(100) NOT NULL,
    region_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sales_representatives (
    sales_rep_id VARCHAR(16) PRIMARY KEY,
    manager_id VARCHAR(16) NOT NULL REFERENCES sales_managers(manager_id),
    sales_rep_name VARCHAR(100) NOT NULL UNIQUE,
    phone VARCHAR(20) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    territory VARCHAR(100) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    hired_on DATE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(16) PRIMARY KEY,
    customer_name VARCHAR(255) NOT NULL UNIQUE,
    industry VARCHAR(100) NOT NULL,
    region_name VARCHAR(100) NOT NULL,
    account_tier VARCHAR(8) NOT NULL CHECK (account_tier IN ('S', 'A', 'B', 'C')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS customer_contacts (
    contact_id VARCHAR(16) PRIMARY KEY,
    customer_id VARCHAR(16) NOT NULL REFERENCES customers(customer_id),
    contact_name VARCHAR(100) NOT NULL,
    job_title VARCHAR(100) NOT NULL,
    phone VARCHAR(20) NOT NULL,
    email VARCHAR(255) NOT NULL,
    is_primary BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (customer_id, contact_name)
);

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
    ADD COLUMN IF NOT EXISTS project_aliases TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS customer_id VARCHAR(16) REFERENCES customers(customer_id),
    ADD COLUMN IF NOT EXISTS customer_contact_id VARCHAR(16) REFERENCES customer_contacts(contact_id),
    ADD COLUMN IF NOT EXISTS sales_rep_id VARCHAR(16) REFERENCES sales_representatives(sales_rep_id),
    ADD COLUMN IF NOT EXISTS project_stage VARCHAR(24),
    ADD COLUMN IF NOT EXISTS health_status VARCHAR(8),
    ADD COLUMN IF NOT EXISTS priority VARCHAR(8),
    ADD COLUMN IF NOT EXISTS contract_value NUMERIC(14, 2),
    ADD COLUMN IF NOT EXISTS win_probability SMALLINT,
    ADD COLUMN IF NOT EXISTS last_activity_date DATE,
    ADD COLUMN IF NOT EXISTS next_followup_date DATE;

CREATE TABLE IF NOT EXISTS project_status_history (
    history_id BIGSERIAL PRIMARY KEY,
    project_id VARCHAR(64) NOT NULL REFERENCES internal_projects(project_id) ON DELETE CASCADE,
    status VARCHAR(16) NOT NULL CHECK (status IN ('ACTIVE', 'COMPLETED')),
    project_stage VARCHAR(24) NOT NULL,
    health_status VARCHAR(8) NOT NULL CHECK (health_status IN ('GREEN', 'AMBER', 'RED')),
    changed_at TIMESTAMPTZ NOT NULL,
    changed_by VARCHAR(100) NOT NULL,
    change_note TEXT NOT NULL,
    UNIQUE (project_id, changed_at)
);

CREATE INDEX IF NOT EXISTS idx_internal_projects_sales_rep ON internal_projects(sales_rep_id);
CREATE INDEX IF NOT EXISTS idx_internal_projects_customer ON internal_projects(customer_id);
CREATE INDEX IF NOT EXISTS idx_internal_projects_stage ON internal_projects(status, project_stage);
CREATE INDEX IF NOT EXISTS idx_project_status_history_project ON project_status_history(project_id, changed_at DESC);
CREATE INDEX IF NOT EXISTS idx_sales_representatives_manager ON sales_representatives(manager_id);

CREATE OR REPLACE VIEW public.sales_rep_project_status_summary AS
SELECT
    sr.sales_rep_id,
    sr.sales_rep_name,
    sr.phone AS sales_rep_phone,
    sm.manager_id,
    sm.manager_name,
    sm.region_name,
    p.status,
    p.project_stage,
    COUNT(p.project_id)::INTEGER AS project_count,
    COALESCE(SUM(p.contract_value), 0)::NUMERIC(16, 2) AS total_contract_value,
    ROUND(AVG(p.win_probability), 1) AS average_win_probability
FROM sales_representatives sr
JOIN sales_managers sm ON sm.manager_id = sr.manager_id
LEFT JOIN internal_projects p ON p.sales_rep_id = sr.sales_rep_id
GROUP BY
    sr.sales_rep_id, sr.sales_rep_name, sr.phone,
    sm.manager_id, sm.manager_name, sm.region_name,
    p.status, p.project_stage;

CREATE OR REPLACE VIEW public.vw_project_overview AS
SELECT
    p.project_id,
    p.project_name,
    p.customer_name,
    c.industry,
    c.region_name AS customer_region,
    c.account_tier,
    cc.contact_name AS customer_contact_name,
    cc.job_title AS customer_contact_title,
    cc.phone AS customer_contact_phone,
    p.status,
    p.project_stage,
    p.health_status,
    p.priority,
    p.contract_value,
    p.win_probability,
    sr.sales_rep_name,
    sr.phone AS sales_rep_phone,
    sr.territory,
    sm.manager_name AS sales_manager_name,
    sm.region_name AS sales_region,
    p.start_date,
    p.end_date,
    p.last_activity_date,
    p.next_followup_date,
    p.description
FROM internal_projects p
JOIN customers c ON c.customer_id = p.customer_id
JOIN customer_contacts cc ON cc.contact_id = p.customer_contact_id
JOIN sales_representatives sr ON sr.sales_rep_id = p.sales_rep_id
JOIN sales_managers sm ON sm.manager_id = sr.manager_id;

CREATE OR REPLACE VIEW public.vw_sales_team_directory AS
SELECT
    sm.manager_id,
    sm.manager_name,
    sm.region_name,
    sm.phone AS manager_phone,
    sr.sales_rep_id,
    sr.sales_rep_name,
    sr.phone AS sales_rep_phone,
    sr.email AS sales_rep_email,
    sr.territory,
    sr.hired_on,
    sr.active
FROM sales_managers sm
JOIN sales_representatives sr ON sr.manager_id = sm.manager_id;

CREATE OR REPLACE VIEW public.vw_sales_rep_workload AS
SELECT
    sr.sales_rep_id,
    sr.sales_rep_name,
    sr.phone AS sales_rep_phone,
    sm.manager_name AS sales_manager_name,
    COUNT(p.project_id)::INTEGER AS total_projects,
    COUNT(p.project_id) FILTER (WHERE p.status = 'ACTIVE')::INTEGER AS active_projects,
    COUNT(p.project_id) FILTER (WHERE p.status = 'COMPLETED')::INTEGER AS completed_projects,
    COUNT(p.project_id) FILTER (WHERE p.health_status = 'RED')::INTEGER AS red_projects,
    COUNT(p.project_id) FILTER (WHERE p.health_status = 'AMBER')::INTEGER AS amber_projects,
    COALESCE(SUM(p.contract_value) FILTER (WHERE p.status = 'ACTIVE'), 0)::NUMERIC(16, 2)
        AS active_pipeline_value,
    COALESCE(
        SUM(p.contract_value * p.win_probability / 100.0) FILTER (WHERE p.status = 'ACTIVE'),
        0
    )::NUMERIC(16, 2) AS weighted_pipeline_value,
    MAX(p.last_activity_date) AS latest_activity_date,
    MIN(p.next_followup_date) FILTER (WHERE p.next_followup_date IS NOT NULL)
        AS nearest_followup_date
FROM sales_representatives sr
JOIN sales_managers sm ON sm.manager_id = sr.manager_id
LEFT JOIN internal_projects p ON p.sales_rep_id = sr.sales_rep_id
GROUP BY sr.sales_rep_id, sr.sales_rep_name, sr.phone, sm.manager_name;

CREATE OR REPLACE VIEW public.vw_customer_project_summary AS
SELECT
    c.customer_id,
    c.customer_name,
    c.industry,
    c.region_name,
    c.account_tier,
    COUNT(p.project_id)::INTEGER AS total_projects,
    COUNT(p.project_id) FILTER (WHERE p.status = 'ACTIVE')::INTEGER AS active_projects,
    COUNT(p.project_id) FILTER (WHERE p.status = 'COMPLETED')::INTEGER AS completed_projects,
    COALESCE(SUM(p.contract_value), 0)::NUMERIC(16, 2) AS total_contract_value,
    MAX(p.last_activity_date) AS latest_activity_date
FROM customers c
LEFT JOIN internal_projects p ON p.customer_id = c.customer_id
GROUP BY c.customer_id, c.customer_name, c.industry, c.region_name, c.account_tier;

CREATE OR REPLACE VIEW public.vw_project_status_timeline AS
SELECT
    h.history_id,
    h.changed_at,
    h.project_id,
    p.project_name,
    p.customer_name,
    sr.sales_rep_name,
    h.status,
    h.project_stage,
    h.health_status,
    h.changed_by,
    h.change_note
FROM project_status_history h
JOIN internal_projects p ON p.project_id = h.project_id
JOIN sales_representatives sr ON sr.sales_rep_id = p.sales_rep_id;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'resource_reader') THEN
        CREATE ROLE resource_reader LOGIN PASSWORD 'resource_reader';
    END IF;
END
$$;

-- Compatibility for legacy Navicat releases that still query the PostgreSQL 14
-- pg_database.datlastsysoid column, which was removed in PostgreSQL 15.
CREATE SCHEMA IF NOT EXISTS navicat_compat;
DROP VIEW IF EXISTS navicat_compat.sales_rep_project_status_summary;
DROP VIEW IF EXISTS navicat_compat.vw_project_overview;
DROP VIEW IF EXISTS navicat_compat.vw_sales_team_directory;
DROP VIEW IF EXISTS navicat_compat.vw_sales_rep_workload;
DROP VIEW IF EXISTS navicat_compat.vw_customer_project_summary;
DROP VIEW IF EXISTS navicat_compat.vw_project_status_timeline;
DROP TABLE IF EXISTS navicat_compat.project_status_history CASCADE;
DROP TABLE IF EXISTS navicat_compat.internal_projects CASCADE;
DROP TABLE IF EXISTS navicat_compat.customer_contacts CASCADE;
DROP TABLE IF EXISTS navicat_compat.customers CASCADE;
DROP TABLE IF EXISTS navicat_compat.sales_representatives CASCADE;
DROP TABLE IF EXISTS navicat_compat.sales_managers CASCADE;
CREATE OR REPLACE VIEW navicat_compat.pg_database AS
SELECT database_catalog.*, 0::OID AS datlastsysoid
FROM pg_catalog.pg_database AS database_catalog;

GRANT USAGE ON SCHEMA navicat_compat TO resource_agent, resource_reader;
GRANT SELECT ON navicat_compat.pg_database TO resource_agent, resource_reader;
GRANT SELECT ON public.vw_project_overview TO resource_reader;
GRANT SELECT ON public.vw_sales_team_directory TO resource_reader;
GRANT SELECT ON public.vw_sales_rep_workload TO resource_reader;
GRANT SELECT ON public.vw_customer_project_summary TO resource_reader;
GRANT SELECT ON public.vw_project_status_timeline TO resource_reader;
ALTER ROLE resource_agent SET search_path = navicat_compat, public, pg_catalog;
ALTER ROLE resource_reader SET search_path = navicat_compat, public, pg_catalog;

GRANT CONNECT ON DATABASE resource_agent TO resource_reader;
GRANT USAGE ON SCHEMA public TO resource_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO resource_reader;
GRANT SELECT ON sales_rep_project_status_summary TO resource_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO resource_reader;
