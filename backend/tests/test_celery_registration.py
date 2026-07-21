from app.tasks.celery_app import celery_app


def test_pipeline_module_is_in_worker_imports() -> None:
    assert "app.tasks.pipeline" in celery_app.conf.imports


def test_pipeline_task_registers_after_module_import() -> None:
    import app.tasks.pipeline  # noqa: F401

    assert "run_research_pipeline" in celery_app.tasks
