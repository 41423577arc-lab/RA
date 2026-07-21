from celery import Celery

from app.config import settings


celery_app = Celery("resource_agent", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_always_eager=settings.celery_task_always_eager,
    imports=("app.tasks.pipeline",),
)
