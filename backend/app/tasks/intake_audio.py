from pathlib import Path

from app.config import settings
from app.database import SessionLocal
from app.models.database import IntakeAudioJob, IntakeSession
from app.services.transcriber import LocalWhisperTranscriber
from app.tasks.celery_app import celery_app


@celery_app.task(name="run_intake_audio_transcription")
def run_intake_audio_transcription(job_id: str) -> None:
    with SessionLocal() as session:
        job = session.get(IntakeAudioJob, job_id)
        if job is None or job.status == "TRANSCRIBED":
            return
        path = Path(job.audio_path) if job.audio_path else None
        job.status = "TRANSCRIBING"
        session.commit()
        try:
            if path is None:
                raise ValueError("录音文件不存在")
            transcript = LocalWhisperTranscriber(settings.whisper_model_path).transcribe(path)
            if not transcript:
                raise ValueError("未识别到有效语音")
            job.transcript = transcript
            job.status = "NEEDS_REVIEW"
            job.error_message = None
        except Exception as exc:
            job.status = "FAILED"
            job.error_message = str(exc)[:1000]
            job.retry_count += 1
            intake_session = session.get(IntakeSession, job.session_id)
            if intake_session is not None:
                intake_session.status = "COLLECTING"
        session.commit()
