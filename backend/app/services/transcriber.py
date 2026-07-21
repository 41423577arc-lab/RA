import subprocess
from pathlib import Path

from opencc import OpenCC


_TRADITIONAL_TO_SIMPLIFIED = OpenCC("t2s")


def normalize_transcript(text: str) -> str:
    return _TRADITIONAL_TO_SIMPLIFIED.convert(text).strip()


class LocalWhisperTranscriber:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None

    def transcribe(self, webm_path: Path) -> str:
        wav_path = webm_path.with_suffix(".wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(webm_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
        try:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self.model_path, device="cpu", compute_type="int8")
            segments, _ = self._model.transcribe(
                str(wav_path), language="zh", vad_filter=True, beam_size=5
            )
            transcript = "".join(segment.text.strip() for segment in segments)
            return normalize_transcript(transcript)
        finally:
            wav_path.unlink(missing_ok=True)
