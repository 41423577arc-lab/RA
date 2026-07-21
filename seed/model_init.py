import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", "/data/models"))


def main() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    whisper_path = MODEL_ROOT / "faster-whisper-small"
    if not (whisper_path / "config.json").exists():
        snapshot_download(
            "Systran/faster-whisper-small", local_dir=whisper_path, local_dir_use_symlinks=False
        )


if __name__ == "__main__":
    main()
