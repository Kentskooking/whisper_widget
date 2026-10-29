"""Explicit model download; does not start inference, the microphone, or a server."""
import os

from app.bark_backend import MODEL_FILES, MODEL_ID, MODEL_REVISION


def main():
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL_ID, revision=os.environ.get("BARK_MODEL_REVISION", MODEL_REVISION),
        allow_patterns=MODEL_FILES,
    )
    print(f"Bark Pro downloaded to: {path}")
    print("Silero VAD weights are included with the silero-vad package.")
    print("See the downloaded model's NOTICE and model card for attribution and license terms.")


if __name__ == "__main__":
    main()
